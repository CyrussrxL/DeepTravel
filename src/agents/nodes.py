"""
DeepTravel - 所有 Agent 节点实现

Agent 架构：
  coordinator → itinerary → budget → safety → review ─┬─→ integrate → END
                                                       └─→ itinerary（REVISE 回退）

每个节点：
1. 可选调用外部 API（高德地图 POI / 心知天气）
2. 调用 LLM（或 mock 模式下用纯函数生成一致性 mock 输出）
3. 返回 state update + sub_reports 条目

Mock 一致性设计：
  所有节点从同一份 _parse_user_input(raw_input) 提取的 ctx（城市/天数/人数/预算/偏好）
  动态生成输出，避免了早期硬编码常量导致跨节点数据不一致的问题。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .common import (
    get_llm,
    is_mock_mode,
    extract_last_user_text,
    LLM_TIMEOUT_SEC,
    ROUTE_END,
    ROUTE_ITINERARY,
    ROUTE_BUDGET,
    ROUTE_SAFETY,
    ROUTE_REVIEW,
    ROUTE_INTEGRATE,
)
from .tools import search_pois, get_weather, get_weather_daily


# --------- 自定义异常 ---------
class LLMTimeoutError(Exception):
    """LLM 调用超时，触发 HITL"""
    pass


# ==========================================================================
# 城市名表 + 用户输入解析（用于 mock 动态生成）
# ==========================================================================

_CITY_NAMES = [
    "北京", "上海", "天津", "重庆",
    "广州", "深圳", "成都", "杭州", "武汉", "南京", "西安", "长沙",
    "青岛", "厦门", "三亚", "丽江", "大理", "桂林", "张家界", "九寨沟",
    "拉萨", "乌鲁木齐", "昆明", "贵阳", "福州", "合肥", "郑州", "济南",
    "沈阳", "大连", "长春", "哈尔滨", "石家庄", "太原", "呼和浩特",
    "银川", "西宁", "兰州", "南昌", "宁波", "苏州", "无锡", "烟台",
    "珠海", "东莞", "佛山", "中山", "惠州", "汕头",
    "香港", "澳门",
]


def _parse_user_input(text: str) -> dict:
    """从用户原始输入提取 mock 生成所需的字段。"""
    result = {"city": "", "days": 3, "nights": 2, "people": 1, "budget": 5000, "preference": "休闲观光"}

    # 城市
    for city in sorted(_CITY_NAMES, key=len, reverse=True):
        if city in text:
            result["city"] = city
            break
    if not result["city"]:
        m = re.search(r"去([\u4e00-\u9fff]{2,6})(?:旅游|旅行|玩|度假)", text)
        result["city"] = m.group(1) if m else "目的地"

    # 天数/晚数
    m = re.search(r"(\d+)\s*天(\d+)\s*晚", text)
    if m:
        result["days"], result["nights"] = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(\d+)\s*天", text)
        if m:
            d = int(m.group(1))
            result["days"], result["nights"] = d, max(1, d - 1)

    # 人数（阿拉伯数字："2个人/3人"；中文数字："两个人/两人/一家三口"）
    m = re.search(r"(\d+)\s*个人", text)
    if not m:
        m = re.search(r"(\d+)\s*人", text)
    if m:
        result["people"] = int(m.group(1))
    if not m:
        cn_num = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        m = re.search(r"([一二两三四五六七八九十])\s*(?:个)?(?:人|口)", text)
        if m and m.group(1) in cn_num:
            result["people"] = cn_num[m.group(1)]
    if "情侣" in text or "两口" in text:
        result["people"] = 2
    if "一个人" in text or "独自" in text or "单身" in text:
        result["people"] = 1

    # 预算
    m = re.search(r"预算[^\d]*(\d+)", text)
    if not m:
        m = re.search(r"(\d+)\s*元", text)
    if m:
        result["budget"] = int(m.group(1))

    # 偏好
    for kw in ["亲子", "情侣", "背包", "商务", "美食", "自然", "文化", "历史",
               "温泉", "海滩", "度假", "摄影", "登山", "熊猫"]:
        if kw in text:
            result["preference"] = kw
            break

    return result


def _get_ctx(state: dict) -> dict:
    """
    获取 mock 生成所需的结构化信息。

    优先从 coordinator 已存入的 parsed_info 取值（adjust 重跑时保持一致），
    没有才 fallback 从原始 user_request 文本重新解析。
    """
    cached = state.get("parsed_info")
    if cached and isinstance(cached, dict) and cached.get("city"):
        return cached
    # Fallback：从最后一条用户消息解析
    raw = extract_last_user_text(state.get("messages", [])) or state.get("user_request", "")
    return _parse_user_input(raw)


def extract_city_name(text: str) -> str:
    """从归一化需求文本提取城市名。"""
    m = re.search(r"目的地[：:]\s*([^\n【]+)", text)
    target = m.group(1).strip() if m else text
    for city in sorted(_CITY_NAMES, key=len, reverse=True):
        if city in target:
            return city
    for line in text.splitlines():
        if "目的地" in line:
            parts = re.findall(r"[\u4e00-\u9fff]+", line)
            if parts:
                return parts[0]
    return ""


def _city_pinyin(city_name: str) -> str:
    """中国城市名 → 心知天气拼音"""
    mapping = {
        "北京": "beijing", "上海": "shanghai", "天津": "tianjin", "重庆": "chongqing",
        "广州": "guangzhou", "深圳": "shenzhen", "成都": "chengdu", "杭州": "hangzhou",
        "武汉": "wuhan", "南京": "nanjing", "西安": "xian", "长沙": "changsha",
        "青岛": "qingdao", "厦门": "xiamen", "三亚": "sanya", "丽江": "lijiang",
        "大理": "dali", "桂林": "guilin", "昆明": "kunming", "贵阳": "guiyang",
        "福州": "fuzhou", "合肥": "hefei", "郑州": "zhengzhou", "济南": "jinan",
        "沈阳": "shenyang", "大连": "dalian", "长春": "changchun", "哈尔滨": "haerbin",
        "石家庄": "shijiazhuang", "太原": "taiyuan", "呼和浩特": "huhehaote",
        "银川": "yinchuan", "西宁": "xining", "兰州": "lanzhou", "拉萨": "lasa",
        "乌鲁木齐": "wulumuqi", "南昌": "nanchang", "宁波": "ningbo", "苏州": "suzhou",
        "香港": "xianggang", "澳门": "aomen",
    }
    return mapping.get(city_name, "")


# ==========================================================================
# Mock 内容动态生成器
# ==========================================================================

_CITY_POI = {
    "成都": [("熊猫基地", "成都大熊猫繁育研究基地", "熊猫大道1375号"),
             ("宽窄巷子", "宽窄巷子景区", "少城街道金河路口"),
             ("锦里古街", "锦里古街", "武侯祠大街中段"),
             ("武侯祠", "武侯祠博物馆", "武侯祠大街231号"),
             ("杜甫草堂", "杜甫草堂博物馆", "青华路38号"),
             ("都江堰", "都江堰景区", "都江堰市"),
             ("青城山", "青城山风景区", "都江堰市青城山镇"),
             ("春熙路", "春熙路步行街", "锦江区"),
             ("九眼桥", "九眼桥酒吧街", "锦江区九眼桥")],
    "北京": [("故宫", "故宫博物院", "景山前街4号"),
             ("长城", "八达岭长城", "延庆区G6高速58号出口"),
             ("颐和园", "颐和园", "新建宫门路19号"),
             ("天坛", "天坛公园", "永定门内东街中里1号"),
             ("天安门", "天安门广场", "东长安街"),
             ("南锣鼓巷", "南锣鼓巷", "东城区"),
             ("什刹海", "什刹海", "西城区"),
             ("环球影城", "北京环球影城", "通州区")],
    "西安": [("兵马俑", "秦始皇兵马俑博物馆", "临潼区秦陵北路"),
             ("大雁塔", "大雁塔·大慈恩寺", "雁塔区雁祥路1号"),
             ("华清池", "华清宫", "临潼区华清路38号"),
             ("城墙", "西安城墙", "碑林区南大街1号"),
             ("回民街", "回民街", "莲湖区北院门"),
             ("钟鼓楼", "钟鼓楼", "碑林区")],
    "三亚": [("亚龙湾", "亚龙湾国家旅游度假区", "吉阳区"),
             ("天涯海角", "天涯海角游览区", "天涯区"),
             ("蜈支洲岛", "蜈支洲岛", "海棠区"),
             ("南山", "南山文化旅游区", "崖州区"),
             ("大东海", "大东海旅游区", "吉阳区")],
    "杭州": [("西湖", "西湖景区", "西湖区"),
             ("灵隐寺", "灵隐寺", "西湖区灵隐路法云弄1号"),
             ("千岛湖", "千岛湖", "淳安县"),
             ("宋城", "宋城景区", "之江路148号"),
             ("西溪湿地", "西溪国家湿地公园", "西湖区")],
}


def _mock_coordinator(ctx: dict) -> str:
    c, d, n = ctx["city"], ctx["days"], ctx["nights"]
    p, b = ctx["people"], ctx["budget"]
    pref = ctx["preference"]
    label = "情侣" if p == 2 else "独自" if p == 1 else f"{p}人团队"
    return (f"## 归一化旅行需求\n"
            f"- 目的地：{c}\n"
            f"- 出行日期：【用户指定起止】（{d}天{n}晚）\n"
            f"- 出行人数：{p}人（{label}）\n"
            f"- 预算范围：总预算 ¥{b:,}\n"
            f"- 旅行偏好：{pref}\n"
            f"- 特殊需求：【缺失】")


def _mock_itinerary(ctx: dict) -> str:
    c = ctx["city"]
    d = ctx["days"]
    pois = _CITY_POI.get(c, [
        ("景点1", f"{c}热门景点", "市中心"),
        ("景点2", f"{c}特色街区", "老城区"),
        ("景点3", f"{c}博物馆", "文化区"),
    ])
    lines = [f"## {c} {d}天行程（POI 参考）\n"]
    for i in range(d):
        a = pois[i % len(pois)]
        b = pois[(i + 2) % len(pois)]
        lines.append(f"### Day {i+1} · {a[0]} & {b[0]}")
        lines.append(f"- 上午：抵达/入住 → {a[1]}（{a[2]}）→ 约3小时")
        lines.append(f"- 下午：{b[1]} → 约2-3小时")
        lines.append(f"- 晚上：{c}本地美食 + 夜游\n")
    return "\n".join(lines)


def _mock_budget(ctx: dict) -> str:
    b, p, n = ctx["budget"], ctx["people"], ctx["nights"]
    c = ctx["city"]
    rows = [
        f"## 预算估算（¥）\n",
        "| 类别 | 估算（人均） | 估算（"+str(p)+"人合计） |",
        "|------|------------|------------------------|",
        f"| 交通（往返 + 当地） | {int(b*0.15/p):,} | {int(b*0.15):,} |",
        f"| 住宿（{n}晚） | {int(b*0.25/p):,} | {int(b*0.25):,} |",
        f"| 餐饮 | {int(b*0.25/p):,} | {int(b*0.25):,} |",
        f"| 门票/活动 | {int(b*0.15/p):,} | {int(b*0.15):,} |",
        f"| 其他（预留 20%） | {int(b*0.20/p):,} | {int(b*0.20):,} |",
        f"| **合计** | **{int(b/p):,}** | **{b:,}** |\n",
        f"> ✅ 估算在用户预算 ¥{b:,} 范围内，比例合理。",
    ]
    return "\n".join(rows)


def _mock_safety(ctx: dict) -> str:
    c = ctx["city"]
    return (f"## {c} 旅行安全评估\n\n"
            f"### ✅ 整体风险：低\n{c}治安良好，自然灾害概率低。\n\n"
            f"### ⚠️ 关注事项\n"
            f"- 当前季节适合户外活动，建议备好防晒/雨具\n"
            f"- 热门景点节假日人流密集，建议提前预约\n"
            f"- 留意当地天气预警信息\n\n"
            f"### 📋 紧急信息\n"
            f"- 报警：110；急救：120\n"
            f"- 推荐购买：国内旅游保险（含医疗紧急救援）\n\n"
            f"### 💊 个人事项\n"
            f"- 常规常备药（感冒药、肠胃药、创可贴）即可")


def _mock_review(ctx: dict, revision_count: int = 0) -> str:
    """
    mock review 根据 revision_count 返回不同结果（方便测试 HITL）：
    revision_count==0 → APPROVED
    revision_count>=1 → REVISE（触发 revise 回退，两次后超限进 HITL）
    """
    c = ctx["city"]
    if revision_count == 0:
        return f"[APPROVED]\n{c}行程完整覆盖需求，预算合理，安全评估充分。无需修订。"
    else:
        return (f"[REVISE]\n"
                f"{c}行程需要进一步优化：\n"
                f"1. 部分景点安排过于密集，建议增加弹性时间\n"
                f"2. 预算中交通费用估算偏紧，建议补充\n"
                f"请 itinerary 节点重新规划。")


# ==========================================================================
# API 辅助函数
# ==========================================================================

def _pois_to_text(pois: list[dict]) -> str:
    if not pois:
        return ""
    lines = ["【高德地图真实 POI 数据】"]
    for i, p in enumerate(pois, 1):
        lines.append(f"{i}. {p['name']}（{p.get('type','')}）| 地址：{p.get('address','')}")
    return "\n".join(lines)


def _weather_to_text(weather: dict | None, daily: list[dict] | None = None) -> str:
    parts = ["【心知天气真实数据】"]
    if weather:
        parts.append(f"📍 {weather['location']}  当前：{weather['text']}  {weather['temperature']}°C")
    if daily:
        parts.append("📅 未来预报：")
        for d in daily:
            parts.append(f"  {d['date'][:10]}：{d['text_day']}/{d['text_night']}  {d['low']}~{d['high']}°C")
    if not weather and not daily:
        parts.append("（天气数据暂不可用）")
    return "\n".join(parts)


# ==========================================================================
# LLM 调用统一封装
# ==========================================================================

def _call_or_mock(system_prompt: str, user_text: str, mock: str, stage: str) -> str:
    """
    mock 模式直接返回 mock；真实模式调 LLM。

    超时行为：抛 LLMTimeoutError（让调用方捕获并触发 HITL）
    其他错误：回退 mock 输出
    """
    if is_mock_mode():
        return mock

    # 用线程池 + timeout 实现精确超时
    def _llm_task() -> str:
        llm = get_llm()
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_text))
        resp = llm.invoke(messages)
        return getattr(resp, "content", str(resp))

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_llm_task)
            content = future.result(timeout=LLM_TIMEOUT_SEC)
        print(f"  [{stage}] LLM 返回 {len(content)} 字符")
        # 空输出 fallback（某些 flash 模型可能返回空）
        if not content or len(content.strip()) < 10:
            print(f"  [{stage}] ⚠️ LLM 输出过短（{len(content) if content else 0}字符），回退 mock 输出")
            return mock
        return content
    except FutureTimeout:
        print(f"  [{stage}] ⏰ LLM 调用超时（{LLM_TIMEOUT_SEC}s），触发 HITL")
        raise LLMTimeoutError(f"{stage} 节点 LLM 调用超时 ({LLM_TIMEOUT_SEC}s)")
    except Exception as exc:
        # 打全量堆栈：只打一行 exc 的话，网络错误 / 鉴权失败 / 响应解析失败无法区分
        import traceback
        print(f"  [{stage}] ⚠️ LLM 调用失败（{type(exc).__name__}），回退 mock 输出")
        traceback.print_exc()
        return mock


def _hitl_on_timeout(stage: str, next_node: str = "itinerary", reason: str | None = None) -> dict:
    """构造 HITL 状态返回值（节点超时/失败时调用）"""
    return {
        "next_node": "hitl",
        "hitl_status": "waiting",
        "hitl_stage": stage,
        "hitl_reason": reason or f"{stage} 节点 LLM 调用超时 ({LLM_TIMEOUT_SEC}s)",
        "hitl_next_node": next_node,  # approve 后往哪走
    }


# ==========================================================================
# Coordinator —— 需求归一化
# ==========================================================================

COORDINATOR_PROMPT = """\
你是 DeepTravel 的旅行需求协调员。请从用户输入中提取并归一化关键信息，**信息缺失时按以下默认值自动填充**，不要标注【缺失】：

| 字段 | 默认填充规则 |
|------|-------------|
| 出行日期 | 标注"【灵活】"即可 |
| 出行人数 | 未提及默认 **1 人**（独自前往） |
| 预算范围 | 未提及默认 **¥1000-2000/人**（经济型） |
| 旅行偏好 | 未提及默认"休闲观光" |
| 特殊需求 | 未提及填"无特殊需求" |

以清晰的 Markdown 列表输出，示例格式：

```
## 归一化旅行需求
- 目的地：成都
- 出行日期：【灵活】（5天4晚）
- 出行人数：1 人（独自前往）
- 预算范围：总预算 ¥1,500 左右（经济型）
- 旅行偏好：休闲观光
- 特殊需求：无特殊需求
```

**重要：即使用户只说了"去成都旅游5天"这种极简需求，你也要按默认值补全，给下游 Agent 完整的规划输入。**
"""


# ==========================================================================
# Coordinator —— 结构化输出主路径（Pydantic Schema 约束解码）
# ==========================================================================

class ParsedTravelInfo(BaseModel):
    """coordinator 需求归一化的输出 Schema。

    真实 LLM 模式下用 with_structured_output 约束解码，一次调用同时产出
    归一化文本（normalized_request）和结构化字段（parsed_info 的数据源），
    替代「自由文本 + 正则解析」路线。正则解析保留为 fallback。
    """

    normalized_request: str = Field(
        description="归一化后的完整需求文本：包含目的地/天数/人数/预算/日期/偏好，"
                    "缺失项按默认值补全（目的地默认成都、天数默认3、人数默认1、预算默认5000）"
    )
    destination: str = Field(description="目的地城市名，如「成都」")
    days: int = Field(ge=1, le=30, description="行程天数")
    people: int = Field(ge=1, le=20, description="出行人数")
    budget_total: int | None = Field(
        default=None, ge=0, description="总预算（元）；用户未明确提及则为 null"
    )
    start_date: str = Field(
        default="",
        description="出发日期，格式 YYYY-MM-DD；未提及则为空字符串。"
                    "若用户提到了月份但没给具体日期，出发日期必须落在该月份内（不可早于该月 1 日）"
    )
    end_date: str = Field(default="", description="返程日期，格式 YYYY-MM-DD；未提及则为空字符串")
    preference: str = Field(default="休闲观光", description="核心旅行偏好关键词（如 美食/亲子/文化）")
    special_prefs: list[str] = Field(default_factory=list, description="特殊要求列表（如 带老人/不吃辣）")


def _validate_dates(start: str, end: str, days: int) -> tuple[str, str, int, bool]:
    """确定性日期校验（代码级，不依赖 LLM）。

    规则：
    - 双日期都合法（可解析、非过去、start<=end）→ 以日期为准反推 days
    - 否则：起点缺失/过去 → 顺延到明天；终点一律按 days 重排为 start+days-1
    返回 (start, end, days, adjusted)，adjusted=True 表示做过修正。
    """
    from datetime import date, timedelta

    def _parse(s: str):
        try:
            y, m, d = s.strip().split("-")
            return date(int(y), int(m), int(d))
        except Exception:
            return None

    today = date.today()
    adjusted = False
    sd, ed = _parse(start), _parse(end)

    if sd and ed and sd >= today and sd <= ed:
        derived = (ed - sd).days + 1
        if derived != days:
            days = derived
            adjusted = True
        return sd.isoformat(), ed.isoformat(), days, adjusted

    if sd is None or sd < today:
        sd = today + timedelta(days=1)
        adjusted = True
    ed = sd + timedelta(days=days - 1)
    if end.strip() != ed.isoformat():
        adjusted = True
    return sd.isoformat(), ed.isoformat(), days, adjusted


_STRUCTURED_SYS = """\
你是旅行需求归一化器，从用户的口语化需求中抽取结构化字段。规则：
1. 用户提到月份但没给具体日期（如「十月份去」）→ start_date 必须落在该月份内
2. 用户提到相对日期（如「下周六出发」「下个月」）→ 换算成今天之后的具体日期
3. 用户完全没提日期 → start_date 和 end_date 都留空字符串（系统会自动补默认值）
4. 默认值：目的地「成都」、天数 3、人数 1、预算 5000 元；只对用户没提的字段用默认值
5. normalized_request 用简洁中文列点呈现全部归一化结果（含日期，若用户提到或可换算）"""

_TODAY_HINT = "（提示：今天是 {today}）"

_CN_MONTH = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
             "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}


def _enforce_mentioned_month(raw: str, start: str) -> tuple[str, bool]:
    """确定性月份校验：用户提到「N月」但 LLM 给的起点月份不符 → 钉到该月 1 日。

    实测 LLM 对「十月份去大理」仍可能给出 9/30（LLM 屡教不改的典型），
    语义校验交给代码。返回 (start, adjusted)。
    """
    from datetime import date

    m = re.search(r"(\d{1,2})\s*月", raw)
    if not m:
        m = re.search(r"([一二三四五六七八九十]{1,2})\s*月", raw)
        month = _CN_MONTH.get(m.group(1)) if m else None
    else:
        month = int(m.group(1))
    if not month or not (1 <= month <= 12):
        return start, False

    # 钉到「未来的该月 1 日」（今年该月已过则取明年）
    today = date.today()
    target = date(today.year, month, 1)
    if target < today:
        target = date(today.year + 1, month, 1)

    try:
        y, mm, d = start.strip().split("-")
        sd = date(int(y), int(mm), int(d))
    except Exception:
        # 起点缺失/非法（LLM 按规则留空）但用户提到了月份 → 直接用该月 1 日
        return target.isoformat(), True

    if sd.month == month:
        return start, False
    # 起点月份早于提到的月份（LLM 给了 9/30 而用户说十月份）→ 钉到该月 1 日
    if sd < target:
        return target.isoformat(), True
    return start, False


def _coordinator_structured(raw: str) -> tuple[str, dict] | None:
    """结构化输出主路径：一次 LLM 调用同时产出归一化文本 + parsed_info。

    成功返回 (user_request, parsed_info)；任何异常/输出异常由调用方捕获后
    回退旧的「自由文本 + 正则」路径。返回 None 表示结果不可信。
    """
    from datetime import date as _date

    llm = get_llm().with_structured_output(ParsedTravelInfo)
    info = llm.invoke([
        SystemMessage(content=_STRUCTURED_SYS + _TODAY_HINT.format(today=_date.today().isoformat())),
        HumanMessage(content=raw),
    ])
    if not isinstance(info, ParsedTravelInfo):
        return None

    days = max(1, min(30, int(info.days)))
    people = max(1, min(20, int(info.people)))
    # 先做月份钉正（用户提到「N月」而 LLM 起点不符时），再走通用日期校验；
    # 钉正后终点作废，按天数重排，避免旧的 end 把 days 反推出错值
    start = info.start_date or ""
    start, month_adjusted = _enforce_mentioned_month(raw, start)
    end = "" if month_adjusted else (info.end_date or "")
    start, end, days, adjusted = _validate_dates(start, end, days)
    adjusted = adjusted or month_adjusted
    budget = int(info.budget_total) if info.budget_total else 5000

    parsed = {
        "city": (info.destination or "").strip() or "目的地",
        "days": days,
        "nights": max(1, days - 1),
        "people": people,
        "budget": budget,
        "preference": (info.preference or "").strip() or "休闲观光",
        "special_prefs": list(info.special_prefs or []),
        "start_date": start,
        "end_date": end,
        "source": "structured",
    }

    user_req = (info.normalized_request or "").strip()
    if len(user_req) < 10:
        return None
    if adjusted:
        user_req += f"\n\n【日期校验】出行日期：{start} ~ {end}（{days}天）"
    return user_req, parsed


def coordinator_node(state: dict[str, Any]) -> dict[str, Any]:
    """归一化用户需求 → itinerary

    真实 LLM 模式优先走结构化输出（ParsedTravelInfo Schema 约束解码），
    失败时回退旧的「COORDINATOR_PROMPT 自由文本 + 正则解析」路径；
    mock 模式完全不走 LLM，保持原行为。
    """
    raw = extract_last_user_text(state.get("messages", [])) or "请为我规划一次旅行"
    ctx = _get_ctx(state)

    # 主路径：结构化输出（仅真实 LLM 模式）
    if not is_mock_mode():
        try:
            structured = _coordinator_structured(raw)
        except Exception as exc:
            print(f"[coordinator] 结构化输出失败，回退旧路径: {exc}", flush=True)
            structured = None
        if structured is not None:
            user_req, parsed = structured
            return {
                "user_request": user_req,
                "parsed_info": parsed,               # ← 结构化信息（Schema 约束解码产出）
                "next_node": ROUTE_ITINERARY,
                "sub_reports": [{"agent": "coordinator", "report": user_req}],
            }

    # fallback / mock：原路径（自由文本 + 正则）
    ctx["source"] = "regex"
    try:
        output = _call_or_mock(COORDINATOR_PROMPT, raw, _mock_coordinator(ctx), "coordinator")
    except LLMTimeoutError:
        return _hitl_on_timeout("coordinator", ROUTE_ITINERARY)
    return {
        "user_request": output,
        "parsed_info": ctx,                    # ← 存结构化信息！所有节点共享
        "next_node": ROUTE_ITINERARY,
        "sub_reports": [{"agent": "coordinator", "report": output}],
    }


# ==========================================================================
# Itinerary —— 行程规划（可选高德 POI）
# ==========================================================================

ITINERARY_PROMPT = """\
你是 DeepTravel 的行程规划专家。基于归一化需求，设计每日行程：

{user_request}

## 行程日期（唯一权威，禁止编造其他日期）
{dates}

{poi_context}

要求：
- 每天三段安排（上午/下午/晚上）
- 每日标题使用上面的具体日期（如 Day 1 · 10月1日）；若未指定日期，用「第N天」相对表述
- 优先从【高德地图真实 POI 数据】选择景点；POI 为空则自行合理安排
- 景点名称 + 简述 + 大致时长 + 交通便利性
- 标注适合的用餐区域 + 推荐的当地特色美食（参考小红书热门攻略的真实美食推荐）
- 每天的行程要参考小红书高质量攻略的风格：实用、有具体店名/区域、标注"必吃""避坑"等小贴士
- 加入小红书常见的 emoji 风格（🐼🍜✨📸）让行程更生动

以 Markdown 格式输出完整日程。
"""


def itinerary_node(state: dict[str, Any]) -> dict[str, Any]:
    """先用高德地图搜索 POI，再生成行程 → budget"""
    raw = extract_last_user_text(state.get("messages", [])) or state.get("user_request", "")
    ctx = _get_ctx(state)

    # 真实 POI
    city = extract_city_name(state.get("user_request", "")) or ctx["city"]
    ctx["city"] = city
    pois_text = ""
    if city:
        all_pois = []
        for kw in ["景点", "博物馆", "美食"]:
            all_pois.extend(search_pois(kw, city, limit=6))
        seen, uniq = set(), []
        for p in all_pois:
            if p["name"] not in seen:
                seen.add(p["name"]); uniq.append(p)
        pois_text = _pois_to_text(uniq[:12])

    try:
        print(f"  [itinerary] 开始调用... mock_mode={is_mock_mode()}", flush=True)
        output = _call_or_mock(
            "",
            ITINERARY_PROMPT.format(
                user_request=state.get("user_request", ""),
                dates=_trip_dates_context(state),
                poi_context=pois_text,
            ),
            _mock_itinerary(ctx),
            "itinerary",
        )
        print(f"  [itinerary] 返回 len={len(output)}", flush=True)
    except LLMTimeoutError:
        return _hitl_on_timeout("itinerary", "itinerary")

    return {
        "itinerary": output,
        "next_node": ROUTE_BUDGET,
        "sub_reports": [{"agent": "itinerary", "report": output, "city": city}],
    }


# ==========================================================================
# Budget —— 预算估算
# ==========================================================================

BUDGET_PROMPT = """\
你是 DeepTravel 的预算专家。按类别拆分估算预算：

## 归一化需求
{user_request}

## 行程日期
{dates}

## 已规划行程
{itinerary}

请基于目的地消费水平、行程天数和用户总预算，按类别拆分（交通/住宿/餐饮/门票活动/其他预留），输出 Markdown 表格 + 与用户预算对比 + 合理性说明。
预算中的日期必须与上面的行程日期一致；若未指定日期，用「第N天」相对表述，不要编造具体日期。
"""


def _trip_dates_context(state: dict[str, Any]) -> str:
    """从 parsed_info 取行程日期给 budget/safety prompt 注入。

    主图三路并行时 budget/safety 读不到行程输出，LLM 会自己编日期
    （与行程日期不一致 → review 抓跨报告不一致）。coordinator 的结构化
    日期是唯一权威来源；无日期时显式告知用相对表述。
    """
    parsed = state.get("parsed_info") or {}
    sd, ed = parsed.get("start_date") or "", parsed.get("end_date") or ""
    if sd and ed:
        return f"{sd} 至 {ed}"
    return "用户未指定具体日期（请用「第N天」相对表述，不要编造具体日期）"


def _budget_fallback_banner(reason: str) -> str:
    """兜底数据显式标注：前端 Markdown 里以 blockquote 警示条呈现。"""
    return f"> ⚠️ **本段为估算兜底数据**（{reason}），由系统按预算比例生成，仅供参考\n\n"


def _cell_amount(cell: str) -> float | None:
    """取单元格里的金额（第一个数值，跳过括号里的补充数字）。"""
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", cell)
    return float(nums[0].replace(",", "")) if nums else None


def _fmt_amount(v: float, bold: bool) -> str:
    s = f"{int(round(v)):,}"
    return f"**{s}**" if bold else s


_BUDGET_ROW_RE = re.compile(r"^\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|\s*$")
# 预算类别关键词：active 表内只处理这些行，防止误改无关表格
_BUDGET_CAT_KW = ("交通", "住宿", "餐饮", "门票", "活动", "其他", "预留", "购物", "娱乐")


def _fix_budget_arithmetics(report: str, people: int) -> str:
    """代码级算术校验：合计 = 人均 × 人数，LLM 算错直接覆盖（不重试）。

    只处理**预算拆分表**：表头含「人均/单人」+「合计/总计」的 3 列 Markdown 表
    （类别 | 人均 | N人合计）。其他 3 列表（如「项目|金额|状态」对比表）、
    4+ 列表一律不碰，避免破坏性误改。
    - 类别行：合计列 = 人均 × people（±1 元容差，容忍取整）
    - 合计行：人均 = Σ类别行人均，合计 = 人均合计 × people
    有修正时在报告末尾追加一行来源说明（前端可见）。
    仅真实 LLM 模式调用（mock 表由代码生成，天然自洽）。
    """
    if not report or "|" not in report or people < 1:
        return report
    lines = report.splitlines()
    per_head_total = 0.0
    changed = False
    in_budget_table = False
    for i, line in enumerate(lines):
        m = _BUDGET_ROW_RE.match(line)
        if not m:
            in_budget_table = False  # 表格中断（空行/文本/其他宽度表格）
            continue
        c0, c1, c2 = (c.strip() for c in m.groups())
        head = c0.replace("*", "")
        bold1, bold2 = "**" in c1, "**" in c2

        if not in_budget_table:
            # 表头检测：人均/单人 + 合计/总计
            if ("人均" in c1 or "单人" in c1) and ("合计" in c2 or "总计" in c2):
                in_budget_table = True
            continue

        if "合计" in head or "总计" in head:
            # 合计行：按类别行人均之和重算两列
            if per_head_total > 0:
                v1, v2 = _cell_amount(c1), _cell_amount(c2)
                if (v1 is None or abs(v1 - per_head_total) > 1
                        or v2 is None or abs(v2 - per_head_total * people) > 1):
                    lines[i] = f"| {c0} | {_fmt_amount(per_head_total, bold1)} | {_fmt_amount(per_head_total * people, bold2)} |"
                    changed = True
            continue

        # 类别行：c0 必须命中预算类别关键词，防误改
        if not any(k in head for k in _BUDGET_CAT_KW):
            continue
        v1, v2 = _cell_amount(c1), _cell_amount(c2)
        if v1 is None:
            continue
        per_head_total += v1
        if v2 is None or abs(v2 - v1 * people) > 1:
            lines[i] = f"| {c0} | {c1} | {_fmt_amount(v1 * people, bold2)} |"
            changed = True
    if not changed:
        return report  # 无修正时保留原文（splitlines/join 会丢尾部换行）
    lines.append(f"\n> 🔧 合计列已按 {people} 人重算（LLM 原始算术有误，系统自动修正）")
    return "\n".join(lines)


def budget_node(state: dict[str, Any]) -> dict[str, Any]:
    """让 LLM 基于目的地消费水平估算预算 → safety

    真实模式两道确定性防线：
    1. 兜底标注：LLM 调用失败/输出异常/缺类别 → 注入 mock 表并显式加 ⚠️ 标注
    2. 算术校验：合计 = 人均 × 人数（people 取自 parsed_info），算错直接覆盖
    """
    raw = extract_last_user_text(state.get("messages", [])) or state.get("user_request", "")
    ctx = _get_ctx(state)
    mock_out = _mock_budget(ctx)

    try:
        print(f"  [budget] 开始调用... mock_mode={is_mock_mode()}", flush=True)
        output = _call_or_mock(
            "",
            BUDGET_PROMPT.format(
                user_request=state.get("user_request", ""),
                dates=_trip_dates_context(state),
                itinerary=state.get("itinerary", ""),
            ),
            mock_out,
            "budget",
        )
        print(f"  [budget] 返回 len={len(output)}", flush=True)

        if not is_mock_mode():
            # 防线 0：_call_or_mock 内部回退了 mock（空输出/调用失败）→ 标注
            if output == mock_out:
                output = _budget_fallback_banner("LLM 调用失败或输出异常") + mock_out

            # 防线 1：类别完整性 —— 缺任何一类就 fallback + 标注
            required = ["交通", "住宿", "餐饮", "门票"]
            missing = [k for k in required if k not in output]
            if missing:
                print(f"  [budget] ⚠️ LLM 输出缺少: {missing}，使用 mock 完整预算表")
                output = _budget_fallback_banner(f"LLM 输出缺失类别：{'、'.join(missing)}") + mock_out

            # 防线 2：算术校验 —— 总计 = 人均 × 人数，算错直接覆盖
            output = _fix_budget_arithmetics(output, int(ctx.get("people") or 1))
    except LLMTimeoutError:
        return _hitl_on_timeout("budget", "itinerary")
    return {
        "budget": output,
        "next_node": ROUTE_SAFETY,
        "sub_reports": [{"agent": "budget", "report": output}],
    }


# ==========================================================================
# Safety —— 安全评估（可选心知天气）
# ==========================================================================

SAFETY_PROMPT = """\
你是 DeepTravel 的安全顾问。评估行程风险并给出建议：

## 行程日期
{dates}

## 行程概览
{itinerary}

{weather_context}

重点：目的地天气形势 / 治安与自然灾害 / 高风险活动 / 紧急联系方式 / 保险建议
输出简洁清单格式。日期表述必须与上面的行程日期一致；若未指定日期，用「第N天」相对表述，不要编造具体日期。
"""


def safety_node(state: dict[str, Any]) -> dict[str, Any]:
    """先查心知天气，再生成安全评估 → review"""
    raw = extract_last_user_text(state.get("messages", [])) or state.get("user_request", "")
    ctx = _get_ctx(state)

    # 拿到 itinerary 写入的 city —— 从后往前找最新记录
    # （增量调整会 append 新 itinerary 报告，从头找会命中旧城市的记录）
    city = ""
    for sub in reversed(state.get("sub_reports", [])):
        if sub.get("agent") == "itinerary" and sub.get("city"):
            city = sub["city"]; break
    if not city:
        city = extract_city_name(state.get("user_request", ""))
    if city:
        ctx["city"] = city

    # 天气
    weather_text = ""
    if city:
        pinyin = _city_pinyin(city)
        if pinyin:
            print(f"  [safety] 心知天气查询 {city}({pinyin})...")
            weather = get_weather(pinyin)
            daily = get_weather_daily(pinyin, days=3)
            weather_text = _weather_to_text(weather, daily)
            print(f"  [safety] {weather['text']} {weather['temperature']}°C" if weather else "  [safety] 天气查询失败")

    try:
        output = _call_or_mock(
            "",
            SAFETY_PROMPT.format(
                dates=_trip_dates_context(state),
                itinerary=state.get("itinerary", ""),
                weather_context=weather_text,
            ),
            _mock_safety(ctx),
            "safety",
        )
    except LLMTimeoutError:
        return _hitl_on_timeout("safety", "itinerary")
    return {
        "safety_report": output,
        "next_node": ROUTE_REVIEW,
        "sub_reports": [{"agent": "safety", "report": output}],
    }


# ==========================================================================
# Review —— 方案审核
# ==========================================================================

REVIEW_PROMPT = """\
你是 DeepTravel 的方案审核专家。综合评审三份子报告：

## 归一化需求
{user_request}

## 行程报告
{itinerary}

## 预算报告
{budget}

## 安全报告
{safety_report}

检查：
1. 行程是否覆盖日期、偏好、特殊需求？
2. 预算是否可接受？
3. 安全风险是否充分提示？

**输出：**
先给出 [APPROVED] 或 [REVISE] 标签；然后简短说明理由。
如果 [REVISE]，明确指出 itinerary 哪些部分需要修改。
"""

MAX_REVISION = 2

# 评审门控阈值（代码级，不依赖 LLM 自由心证）
SCORE_PASS = 7    # 放行阈值：verdict=approved 且四维均 ≥7 才真正 approved
SCORE_HITL = 5    # HITL 阈值：任一维 <5 直接进人工审核


class ReviewScores(BaseModel):
    """四维评分 0-10"""
    completeness: int = Field(ge=0, le=10, description="完整性：行程是否覆盖全部天数、偏好与特殊需求")
    feasibility: int = Field(ge=0, le=10, description="可行性：每日节奏、交通衔接、时间安排是否现实")
    consistency: int = Field(ge=0, le=10, description="一致性：行程/预算/安全三份报告是否相互自洽")
    budget_fit: int = Field(ge=0, le=10, description="预算匹配：预算拆分是否贴合用户总预算、类别比例是否合理")


class ReviewVerdict(BaseModel):
    """review 结构化评审输出 Schema"""
    scores: ReviewScores
    verdict: Literal["approved", "revise"]
    needs_human: bool = Field(
        default=False,
        description="是否存在你无法替用户决定、必须由用户决策的事项"
                    "（如需求互相矛盾、预算与需求严重冲突）；一般情况为 false",
    )
    reasons: list[str] = Field(default_factory=list, description="各维度评分理由")
    revision_suggestions: str = Field(default="", description="revise 时的具体修改建议")


_REVIEW_SYS = """\
你是旅行方案审核专家。对三份子报告做四维评分（0-10）并给出裁定。

评分维度：
- completeness 完整性：行程是否覆盖全部天数、偏好与特殊需求
- feasibility 可行性：每日节奏、交通衔接、时间安排是否现实
- consistency 一致性：行程/预算/安全三份报告是否相互自洽（天数/城市/消费水平）
- budget_fit 预算匹配：预算拆分是否贴合用户总预算、类别比例是否合理

裁定规则：
- 四维均 ≥7 且无硬伤 → verdict=approved
- 任一维 <7 或存在明显问题 → verdict=revise，并在 revision_suggestions 给出具体修改建议
- 只有存在你无法替用户决定的事项（需求矛盾/预算严重冲突）才 needs_human=true，否则 false

【系统交叉校验结果】（代码级确定性校验，若有问题必须体现在评分中）：
{findings}"""


_CN_NUM = _CN_MONTH  # 一~十二 的中文数字映射，与月份共用


def _cross_check(state: dict) -> list[str]:
    """确定性交叉校验（代码级，不花 LLM token，100% 准）：

    1) 预算类别合计 vs 用户总预算（±5% 容差）
    2) 行程 Day 数 vs parsed_info.days

    返回问题列表；空列表 = 全部通过（或报告里没有可校验的结构）。
    """
    findings: list[str] = []
    parsed = state.get("parsed_info") or {}
    budget_text = state.get("budget", "") or ""
    itinerary_text = state.get("itinerary", "") or ""

    # 1) 预算：Markdown 表格行最后一个数值列（合计列）求和
    target = parsed.get("budget")
    if target and budget_text:
        try:
            target_f = float(target)
            cat_total, has_row = 0.0, False
            for cells in re.findall(r"^\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|\s*$", budget_text, re.M):
                if "合计" in cells[0] or "总计" in cells[0]:
                    continue
                nums = re.findall(r"\d[\d,]*(?:\.\d+)?", cells[-1])
                if nums:
                    cat_total += float(nums[-1].replace(",", ""))
                    has_row = True
            if has_row and abs(cat_total - target_f) / target_f > 0.05:
                findings.append(
                    f"预算校验失败：类别合计 ¥{cat_total:,.0f} 与用户总预算 ¥{target_f:,.0f} 偏差超过 5%"
                )
        except (TypeError, ValueError):
            pass

    # 2) 天数：统计行程中 Day N / 第N天 / 第N天(中文) 的不同天数
    days = parsed.get("days")
    if days and itinerary_text:
        found = set()
        for m in re.finditer(r"[Dd]ay\s*(\d+)", itinerary_text):
            found.add(int(m.group(1)))
        for m in re.finditer(r"第\s*(\d+)\s*天", itinerary_text):
            found.add(int(m.group(1)))
        for m in re.finditer(r"第([一二三四五六七八九十])天", itinerary_text.replace(" ", "")):
            found.add(_CN_NUM[m.group(1)])
        if found and len(found) != days:
            findings.append(
                f"天数校验失败：行程覆盖 {len(found)} 天（{sorted(found)}），与需求 {days} 天不符"
            )

    # 3) 日期区间：报告中的具体日期必须落在 parsed_info.start_date ~ end_date 内
    #    抓 LLM 幻觉日期（如 2 天行程正文里冒出「10月3日」）。月份无年份的表述
    #    按 start 年份（及跨年 +1）组合后再比较。宁可误抓（如「10月1日-7日人多」
    #    这类背景描述），由 HITL 人工复核——比漏抓安全。
    sd_raw, ed_raw = parsed.get("start_date") or "", parsed.get("end_date") or ""
    safety_text = state.get("safety", "") or ""
    if sd_raw and ed_raw:
        try:
            from datetime import date as _date

            sd = _date.fromisoformat(sd_raw)
            ed = _date.fromisoformat(ed_raw)
        except ValueError:
            sd = ed = None  # type: ignore[assignment]
        if sd and ed:
            offenders: set[str] = set()
            for text, label in (
                (itinerary_text, "行程"),
                (budget_text, "预算"),
                (safety_text, "安全"),
            ):
                if not text:
                    continue
                # 完整日期 YYYY-MM-DD
                for m in re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})", text):
                    try:
                        d = _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    except ValueError:
                        continue
                    if d < sd or d > ed:
                        offenders.add(f"{label}报告 {d.isoformat()}")
                # 中文日期 N月N日（无年份，试 start 年份与跨年 +1）
                for m in re.finditer(r"(\d{1,2})月(\d{1,2})日", text):
                    mo, dy = int(m.group(1)), int(m.group(2))
                    ok = False
                    for yr in (sd.year, sd.year + 1):
                        try:
                            d = _date(yr, mo, dy)
                        except ValueError:
                            continue
                        if sd <= d <= ed:
                            ok = True
                            break
                    if not ok:
                        offenders.add(f"{label}报告 {mo}月{dy}日")
            if offenders:
                findings.append(
                    f"日期区间校验失败：{'、'.join(sorted(offenders)[:5])} 不在行程区间 {sd_raw}~{ed_raw} 内"
                )

    return findings


def _review_structured(state: dict, findings: list[str]) -> ReviewVerdict | None:
    """结构化评审主路径：四维评分 + verdict + needs_human。

    任何异常由调用方捕获后回退旧路径（[APPROVED]/[REVISE] 标签 + 关键词触发）。
    """
    findings_text = "\n".join(f"- {f}" for f in findings) if findings else "- 全部通过，未发现问题"
    reports = (
        f"## 归一化需求\n{state.get('user_request', '')}\n\n"
        f"## 行程报告\n{(state.get('itinerary', '') or '')[:3000]}\n\n"
        f"## 预算报告\n{(state.get('budget', '') or '')[:1500]}\n\n"
        f"## 安全报告\n{(state.get('safety_report', '') or '')[:1500]}"
    )
    llm = get_llm().with_structured_output(ReviewVerdict)
    verdict = llm.invoke([
        SystemMessage(content=_REVIEW_SYS.format(findings=findings_text)),
        HumanMessage(content=reports),
    ])
    return verdict if isinstance(verdict, ReviewVerdict) else None


def review_node(state: dict[str, Any]) -> dict[str, Any]:
    """审核三份子报告，决定 approve/revise。

    真实 LLM 模式走结构化评审（四维评分 + needs_human），
    代码级门控：approved 需 verdict=approved 且四维均 ≥SCORE_PASS 且无交叉校验问题；
    needs_human 或任一维 <SCORE_HITL → hitl_flag=True 直接进人工审核；
    交叉校验发现问题 → consistency 顶格降到 6（必然 revise，走自动修订而非直接 HITL）。
    """
    revision_count = state.get("revision_count", 0)

    raw = extract_last_user_text(state.get("messages", [])) or ""
    ctx = _get_ctx(state)

    # 确定性交叉校验（不花 token）
    findings = _cross_check(state)

    # 主路径：结构化评审（仅真实 LLM 模式）
    verdict = None
    if not is_mock_mode():
        try:
            verdict = _review_structured(state, findings)
        except Exception as exc:
            print(f"[review] 结构化评审失败，回退旧路径: {exc}", flush=True)
            verdict = None

    if verdict is not None:
        s = verdict.scores
        scores = {
            "completeness": s.completeness,
            "feasibility": s.feasibility,
            "consistency": s.consistency,
            "budget_fit": s.budget_fit,
        }
        # 事实问题 → consistency 顶格 6（必然 revise；两轮改不好自然进 HITL）
        if findings:
            scores["consistency"] = min(scores["consistency"], 6)
        min_score = min(scores.values())
        needs_hitl = bool(verdict.needs_human) or min_score < SCORE_HITL
        approved = (verdict.verdict == "approved"
                    and min_score >= SCORE_PASS
                    and not findings)

        # 组装可读反馈（保留 [APPROVED]/[REVISE] 标签，兼容旧消费方）
        tag = "[APPROVED]" if approved else "[REVISE]"
        fb_parts = [
            tag,
            f"评分：完整性 {scores['completeness']} | 可行性 {scores['feasibility']} | "
            f"一致性 {scores['consistency']} | 预算匹配 {scores['budget_fit']}",
        ]
        if findings:
            fb_parts.append("交叉校验：" + "；".join(findings))
        if verdict.reasons:
            fb_parts.append("理由：" + "；".join(verdict.reasons))
        if not approved and verdict.revision_suggestions:
            fb_parts.append("修改建议：" + verdict.revision_suggestions)
        output = "\n".join(fb_parts)

        new_count = revision_count if approved else revision_count + 1
        return {
            "review_status": "approved" if approved else "revise",
            "review_feedback": output,
            "review_scores": scores,
            "revision_count": new_count,
            "hitl_flag": needs_hitl,
            # hitl_flag=True 时由 hitl_node 透传给前端
            "hitl_reason": (f"审核需人工决策（最低分 {min_score}）："
                            + "；".join(verdict.reasons)[:100]) if needs_hitl else "",
            "next_node": ROUTE_INTEGRATE if approved else ROUTE_ITINERARY,
            "sub_reports": [{"agent": "review", "report": output}],
        }

    # fallback / mock：原路径（标签判定 + revision_count 驱动）
    try:
        output = _call_or_mock(
            "",
            REVIEW_PROMPT.format(
                user_request=state.get("user_request", ""),
                itinerary=state.get("itinerary", ""),
                budget=state.get("budget", ""),
                safety_report=state.get("safety_report", ""),
            ),
            _mock_review(ctx, revision_count),
            "review",
        )
    except LLMTimeoutError:
        return _hitl_on_timeout("review", ROUTE_INTEGRATE)

    approved = "[APPROVED]" in output.upper()
    new_count = revision_count if approved else revision_count + 1

    # mock 模式给展示用假评分（门控不消费）；真实模式 fallback 不给评分
    # （review_scores 为空时 hitl_gate 才启用关键词兜底，这里不能伪造非空）
    mock_scores = (
        {"completeness": 8, "feasibility": 8, "consistency": 8, "budget_fit": 8} if approved
        else {"completeness": 6, "feasibility": 6, "consistency": 6, "budget_fit": 6}
    ) if is_mock_mode() else {}

    return {
        "review_status": "approved" if approved else "revise",
        "review_feedback": output,
        "review_scores": mock_scores,
        "revision_count": new_count,
        "hitl_flag": False,
        "next_node": ROUTE_INTEGRATE if approved else ROUTE_ITINERARY,
        "sub_reports": [{"agent": "review", "report": output}],
    }


# ==========================================================================
# Integrate —— 整合最终方案
# ==========================================================================

INTEGRATE_PROMPT = """\
你是 DeepTravel 的方案整合专家。将以下报告整合成面向用户的完整方案：

## 1. 归一化需求
{user_request}

## 2. 行程规划
{itinerary}

## 3. 预算估算
{budget}

## 4. 安全提示
{safety_report}

## 5. 审核意见
{review_feedback}

要求：开头一句话总结 + 各章节 Markdown 分隔 + 结尾方案摘要卡片。
"""


def integrate_node(state: dict[str, Any]) -> dict[str, Any]:
    """整合所有子报告，输出最终方案 → END"""
    raw = extract_last_user_text(state.get("messages", [])) or ""
    ctx = _get_ctx(state)
    c, d, n = ctx["city"], ctx["days"], ctx["nights"]
    p, b, pref = ctx["people"], ctx["budget"], ctx["preference"]
    rev = state.get("revision_count", 0)

    # 动态拼接（各章节已来自真实上游节点）
    mock_integrated = f"""# 🗺️ DeepTravel 旅行方案

**目的地：** {c}  |  **天数：** {d}天{n}晚  |  **人数：** {p}人

---

## 📋 归一化需求

{state.get('user_request', '')}

---

## 📅 详细行程

{state.get('itinerary', '')}

---

## 💰 预算估算

{state.get('budget', '')}

---

## 🛡️ 安全提示

{state.get('safety_report', '')}

---

## ✅ 审核意见

{state.get('review_feedback', '')}

---

## 📑 方案摘要卡片

| 项目 | 内容 |
|------|------|
| 目的地 | {c} |
| 日期 | 【用户指定】{d}天{n}晚 |
| 人数 | {p}人 |
| 偏好 | {pref} |
| 预算 | ¥{b:,} |
| 修订次数 | {rev} |"""

    try:
        output = _call_or_mock(
            "",
            INTEGRATE_PROMPT.format(
                user_request=state.get("user_request", ""),
                itinerary=state.get("itinerary", ""),
                budget=state.get("budget", ""),
                safety_report=state.get("safety_report", ""),
                review_feedback=state.get("review_feedback", ""),
            ),
            mock_integrated,
            "integrate",
        )
    except LLMTimeoutError:
        return _hitl_on_timeout("integrate", ROUTE_END)

    return {
        "final_plan": output,
        "next_node": ROUTE_END,
        "sub_reports": [{"agent": "integrate", "report": "方案整合完成"}],
    }

