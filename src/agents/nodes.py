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
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

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
from .mcp_tools import query_flights, query_hotels


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
    result = {"city": "", "days": 3, "nights": 2, "people": 2, "budget": 5000, "preference": "休闲观光"}

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

    # 人数
    m = re.search(r"(\d+)\s*个人", text)
    if not m:
        m = re.search(r"(\d+)\s*人", text)
    if m:
        result["people"] = int(m.group(1))
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
        return content
    except FutureTimeout:
        print(f"  [{stage}] ⏰ LLM 调用超时（{LLM_TIMEOUT_SEC}s），触发 HITL")
        raise LLMTimeoutError(f"{stage} 节点 LLM 调用超时 ({LLM_TIMEOUT_SEC}s)")
    except Exception as exc:
        print(f"  [{stage}] ⚠️ LLM 调用失败（{type(exc).__name__}），回退 mock 输出: {exc}")
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
你是 DeepTravel 的旅行需求协调员。请从用户输入中提取并归一化以下关键信息：
1. 目的地（城市/国家）
2. 出行日期（起止）
3. 出行人数
4. 预算范围（总预算或人均）
5. 旅行偏好（亲子/情侣/背包/商务/美食/自然风光 等）
6. 特殊需求（无障碍/宗教禁忌/过敏 等）

以清晰的结构化文本输出，信息缺失时标注 【缺失】。
"""


def coordinator_node(state: dict[str, Any]) -> dict[str, Any]:
    """归一化用户需求 → itinerary"""
    raw = extract_last_user_text(state.get("messages", [])) or "请为我规划一次旅行"
    ctx = _parse_user_input(raw)
    try:
        output = _call_or_mock(COORDINATOR_PROMPT, raw, _mock_coordinator(ctx), "coordinator")
    except LLMTimeoutError:
        return _hitl_on_timeout("coordinator", ROUTE_ITINERARY)
    return {
        "user_request": output,
        "next_node": ROUTE_ITINERARY,
        "sub_reports": [{"agent": "coordinator", "report": output}],
    }


# ==========================================================================
# Itinerary —— 行程规划（可选高德 POI）
# ==========================================================================

ITINERARY_PROMPT = """\
你是 DeepTravel 的行程规划专家。基于归一化需求，设计每日行程：

{user_request}

{poi_context}

要求：
- 每天三段安排（上午/下午/晚上）
- 优先从【高德地图真实 POI 数据】选择景点；POI 为空则自行合理安排
- 景点名称 + 简述 + 大致时长 + 交通便利性
- 标注适合的用餐区域

以 Markdown 格式输出完整日程。
"""


def itinerary_node(state: dict[str, Any]) -> dict[str, Any]:
    """先用高德地图搜索 POI，再生成行程 → budget"""
    raw = extract_last_user_text(state.get("messages", [])) or state.get("user_request", "")
    ctx = _parse_user_input(raw)

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
        output = _call_or_mock(
            "",
            ITINERARY_PROMPT.format(user_request=state.get("user_request", ""), poi_context=pois_text),
            _mock_itinerary(ctx),
            "itinerary",
        )
    except LLMTimeoutError:
        return _hitl_on_timeout("itinerary", ROUTE_BUDGET)

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

## 已规划行程
{itinerary}

{price_context}

类别：交通（往返+当地）/ 住宿 / 餐饮 / 门票活动 / 其他（预留10%）
输出 Markdown 表格 + 与用户预算对比。
"""


def _extract_travel_dates(raw: str) -> tuple[str | None, str | None]:
    """从用户文本里粗粒度提取出发/返程日期"""
    import re
    # 匹配 "2025-10-01" 或 "10月1日" / "10/1"
    date_re = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
    m = date_re.search(raw)
    if m:
        return m.group(0), None
    return None, None


def _flight_price_to_text(data: dict) -> str:
    """把 MCP 返回的航班数据压缩成 prompt 能吃的文本"""
    if not data or not data.get("dep"):
        return ""
    lines = ["### 航班参考价"]
    for f in data["dep"][:3]:
        p = f["price_range"]
        lines.append(
            f"- {f['airline']} {f['flight_no']} {f['dep_time']} "
            f"{f['dep_city']}→{f['arr_city']} "
            f"经济舱 ¥{p['economy_low']}~¥{p['economy_high']} "
            f"(¥{(p['economy_low']+p['economy_high'])//2}均价)"
        )
    if data.get("ret"):
        for f in data["ret"][:2]:
            p = f["price_range"]
            lines.append(
                f"- 返程 {f['airline']} {f['flight_no']} "
                f"经济舱 ¥{p['economy_low']}~¥{p['economy_high']}"
            )
    return "\n".join(lines)


def _hotel_price_to_text(data: dict) -> str:
    """把 MCP 返回的酒店数据压缩成 prompt 能吃的文本"""
    if not data or not data.get("hotels"):
        return ""
    lines = [f"### 酒店参考价（{data['nights']} 晚）"]
    # 按档次聚合
    tier_prices: dict[str, list[int]] = {}
    for h in data["hotels"]:
        tier_prices.setdefault(h["tier"], []).append(h["price_per_night"])
    for tier in ("经济", "舒适", "高档", "豪华"):
        if tier in tier_prices:
            prices = tier_prices[tier]
            avg = sum(prices) // len(prices)
            low, high = min(prices), max(prices)
            lines.append(f"- {tier}档: ¥{low}~¥{high}/晚 (平均 ¥{avg}, {len(prices)}家)")
            lines.append(f"  - {tier}{data['nights']}晚合计 ≈ ¥{avg * data['nights']}/间")
    return "\n".join(lines)


def budget_node(state: dict[str, Any]) -> dict[str, Any]:
    """先查飞猪真实航班/酒店价格，再生成预算 → safety"""
    raw = extract_last_user_text(state.get("messages", [])) or state.get("user_request", "")
    ctx = _parse_user_input(raw)

    # 1. 从 itinerary sub_report 拿 city（和 safety_node 同样的技巧）
    city = ""
    for sub in state.get("sub_reports", []):
        if sub.get("agent") == "itinerary" and sub.get("city"):
            city = sub["city"]; break
    if not city:
        city = extract_city_name(state.get("user_request", ""))

    # 2. MCP 查询（mock 模式或没 city 时跳过）
    price_context = ""
    if not is_mock_mode() and city:
        print(f"  [budget] MCP 查 {city} 航班+酒店...")
        dep_date, ret_date = _extract_travel_dates(raw)
        # 没日期也查酒店，航班用 ctx 里的信息兜底
        if not dep_date and ctx.get("days"):
            import datetime
            dep_date = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
            if ctx["days"] >= 2:
                ret_date = (datetime.date.today() + datetime.timedelta(days=30 + ctx["days"])).isoformat()

        flights = query_flights(city, "上海", dep_date or "2025-10-01", ret_date) if dep_date else None
        hotels = query_hotels(city, dep_date or "2025-10-01", ret_date or "2025-10-03") if dep_date else None

        price_chunks = []
        if flights:
            txt = _flight_price_to_text(flights)
            if txt:
                price_chunks.append(txt)
                print(f"  [budget] 航班 {len(flights['dep'])} 班")
        if hotels:
            txt = _hotel_price_to_text(hotels)
            if txt:
                price_chunks.append(txt)
                print(f"  [budget] 酒店 {len(hotels['hotels'])} 家")

        if price_chunks:
            price_context = "\n## 真实价格参考（飞猪 MCP）\n" + "\n".join(price_chunks)

    try:
        output = _call_or_mock(
            "",
            BUDGET_PROMPT.format(
                user_request=state.get("user_request", ""),
                itinerary=state.get("itinerary", ""),
                price_context=price_context,
            ),
            _mock_budget(ctx),
            "budget",
        )
    except LLMTimeoutError:
        return _hitl_on_timeout("budget", ROUTE_SAFETY)
    return {
        "budget": output,
        "next_node": ROUTE_SAFETY,
        "sub_reports": [{"agent": "budget", "report": output, "city": city}],
    }


# ==========================================================================
# Safety —— 安全评估（可选心知天气）
# ==========================================================================

SAFETY_PROMPT = """\
你是 DeepTravel 的安全顾问。评估行程风险并给出建议：

## 行程概览
{itinerary}

{weather_context}

重点：目的地天气形势 / 治安与自然灾害 / 高风险活动 / 紧急联系方式 / 保险建议
输出简洁清单格式。
"""


def safety_node(state: dict[str, Any]) -> dict[str, Any]:
    """先查心知天气，再生成安全评估 → review"""
    raw = extract_last_user_text(state.get("messages", [])) or state.get("user_request", "")
    ctx = _parse_user_input(raw)

    # 拿到 itinerary 写入的 city
    city = ""
    for sub in state.get("sub_reports", []):
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
            SAFETY_PROMPT.format(itinerary=state.get("itinerary", ""), weather_context=weather_text),
            _mock_safety(ctx),
            "safety",
        )
    except LLMTimeoutError:
        return _hitl_on_timeout("safety", ROUTE_REVIEW)
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


def review_node(state: dict[str, Any]) -> dict[str, Any]:
    """审核三份子报告，决定 approve/revise"""
    revision_count = state.get("revision_count", 0)

    raw = extract_last_user_text(state.get("messages", [])) or ""
    ctx = _parse_user_input(raw)
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

    return {
        "review_status": "approved" if approved else "revise",
        "review_feedback": output,
        "revision_count": new_count,
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
    ctx = _parse_user_input(raw)
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
