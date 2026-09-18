"""
DeepTravel 六阶段 Agent 节点

每个节点职责：
1. 读取 state（主要是 messages + user_request）
2. 调用 LLM（或 mock）产出结构化子报告
3. 返回 partial state：写入对应报告字段 + sub_reports + next_node

设计原则：
- 节点是纯函数风格：输入 state dict，输出 partial state dict
- 节点绝不直接路由，只写 next_node，由 graph.py 中的 route_decision 统一调度
- System Prompt 约束输出格式，LLM 编造时 mock 兜底
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from .common import (
    get_llm,
    is_mock_mode,
    extract_last_user_text,
    ROUTE_ITINERARY,
    ROUTE_BUDGET,
    ROUTE_SAFETY,
    ROUTE_REVIEW,
    ROUTE_INTEGRATE,
    ROUTE_END,
)
from .tools import search_pois, get_weather, get_weather_daily


# 中国主要城市名表（用于从目的地字符串中提取纯城市名）
# 覆盖直辖市 + 省会 + 热门旅游城市
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


def extract_city_name(text: str) -> str:
    """
    从归一化需求文本中提取目的地城市名。

    匹配 "目的地：XXX" 或直接在全文中搜索已知城市名。
    先匹配直辖市/省会/热门城市（优先长名如"呼和浩特"再短名）。
    """
    # 先看是否有 "目的地：" 这样的行
    m = re.search(r"目的地[：:]\s*([^\n【]+)", text)
    if m:
        target = m.group(1).strip()
    else:
        target = text

    # 关键词匹配（长名优先，避免 "呼和" 先匹配了 "呼和浩特"）
    for city in sorted(_CITY_NAMES, key=len, reverse=True):
        if city in target:
            return city
    # 兜底：取 "目的地" 行第一个中文词组
    for line in text.splitlines():
        if "目的地" in line:
            parts = re.findall(r"[\u4e00-\u9fff]+", line)
            if parts:
                return parts[0]
    return ""


def _pois_to_text(pois: list[dict]) -> str:
    """把高德 POI 列表格式化为可读文本，供 LLM 拼入 prompt"""
    if not pois:
        return ""
    lines = ["【高德地图真实 POI 数据】"]
    for i, p in enumerate(pois, 1):
        addr = p.get("address", "")
        ptype = p.get("type", "")
        lines.append(f"{i}. {p['name']}（{ptype}）| 地址：{addr}")
    return "\n".join(lines)


def _weather_to_text(weather: dict | None, daily: list[dict] | None = None) -> str:
    """把心知天气数据格式化为可读文本"""
    parts = ["【心知天气真实数据】"]
    if weather:
        parts.append(f"📍 {weather['location']}（{weather.get('country','')}）")
        parts.append(f"🌤 当前：{weather['text']}  {weather['temperature']}°C  湿度{weather['humidity']}%")
        if weather.get("wind_direction"):
            parts.append(f"💨 风：{weather['wind_direction']} {weather['wind_speed']}km/h")
    if daily:
        parts.append("📅 未来预报：")
        for d in daily:
            parts.append(f"  {d['date'][:10]}：{d['text_day']}/{d['text_night']}  {d['low']}~{d['high']}°C")
    if not weather and not daily:
        parts.append("（天气数据暂不可用，API 未返回有效结果）")
    return "\n".join(parts)


# ==========================================================================
# Coordinator —— 需求归一化 & 任务拆解
# ==========================================================================

COORDINATOR_PROMPT = """\
你是 DeepTravel 的旅行需求协调员。请从用户输入中提取并归一化以下关键信息：

1. 目的地（城市/国家）
2. 出行日期（起止）
3. 出行人数
4. 预算范围（总预算或人均）
5. 旅行偏好（如：亲子、情侣、背包、商务、美食、自然风光等）
6. 特殊需求（如：无障碍、宗教禁忌、过敏等）

请以清晰的结构化文本输出归一化后的需求摘要。如果用户信息有缺失，请在输出中标注 "【缺失】"。
"""

MOCK_COORDINATOR_OUTPUT = """\
## 归一化旅行需求
- 目的地：日本东京
- 出行日期：2026-11-01 至 2026-11-07（7天6晚）
- 出行人数：2人（情侣）
- 预算范围：总预算 ¥25,000
- 旅行偏好：自然风光 + 美食 + 温泉
- 特殊需求：【缺失】
"""


def coordinator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Coordinator：归一化用户需求，设置 next_node='itinerary'"""
    user_text = extract_last_user_text(state.get("messages", []))

    output = _call_or_mock(
        system_prompt=COORDINATOR_PROMPT,
        user_text=user_text or "请为我规划一次旅行",
        mock=MOCK_COORDINATOR_OUTPUT,
        stage="coordinator",
    )

    return {
        "user_request": output,
        "next_node": ROUTE_ITINERARY,
        "sub_reports": [{"agent": "coordinator", "report": output}],
    }


# ==========================================================================
# Itinerary —— 行程规划
# ==========================================================================

ITINERARY_PROMPT = """\
你是 DeepTravel 的行程规划专家。基于以下归一化需求，设计一份详细的每日行程：

{user_request}

{poi_context}

要求：
1. 每天包含：上午 / 下午 / 晚上 三段安排
2. 优先从上面的【高德地图真实 POI 数据】中选择景点；如果 POI 为空则自行合理安排
3. 每个景点/活动要具体（名称 + 简述 + 大致时长）
4. 考虑景点间的距离和交通便利性
5. 标注适合的用餐区域

以 Markdown 格式输出完整日程。
"""

MOCK_ITINERARY_OUTPUT = """\
## 成都 3天2晚 行程（真实 POI 参考）

### Day 1 · 熊猫 & 宽窄巷子
- 上午：抵达成都 → 入住春熙路附近酒店
- 下午：成都大熊猫繁育研究基地（熊猫大道1375号，国家级景点） → 约3小时
- 晚上：宽窄巷子景区（少城街道金河路口，特色商业街）晚餐 + 夜游

### Day 2 · 锦里 & 武侯祠
- 上午：武侯祠博物馆 → 锦里古街
- 下午：杜甫草堂博物馆
- 晚上：九眼桥酒吧街 / 合江亭

### Day 3 · 青城山 / 都江堰（二选一）& 返程
- 上午：都江堰景区（国家5A）或青城山一日游
- 下午：返程回市区 → 机场
"""


def itinerary_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Itinerary：先用高德地图搜索目的地 POI，再让 LLM 生成每日行程。

    真实 API 调用失败时（如海外城市权限不足），POI 列表为空，
    LLM 会退化为纯文本规划（prompt 里有说明）。
    """
    user_request = state.get("user_request", "")
    city = extract_city_name(user_request)

    # 真实 POI 搜索（同时搜景点/美食/地标三个关键词）
    pois_text = ""
    if city:
        print(f"  [itinerary] 正在用高德地图搜索 {city} 的 POI...")
        all_pois = []
        for kw in ["景点", "博物馆", "美食"]:
            all_pois.extend(search_pois(kw, city, limit=6))
        # 去重（按 name）
        seen = set()
        unique = []
        for p in all_pois:
            if p["name"] not in seen:
                seen.add(p["name"])
                unique.append(p)
        pois_text = _pois_to_text(unique[:12])
        if unique:
            print(f"  [itinerary] 搜到 {len(unique)} 个真实 POI")
        else:
            print(f"  [itinerary] 未搜到 POI（可能是海外城市或 API 权限问题）")

    output = _call_or_mock(
        system_prompt="",
        user_text=ITINERARY_PROMPT.format(
            user_request=user_request,
            poi_context=pois_text,
        ),
        mock=MOCK_ITINERARY_OUTPUT,
        stage="itinerary",
    )

    return {
        "itinerary": output,
        "next_node": ROUTE_BUDGET,
        "sub_reports": [{"agent": "itinerary", "report": output, "city": city}],
    }


# ==========================================================================
# Budget —— 预算估算
# ==========================================================================

BUDGET_PROMPT = """\
你是 DeepTravel 的预算专家。基于以下行程，按类别拆分估算总预算：

## 归一化需求
{user_request}

## 已规划行程
{itinerary}

请按以下类别估算（人民币）：
1. 交通（国际机票 + 当地交通）
2. 住宿（每晚均价 × 晚数）
3. 餐饮（每天人均 × 天数 × 人数）
4. 门票/活动
5. 其他（购物、应急预留 10%）

输出 Markdown 表格形式，并给出总预算和与用户预算的对比。
"""

MOCK_BUDGET_OUTPUT = """\
## 预算估算（¥）

| 类别 | 估算（人均） | 估算（2人） |
|------|------------|------------|
| 国际机票（上海⇄东京） | 3,500 | 7,000 |
| 当地交通（JR Pass + 地铁） | 1,200 | 2,400 |
| 住宿（6晚 × ¥1,200/晚） | 3,600 | 7,200 |
| 餐饮（7天 × ¥400/天） | 2,800 | 5,600 |
| 门票/活动 | 800 | 1,600 |
| 其他（预留 10%） | 1,190 | 2,380 |
| **合计** | **13,090** | **26,180** |

> 用户预算 ¥25,000，超出约 ¥1,180（+4.7%），可考虑调整住宿档次或减少一晚温泉旅馆。
"""


def budget_node(state: dict[str, Any]) -> dict[str, Any]:
    """Budget：预算估算，设置 next_node='safety'"""
    output = _call_or_mock(
        system_prompt="",
        user_text=BUDGET_PROMPT.format(
            user_request=state.get("user_request", ""),
            itinerary=state.get("itinerary", ""),
        ),
        mock=MOCK_BUDGET_OUTPUT,
        stage="budget",
    )

    return {
        "budget": output,
        "next_node": ROUTE_SAFETY,
        "sub_reports": [{"agent": "budget", "report": output}],
    }


# ==========================================================================
# Safety —— 安全评估
# ==========================================================================

SAFETY_PROMPT = """\
你是 DeepTravel 的旅行安全顾问。请评估以下行程的安全风险点并给出建议：

## 行程概览
{itinerary}

{weather_context}

请结合上面的天气真实数据重点关注：
1. 目的地当前天气形势，是否需要调整行程安排（暴雨/台风/极端高温等）
2. 安全形势（治安、自然灾害、公共卫生）
3. 行程中的高风险地点或活动
4. 紧急联系方式与保险建议
5. 个人特殊需求（如过敏、慢性病）的应对

输出简洁的安全清单格式。
"""

MOCK_SAFETY_OUTPUT = """\
## 旅行安全评估

### ✅ 整体风险：低
目的地治安良好，自然灾害概率低。

### ⚠️ 天气关注
- 当前多云 22°C，湿度适中，适合户外活动
- 出行前再次确认未来 3 天预报

### 📋 紧急信息
- 中国报警：110；急救：120
- 推荐购买：国内旅游保险

### 💊 个人事项
- 常规常备药（感冒药、肠胃药、创可贴）即可
"""


def safety_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Safety：先调心知天气获取目的地天气，再让 LLM 生成安全评估。
    天气 API 不通时降级为常规安全建议。
    """
    # 从 sub_reports 中找到 itinerary 节点的 city 字段（sub_reports 是累加列表）
    city = ""
    for sub in state.get("sub_reports", []):
        if sub.get("agent") == "itinerary" and sub.get("city"):
            city = sub["city"]
            break
    # 兜底：从 user_request 文本提取
    if not city:
        city = extract_city_name(state.get("user_request", ""))

    weather_text = ""
    if city:
        # 心知天气用拼音（如 "成都" → "chengdu"）
        pinyin = _city_pinyin(city)
        if pinyin:
            print(f"  [safety] 正在用心知天气查询 {city}({pinyin}) 的天气...")
            weather = get_weather(pinyin)
            daily = get_weather_daily(pinyin, days=3)
            weather_text = _weather_to_text(weather, daily)
            if weather:
                print(f"  [safety] 天气查询成功：{weather['text']} {weather['temperature']}°C")
            else:
                print(f"  [safety] 天气查询失败（可能是海外城市或权限问题）")

    output = _call_or_mock(
        system_prompt="",
        user_text=SAFETY_PROMPT.format(
            itinerary=state.get("itinerary", ""),
            weather_context=weather_text,
        ),
        mock=MOCK_SAFETY_OUTPUT,
        stage="safety",
    )

    return {
        "safety_report": output,
        "next_node": ROUTE_REVIEW,
        "sub_reports": [{"agent": "safety", "report": output}],
    }


def _city_pinyin(city_name: str) -> str:
    """
    中国城市名 → 心知天气拼音 location 映射表。
    心知天气 v3 的 location 参数要求城市拼音（小写，无空格）。
    覆盖主要热门城市 + 直辖市。
    """
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
# Review —— 方案审核 & 修订判断
# ==========================================================================

REVIEW_PROMPT = """\
你是 DeepTravel 的方案审核专家。请综合评审以下三份子报告，判断是否通过：

## 归一化需求
{user_request}

## 行程报告
{itinerary}

## 预算报告
{budget}

## 安全报告
{safety_report}

请逐条检查：
1. 行程是否完整覆盖需求中的日期、偏好、特殊需求？
2. 预算是否在用户可接受范围内？
3. 安全风险是否已充分提示？

**输出格式要求：**
先给出 [APPROVED] 或 [REVISE] 标签；
然后用简短文字说明理由；
如果是 [REVISE]，请明确指出需要修改 itinerary 中的哪些部分。
"""

MOCK_REVIEW_APPROVED_OUTPUT = """\
[APPROVED]
整体方案完整覆盖 7天6晚 东京 + 富士山行程，预算超出在 5% 以内可接受范围，安全评估充分。无需修订。
"""

MOCK_REVIEW_REVISE_OUTPUT = """\
[REVISE]
预算超出约 ¥6,000（+31%），建议调整：将温泉旅馆从富士山移出改为一日往返，可节省住宿 + 交通费用约 ¥4,500。
"""

MAX_REVISION = 2  # 最多允许 2 次修订，超过则强制通过或终止


def review_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Review：审核三份子报告，写 review_status（approved/revise）并决定 next_node

    路由规则：
    - review_status=approved → next_node='integrate'
    - review_status=revise  → next_node='itinerary'（回退重跑）
    - revision_count 超过 MAX_REVISION → 强制 integrate 防死循环
    """
    revision_count = state.get("revision_count", 0)

    # 超限保护：达到上限则强制通过
    if revision_count >= MAX_REVISION:
        return {
            "review_status": "approved",
            "review_feedback": f"已达到最大修订次数 ({MAX_REVISION})，强制通过审核",
            "revision_count": revision_count,
            "next_node": ROUTE_INTEGRATE,
            "sub_reports": [{"agent": "review", "report": "[APPROVED-强制] 超过修订上限"}],
        }

    output = _call_or_mock(
        system_prompt="",
        user_text=REVIEW_PROMPT.format(
            user_request=state.get("user_request", ""),
            itinerary=state.get("itinerary", ""),
            budget=state.get("budget", ""),
            safety_report=state.get("safety_report", ""),
        ),
        # 第一次 review 用 approved；第二次如果回退过来，让它 revise 触发上限检查
        mock=(
            MOCK_REVIEW_REVISE_OUTPUT if revision_count > 0 else MOCK_REVIEW_APPROVED_OUTPUT
        ),
        stage="review",
    )

    # 解析审核结果
    is_approved = "[APPROVED]" in output.upper()
    next_node = ROUTE_INTEGRATE if is_approved else ROUTE_ITINERARY
    new_revision = revision_count if is_approved else revision_count + 1

    return {
        "review_status": "approved" if is_approved else "revise",
        "review_feedback": output,
        "revision_count": new_revision,
        "next_node": next_node,
        "sub_reports": [{"agent": "review", "report": output}],
    }


# ==========================================================================
# Integrate —— 整合最终方案
# ==========================================================================

INTEGRATE_PROMPT = """\
你是 DeepTravel 的方案整合专家。请将以下三份子报告整合成一份结构清晰、
面向用户的完整旅行方案：

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

要求：
- 开头有一句话总结（目的地、天数、人数）
- 各章节用清晰的 Markdown 标题分隔
- 结尾给出"方案摘要卡片"（关键信息一览表）
"""


def integrate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Integrate：整合最终方案，设置 next_node='end'"""
    output = _call_or_mock(
        system_prompt="",
        user_text=INTEGRATE_PROMPT.format(
            user_request=state.get("user_request", ""),
            itinerary=state.get("itinerary", ""),
            budget=state.get("budget", ""),
            safety_report=state.get("safety_report", ""),
            review_feedback=state.get("review_feedback", ""),
        ),
        # 整合阶段用 mock 拼接已有子报告，避免 LLM 调用
        mock=_mock_integrated_plan(state),
        stage="integrate",
    )

    return {
        "final_plan": output,
        "next_node": ROUTE_END,
        "sub_reports": [{"agent": "integrate", "report": "方案整合完成"}],
    }


def _mock_integrated_plan(state: dict[str, Any]) -> str:
    """mock 整合：直接拼接已有子报告，保证即使 mock 模式也能输出完整方案"""
    parts = [
        "# 🗺️ DeepTravel 旅行方案\n",
        f"**目的地：** 日本东京  |  **天数：** 7天6晚  |  **人数：** 2人\n",
        "---\n",
        "## 📋 归一化需求\n",
        state.get("user_request", ""),
        "\n---\n",
        "## 📅 详细行程\n",
        state.get("itinerary", ""),
        "\n---\n",
        "## 💰 预算估算\n",
        state.get("budget", ""),
        "\n---\n",
        "## 🛡️ 安全提示\n",
        state.get("safety_report", ""),
        "\n---\n",
        "## ✅ 审核意见\n",
        state.get("review_feedback", ""),
        "\n---\n",
        "## 📑 方案摘要卡片\n",
        "| 项目 | 内容 |\n",
        "|------|------|\n",
        "| 目的地 | 日本东京 + 富士山 |\n",
        "| 出行日期 | 2026-11-01 ~ 2026-11-07 |\n",
        "| 总预算 | ≈ ¥26,180（略超用户预算）|\n",
        "| 修订次数 | " + str(state.get("revision_count", 0)) + " |\n",
    ]
    return "\n".join(parts)


# ==========================================================================
# 内部辅助
# ==========================================================================

def _call_or_mock(
    system_prompt: str,
    user_text: str,
    mock: str,
    stage: str,
) -> str:
    """
    统一的 LLM 调用封装：
    - mock 模式 → 直接返回预设 mock 输出
    - 真实 LLM → 调用后返回 content
    - LLM 异常 → 记录并返回 mock 兜底，保证流程不中断
    """
    if is_mock_mode():
        return mock

    try:
        llm = get_llm()
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_text))
        resp = llm.invoke(messages)
        content = getattr(resp, "content", str(resp)) or ""
        # LLM 返回为空则兜底
        if not content.strip():
            print(f"  [{stage}] LLM 返回空，使用 mock 兜底")
            return mock
        return content
    except Exception as exc:
        print(f"  [{stage}] LLM 调用失败 ({exc})，使用 mock 兜底")
        return mock
