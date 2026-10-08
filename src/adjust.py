"""
DeepTravel 独立调整子图 (Adjust Subgraph，并行拓扑)

架构：
    用户修改请求 → adjust_coordinator (LLM 关键词分析)
                        │
                  ┌─────┴──────┐
                  │ 确定起点    │
                  └─────┬──────┘
                        ▼
            ┌─ 裁剪后的并行子图 ─┐
            │ (对齐主图 fan-out， │
            │  只重跑受影响的段)  │
            └────────┬──────────┘
                     ▼
              integrate → END

按起点的并行拓扑（budget/safety 的 prompt 读 state["itinerary"]，
所以 itinerary 起点必须先跑行程、再让预算/安全并行）：

    coordinator:  coordinator → {itinerary, budget, safety} ∥ → review
                  （同主图：budget/safety 基于归一化需求估算，不依赖行程输出）
    itinerary:    itinerary → {budget, safety} ∥ → review
                  （新行程先落 state，预算/安全基于新行程并行重算）
    budget:       {budget, safety} ∥ → review
    safety:       safety → review
    review:       review → HITL gate
    integrate:    integrate → END

revise / HITL-revise 回退目标 = 起点所在 superstep（revise_targets），
后续拓扑自动并行，语义与旧串行版一致（回到最早受影响节点）。
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START

from .state import TravelState
from .agents.common import get_llm

load_dotenv()


# --------- 依赖图：每个节点依赖哪些上游 ---------

AGENT_ORDER = ["coordinator", "itinerary", "budget", "safety", "review", "integrate"]

# 节点 → 它需要重跑时，必须重跑的起点
# （起点本身及之后的所有节点都要跑）
NODE_INDEX = {n: i for i, n in enumerate(AGENT_ORDER)}

# --------- 按起点的并行子图拓扑表 ---------
# entry:      START 扇出的入口节点（可多个，同一 superstep 并行）
# fanout:     节点 → 它完成后扇出到哪些节点（缺省 = 直通 review）
# revise:     review revise / HITL revise 的回退目标（回到起点所在 superstep；
#             coordinator 起点的 revise 不重跑归一化，同主图）
_ADJUST_TOPO = {
    "coordinator": {
        "entry": ["coordinator"],
        "fanout": {"coordinator": ["itinerary", "budget", "safety"]},
        "revise": ["itinerary", "budget", "safety"],
    },
    "itinerary": {
        "entry": ["itinerary"],
        "fanout": {"itinerary": ["budget", "safety"]},
        "revise": ["itinerary"],
    },
    "budget": {
        "entry": ["budget", "safety"],
        "fanout": {},
        "revise": ["budget", "safety"],
    },
    "safety": {
        "entry": ["safety"],
        "fanout": {},
        "revise": ["safety"],
    },
    "review": {
        "entry": ["review"],
        "fanout": {},
        "revise": ["review"],  # 无上游 workers 可回退，就地重跑（与旧逻辑一致）
    },
    "integrate": {
        "entry": ["integrate"],
        "fanout": {},
        "revise": [],
    },
}


def determine_adjust_start(original_state: dict, modify_request: str) -> str:
    """
    LLM 分析用户修改请求，决定从哪个 Agent 开始重跑。

    依赖规则：
    - 改目的地/天数/人数/核心偏好 → coordinator（全重跑）
    - 改行程安排/景点/每日路线 → itinerary
    - 改预算/消费偏好 → itinerary（行程里的消费推荐必须与预算联动，
      否则 review 交叉校验会打一致性低分 → revise 双轮反而更慢；
      实测 budget 起点 96s vs itinerary 起点单轮 ~70-90s）
    - 改出行时间/季节/安全要求 → safety
    - 改最终方案呈现/格式 → review 或 integrate
    """
    # Mock 模式：关键词规则匹配
    import os
    if os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes"):
        text = modify_request.lower()
        rules = [
            ("coordinator", ["目的地", "去哪", "去哪里", "城市", "换个地方", "换地方", "天数", "几天", "人数", "多少人", "偏好", "爱好", "出行方式", "飞机", "高铁"]),
            # 预算/消费类也判 itinerary：行程消费推荐需与预算联动（见 docstring）
            ("itinerary", ["行程", "景点", "路线", "每日", "每天", "节奏", "住宿", "酒店位置", "怎么走",
                           "预算", "钱", "贵", "便宜", "节省", "经济", "省钱", "消费", "价格", "费用"]),
            ("safety", ["天气", "季节", "几月", "月份", "安全", "温度", "冷", "热"]),
        ]
        for node, keywords in rules:
            for kw in keywords:
                if kw in text:
                    print(f"[adjust] mock 关键词匹配: '{kw}' → {node}")
                    return node
        return "integrate"

    # 真实 LLM 模式
    llm = get_llm()

    system = SystemMessage(content="""你是一个旅行规划系统的调整管理器。
用户想要修改一个已生成的旅行方案，你需要判断：从哪个阶段开始重新规划。

可选起点（按依赖顺序）：
- coordinator: 改了目的地、天数、人数、核心偏好、出行方式等基础需求
- itinerary: 改了行程安排、景点、路线、每日节奏、住宿位置；
  也包括预算金额或消费偏好的变化 —— 行程里的消费推荐必须与预算联动更新
- safety: 改了出行时间/季节、天气、安全相关要求
- review: 想修改整体方案的结构或需要重新审核
- integrate: 只想修改最终方案的呈现方式

规则：
1. 选最精准的起点（不要从头跑不必要的节点）
2. 预算变化必须选 itinerary，不要选 budget —— 否则行程与预算不一致，
   评审会打回重做（反而更慢）
3. 如果不确定，选更靠前的起点（保证一致性）
4. 只返回一个英文节点名，不要其他内容""")

    context_parts = []
    if original_state.get("user_request"):
        context_parts.append(f"原需求: {original_state['user_request']}")
    if original_state.get("final_plan"):
        # 只取前 300 字，省 token
        context_parts.append(f"原方案摘要: {original_state['final_plan'][:300]}...")
    context = "\n".join(context_parts)

    user = HumanMessage(content=f"""{context}

用户修改请求: {modify_request}

请返回起点节点名:""")

    try:
        resp = llm.invoke([system, user])
        raw = resp.content.strip().lower()
        # 提取第一个匹配的节点名
        for node in AGENT_ORDER:
            if node in raw:
                return node
        return "integrate"  # fallback
    except Exception as exc:
        print(f"[adjust] LLM 分析失败: {exc}，fallback 到 integrate")
        return "integrate"


# --------- 调整请求中的新预算提取 ---------
# 增量改预算时 parsed_info.budget 不会更新（子图不含 coordinator），
# review 的 _cross_check 会拿旧预算对比新预算报告 → 必然 finding → revise 双轮。
# 所以注入调整请求时用正则提取新预算，同步覆盖 parsed_info["budget"]。

import re as _re

_BUDGET_PATTERNS = [
    # 「预算改成/改为/调整到/控制在/只要/不要超过 4000（元/块）」，支持千分位 5,000
    _re.compile(r"(?:预算|花销|花费|费用|开支)[^\d\n]{0,6}(?:改成|改为|换成|调整到|调整成|调整至|控制在|控制到|只要|不超过|不要超过|最多|最多花|上限)[^\d\n]{0,4}(\d{1,3}(?:,\d{3})+|\d{3,6})"),
    # 「预算 4000」直接声明
    _re.compile(r"(?:预算|总预算)[^\d\n]{0,4}(\d{1,3}(?:,\d{3})+|\d{3,6})\s*(?:元|块|¥)?"),
]
# 「预算 3 万 / 2.5 万」
_BUDGET_WAN = _re.compile(r"(?:预算|总预算|花销|花费)[^\d\n]{0,8}(\d{1,3}(?:\.\d)?)\s*万")


def extract_new_budget(modify_request: str) -> int | None:
    """从调整请求提取新的总预算；提取不到返回 None（保持原值）。

    返回值单位：元。匹配 3-6 位数字（100 ~ 999999，支持千分位），
    避免误抓「2 人」「3 天」；另支持「N 万」表达。
    """
    if not modify_request:
        return None
    m = _BUDGET_WAN.search(modify_request)
    if m:
        return int(float(m.group(1)) * 10000)
    for pat in _BUDGET_PATTERNS:
        m = pat.search(modify_request)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def apply_budget_to_state(state: dict, modify_request: str, base: dict | None = None) -> dict:
    """调整请求注入 state 时调用：提取到新预算则覆盖 parsed_info.budget。

    state 是完整 state（/api/adjust）或增量 update dict（/api/hitl 的
    human_update）。当 state 是增量 update 时必须传 base=完整 state，
    否则 parsed_info 从空 dict 起建，写回会丢 city/days/people 等字段。
    """
    new_budget = extract_new_budget(modify_request)
    if new_budget:
        src = base if base is not None else state
        parsed = dict(src.get("parsed_info") or {})
        if parsed.get("budget") != new_budget:
            print(f"[adjust] 检测到新预算 ¥{new_budget}，更新 parsed_info.budget（原值 {parsed.get('budget')}）")
            parsed["budget"] = new_budget
            state["parsed_info"] = parsed
    return state


def build_adjust_graph(start_node: str):
    """
    构建裁剪后的**并行**调整子图（对齐主图 fan-out 拓扑，见模块 docstring）。

    路由：START 扇出入口节点 → ... → review → HITL gate ─┬─ hitl（等待人工）
                                                          ├─ revise_targets（并行回退）
                                                          └─ integrate（自动 approve）
    """
    from .graph import build_graph as _build_full_graph
    from .agents.nodes import (
        coordinator_node, itinerary_node, budget_node,
        safety_node, review_node, integrate_node,
    )
    from .agents.common import route_decision
    from .hitl import hitl_node, hitl_gate_route

    topo = _ADJUST_TOPO.get(start_node)
    if topo is None:
        # 未知起点：与旧逻辑一致，退化到只重跑 integrate
        topo = _ADJUST_TOPO["integrate"]
    entry_nodes = topo["entry"]
    fanout = topo["fanout"]
    revise_targets = topo["revise"]

    node_fns = {
        "coordinator": coordinator_node,
        "itinerary": itinerary_node,
        "budget": budget_node,
        "safety": safety_node,
        "review": review_node,
        "integrate": integrate_node,
        "hitl": hitl_node,
    }
    # 节点集合 = 入口 ∪ 扇出目标 ∪ 尾部（review/integrate 按起点裁剪）∪ hitl
    all_nodes = set(entry_nodes)
    for targets in fanout.values():
        all_nodes.update(targets)
    idx = NODE_INDEX.get(start_node, 5)
    all_nodes.update(AGENT_ORDER[idx:])  # review / integrate（start 靠后时自动裁掉 review）
    all_nodes.add("hitl")

    sg = StateGraph(TravelState)
    for node in all_nodes:
        sg.add_node(node, node_fns[node])

    # 通用 path_map：图内所有节点 + end → END
    path_map = {n: n for n in all_nodes}
    path_map["end"] = END

    # START → 入口扇出（单/多入口同一写法；多入口时同一 superstep 并行）
    sg.add_conditional_edges(
        source=START,
        path=lambda state: entry_nodes,
        path_map={e: e for e in entry_nodes},
    )

    # 入口/中间节点 → 扇出目标 或 直通 review。
    # worker 超时返回的 next_node='hitl' 在这里被有意忽略 —— 并行分支的路由无法
    # 感知兄弟分支，统一走 review，由 review 后的 hitl_gate_route 读取超时标记接管
    # （与主图同款设计）
    for node in all_nodes - {"hitl", "review", "integrate"}:
        targets = fanout.get(node)
        if targets:
            sg.add_conditional_edges(
                source=node,
                path=lambda state, t=targets: t,
                path_map={t: t for t in targets},
            )
        else:
            sg.add_conditional_edges(
                source=node,
                path=lambda state: "review",
                path_map={"review": "review"},
            )

    # review → HITL gate：revise 时并行回到 revise_targets（起点所在 superstep）。
    # 注意：hitl_gate_route 返回的是 path_map 的「键」（hitl/itinerary/integrate），
    # "itinerary" 键在这里展开为 revise_targets 列表，不能再手动替换成节点名
    if "review" in all_nodes:
        def _review_gate(state: dict):
            key = hitl_gate_route(state)
            if key == "itinerary":
                return revise_targets or "review"
            return key

        sg.add_conditional_edges(
            source="review",
            path=_review_gate,
            path_map=path_map,
        )

    # hitl → 恢复路由：approve 按 next_node（hitl_next_node/integrate）续跑，
    # revise（next_node='itinerary'）→ 并行回到 revise_targets；
    # 等待时 hitl_node 返回 next_node='end' → END（图暂停，等 /api/hitl 决策）
    def _hitl_route(state: dict):
        key = route_decision(state)
        if key == "itinerary" and revise_targets:
            return revise_targets
        return key

    sg.add_conditional_edges(
        source="hitl",
        path=_hitl_route,
        path_map=path_map,
    )

    # integrate 收尾 → END
    if "integrate" in all_nodes:
        sg.add_conditional_edges(
            source="integrate",
            path=route_decision,
            path_map=path_map,
        )

    # 复用主 graph 的 checkpointer 逻辑
    _build_full_graph()
    from .graph import _get_checkpointer
    return sg.compile(checkpointer=_get_checkpointer())


def run_adjust(
    thread_id: str,
    modify_request: str,
    mock: bool = False,
) -> tuple[list[dict], str]:
    """
    运行调整子图。

    Args:
        thread_id: 已有会话 ID（从 checkpoint 恢复 state）
        modify_request: 用户的修改请求
        mock: 是否 mock LLM

    Returns:
        (节点事件列表, 最终方案)
    """
    if mock:
        os.environ["MOCK_LLM"] = "true"
    else:
        os.environ.pop("MOCK_LLM", None)

    from .graph import build_graph

    # 1. 恢复原始 state
    full_graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state_obj = full_graph.get_state(config)
    if state_obj is None or state_obj.values is None:
        raise ValueError(f"会话 {thread_id} 未找到或无 state")
    original_state = dict(state_obj.values)

    # 2. 决定起点
    start_node = determine_adjust_start(original_state, modify_request)
    print(f"[adjust] LLM 决定起点 = {start_node}（修改请求: {modify_request}）")

    # 3. 准备调整后的 state：追加新消息、重置 revision、清空待重跑节点的旧子报告
    adjust_state = dict(original_state)

    # 追加用户新请求到 messages
    from langchain_core.messages import HumanMessage
    adjust_state["messages"] = list(original_state.get("messages", [])) + [
        HumanMessage(content=f"[用户调整] {modify_request}")
    ]

    # 如果起点是 coordinator，重置 user_request（让 coordinator 重新归一化）；
    # 否则附加调整意见——下游节点的 prompt 读 state["user_request"]，
    # 不附加上它们就看不到新的调整要求
    if start_node == "coordinator":
        adjust_state["user_request"] = modify_request
    else:
        adjust_state["user_request"] = (
            (original_state.get("user_request", "") or "")
            + f"\n[用户调整] {modify_request}"
        )

    # 清空待重跑节点的旧子报告
    NODE_TO_FIELD = {
        "itinerary": "itinerary",
        "budget": "budget",
        "safety": "safety_report",
        "review": "review_feedback",
    }
    start_idx = NODE_INDEX.get(start_node, 5)
    for node_name in AGENT_ORDER[start_idx:]:
        field = NODE_TO_FIELD.get(node_name)
        if field and field in adjust_state:
            # 不删除，保留旧值做 fallback（节点自己会覆盖）
            pass

    adjust_state["revision_count"] = 0
    adjust_state["review_status"] = "pending"
    # 记录子图起点：万一中途进 HITL，恢复时要用同一张子图
    adjust_state["hitl_subgraph_start"] = start_node

    # 4. 构建裁剪后的子图并运行
    adjust_graph = build_adjust_graph(start_node)

    node_events: list[dict] = []
    final_plan = ""

    for event in adjust_graph.stream(adjust_state, config):
        for node_name, state_update in event.items():
            node_events.append({
                "node": node_name,
                "next_node": state_update.get("next_node", ""),
                "revision_count": state_update.get("revision_count", 0),
            })
            if "final_plan" in state_update:
                final_plan = state_update["final_plan"]

    print(f"[adjust] 完成，重跑 {len(node_events)} 个节点，起点 {start_node}")
    return node_events, final_plan
