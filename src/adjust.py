"""
DeepTravel 独立调整子图 (Adjust Subgraph)

架构：
    用户修改请求 → adjust_coordinator (LLM 关键词分析)
                        │
                  ┌─────┴──────┐
                  │ 确定起点    │
                  │  + 影响范围 │
                  └─────┬──────┘
                        ▼
            ┌─ 裁剪后的有向图 ─┐
            │                 │
            │  (只重跑受影响   │
            │   起点之后的     │
            │   Agent 节点)   │
            │                 │
            └────────┬────────┘
                     ▼
              integrate → END

依赖关系（DAG）：
    coordinator → itinerary → budget → safety → review → integrate
                                  └───────────────┘
         ↑ (review revise 回退)

所以：
  - "改预算" → 起点 budget → 重跑 budget, safety, review, integrate
  - "改行程" → 起点 itinerary → 重跑 itinerary, budget, safety, review, integrate
  - "改目的地/天数" → 起点 coordinator → 全重跑
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from .state import TravelState
from .agents.common import get_llm

load_dotenv()


# --------- 依赖图：每个节点依赖哪些上游 ---------

AGENT_ORDER = ["coordinator", "itinerary", "budget", "safety", "review", "integrate"]

# 节点 → 它需要重跑时，必须重跑的起点
# （起点本身及之后的所有节点都要跑）
NODE_INDEX = {n: i for i, n in enumerate(AGENT_ORDER)}


def determine_adjust_start(original_state: dict, modify_request: str) -> str:
    """
    LLM 分析用户修改请求，决定从哪个 Agent 开始重跑。

    依赖规则：
    - 改目的地/天数/人数/核心偏好 → coordinator（全重跑）
    - 改行程安排/景点/每日路线 → itinerary
    - 改预算/消费偏好 → budget
    - 改出行时间/季节/安全要求 → safety
    - 改最终方案呈现/格式 → review 或 integrate
    """
    # Mock 模式：关键词规则匹配
    import os
    if os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes"):
        text = modify_request.lower()
        rules = [
            ("coordinator", ["目的地", "去哪", "去哪里", "城市", "换个地方", "换地方", "天数", "几天", "人数", "多少人", "偏好", "爱好", "出行方式", "飞机", "高铁"]),
            ("itinerary", ["行程", "景点", "路线", "每日", "每天", "节奏", "住宿", "酒店位置", "怎么走"]),
            ("budget", ["预算", "钱", "贵", "便宜", "节省", "经济", "省钱", "消费", "价格", "费用"]),
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
- itinerary: 改了行程安排、景点、路线、每日节奏、住宿位置
- budget: 改了预算金额、消费偏好、价格敏感度
- safety: 改了出行时间/季节、天气、安全相关要求
- review: 想修改整体方案的结构或需要重新审核
- integrate: 只想修改最终方案的呈现方式

规则：
1. 选最精准的起点（不要从头跑不必要的节点）
2. 如果不确定，选更靠后的起点（更保守）
3. 只返回一个英文节点名，不要其他内容""")

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


def build_adjust_graph(start_node: str):
    """
    构建裁剪后的调整子图。

    只保留 start_node 及之后的节点。
    路由：start → ... → review → integrate → END
          （review 仍可 revise 回退，但只允许回退到 start_node）
    """
    from .graph import build_graph as _build_full_graph
    from .agents.nodes import (
        coordinator_node, itinerary_node, budget_node,
        safety_node, review_node, integrate_node,
    )
    from .agents.common import route_decision

    start_idx = NODE_INDEX.get(start_node, 5)
    active_nodes = AGENT_ORDER[start_idx:]

    # 构造新的路由：只允许在 active_nodes 内部跳转
    # 如果 revise 目标（itinerary）不在 active_nodes 里，就跳到 active_nodes 的第一个
    revise_target = "itinerary"
    if revise_target not in active_nodes:
        revise_target = active_nodes[0]  # 就地重跑起点

    sg = StateGraph(TravelState)

    # 添加起点节点
    sg.set_entry_point(start_node)

    # 添加所有激活的节点
    for node in active_nodes:
        node_fn = {
            "coordinator": coordinator_node,
            "itinerary": itinerary_node,
            "budget": budget_node,
            "safety": safety_node,
            "review": review_node,
            "integrate": integrate_node,
        }[node]
        sg.add_node(node, node_fn)

    # 定义局部路由函数
    def local_route(state: dict) -> str:
        next_node = state.get("next_node", "end")
        # 如果 revise 目标不在 active_nodes 里，跳 revise_target
        if next_node == "itinerary" and "itinerary" not in active_nodes:
            return revise_target
        # 如果目标是 integrate 且不在 active_nodes（不应该发生，integrate 总是最后）
        if next_node == "end" or next_node not in active_nodes:
            return END
        return next_node

    path_map = {n: n for n in active_nodes}
    path_map[END] = END

    for node in active_nodes:
        sg.add_conditional_edges(
            source=node,
            path=local_route,
            path_map=path_map,
        )

    # 复用主 graph 的 checkpointer 逻辑（直接 import build_graph 以触发全局 checkpointer 初始化）
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

    # 如果起点是 coordinator，重置 user_request（让 coordinator 重新归一化）
    if start_node == "coordinator":
        adjust_state["user_request"] = modify_request

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
