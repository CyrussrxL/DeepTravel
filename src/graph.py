"""
DeepTravel 六阶段有向图构建

架构：
            ┌──────────────┐
            │  coordinator │ ← START
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │  itinerary   │ ◄──┐
            └──────┬───────┘    │
                   │            │ (review=revise)
                   ▼            │
            ┌──────────────┐    │
            │    budget    │    │
            └──────┬───────┘    │
                   │            │
                   ▼            │
            ┌──────────────┐    │
            │    safety    │    │
            └──────┬───────┘    │
                   │            │
                   ▼            │
            ┌──────────────┐    │
            │    review    │─────┘
            └──┬───────┬───┘
               │       │
          approve   revise
               │       │
               ▼       │
        ┌────────────┐ │
        │  integrate │ │
        └─────┬──────┘ │
              │        │
              ▼        │
            END        │

路由规则：
- 所有节点只写 state['next_node']
- route_decision(state) 读取 next_node 决定下一跳
- 每个节点的 add_conditional_edges 提供 path_map 映射
- 不混用 add_edge 和 add_conditional_edges（来自失败经验）
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from .state import TravelState
from .agents.common import (
    route_decision,
    ROUTE_ITINERARY,
    ROUTE_BUDGET,
    ROUTE_SAFETY,
    ROUTE_REVIEW,
    ROUTE_INTEGRATE,
    ROUTE_END,
)
from .agents.nodes import (
    coordinator_node,
    itinerary_node,
    budget_node,
    safety_node,
    review_node,
    integrate_node,
)


def build_graph():
    """构建并返回编译后的 LangGraph StateGraph"""

    # 1. 初始化图
    sg = StateGraph(TravelState)

    # 2. 添加六个节点
    sg.add_node("coordinator", coordinator_node)
    sg.add_node("itinerary", itinerary_node)
    sg.add_node("budget", budget_node)
    sg.add_node("safety", safety_node)
    sg.add_node("review", review_node)
    sg.add_node("integrate", integrate_node)

    # 3. 设置起点
    sg.set_entry_point("coordinator")

    # 4. 为每个节点添加条件边（统一路由）
    # path_map: 将 next_node 字符串映射到实际图节点名（或 END）
    PATH_MAP = {
        ROUTE_ITINERARY: "itinerary",
        ROUTE_BUDGET: "budget",
        ROUTE_SAFETY: "safety",
        ROUTE_REVIEW: "review",
        ROUTE_INTEGRATE: "integrate",
        ROUTE_END: END,
    }

    for node_name in ("coordinator", "itinerary", "budget", "safety", "review", "integrate"):
        sg.add_conditional_edges(
            source=node_name,
            path=route_decision,
            path_map=PATH_MAP,
        )

    # 5. 编译返回
    return sg.compile()
