"""
DeepTravel 共享状态定义

核心设计：
- 使用 TypedDict（而非自定义可变对象）避免 LangGraph 状态合并异常
- messages 字段使用 add_messages reducer 自动追加历史消息
- 各 Agent 子报告字段使用 Annotated 显式声明 reducer
- next_node 显式维护路由信号，路由函数为纯函数
"""

from typing import TypedDict, Annotated, Literal, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# --------- 简单 reducer 定义 ---------

def _override_reducer(existing: Any, updates: Any) -> Any:
    """覆盖式 reducer：新值直接替换旧值"""
    return updates


def _append_reducer(existing: list, updates: list) -> list:
    """追加式 reducer：将新 list 拼接到旧 list 末尾"""
    return (existing or []) + (updates or [])


# --------- 阶段枚举 ---------

Stage = Literal[
    "coordinator",
    "itinerary",
    "budget",
    "safety",
    "review",
    "integrate",
    "end",
]

ReviewStatus = Literal[
    "pending",    # 未审核
    "approved",   # 审核通过
    "revise",     # 需要修订（回退到 itinerary）
]


# --------- 主状态 ---------

class TravelState(TypedDict, total=False):
    """
    旅行规划有向图共享状态

    使用 Annotated[..., reducer] 声明合并策略：
    - messages:   add_messages —— 自动追加对话历史
    - user_request:   覆盖 —— Coordinator 写入后不再变化
    - itinerary:      覆盖 —— Itinerary Agent 写入/修订
    - budget:         覆盖 —— Budget Agent 写入
    - safety_report:  覆盖 —— Safety Agent 写入
    - review_feedback: 覆盖 —— Review Agent 写入审核意见
    - review_status:  覆盖 —— Review Agent 写入审核结果（approved/revise）
    - revision_count: 覆盖 —— 每次修订 +1，上限保护
    - next_node:      覆盖 —— 显式路由信号
    - final_plan:     覆盖 —— Integrate 节点写入最终方案
    - sub_reports:    追加 —— 各 Agent 写入子报告（持久化用）
    """

    # 对话历史
    messages: Annotated[list[AnyMessage], add_messages]

    # 用户原始需求（Coordinator 归一化后写入）
    user_request: Annotated[str, _override_reducer]

    # 各 Agent 子报告
    itinerary: Annotated[str, _override_reducer]
    budget: Annotated[str, _override_reducer]
    safety_report: Annotated[str, _override_reducer]
    review_feedback: Annotated[str, _override_reducer]

    # 审核状态 & 修订计数
    review_status: Annotated[ReviewStatus, _override_reducer]
    revision_count: Annotated[int, _override_reducer]

    # 路由信号 & 最终方案
    next_node: Annotated[Stage, _override_reducer]
    final_plan: Annotated[str, _override_reducer]

    # 所有 Agent 子报告的汇总（便于持久化 & 历史检索）
    sub_reports: Annotated[list[dict], _append_reducer]
