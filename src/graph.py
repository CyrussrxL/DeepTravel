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

from pathlib import Path

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
from .hitl import hitl_node, hitl_gate_route

# SQLite 持久化文件位置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "checkpoints.sqlite"


def _get_checkpointer(db_path: str | None = None):
    """每次新建 SqliteSaver + sqlite3.Connection；传 None 则不持久化"""
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    if db_path is None:
        db_path = str(DB_PATH)

    if str(db_path) == ":memory:":
        conn = sqlite3.connect(":memory:", check_same_thread=False)
    else:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def build_graph(checkpointer_path: str | None = None):
    """
    构建并返回编译后的 LangGraph StateGraph（带 SQLite Checkpointer + HITL 节点）

    架构更新：review 之后走 hitl_gate 条件路由
      review → HITL gate ─┬─ hitl（等待人工）
                          ├─ itinerary（自动 revise）
                          └─ integrate（自动 approve）
    """
    sg = StateGraph(TravelState)

    # 七个节点（新增 hitl）
    sg.add_node("coordinator", coordinator_node)
    sg.add_node("itinerary", itinerary_node)
    sg.add_node("budget", budget_node)
    sg.add_node("safety", safety_node)
    sg.add_node("review", review_node)
    sg.add_node("hitl", hitl_node)
    sg.add_node("integrate", integrate_node)

    sg.set_entry_point("coordinator")

    # 通用路由 path_map
    PATH_MAP = {
        ROUTE_ITINERARY: "itinerary",
        ROUTE_BUDGET: "budget",
        ROUTE_SAFETY: "safety",
        ROUTE_REVIEW: "review",
        ROUTE_INTEGRATE: "integrate",
        ROUTE_END: END,
        "hitl": "hitl",  # 任意节点超时/异常都能触发 HITL
    }

    # 除了 review，其他节点用 route_decision
    # hitl 节点自己返回 next_node='end'，route_decision 自然走 END
    for node_name in ("coordinator", "itinerary", "budget", "safety", "integrate", "hitl"):
        sg.add_conditional_edges(
            source=node_name,
            path=route_decision,
            path_map=PATH_MAP,
        )

    # review → HITL gate 专用路由
    sg.add_conditional_edges(
        source="review",
        path=hitl_gate_route,
        path_map={
            "hitl": "hitl",
            "itinerary": "itinerary",
            "integrate": "integrate",
        },
    )

    checkpointer = _get_checkpointer(checkpointer_path)
    return sg.compile(checkpointer=checkpointer)


def list_sessions():
    """列出所有持久化会话（从 SQLite checkpoints 表检索）"""
    import sqlite3

    db_path = DB_PATH
    if not db_path.exists():
        return []

    sessions = []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread_id, MIN(checkpoint_id) as first, MAX(checkpoint_id) as last "
            "FROM checkpoints GROUP BY thread_id ORDER BY first DESC LIMIT 50"
        ).fetchall()
        for row in rows:
            sessions.append({
                "thread_id": row[0],
                "checkpoint_ids": [row[1], row[2]],
            })
    finally:
        conn.close()
    return sessions


def get_session_state(thread_id: str) -> dict | None:
    """从持久化存储中检索某个会话的最新 state"""
    graph = build_graph()
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        if state is None:
            return None
        values = state.values
        # 只返回前端关心的字段
        return {
            "thread_id": thread_id,
            "messages": [{"role": m.type, "content": m.content[:200] + "..." if len(m.content) > 200 else m.content}
                         for m in values.get("messages", [])],
            "user_request": values.get("user_request", ""),
            "final_plan": values.get("final_plan", ""),
            "revision_count": values.get("revision_count", 0),
            "last_node": state.next[0] if state.next else "end",
        }
    except Exception:
        return None
