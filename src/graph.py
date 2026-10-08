"""
DeepTravel 六阶段有向图构建（fan-out 并行拓扑）

架构：
            coordinator ← START
                 │ 三路 fan-out（同一 superstep 并发执行）
     ┌───────────┼───────────┐
     ▼           ▼           ▼
 itinerary    budget      safety   ◄──┐
     └───────────┼───────────┘        │
                 ▼ fan-in            │ review=revise：
              review ────────────────┘ 三路并行重跑
                 │       │
            approve    hitl gate
                 │       │
                 ▼       ▼
            integrate   hitl（人工决策，revise 恢复 → 三路并行重跑）
                 │
                END

并行依据：budget（目的地消费水平估算）与 safety（天气+城市级安全）
不依赖 itinerary 的输出，只依赖 coordinator 的归一化结果。
sub_reports 为 _append_reducer，并行写入安全；review 天然充当 fan-in 汇聚点。

路由规则：
- 串行段节点只写 state['next_node']，route_decision(state) 决定下一跳
- fan-out 段（coordinator / revise / hitl 恢复）用 path 返回节点列表扇出
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


def cleanup_old_checkpoints(keep_last: int = 20) -> int:
    """每个 thread 只保留最近 keep_last 条 checkpoint，防止 SQLite 库无限膨胀。

    langgraph-checkpoint-sqlite 没有内置 TTL/深度清理，而每个 checkpoint 存的是
    全量 state（含 final_plan 和 sub_reports），一轮 6 节点 + 若干次修订就是
    十几条，debug 场景曾把库撑到 873MB。

    Returns:
        删除的 checkpoint 行数（0 表示无需清理）
    """
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH))
    try:
        old = conn.execute(
            "SELECT rowid FROM ("
            "  SELECT rowid,"
            "         ROW_NUMBER() OVER (PARTITION BY thread_id ORDER BY rowid DESC) AS rn"
            "  FROM checkpoints"
            ") WHERE rn > ?",
            (keep_last,),
        ).fetchall()
        if not old:
            return 0
        conn.executemany("DELETE FROM checkpoints WHERE rowid=?", old)
        # 清掉引用已删 checkpoint 的 writes；最新的 keep_last 条及其 writes 都保留，
        # HITL 暂停线程的 pending writes 挂在最新 checkpoint 上，不受影响
        conn.execute(
            "DELETE FROM writes WHERE NOT EXISTS ("
            "  SELECT 1 FROM checkpoints c"
            "  WHERE c.thread_id = writes.thread_id AND c.checkpoint_id = writes.checkpoint_id"
            ")"
        )
        conn.commit()
        # 回收磁盘空间（否则 sqlite 文件不会自动缩小）
        try:
            conn.execute("VACUUM")
        except Exception as exc:
            print(f"[cleanup] VACUUM 失败（不影响数据）: {exc}")
        return len(old)
    finally:
        conn.close()


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

    # ---- fan-out 并行拓扑 ----
    # 依赖分析：budget（目的地消费水平估算）与 safety（天气+城市级安全提示）
    # 只依赖 coordinator 的归一化结果，不依赖 itinerary 的输出 → 三个 worker 可并行。
    # LangGraph 同一 superstep 内的节点并发执行、全部完成后才进入下一 superstep，
    # review 恰好充当 fan-in 汇聚点（sub_reports 是 _append_reducer，并行写入安全）。
    WORKERS = ["itinerary", "budget", "safety"]

    # coordinator → 三路 fan-out（固定扇出，不读 next_node）
    sg.add_conditional_edges(
        source="coordinator",
        path=lambda state: WORKERS,
        path_map={w: w for w in WORKERS},
    )

    # 三个 worker → review：固定路由（不走串行链）。
    # worker 超时返回的 next_node='hitl' 在这里被有意忽略 —— 并行分支的路由无法
    # 感知兄弟分支，统一走 review，由 review 后的 hitl_gate_route 读取超时标记接管
    for node_name in WORKERS:
        sg.add_conditional_edges(
            source=node_name,
            path=lambda state: ROUTE_REVIEW,
            path_map={ROUTE_REVIEW: "review"},
        )

    # review → HITL gate：revise 时三路并行重跑
    def _revise_fan_out(state) -> str | list[str]:
        key = hitl_gate_route(state)  # 'hitl' / 'itinerary' / 'integrate'
        if key == "itinerary":
            return WORKERS
        return key

    sg.add_conditional_edges(
        source="review",
        path=_revise_fan_out,
        path_map={
            "hitl": "hitl",
            **{w: w for w in WORKERS},
            "integrate": "integrate",
        },
    )

    # hitl → 恢复路由：revise（next_node='itinerary'）时同样三路并行重跑
    def _hitl_fan_out(state) -> str | list[str]:
        key = route_decision(state)
        if key == "itinerary":
            return WORKERS
        return key

    sg.add_conditional_edges(
        source="hitl",
        path=_hitl_fan_out,
        path_map=PATH_MAP,
    )

    # integrate 保持通用路由（收尾 → END）
    sg.add_conditional_edges(
        source="integrate",
        path=route_decision,
        path_map=PATH_MAP,
    )

    checkpointer = _get_checkpointer(checkpointer_path)
    return sg.compile(checkpointer=checkpointer)


def _extract_session_summary_from_graph(thread_id: str) -> dict:
    """通过 LangGraph API 获取最新 state 的摘要（比直接 pickle blob 更可靠）"""
    try:
        graph = build_graph()
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        if state is None:
            return {"title": "", "city": "", "days": 0, "people": 0, "budget": 0, "final_plan_len": 0}
        values = state.values or {}

        parsed = values.get("parsed_info") or {}
        final_plan = values.get("final_plan", "") or ""
        user_request = values.get("user_request", "") or ""

        city = parsed.get("city", "") if isinstance(parsed, dict) else ""
        days = parsed.get("days", 0) if isinstance(parsed, dict) else 0
        people = parsed.get("people", 0) if isinstance(parsed, dict) else 0
        budget = parsed.get("budget", 0) if isinstance(parsed, dict) else 0

        # 兜底：按字段逐个补——原来 if not parts 整体判断，
        # city 有值但 days 缺失时兜底永远不触发，标题只剩城市
        import re
        if not city:
            for cn in ["北京", "上海", "成都", "南京", "杭州", "广州", "深圳", "三亚", "西安",
                       "重庆", "武汉", "长沙", "青岛", "厦门", "大理", "丽江", "香格里拉", "九寨"]:
                if cn in user_request or cn in final_plan:
                    city = cn
                    break
        if not days:
            m = re.search(r"(\d+)\s*天", user_request + final_plan[:1000])
            if m:
                days = int(m.group(1))

        parts = []
        if city:
            parts.append(city)
        if days:
            parts.append(f"{days}天")
        if people:
            parts.append(f"{people}人")
        if budget:
            parts.append(f"¥{budget:,}")

        return {
            "title": " ".join(parts) if parts else "未命名方案",
            "city": city,
            "days": days,
            "people": people,
            "budget": budget,
            "final_plan_len": len(final_plan) if isinstance(final_plan, str) else 0,
        }
    except Exception:
        return {"title": "", "city": "", "days": 0, "people": 0, "budget": 0, "final_plan_len": 0}


def list_sessions():
    """列出所有持久化会话（从 SQLite checkpoints 表检索，附带摘要）"""
    import sqlite3

    db_path = DB_PATH
    if not db_path.exists():
        return []

    sessions = []
    conn = sqlite3.connect(str(db_path))
    try:
        # SQLiteSaver 用 checkpoint_ns + checkpoint_id，按 rowid 取最新
        rows = conn.execute(
            "SELECT thread_id, MAX(rowid) FROM checkpoints GROUP BY thread_id ORDER BY MAX(rowid) DESC LIMIT 50"
        ).fetchall()
        for row in rows:
            tid = row[0]
            # 用 graph API 获取真实摘要（比直接 pickle blob 可靠）
            summary = _extract_session_summary_from_graph(tid)

            # 取 created_at（从 metadata blob 里解析？SQLiteSaver 没有直接的 created_at 列）
            sessions.append({
                "thread_id": tid,
                "title": summary["title"],
                "city": summary["city"],
                "days": summary["days"],
                "people": summary["people"],
                "budget": summary["budget"],
                "final_plan_len": summary["final_plan_len"],
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
