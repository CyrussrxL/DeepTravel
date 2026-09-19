"""
DeepTravel FastAPI 服务

提供：
- POST /api/plan     — 提交旅行规划请求（SSE 流式推送各 Agent 进度）
- GET  /api/health   — 健康检查
- GET  /             — 返回前端 index.html
- /static/*          — 静态资源

运行：
    uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .graph import build_graph, list_sessions, get_session_state

load_dotenv()

# --------- FastAPI App ---------

app = FastAPI(
    title="DeepTravel",
    description="基于 LangGraph 的多 Agent 旅行规划管家",
    version="0.1.0",
)

# CORS（开发时放开全部，生产环境限制域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- 路径常量 ---------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

# 确保 static 目录存在
STATIC_DIR.mkdir(exist_ok=True)


# --------- Agent 元信息（前端渲染用） ---------

AGENT_META = {
    "coordinator": {
        "label": "需求协调",
        "icon": "🎯",
        "color": "#6366f1",
        "desc": "归一化用户需求",
    },
    "itinerary": {
        "label": "行程规划",
        "icon": "📅",
        "color": "#0ea5e9",
        "desc": "生成每日行程（接入高德地图）",
    },
    "budget": {
        "label": "预算估算",
        "icon": "💰",
        "color": "#10b981",
        "desc": "按类别拆分预算",
    },
    "safety": {
        "label": "安全评估",
        "icon": "🛡️",
        "color": "#f59e0b",
        "desc": "天气 + 安全提示（接入心知天气）",
    },
    "review": {
        "label": "方案审核",
        "icon": "🔍",
        "color": "#ef4444",
        "desc": "综合评审三份子报告",
    },
    "integrate": {
        "label": "方案整合",
        "icon": "✨",
        "color": "#8b5cf6",
        "desc": "整合最终方案",
    },
    "hitl": {
        "label": "人工审核",
        "icon": "👤",
        "color": "#d946ef",
        "desc": "等待人工审核决策",
    },
}


# --------- 请求/响应模型 ---------

class PlanRequest(BaseModel):
    user_input: str
    mock: bool = False
    thread_id: str | None = None  # 可选：恢复已有会话继续
    initial_revision_count: int = 0  # 可选：测试用，让 review 更快触发 HITL


class AdjustRequest(BaseModel):
    thread_id: str                # 必须：要调整的会话
    modify_request: str           # 用户修改请求
    mock: bool = False


# --------- 辅助：把 LangGraph 事件转为 SSE ---------

def _event_from_node(node_name: str, state_update: dict, duration_ms: int) -> dict:
    """把一个节点的输出包装成前端友好的事件"""
    meta = AGENT_META.get(node_name, {"label": node_name, "icon": "🤖", "color": "#888", "desc": "未知节点"})

    # 提取该节点产出的主要内容（子报告）
    report_content = ""
    sub_reports = state_update.get("sub_reports", [])
    # 找到最后一个属于本节点的 sub_report
    for sr in reversed(sub_reports):
        if sr.get("agent") == node_name:
            report_content = sr.get("report", "")
            break

    # 路由决定
    next_node = state_update.get("next_node", "")
    review_status = state_update.get("review_status", "")
    revision_count = state_update.get("revision_count", 0)

    # HITL 信息（超时/审核触发时携带）
    hitl_status = state_update.get("hitl_status", "")
    hitl_stage = state_update.get("hitl_stage", "")
    hitl_reason = state_update.get("hitl_reason", "")
    hitl_next_node = state_update.get("hitl_next_node", "")

    return {
        "type": "node",
        "node": node_name,
        "label": meta["label"],
        "icon": meta["icon"],
        "color": meta["color"],
        "desc": meta["desc"],
        "duration_ms": duration_ms,
        "report": report_content,
        "next_node": next_node,
        "review_status": review_status,
        "revision_count": revision_count,
        "hitl_status": hitl_status,
        "hitl_stage": hitl_stage,
        "hitl_reason": hitl_reason,
        "hitl_next_node": hitl_next_node,
    }


# --------- SSE 端点 ---------

@app.post("/api/plan")
def plan(req: PlanRequest) -> StreamingResponse:
    """
    提交旅行规划请求，通过 SSE 逐节点推送进度事件（同步 StreamingResponse）。
    """
    print(f"[plan] 进入 plan() mock={req.mock} user_input={req.user_input[:20]}", flush=True)
    # mock 开关：req.mock 优先；否则看环境变量；都没有就不 mock
    if req.mock:
        os.environ["MOCK_LLM"] = "true"
    elif os.getenv("MOCK_LLM", "").lower() not in ("true", "1", "yes"):
        os.environ.pop("MOCK_LLM", None)

    print(f"[plan] 调 build_graph()...", flush=True)
    graph = build_graph()
    print(f"[plan] build_graph 完成", flush=True)
    thread_id = req.thread_id or f"deeptravel-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=req.user_input)],
        "revision_count": req.initial_revision_count,
    }
    print(f"[plan] 返回 StreamingResponse, thread_id={thread_id}", flush=True)

    def generate():
        all_sub_reports: list[dict] = []
        final_plan = ""
        last_revision_count = 0
        total_start = time.time()

        # 1. start 事件
        start_evt = {
            "type": "start",
            "thread_id": thread_id,
            "user_input": req.user_input,
            "agents": [
                {"node": name, **AGENT_META[name]}
                for name in ("coordinator", "itinerary", "budget", "safety", "review", "integrate")
            ],
            "mock": req.mock or os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes"),
        }
        chunk = f"data: {json.dumps(start_evt, ensure_ascii=False)}\n\n"
        print(f"  [server.generate] yield start ({len(chunk)} bytes)")
        yield chunk

        # 2. stream
        hitl_triggered = False
        hitl_info: dict = {}
        try:
            print(f"  [server.generate] graph.stream start...")
            for event in graph.stream(initial_state, config):
                print(f"  [server.generate] 收到事件 {list(event.keys())}")
                for node_name, state_update in event.items():
                    if "sub_reports" in state_update:
                        all_sub_reports.extend(state_update["sub_reports"])

                    cur_rev = state_update.get("revision_count", 0)
                    if cur_rev > last_revision_count:
                        yield f"data: {json.dumps({'type':'revise','from_node':'review','to_node':'itinerary','count':cur_rev}, ensure_ascii=False)}\n\n"
                    last_revision_count = cur_rev

                    if "final_plan" in state_update:
                        final_plan = state_update["final_plan"]

                    # 先 yield node 事件（已携带 hitl 字段）
                    yield f"data: {json.dumps(_event_from_node(node_name, state_update, 0), ensure_ascii=False)}\n\n"

                    # 检测 HITL 触发
                    if (node_name == "hitl"
                        or state_update.get("hitl_status") == "waiting"
                        or state_update.get("next_node") == "hitl"):
                        hitl_triggered = True
                        hitl_info = {
                            "hitl_stage": state_update.get("hitl_stage", "") or "review",
                            "hitl_reason": state_update.get("hitl_reason", "") or "需要人工审核",
                            "hitl_next_node": state_update.get("hitl_next_node", ""),
                            "revision_count": cur_rev,
                        }
                        # 专门的 hitl_start 事件
                        yield f"data: {json.dumps({'type':'hitl_start', **hitl_info}, ensure_ascii=False)}\n\n"

                    if state_update.get("next_node") == "end":
                        break
        except Exception as exc:
            err_evt = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err_evt, ensure_ascii=False)}\n\n"
            return

        # 3. done（HITL 场景下 final_plan 为空、hitl_waiting=True）
        done_evt = {
            "type": "done",
            "thread_id": thread_id,
            "final_plan": final_plan,
            "sub_reports": all_sub_reports,
            "total_ms": int((time.time() - total_start) * 1000),
            "revision_count": last_revision_count,
            "hitl_triggered": hitl_triggered,
            "hitl_stage": hitl_info.get("hitl_stage", ""),
            "hitl_reason": hitl_info.get("hitl_reason", ""),
        }
        yield f"data: {json.dumps(done_evt, ensure_ascii=False)}\n\n"
        print(f"  [server.generate] done hitl={hitl_triggered}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# --------- 其他路由 ---------

@app.get("/api/health")
def health():
    mock = os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")
    return {
        "status": "ok",
        "mock_mode": mock,
        "model": os.getenv("DASHSCOPE_MODEL", "qwen3.8-27b"),
    }


# --------- 会话持久化 API ---------

@app.get("/api/sessions")
def sessions_list():
    """列出所有持久化会话"""
    sessions = list_sessions()
    return {"count": len(sessions), "sessions": sessions}


@app.get("/api/sessions/{thread_id}")
def session_state(thread_id: str):
    """获取某个会话的最新 state"""
    state = get_session_state(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"会话 {thread_id} 未找到")
    return state


@app.delete("/api/sessions/{thread_id}")
def session_delete(thread_id: str):
    """删除某个会话"""
    import sqlite3
    from .graph import DB_PATH
    if not DB_PATH.exists():
        return {"ok": True}
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "thread_id": thread_id}


# --------- 调整子图 API ---------

@app.post("/api/adjust")
def adjust(req: AdjustRequest) -> StreamingResponse:
    """交互式调整已有方案（同步 StreamingResponse）。"""
    if req.mock:
        os.environ["MOCK_LLM"] = "true"
    elif os.getenv("MOCK_LLM", "").lower() not in ("true", "1", "yes"):
        os.environ.pop("MOCK_LLM", None)

    from .adjust import run_adjust, determine_adjust_start, build_adjust_graph

    # 先同步恢复 state
    from .graph import build_graph
    full_graph = build_graph()
    state_obj = full_graph.get_state({"configurable": {"thread_id": req.thread_id}})
    if state_obj is None:
        raise HTTPException(status_code=404, detail=f"会话 {req.thread_id} 不存在")
    original_state = dict(state_obj.values)
    start_node = determine_adjust_start(original_state, req.modify_request)

    adjust_graph = build_adjust_graph(start_node)
    config = {"configurable": {"thread_id": req.thread_id}}
    from langchain_core.messages import HumanMessage
    adjust_state = dict(original_state)
    adjust_state["messages"] = list(original_state.get("messages", [])) + [
        HumanMessage(content=f"[用户调整] {req.modify_request}")
    ]
    if start_node == "coordinator":
        adjust_state["user_request"] = req.modify_request
    # 保留已有 revision_count——多次 adjust 可以累积触发 HITL
    # 但第一次 adjust 如果 revision_count 是 0，那 review mock 会 APPROVED（正常）
    # 连续多次 adjust 后 revision_count 累积 → 超限 → HITL
    adjust_state["review_status"] = "pending"

    def generate():
        total_start = time.time()

        start_evt = {
            "type": "start_adjust",
            "thread_id": req.thread_id,
            "modify_request": req.modify_request,
            "start_node": start_node,
            "mock": req.mock or os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes"),
        }
        yield f"data: {json.dumps(start_evt, ensure_ascii=False)}\n\n"

        all_sub_reports: list[dict] = []
        final_plan = ""
        last_revision = 0

        try:
            for event in adjust_graph.stream(adjust_state, config):
                for node_name, state_update in event.items():
                    if "sub_reports" in state_update:
                        all_sub_reports.extend(state_update["sub_reports"])

                    cur_rev = state_update.get("revision_count", 0)
                    if cur_rev > last_revision:
                        revise_evt = {"type": "revise", "from_node": "review", "to_node": start_node, "count": cur_rev}
                        yield f"data: {json.dumps(revise_evt, ensure_ascii=False)}\n\n"
                    last_revision = cur_rev

                    if "final_plan" in state_update:
                        final_plan = state_update["final_plan"]

                    node_evt = _event_from_node(node_name, state_update, 0)
                    yield f"data: {json.dumps(node_evt, ensure_ascii=False)}\n\n"

                    if state_update.get("next_node") == "end":
                        break
        except Exception as exc:
            err_evt = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err_evt, ensure_ascii=False)}\n\n"
            return

        done_evt = {
            "type": "done",
            "final_plan": final_plan,
            "sub_reports": all_sub_reports,
            "total_ms": int((time.time() - total_start) * 1000),
            "revision_count": last_revision,
            "adjusted": True,
        }
        yield f"data: {json.dumps(done_evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------- HITL 人工审核 API ---------

class HitlDecisionRequest(BaseModel):
    decision: str  # "approve" | "revise"
    note: str = ""  # 可选：人工附加说明


@app.post("/api/hitl/{thread_id}")
def hitl_decision(thread_id: str, req: HitlDecisionRequest) -> StreamingResponse:
    """人工审核决策（同步 StreamingResponse）。"""
    from .graph import build_graph
    from .hitl import hitl_approve_state, hitl_revise_state

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state_obj = graph.get_state(config)
    if state_obj is None:
        raise HTTPException(status_code=404, detail=f"会话 {thread_id} 不存在")

    # 注入人工决策（修改 checkpoint 里的 state）
    if req.decision == "approve":
        human_update = hitl_approve_state(dict(state_obj.values))
    elif req.decision == "revise":
        human_update = hitl_revise_state(dict(state_obj.values))
        from langchain_core.messages import HumanMessage
        if req.note:
            human_update["messages"] = [HumanMessage(content=f"[人工修订] {req.note}")]
    else:
        def gen_err():
            err_evt = {"type": "error", "message": f"无效决策: {req.decision}"}
            yield f"data: {json.dumps(err_evt, ensure_ascii=False)}\n\n"
        return StreamingResponse(gen_err(), media_type="text/event-stream")

    # 修改 checkpoint 中的 state（让 hitl_node 下次执行时能看到 resolved 状态）
    graph.update_state(config, human_update)

    def generate():
        total_start = time.time()

        hitl_start_evt = {
            "type": "hitl_decision",
            "thread_id": thread_id,
            "decision": req.decision,
            "note": req.note,
        }
        yield f"data: {json.dumps(hitl_start_evt, ensure_ascii=False)}\n\n"

        all_sub_reports: list[dict] = []
        final_plan = ""
        try:
            for event in graph.stream(None, config):
                for node_name, state_update in event.items():
                    if "sub_reports" in state_update:
                        all_sub_reports.extend(state_update["sub_reports"])
                    if "final_plan" in state_update:
                        final_plan = state_update["final_plan"]

                    node_evt = _event_from_node(node_name, state_update, 0)
                    yield f"data: {json.dumps(node_evt, ensure_ascii=False)}\n\n"

                    if state_update.get("next_node") == "end":
                        break
        except Exception as exc:
            err_evt = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err_evt, ensure_ascii=False)}\n\n"
            return

        done_evt = {
            "type": "done",
            "final_plan": final_plan,
            "sub_reports": all_sub_reports,
            "total_ms": int((time.time() - total_start) * 1000),
            "from_hitl": True,
        }
        yield f"data: {json.dumps(done_evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------- 静态文件挂载（前端） ---------

# 先 mount static 目录（前端资源）
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    """返回前端首页"""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="前端 index.html 未找到，请先创建")
    return FileResponse(str(index))
