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
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .graph import build_graph

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
}


# --------- 请求/响应模型 ---------

class PlanRequest(BaseModel):
    user_input: str
    mock: bool = False  # 是否强制 mock（忽略 .env 里的 MOCK_LLM）


# --------- 辅助：把 LangGraph 事件转为 SSE ---------

def _event_from_node(node_name: str, state_update: dict, duration_ms: int) -> dict:
    """把一个节点的输出包装成前端友好的事件"""
    meta = AGENT_META.get(node_name, {"label": node_name, "icon": "🤖", "color": "#888"})

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
    }


# --------- SSE 端点 ---------

@app.post("/api/plan")
async def plan(req: PlanRequest) -> StreamingResponse:
    """
    提交旅行规划请求，通过 SSE 逐节点推送进度事件。

    SSE 事件格式（data: {...}）：
      - 开始事件    {"type":"start", "user_input": "...", "agents":[...]}
      - 节点完成    {"type":"node", "node":"itinerary", "label":"...", "report":"...", ...}
      - 修订回退    {"type":"revise", "from_node":"review", "to_node":"itinerary", "count":1}
      - 最终结果    {"type":"done", "final_plan":"...", "sub_reports":[...], "total_ms":12345}
      - 错误        {"type":"error", "message":"..."}
    """
    # 强制 mock 开关
    if req.mock:
        os.environ["MOCK_LLM"] = "true"
    else:
        os.environ.pop("MOCK_LLM", None)

    async def generate() -> AsyncGenerator[str, None]:
        graph = build_graph()
        initial_state = {
            "messages": [HumanMessage(content=req.user_input)],
            "revision_count": 0,
        }

        all_sub_reports: list[dict] = []
        final_plan = ""
        total_start = time.time()
        last_revision_count = 0

        # 1. 发送 start 事件
        start_evt = {
            "type": "start",
            "user_input": req.user_input,
            "agents": [
                {"node": name, **AGENT_META[name]}
                for name in ("coordinator", "itinerary", "budget", "safety", "review", "integrate")
            ],
            "mock": req.mock or os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes"),
        }
        yield f"data: {json.dumps(start_evt, ensure_ascii=False)}\n\n"

        # 2. 逐节点 stream
        try:
            for event in graph.stream(initial_state):
                for node_name, state_update in event.items():
                    node_start = time.time()

                    # 收集子报告
                    if "sub_reports" in state_update:
                        all_sub_reports.extend(state_update["sub_reports"])

                    # 捕获修订变化
                    cur_rev = state_update.get("revision_count", 0)
                    if cur_rev > last_revision_count:
                        revise_evt = {
                            "type": "revise",
                            "from_node": "review",
                            "to_node": "itinerary",
                            "count": cur_rev,
                        }
                        yield f"data: {json.dumps(revise_evt, ensure_ascii=False)}\n\n"
                    last_revision_count = cur_rev

                    # 收集最终方案
                    if "final_plan" in state_update:
                        final_plan = state_update["final_plan"]

                    duration_ms = int((time.time() - node_start) * 1000)
                    node_evt = _event_from_node(node_name, state_update, duration_ms)
                    yield f"data: {json.dumps(node_evt, ensure_ascii=False)}\n\n"

                    # 如果节点已经跳到 END，就不再继续（END 不会出现在 graph.stream 输出里，
                    # 这里用 next_node=="end" 作为信号）
                    if state_update.get("next_node") == "end":
                        break

        except Exception as exc:
            err_evt = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err_evt, ensure_ascii=False)}\n\n"
            return

        # 3. done 事件
        done_evt = {
            "type": "done",
            "final_plan": final_plan,
            "sub_reports": all_sub_reports,
            "total_ms": int((time.time() - total_start) * 1000),
            "revision_count": last_revision_count,
        }
        yield f"data: {json.dumps(done_evt, ensure_ascii=False)}\n\n"

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
