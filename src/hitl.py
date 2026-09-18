"""
DeepTravel HITL (Human-In-The-Loop) 子模块

触发 HITL 的场景：
1. MAX_REVISION 超限（默认 2 次）→ 进入人工审核
2. Review 审核结果带 "不确定" / "存疑" → 进入人工审核

实现方式：
- 新增 review 后面的 hitl_gate 条件路由
- hitl 节点写入 next_node="hitl" 并暂停
- 通过 LangGraph checkpointer 恢复会话，等待人工决策
- 用户 approve → integrate；用户 revise → itinerary（重置 revision_count 让 LLM 再试一次）
"""

from __future__ import annotations

from typing import Any

from .state import TravelState


# --------- HITL 决策 ---------

MAX_REVISION_BEFORE_HITL = 2  # 超过 2 次自动 revise 就进 HITL


def hitl_gate_route(state: TravelState) -> str:
    """
    review 节点之后的条件路由。

    返回值：
    - "hitl":   进入人工审核
    - "integrate": 自动通过（review=approved 或 revise 且 revision_count < 上限）
    - "itinerary": revise 且未达上限 → 自动回退
    """
    review_status = state.get("review_status", "pending")
    revision_count = state.get("revision_count", 0)
    review_feedback = state.get("review_feedback", "")

    # 触发 HITL 的条件
    hitl_triggered = False
    reason = ""

    # 条件 1: revise 且 revision_count 超限
    if review_status == "revise" and revision_count >= MAX_REVISION_BEFORE_HITL:
        hitl_triggered = True
        reason = f"连续修订 {revision_count} 次，已达自动修订上限"

    # 条件 2: feedback 包含不确定性关键词
    uncertain_kw = ["不确定", "存疑", "可能", "有疑问", "难以判断", "需要确认", "保守"]
    if review_status == "approved":
        for kw in uncertain_kw:
            if kw in review_feedback:
                hitl_triggered = True
                reason = f"审核反馈包含不确定词「{kw}」：{review_feedback[:100]}"
                break

    if hitl_triggered:
        return "hitl"
    elif review_status == "revise":
        return "itinerary"
    else:  # approved 或 pending 默认
        return "integrate"


# --------- HITL 节点 ---------

def hitl_node(state: TravelState) -> dict:
    """
    人工审核节点。

    hitl 是一个"暂停点"——执行完这个节点后图就停止了（next_node='end'）。
    用户通过 /api/hitl/{thread_id} POST 注入人工决策，
    LangGraph 会从 checkpoint 恢复，带着新 state 继续执行。
    """
    print(f"[HITL] 会话等待人工审核 — revision_count={state.get('revision_count',0)}")
    return {
        "next_node": "end",  # 停止图执行
        "hitl_status": "waiting",  # 标记：前端显示审核面板
    }


# --------- 人工决策：恢复执行 ---------

def hitl_approve_state(state: TravelState) -> dict:
    """人工审核：批准（写入 state，让后续 integrate 节点消费）"""
    return {
        "review_status": "approved",
        "next_node": "integrate",
        "hitl_status": "resolved_approve",
    }


def hitl_revise_state(state: TravelState) -> dict:
    """人工审核：要求继续修订（重置 revision_count 给 LLM 再试一次）"""
    return {
        "review_status": "revise",
        "revision_count": 0,  # 重置让 LLM 再跑
        "next_node": "itinerary",
        "hitl_status": "resolved_revise",
    }
