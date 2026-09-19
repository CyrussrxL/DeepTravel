"""
DeepTravel HITL (Human-In-The-Loop) 子模块

触发 HITL 的场景：
1. MAX_REVISION 超限（默认 2 次）→ 进入人工审核
2. Review 审核结果带 "不确定" / "存疑" → 进入人工审核
3. ⭐ LLM 调用超时（新增）→ 任意节点都能触发

实现方式：
- 新增 review 后面的 hitl_gate 条件路由（原有）
- 任意节点 LLM 调用超时 → 返回 next_node="hitl" + hitl_status="waiting"（新增）
- hitl 节点写入 next_node='end' 并暂停
- 通过 LangGraph checkpointer 恢复会话，等待人工决策
- 用户 approve → integrate / 注入 mock 继续；用户 revise → itinerary（重置 revision_count）
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

    # 条件 3: 任意节点超时（由节点自身 next_node="hitl" 触发，这里兜底检查）
    if state.get("hitl_status") == "waiting" and state.get("hitl_reason", "").find("超时") >= 0:
        hitl_triggered = True
        reason = state.get("hitl_reason", "")

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

    恢复执行时（hitl_status != "waiting"）：
      - approved  → 走 hitl_next_node 或默认 integrate
      - revised   → 走 itinerary 重跑
    """
    # 如果已经有决策，跳过等待直接继续
    hitl_status = state.get("hitl_status", "")
    if hitl_status in ("resolved_approve", "resolved_revise"):
        if hitl_status == "resolved_approve":
            next_node = state.get("hitl_next_node") or state.get("next_node") or "integrate"
            # 如果 review_status 被设成 approved，那就直接到 integrate
            if state.get("review_status") == "approved":
                next_node = "integrate"
        else:  # resolved_revise
            next_node = "itinerary"
        print(f"[HITL] 恢复执行 — status={hitl_status}, 前往 {next_node}")
        return {
            "next_node": next_node,
            "hitl_status": hitl_status,
        }

    # 正常等待流程
    revision_count = state.get("revision_count", 0)
    review_feedback = state.get("review_feedback", "")
    hitl_stage = state.get("hitl_stage", "")
    hitl_reason = state.get("hitl_reason", "")

    if not hitl_reason:
        if revision_count >= MAX_REVISION_BEFORE_HITL:
            hitl_stage = hitl_stage or "review"
            hitl_reason = f"连续修订 {revision_count} 次已达上限 ({MAX_REVISION_BEFORE_HITL})"
        elif state.get("hitl_status") == "waiting":
            hitl_stage = hitl_stage or "unknown"
            hitl_reason = hitl_reason or state.get("hitl_reason", "需要人工审核")
        elif hitl_stage in ("", "review") and review_feedback:
            hitl_stage = "review"
            hitl_reason = f"审核结果包含不确定判断：{review_feedback[:80]}"
        else:
            hitl_stage = hitl_stage or "unknown"
            hitl_reason = hitl_reason or "需要人工审核"

    print(f"[HITL] 会话等待人工审核 — stage={hitl_stage}, 原因={hitl_reason[:60]}, rev={revision_count}")
    return {
        "next_node": "end",  # 停止图执行
        "hitl_status": "waiting",
        "hitl_stage": hitl_stage,
        "hitl_reason": hitl_reason,
    }


# --------- 人工决策：恢复执行 ---------

def hitl_approve_state(state: TravelState) -> dict:
    """
    人工审核：批准（写入 state，让后续节点消费）。

    超时场景：hitl_stage 非空 → 清除 hitl 标记，保留用户可能已手动补充的字段，
             让图根据 hitl_next_node 或 review_status 继续执行。
    review 场景：review_status=approved → integrate。
    """
    update: dict[str, Any] = {
        "hitl_status": "resolved_approve",
        "hitl_stage": "",
        "hitl_reason": "",
    }

    # 超时场景：有 hitl_next_node 就按它走，否则默认 integrate
    if state.get("hitl_next_node"):
        update["next_node"] = state["hitl_next_node"]
    else:
        # review 场景
        update["review_status"] = "approved"
        update["next_node"] = "integrate"

    return update


def hitl_revise_state(state: TravelState) -> dict:
    """人工审核：要求继续修订（重置 revision_count 给 LLM 再试一次）"""
    update: dict[str, Any] = {
        "review_status": "revise",
        "revision_count": 0,  # 重置让 LLM 再跑
        "next_node": "itinerary",
        "hitl_status": "resolved_revise",
        "hitl_stage": "",
        "hitl_reason": "",
    }
    # 清除可能存在的超时残留
    if state.get("hitl_next_node"):
        update["hitl_next_node"] = ""
    return update
