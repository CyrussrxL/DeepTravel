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

    # 条件 1（主路径）: review 结构化评审主动要求人工
    # （LLM needs_human=true，或任一维评分 < 5 —— 代码级门控，见 review_node）
    if state.get("hitl_flag"):
        hitl_triggered = True

    # 条件 2: revise 且 revision_count 超限
    if not hitl_triggered and review_status == "revise" and revision_count >= MAX_REVISION_BEFORE_HITL:
        hitl_triggered = True

    # 条件 3: 任意节点超时（由节点自身 next_node="hitl" 触发，这里兜底检查）
    if not hitl_triggered and state.get("hitl_status") == "waiting" and state.get("hitl_reason", "").find("超时") >= 0:
        hitl_triggered = True

    # 条件 4（fallback）: 不确定性关键词 —— 仅当 review 走了旧路径
    # （结构化失败回退，review_scores 为空）时才用词典匹配，避免误触发
    uncertain_kw = ["不确定", "存疑", "可能", "有疑问", "难以判断", "需要确认", "保守"]
    if not hitl_triggered and review_status == "approved" and not state.get("review_scores"):
        for kw in uncertain_kw:
            if kw in review_feedback:
                hitl_triggered = True
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
    人工审核节点 —— 唯一的语义是「暂停并等待人工决策」。

    hitl 是一个"暂停点"——执行完这个节点后图就停止了（next_node='end'）。
    用户通过 /api/hitl/{thread_id} POST 注入人工决策，
    LangGraph 会从 checkpoint 恢复，带着新 state 继续执行。

    ⚠️ 恢复时本节点不会再执行：/api/hitl 用 graph.update_state 写入决策后，
    LangGraph 会依据 hitl 的出边重新计算 next，直接从 integrate / 子图起点续跑。
    因此本节点不需要、也不能依赖 「resolved_*」 之类的标记来判断是否恢复
    （那类标记会在 state 里残留，导致下一次触发 HITL 时被误判成"正在恢复"，
    从而跳过暂停、直接回退重跑）。
    """
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
    人工审核：批准。

    只写业务字段（next_node / review_status），命中后续节点的路由；
    同时把 hitl_status 清空，避免等待标记残留到下一次触发。
    超时场景：按 hitl_next_node 续跑；review 场景：approved → integrate。
    """
    update: dict[str, Any] = {
        "hitl_status": "",
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
        "hitl_status": "",
        "hitl_stage": "",
        "hitl_reason": "",
    }
    # 清除可能存在的超时残留
    if state.get("hitl_next_node"):
        update["hitl_next_node"] = ""
    return update
