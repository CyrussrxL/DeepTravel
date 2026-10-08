"""
Agent 共享工具：LLM 初始化、消息提取、路由函数等

关键设计：
- 路由函数 route_decision 为纯函数，仅读取 state.next_node
- 统一从 state["messages"] 中提取用户最新文本
- 支持 MOCK_LLM 环境变量切换 mock 响应（用于无 API 的端到端测试）
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AnyMessage

load_dotenv()

# --------- 超时常量 ---------
LLM_TIMEOUT_SEC = 300  # LLM 调用超时（秒），qwen3.8-27b 大模型长输出需要更久

# --------- LLM 初始化 ---------

_mock_flag = os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")


def _make_http_client():
    """
    创建禁用系统代理的 httpx.Client。

    背景：部分代理工具（Clash/v2rayN 等）会给所有请求注入
    127.0.0.1 代理，但代理可能对阿里云域名 SSL 转发不兼容
    （表现为 SSL: UNEXPECTED_EOF_WHILE_READING）。
    LLM API 端点通常走国内直连，绕过系统代理更稳定。
    """
    import httpx
    transport = httpx.HTTPTransport(proxy=None)
    return httpx.Client(transport=transport, timeout=httpx.Timeout(90))


def get_llm():
    """
    获取 LLM 实例。

    优先级：
    1. MOCK_LLM=true → 抛异常，调用方回退 mock
    2. DASHSCOPE_API_KEY 存在 → ChatOpenAI + DashScope 兼容端点
       （qwen3.8 新系列模型必须走 compatible-mode，旧 ChatTongyi 端点不认）
    3. OPENAI_API_KEY 兜底

    所有路径均禁用系统代理（通过自定义 http_client），
    避免 Clash/v2rayN 等代理工具的 SSL 干扰。
    """
    if is_mock_mode():
        raise RuntimeError("MOCK_LLM=true, skip real LLM call")

    from langchain_openai import ChatOpenAI

    client = _make_http_client()

    # 主路径：DashScope 兼容端点
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_key:
        # qwen3 系列默认开启思考模式，reasoning 内容会占用 output token 预算。
        # max_tokens 被思考吃光时 content 会返回空字符串（实测 2000 不够用），
        # 因此默认关闭思考。中文长报告 ≈1.5 字/token，6000 实测会在 Day 2/3
        # 截断（4325 字报告即触顶），提高到 8000。
        enable_thinking = os.getenv("DASHSCOPE_ENABLE_THINKING", "false").lower() in ("true", "1", "yes")
        return ChatOpenAI(
            api_key=dashscope_key,
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model=os.getenv("DASHSCOPE_MODEL", "qwen3.8-27b"),
            temperature=0.2,
            max_tokens=int(os.getenv("DASHSCOPE_MAX_TOKENS", "8000")),
            http_client=client,
            extra_body={"enable_thinking": enable_thinking},
        )

    # 兜底：原生 OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=2000,
            http_client=client,
        )

    raise RuntimeError("No LLM API key found (DASHSCOPE_API_KEY or OPENAI_API_KEY)")


def is_mock_mode() -> bool:
    return os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")


# --------- 消息提取 ---------

def extract_last_user_text(messages: list[AnyMessage]) -> str:
    """
    从 messages 中提取最后一条用户文本。

    LangGraph 的 add_messages 会将消息规范化为 langchain_core 的
    HumanMessage / AIMessage 对象。优先按 type=='human' 查找，
    再兜底 (role, content) tuple 格式。
    """
    if not messages:
        return ""
    for msg in reversed(messages):
        # langchain_core 对象
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content or ""
        # 兜底 tuple
        if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] in ("human", "user"):
            return msg[1] or ""
    return ""


# --------- 路由函数 ---------

# 路由目标常量
ROUTE_ITINERARY = "itinerary"
ROUTE_BUDGET = "budget"
ROUTE_SAFETY = "safety"
ROUTE_REVIEW = "review"
ROUTE_INTEGRATE = "integrate"
ROUTE_END = "end"


def route_decision(state: dict[str, Any]) -> str:
    """
    纯函数路由：仅读取 state['next_node'] 决定下一跳。

    所有节点都只写 state['next_node']，本函数作为唯一出口，
    避免同一节点上同时出现 add_edge 和 add_conditional_edges 的冲突。
    """
    return state.get("next_node", ROUTE_END)
