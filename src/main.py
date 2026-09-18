"""
DeepTravel MVP 入口

使用：
    # mock 模式（无需 API key，快速验证图路由）
    $ set MOCK_LLM=true && python -m src.main

    # 真实 LLM 模式
    $ copy .env.example .env   # 填入 API key
    $ python -m src.main
"""

from __future__ import annotations

import os
import sys

from langchain_core.messages import HumanMessage

from .graph import build_graph


def run(user_input: str, mock: bool = False) -> str:
    """
    运行 DeepTravel 规划管线，返回最终方案文本。

    Args:
        user_input: 用户旅行需求
        mock: 是否使用 mock LLM（跳过真实 API 调用）
    """
    if mock:
        os.environ["MOCK_LLM"] = "true"

    app = build_graph()

    # 初始状态：只有用户消息
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "revision_count": 0,
    }

    print("=" * 60)
    print("  DeepTravel —— 多Agent旅行规划管家 (MVP)")
    print("=" * 60)
    print(f"\n用户输入：{user_input}")
    print(f"模式：     {'MOCK（无API调用）' if mock else '真实LLM'}")
    print("\n" + "-" * 40)

    # 执行图，逐节点打印输出
    result = app.invoke(initial_state)

    print("-" * 40)
    print(f"\n总修订次数：{result.get('revision_count', 0)}")
    print(f"子报告数量：{len(result.get('sub_reports', []))}")
    print("\n" + "=" * 60)
    print("  最终方案")
    print("=" * 60)
    print(result.get("final_plan", "（未生成最终方案）"))

    return result.get("final_plan", "")


if __name__ == "__main__":
    # 命令行传入或使用默认测试需求
    user_req = sys.argv[1] if len(sys.argv) > 1 else "我想去日本东京旅行7天，两个人，预算25000元人民币，喜欢自然风光和美食"

    mock_mode = os.getenv("MOCK_LLM", "true").lower() in ("true", "1", "yes")
    run(user_req, mock=mock_mode)
