"""
飞猪 MCP Client 封装

budget 子 Agent 通过这里的同步函数查询航班/酒店价格。
内部每次拉起 fliggy_server.py 子进程（stdio 传输），拿到结构化数据后返回。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# MCP Server 位置
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SERVER_SCRIPT = _PROJECT_ROOT / "mcp_server" / "fliggy_server.py"


async def _call_tool_async(tool_name: str, args: dict[str, Any]) -> Any:
    """异步核心：stdio 拉起 MCP Server，call_tool"""
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(_SERVER_SCRIPT)],
        cwd=str(_PROJECT_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.call_tool(tool_name, args)
            # FastMCP tool 返回 text 内容
            if resp.content and resp.content[0].type == "text":
                return json.loads(resp.content[0].text)
            return None


def _call_tool_sync(tool_name: str, args: dict[str, Any]) -> Any:
    """同步包装：在新事件循环里跑 async"""
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_call_tool_async(tool_name, args))
        loop.close()
        return result
    except Exception as exc:
        print(f"  [mcp] 调用 {tool_name} 失败: {exc}")
        return None


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------

def query_flights(dep_city: str, arr_city: str, dep_date: str,
                  ret_date: str | None = None, adult_num: int = 1) -> dict | None:
    """
    查询航班价格（同步）。
    返回 MCP Server 的原始 dict（含 dep / ret / note 字段）；失败返回 None。
    """
    return _call_tool_sync("search_flights", {
        "dep_city": dep_city,
        "arr_city": arr_city,
        "dep_date": dep_date,
        "ret_date": ret_date or "",
        "adult_num": adult_num,
    })


def query_hotels(city: str, checkin: str, checkout: str, guest_num: int = 2) -> dict | None:
    """
    查询酒店价格（同步）。
    返回 MCP Server 的原始 dict（含 hotels / nights / note 字段）；失败返回 None。
    """
    return _call_tool_sync("search_hotels", {
        "city": city,
        "checkin_date": checkin,
        "checkout_date": checkout,
        "guest_num": guest_num,
    })


if __name__ == "__main__":
    # 独立验证
    print("=== query_flights(成都→上海, 2025-10-01) ===")
    f = query_flights("成都", "上海", "2025-10-01", "2025-10-05")
    if f:
        print(f"  ✅ {len(f['dep'])} 班航班, 示例 ¥{f['dep'][0]['price_range']['economy_low']}")
    else:
        print("  ❌ 查询失败")

    print("=== query_hotels(成都, 2025-10-01~2025-10-03) ===")
    h = query_hotels("成都", "2025-10-01", "2025-10-03")
    if h:
        print(f"  ✅ {len(h['hotels'])} 家酒店, {h['nights']} 晚")
    else:
        print("  ❌ 查询失败")
