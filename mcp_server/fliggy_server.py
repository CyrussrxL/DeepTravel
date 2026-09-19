"""
飞猪 MCP Server（DeepTravel 预算查询）

传输协议: stdio（进程间通信，LangGraph 直接子进程拉起）
MCP SDK: mcp 1.x FastMCP

暴露两个 tool:
  1. search_flights   — 航班列表（出发/到达/日期 → 价格+时刻）
  2. search_hotels    — 酒店列表（城市/入住/退房 → 价格区间+评分）

当前实现: mock-first（返回合理模拟数据，带价格区间）
后续接入淘宝开放平台 TOP 时，只需替换 _do_flights / _do_hotels 内部实现。
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# MCP Server 实例
# ---------------------------------------------------------------------------

mcp = FastMCP(name="deeptravel-fliggy")

# ---------------------------------------------------------------------------
# 城市 ↔ 三字码 映射（简化版，覆盖主流旅游城市）
# ---------------------------------------------------------------------------

_CITY_CODE: dict[str, str] = {
    "北京": "BJS", "上海": "SHA", "广州": "CAN", "深圳": "SZX",
    "成都": "CTU", "杭州": "HGH", "武汉": "WUH", "南京": "NKG",
    "西安": "XIY", "长沙": "CSX", "重庆": "CKG", "昆明": "KMG",
    "三亚": "SYX", "厦门": "XMN", "青岛": "TAO", "大连": "DLC",
    "丽江": "LJG", "大理": "DLU", "桂林": "KWL", "张家界": "DYG",
    "拉萨": "LXA", "乌鲁木齐": "URC", "哈尔滨": "HRB", "沈阳": "SHE",
}


def _city_to_code(city: str) -> str:
    """中文城市 → 三字码，找不到原样返回（TOP 要求大写）"""
    return _CITY_CODE.get(city, city.upper())


def _code_to_city(code: str) -> str:
    """三字码 → 中文城市，找不到原样返回"""
    # 反向字典
    rev = {v: k for k, v in _CITY_CODE.items()}
    return rev.get(code.upper(), code)


# ---------------------------------------------------------------------------
# 航班 tool
# ---------------------------------------------------------------------------

_AIRLINES = ["国航 CA", "东航 MU", "南航 CZ", "海航 HU", "川航 3U", "春秋 9C", "厦航 MF"]
_TYPES = ["直飞", "经停", "中转"]


def _mock_flights(dep_code: str, arr_code: str, dep_date: str, count: int = 6) -> list[dict]:
    """生成模拟航班列表（价格根据航线距离合理分布）"""
    base_price = 300 if dep_code == arr_code else random.choice([450, 680, 920, 1280])
    flights = []
    for i in range(count):
        airline = random.choice(_AIRLINES)
        dep_h = random.randint(6, 22)
        dur = random.randint(70, 200)  # 分钟
        flights.append({
            "flight_no": f"{airline.split()[1]}{random.randint(1000, 9999)}",
            "airline": airline,
            "dep_city": _code_to_city(dep_code),
            "arr_city": _code_to_city(arr_code),
            "dep_time": f"{dep_h:02d}:{random.choice([0, 15, 30, 45]):02d}",
            "dur_minutes": dur,
            "price_range": {
                "economy_low": base_price - random.randint(0, 100),
                "economy_high": base_price + random.randint(100, 400),
                "business": base_price + random.randint(600, 1200),
            },
            "type": random.choice(_TYPES),
        })
    return flights


@mcp.tool(description="查询航班列表（用于预算估算）。返回航班号、价格区间、出发到达时刻。")
def search_flights(
    dep_city: str,
    arr_city: str,
    dep_date: str,
    ret_date: str | None = None,
    adult_num: int = 1,
) -> dict[str, Any]:
    """
    查询航班列表（用于预算估算）。

    Args:
        dep_city: 出发城市，支持中文名 "成都" 或三字码 "CTU"
        arr_city: 到达城市，同上
        dep_date: 出发日期 "YYYY-MM-DD"
        ret_date: 返程日期（可选）
        adult_num: 成人数量（默认 1）

    Returns:
        {
          "dep": [...航班列表...],
          "ret": [...航班列表 或 None...],
          "note": "数据来源说明",
        }
    """
    dep_code = _city_to_code(dep_city)
    arr_code = _city_to_code(arr_city)

    dep_flights = _mock_flights(dep_code, arr_code, dep_date)

    ret_flights = None
    if ret_date:
        ret_flights = _mock_flights(arr_code, dep_code, ret_date)

    return {
        "dep": dep_flights,
        "ret": ret_flights,
        "passengers": adult_num,
        "note": "mock 数据（飞猪 MCP Server，暂未接入 TOP 签名）",
    }


# ---------------------------------------------------------------------------
# 酒店 tool
# ---------------------------------------------------------------------------

_HOTEL_NAMES = [
    "如家商旅", "维也纳酒店", "全季酒店", "汉庭优佳", "锦江都城",
    "亚朵酒店", "桔子水晶", "希尔顿花园", "喜来登", "丽思卡尔顿",
    "精品民宿", "度假公寓",
]


def _mock_hotels(city: str, checkin: str, checkout: str, count: int = 8) -> list[dict]:
    """生成模拟酒店列表（不同档次 → 不同价格）"""
    names_by_tier = {
        "经济": _HOTEL_NAMES[:5],   # 如家/汉庭
        "舒适": _HOTEL_NAMES[5:7],  # 亚朵/桔子
        "高档": _HOTEL_NAMES[7:10], # 希尔顿/喜来登
        "豪华": _HOTEL_NAMES[9:10], # 丽思
    }
    price_base = {"经济": 180, "舒适": 350, "高档": 650, "豪华": 1200}
    hotels = []
    tiers = list(price_base.keys())
    for i in range(count):
        tier = tiers[i % len(tiers)]
        p = price_base[tier] + random.randint(-50, 200)
        name = random.choice(names_by_tier[tier])
        hotels.append({
            "name": f"{name}（{city}分店）",
            "tier": tier,
            "price_per_night": p,
            "rating": round(random.uniform(3.5, 5.0), 1),
            "distance_from_center_km": round(random.uniform(0.5, 12.0), 1),
            "breakfast_included": random.choice([True, False]),
            "cancel_free_before_hours": random.choice([24, 48, 72]),
        })
    return hotels


@mcp.tool(description="查询酒店列表（用于预算估算）。返回档次、每晚价格、评分。")
def search_hotels(
    city: str,
    checkin_date: str,
    checkout_date: str,
    guest_num: int = 2,
) -> dict[str, Any]:
    """
    查询酒店列表（用于预算估算）。

    Args:
        city: 城市名 "成都"
        checkin_date: 入住日期 "YYYY-MM-DD"
        checkout_date: 退房日期
        guest_num: 入住人数（默认 2）

    Returns:
        {
          "hotels": [...酒店列表...],
          "nights": 计算出的住宿天数,
          "city": "成都",
        }
    """
    # 算天数
    try:
        ci = date.fromisoformat(checkin_date)
        co = date.fromisoformat(checkout_date)
        nights = max(1, (co - ci).days)
    except Exception:
        nights = 2

    return {
        "hotels": _mock_hotels(city, checkin_date, checkout_date),
        "nights": nights,
        "guest_num": guest_num,
        "note": "mock 数据（飞猪 MCP Server，暂未接入 TOP 签名）",
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # stdio 传输：LangGraph 作为 client 子进程拉起
    mcp.run(transport="stdio")
