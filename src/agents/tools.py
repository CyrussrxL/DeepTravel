"""
DeepTravel 第三方 API 工具封装

所有工具函数遵循的契约：
- 成功返回结构化 dict / list
- 失败（HTTP 错误、Key 无效、权限不足）不抛异常，返回 None 或空结构
- 调用方可通过 `if result:` 判断是否有真实数据
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

AMAP_KEY = os.getenv("AMAP_API_KEY", "")
SENIVERSE_KEY = os.getenv("SENIVERSE_API_KEY", "")
FLIGGY_KEY = os.getenv("FLIGGY_API_KEY", "")

REQUEST_TIMEOUT = 10


# ==========================================================================
# 高德地图（https://lbs.amap.com/api/webservice/summary）
# ==========================================================================

def search_pois(keywords: str, city: str, limit: int = 8) -> list[dict]:
    """
    高德地图 POI 搜索（国内城市）。

    Args:
        keywords: POI 关键词，如 "大熊猫基地"、"博物馆"
        city: 城市名，如 "成都"、"北京"
        limit: 返回数量上限

    Returns:
        list[dict]，每个 dict 含 name / address / type / location / tel
        失败时返回 []
    """
    if not AMAP_KEY:
        return []
    try:
        r = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={
                "key": AMAP_KEY,
                "keywords": keywords,
                "city": city,
                "output": "JSON",
                "extensions": "base",
                "offset": limit,
            },
            timeout=REQUEST_TIMEOUT,
        )
        d = r.json()
        if d.get("status") != "1":
            return []
        pois = []
        for p in d.get("pois", []):
            pois.append({
                "name": p.get("name", ""),
                "type": p.get("type", ""),
                "address": p.get("address", ""),
                "location": p.get("location", ""),  # lng,lat
                "tel": p.get("tel", ""),
            })
        return pois[:limit]
    except Exception as exc:
        print(f"  [tools.search_pois] 失败: {exc}")
        return []


def driving_route(origin_lnglat: str, dest_lnglat: str, city: str = "") -> dict | None:
    """
    高德驾车路径规划。

    Args:
        origin_lnglat: "经度,纬度" 如 "104.065735,30.659462"
        dest_lnglat: 同上
        city: 城市名（可选）

    Returns:
        dict 含 distance(米) / duration(秒) / traffic_lights / steps 摘要
        失败返回 None
    """
    if not AMAP_KEY:
        return None
    try:
        params = {
            "key": AMAP_KEY,
            "origin": origin_lnglat,
            "destination": dest_lnglat,
            "output": "JSON",
        }
        if city:
            params["city"] = city
        r = requests.get(
            "https://restapi.amap.com/v3/direction/driving",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        d = r.json()
        if d.get("status") != "1":
            return None
        path = d.get("route", {}).get("paths", [{}])[0]
        return {
            "distance_m": path.get("distance"),
            "duration_s": path.get("duration"),
            "traffic_lights": path.get("traffic_lights"),
            "strategy": path.get("strategy"),
        }
    except Exception as exc:
        print(f"  [tools.driving_route] 失败: {exc}")
        return None


def walking_route(origin_lnglat: str, dest_lnglat: str) -> dict | None:
    """高德步行路径规划"""
    if not AMAP_KEY:
        return None
    try:
        r = requests.get(
            "https://restapi.amap.com/v3/direction/walking",
            params={
                "key": AMAP_KEY,
                "origin": origin_lnglat,
                "destination": dest_lnglat,
                "output": "JSON",
            },
            timeout=REQUEST_TIMEOUT,
        )
        d = r.json()
        if d.get("status") != "1":
            return None
        path = d.get("route", {}).get("paths", [{}])[0]
        return {
            "distance_m": path.get("distance"),
            "duration_s": path.get("duration"),
        }
    except Exception as exc:
        print(f"  [tools.walking_route] 失败: {exc}")
        return None


# ==========================================================================
# 心知天气（https://docs.seniverse.com/api/index.html）
# ==========================================================================

def get_weather(city_location: str) -> dict | None:
    """
    心知天气实时天气查询。

    Args:
        city_location: 城市拼音（如 "chengdu"）或城市 ID 或经纬度
                       注意：免费版只支持国内城市

    Returns:
        dict 含 location / text / temperature / humidity / wind / last_update
        失败返回 None
    """
    if not SENIVERSE_KEY:
        return None
    try:
        r = requests.get(
            "https://api.seniverse.com/v3/weather/now.json",
            params={
                "key": SENIVERSE_KEY,
                "location": city_location,
                "language": "zh-Hans",
                "unit": "c",
            },
            timeout=REQUEST_TIMEOUT,
        )
        d = r.json()
        results = d.get("results")
        if not results:
            # status_code=AP010006 等权限问题
            print(f"  [tools.get_weather] API 返回异常: {d}")
            return None
        loc = results[0]["location"]
        now = results[0]["now"]
        return {
            "location": loc["name"],
            "country": loc.get("country", ""),
            "path": loc.get("path", ""),
            "text": now["text"],
            "temperature": now["temperature"],
            "humidity": now["humidity"],
            "wind_direction": now.get("wind_direction", ""),
            "wind_speed": now.get("wind_speed", ""),
            "last_update": results[0].get("last_update", ""),
        }
    except Exception as exc:
        print(f"  [tools.get_weather] 失败: {exc}")
        return None


def get_weather_daily(city_location: str, days: int = 3) -> list[dict]:
    """
    心知天气逐日预报。

    Args:
        city_location: 同上
        days: 预报天数（免费版最多 3 天）

    Returns:
        list[dict]，每个 dict 含 date / text_day / text_night / high / low
    """
    if not SENIVERSE_KEY:
        return []
    try:
        r = requests.get(
            "https://api.seniverse.com/v3/weather/daily.json",
            params={
                "key": SENIVERSE_KEY,
                "location": city_location,
                "language": "zh-Hans",
                "unit": "c",
                "start": 0,
                "days": days,
            },
            timeout=REQUEST_TIMEOUT,
        )
        d = r.json()
        results = d.get("results")
        if not results:
            return []
        daily = []
        for day in results[0].get("daily", []):
            daily.append({
                "date": day["date"],
                "text_day": day["text_day"],
                "text_night": day["text_night"],
                "high": day["high"],
                "low": day["low"],
            })
        return daily
    except Exception as exc:
        print(f"  [tools.get_weather_daily] 失败: {exc}")
        return []


# ==========================================================================
# 飞猪（占位，无公开 HTTP API）
# ==========================================================================

def search_flights(*args: Any, **kwargs: Any) -> list[dict]:
    """飞猪航班查询（预留占位，无公开 HTTP API 可直连）"""
    return []


def search_hotels(*args: Any, **kwargs: Any) -> list[dict]:
    """飞猪酒店查询（预留占位，无公开 HTTP API 可直连）"""
    return []
