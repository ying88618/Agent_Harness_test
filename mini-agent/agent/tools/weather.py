"""天气工具（mock）。

故意保留两种失败形态，方便测工具错误分支：
1. 参数非法（空城市）→ 抛 ToolError，走"参数不合法"回灌路径
2. 城市未收录 → 正常返回一句说明 + 可用城市，模型能据此改问法
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolError
from ..registry import tool
from ..types import ToolContext

_CITIES: dict[str, dict[str, Any]] = {
    "北京": {"today": ("晴", 6, 17), "tomorrow": ("多云", 8, 19)},
    "上海": {"today": ("小雨", 15, 21), "tomorrow": ("阴", 16, 22)},
    "广州": {"today": ("多云", 23, 29), "tomorrow": ("阵雨", 24, 30)},
    "厦门": {"today": ("晴", 22, 28), "tomorrow": ("多云", 23, 29)},
    "深圳": {"today": ("多云", 24, 30), "tomorrow": ("雷阵雨", 25, 30)},
    "杭州": {"today": ("阴", 16, 23), "tomorrow": ("小雨", 15, 21)},
    "成都": {"today": ("小雨", 14, 20), "tomorrow": ("阴", 15, 21)},
}

_ALIASES = {
    "beijing": "北京",
    "shanghai": "上海",
    "guangzhou": "广州",
    "xiamen": "厦门",
    "shenzhen": "深圳",
    "hangzhou": "杭州",
    "chengdu": "成都",
    "北京市": "北京",
    "上海市": "上海",
    "广州市": "广州",
    "厦门市": "厦门",
}


@tool(
    name="weather",
    description="查询指定城市的天气（今日与明日）。用户问天气、气温、是否下雨时调用。",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，如「厦门」或「Xiamen」"},
            "date": {
                "type": "string",
                "enum": ["today", "tomorrow"],
                "description": "查询哪一天，默认 today",
            },
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)
def weather(args: dict[str, Any], ctx: ToolContext) -> str:
    raw_city = str(args.get("city", "")).strip()
    if not raw_city:
        raise ToolError("city 不能为空")

    city = _ALIASES.get(raw_city.lower(), _ALIASES.get(raw_city, raw_city))
    if city not in _CITIES:
        return (
            f"抱歉，暂未收录「{raw_city}」的天气数据。"
            f"当前支持：{'、'.join(_CITIES)}。可以换个城市再查。"
        )

    day = str(args.get("date", "today") or "today").lower()
    if day not in ("today", "tomorrow"):
        day = "today"

    cond, low, high = _CITIES[city][day]
    label = "今天" if day == "today" else "明天"
    return f"{city}{label}：{cond}，气温 {low}~{high}℃，适合{'外出' if '雨' not in cond else '带伞'}"
