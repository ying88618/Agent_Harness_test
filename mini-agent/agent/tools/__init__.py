"""内置工具集。

`build_default_registry()` 是唯一的装配入口，main.py 和 tests 都从这里拿，
保证测试跑的就是线上那套工具。
"""

from __future__ import annotations

from ..registry import ToolRegistry
from . import calculator, search, todo, weather

__all__ = ["build_default_registry", "calculator", "search", "todo", "weather"]


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for module in (calculator, search, todo, weather):
        registry.register_all(module)
    return registry
