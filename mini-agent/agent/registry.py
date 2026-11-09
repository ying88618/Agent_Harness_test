"""工具注册表。

三个职责：
1. 登记：工具自报「名称 / 描述 / 参数 Schema」，自动转成 OpenAI function schema
2. 校验：调用前按 schema 检查必填项、类型、枚举值
3. 兜底：任何工具异常都转成 `ToolResult(ok=False)` 回灌给模型 —— 而不是抛给用户

第 3 点决定了 Agent 能不能"自我修正"：模型把参数写错的正确反应是让它看见错误
再改一次，而不是整个对话崩掉。
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .errors import ToolArgError, ToolError, ToolNotFound
from .types import ToolCall, ToolContext, ToolResult

DEFAULT_MAX_OUTPUT_CHARS = 4000

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[[dict[str, Any], ToolContext], Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or _EMPTY_SCHEMA,
            },
        }


def tool(name: str, description: str, parameters: dict[str, Any] | None = None):
    """装饰器：把普通函数登记成工具。

    函数签名必须是 `func(args, ctx)`，ctx 里带着 session_id，
    这样有状态的工具（todo）才能做到按窗口隔离。
    """

    def deco(fn: Callable[[dict[str, Any], ToolContext], Any]):
        fn.__tool_spec__ = ToolSpec(
            name=name,
            description=description,
            parameters=parameters or _EMPTY_SCHEMA,
            func=fn,
        )
        return fn

    return deco


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # 注册
    def register(self, spec: ToolSpec) -> None:
        if not spec.name or not spec.name.strip():
            raise ValueError("工具名不能为空")
        if spec.name in self._tools:
            raise ValueError(f"工具 {spec.name!r} 已注册，不允许重复")
        if not spec.description or not spec.description.strip():
            # 描述是模型决策的唯一依据，空描述等于工具不可用
            raise ValueError(f"工具 {spec.name!r} 缺少 description，模型无法判断何时调用")
        if not isinstance(spec.parameters, dict) or spec.parameters.get("type") != "object":
            raise ValueError(f"工具 {spec.name!r} 的 parameters 必须是 type=object 的 JSON Schema")
        self._tools[spec.name] = spec

    def register_function(self, fn: Callable[..., Any]) -> None:
        spec = getattr(fn, "__tool_spec__", None)
        if spec is None:
            raise ValueError(f"{getattr(fn, '__name__', fn)} 不是通过 @tool 注册的函数")
        self.register(spec)

    def register_all(self, module_or_iterable: Any) -> None:
        items = (
            vars(module_or_iterable).values()
            if inspect.ismodule(module_or_iterable)
            else module_or_iterable
        )
        for item in items:
            if getattr(item, "__tool_spec__", None) is not None:
                self.register_function(item)

    # 查询
    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def openai_tools(self) -> list[dict[str, Any]]:
        """喂给 LLM 的 tools 参数。"""
        return [spec.to_openai() for spec in self._tools.values()]

    def describe(self) -> str:
        """给模型看的纯文本工具清单（system prompt 里用）。"""
        lines = [f"- {s.name}: {s.description}" for s in self._tools.values()]
        return "\n".join(lines)

    # 执行
    def execute(
        self,
        call: ToolCall,
        ctx: ToolContext,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> ToolResult:
        started = time.perf_counter()
        spec = self._tools.get(call.name)

        if spec is None:
            available = self.names()
            result = ToolResult(
                ok=False,
                tool=call.name,
                error="tool_not_found",
                content=(
                    f"工具 {call.name!r} 不存在。可用工具：{', '.join(available)}。"
                    "请从可用工具中重新选择，或直接回答用户。"
                ),
            )
            return self._finish(result, ctx, started)

        try:
            args = call.args
            if not args and call.raw_args:
                from .parser import parse_tool_args  # 延迟导入，避免顶层循环依赖

                args, _ = parse_tool_args(call.raw_args)
            args = args if isinstance(args, dict) else {}

            self._validate(spec, args)

            raw_out = spec.func(args, ctx)
            content = (
                raw_out
                if isinstance(raw_out, str)
                else json.dumps(raw_out, ensure_ascii=False, default=str)
            )
            content, truncated = _truncate(content, max_output_chars)
            return self._finish(
                ToolResult(ok=True, tool=call.name, content=content, truncated=truncated),
                ctx,
                started,
            )

        except ToolArgError as exc:
            return self._finish(
                ToolResult(
                    ok=False,
                    tool=call.name,
                    error="bad_arguments",
                    content=(
                        f"参数不合法：{exc}\n参数 schema："
                        f"{json.dumps(spec.parameters, ensure_ascii=False)}"
                    ),
                ),
                ctx,
                started,
            )
        except ToolError as exc:
            return self._finish(
                ToolResult(ok=False, tool=call.name, error="tool_error", content=f"执行失败：{exc}"),
                ctx,
                started,
            )
        except Exception as exc:  # 工具内部 bug 也不能炸掉整轮对话
            return self._finish(
                ToolResult(
                    ok=False,
                    tool=call.name,
                    error="internal_error",
                    content=f"工具内部异常：{type(exc).__name__}: {exc}",
                ),
                ctx,
                started,
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _finish(result: ToolResult, ctx: ToolContext, started: float) -> ToolResult:
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        ctx.log(
            "tool_result",
            tool=result.tool,
            ok=result.ok,
            error=result.error or None,
            elapsed_ms=result.elapsed_ms,
            preview=result.content[:200],
        )
        return result

    @staticmethod
    def _validate(spec: ToolSpec, args: dict[str, Any]) -> None:
        schema = spec.parameters or _EMPTY_SCHEMA
        props: dict[str, Any] = schema.get("properties") or {}
        problems: list[str] = []

        for key in schema.get("required") or []:
            if args.get(key) in (None, ""):
                problems.append(f"缺少必填参数 {key!r}")

        for key, val in args.items():
            if key not in props:
                continue  # 宽容：多余参数不拦，交给函数自己忽略
            rule = props[key] or {}

            expected = rule.get("type")
            if expected and not _type_ok(expected, val):
                problems.append(f"参数 {key!r} 应为 {expected}，实际是 {type(val).__name__}")
                continue

            enum = rule.get("enum")
            if enum and val not in enum:
                problems.append(f"参数 {key!r} 必须是 {enum} 之一，实际是 {val!r}")

        if problems:
            raise ToolArgError("；".join(problems))


def _type_ok(expected: str, val: Any) -> bool:
    types = _TYPE_MAP.get(expected)
    if types is None:
        return True  # 未知类型不做判断，交给工具自己
    if isinstance(val, bool) and expected in ("integer", "number"):
        return False  # bool 是 int 的子类，这里必须排除
    return isinstance(val, types)


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """头尾各留一半：工具输出的关键信息常在开头（结论）和结尾（汇总）。"""
    if limit <= 0 or len(text) <= limit:
        return text, False
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    omitted = len(text) - limit
    return f"{head}\n...[已截断 {omitted} 字符]...\n{tail}", True