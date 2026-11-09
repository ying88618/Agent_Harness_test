"""registry 单测：注册约束 + schema 生成 + 执行兜底。

核心断言是"工具出错不抛异常"，而是转成 ok=False 的 ToolResult。
"""

from __future__ import annotations

import json

import pytest

from agent.errors import ToolError
from agent.registry import ToolRegistry, ToolSpec, tool
from agent.tools import build_default_registry
from agent.types import ToolCall, ToolContext


class FakeTrace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def event(self, name: str, **data):
        self.events.append((name, data))


def make_ctx(session_id: str = "s1") -> ToolContext:
    return ToolContext(session_id=session_id, store={}, trace=FakeTrace())


# 注册约束
def test_duplicate_name_rejected():
    reg = ToolRegistry()
    reg.register(ToolSpec("a", "desc", {"type": "object", "properties": {}}, lambda args, ctx: ""))
    with pytest.raises(ValueError, match="已注册"):
        reg.register(ToolSpec("a", "desc2", {"type": "object", "properties": {}}, lambda args, ctx: ""))


def test_empty_description_rejected():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="description"):
        reg.register(ToolSpec("a", "  ", {"type": "object", "properties": {}}, lambda args, ctx: ""))


def test_parameters_must_be_object_schema():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="object"):
        reg.register(ToolSpec("a", "desc", {"type": "array"}, lambda args, ctx: ""))


def test_openai_schema_shape():
    reg = build_default_registry()
    tools = reg.openai_tools()

    assert {t["function"]["name"] for t in tools} == {"calculator", "search", "todo", "weather"}
    for item in tools:
        assert item["type"] == "function"
        fn = item["function"]
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert isinstance(fn["parameters"]["properties"], dict)


def test_default_registry_has_required_tools():
    names = build_default_registry().names()
    assert "todo" in names
    assert len(names) >= 3


# 执行兜底
def test_unknown_tool_returns_helpful_message():
    reg = build_default_registry()
    result = reg.execute(ToolCall(name="teleport", args={}), make_ctx())

    assert result.ok is False
    assert result.error == "tool_not_found"
    assert "calculator" in result.content  # 告诉模型有哪些可用工具


def test_missing_required_arg_detected_before_call():
    reg = build_default_registry()
    result = reg.execute(ToolCall(name="weather", args={}), make_ctx())

    assert result.ok is False
    assert result.error == "bad_arguments"
    assert "city" in result.content
    assert "schema" in result.content  # 把 schema 一起回灌，模型才知道怎么改


def test_enum_violation_detected():
    reg = build_default_registry()
    result = reg.execute(
        ToolCall(name="todo", args={"action": "explode"}), make_ctx()
    )
    assert result.ok is False
    assert result.error == "bad_arguments"


def test_type_mismatch_detected():
    reg = build_default_registry()
    result = reg.execute(
        ToolCall(name="calculator", args={"expression": 123}), make_ctx()
    )
    assert result.ok is False
    assert result.error == "bad_arguments"


def test_bool_is_not_accepted_as_integer():
    reg = ToolRegistry()

    @tool("echo_id", "回显 id", {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]})
    def echo_id(args, ctx):
        return str(args["id"])

    reg.register_function(echo_id)
    result = reg.execute(ToolCall(name="echo_id", args={"id": True}), make_ctx())
    assert result.ok is False
    assert result.error == "bad_arguments"


def test_tool_error_becomes_result_not_exception():
    reg = ToolRegistry()

    @tool("boom", "总是失败", {"type": "object", "properties": {}})
    def boom(args, ctx):
        raise ToolError("故意失败")

    reg.register_function(boom)
    result = reg.execute(ToolCall(name="boom", args={}), make_ctx())
    assert result.ok is False
    assert result.error == "tool_error"
    assert "故意失败" in result.content


def test_internal_exception_is_contained():
    reg = ToolRegistry()

    @tool("crash", "内部 bug", {"type": "object", "properties": {}})
    def crash(args, ctx):
        return {}["missing"]

    reg.register_function(crash)
    result = reg.execute(ToolCall(name="crash", args={}), make_ctx())
    assert result.ok is False
    assert result.error == "internal_error"


def test_non_str_output_is_json_encoded():
    reg = build_default_registry()
    result = reg.execute(ToolCall(name="search", args={"query": "session"}), make_ctx())
    assert result.ok is True
    assert json.loads(result.content)[0]["title"]


def test_long_output_is_truncated():
    reg = ToolRegistry()

    @tool("big", "输出很长", {"type": "object", "properties": {}})
    def big(args, ctx):
        return "A" * 5000 + "TAIL"

    reg.register_function(big)
    result = reg.execute(ToolCall(name="big", args={}), make_ctx(), max_output_chars=200)

    assert result.ok is True
    assert result.truncated is True
    assert len(result.content) < 400
    assert "TAIL" in result.content  # 尾部信息没被丢掉


def test_execution_is_traced():
    reg = build_default_registry()
    ctx = make_ctx()
    reg.execute(ToolCall(name="weather", args={"city": "厦门"}), ctx)

    events = [e for e, _ in ctx.trace.events]
    assert "tool_result" in events


def test_raw_args_are_parsed_when_args_empty():
    reg = build_default_registry()
    result = reg.execute(
        ToolCall(name="weather", args={}, raw_args='{"city": "厦门"}'), make_ctx()
    )
    assert result.ok is True
    assert "厦门" in result.content
