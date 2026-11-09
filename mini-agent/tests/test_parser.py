"""parser 的单测：这是整个项目最先要跑绿的一批。

覆盖三类输入 × 若干畸形输出，重点是"模型写坏 JSON"时能不能救回来。
"""

from __future__ import annotations

import json

import pytest

from agent.parser import parse_response, parse_tool_args
from agent.types import LLMResponse, ToolCall, Usage


def resp(content: str = "", reasoning: str = "", tool_calls=None) -> LLMResponse:
    return LLMResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=list(tool_calls or []),
        usage=Usage(),
        model="test-model",
        finish_reason="stop",
    )


def tc(name: str, raw_args: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(name=name, raw_args=raw_args, id=call_id)


# 通道 A：原生 tool_calls
def test_native_single_tool_call():
    out = parse_response(resp(content="", tool_calls=[tc("weather", '{"city": "厦门"}')]))

    assert out.has_tool_call and not out.is_final
    assert len(out.tool_calls) == 1
    call = out.tool_calls[0]
    assert call.name == "weather"
    assert call.args == {"city": "厦门"}
    assert call.id == "call_1"


def test_native_multiple_tool_calls_keep_order_and_ids():
    out = parse_response(
        resp(
            tool_calls=[
                tc("weather", '{"city": "厦门"}', "call_a"),
                tc("todo", '{"action": "add", "text": "买伞"}', "call_b"),
            ]
        )
    )
    assert [c.name for c in out.tool_calls] == ["weather", "todo"]
    assert [c.id for c in out.tool_calls] == ["call_a", "call_b"]


def test_native_generates_id_when_missing():
    out = parse_response(resp(tool_calls=[tc("weather", "{}", call_id="")]))
    assert out.tool_calls[0].id.startswith("call_")


def test_native_reasoning_and_content_both_kept():
    out = parse_response(
        resp(
            content="我先查一下厦门天气。",
            reasoning="用户问天气，需要调用 weather 工具。",
            tool_calls=[tc("weather", '{"city": "厦门"}')],
        )
    )
    assert "需要调用 weather" in out.thought
    assert "先查一下厦门天气" in out.thought


def test_native_malformed_args_are_repaired():
    # 尾逗号是模型最常犯的错
    out = parse_response(resp(tool_calls=[tc("weather", '{"city": "厦门",}')]))
    assert out.tool_calls[0].args == {"city": "厦门"}
    assert out.warnings == []


def test_native_empty_arguments_string():
    out = parse_response(resp(tool_calls=[tc("todo", "")]))
    assert out.tool_calls[0].args == {}


def test_native_unrecoverable_args_fall_back_to_raw():
    out = parse_response(resp(tool_calls=[tc("weather", "城市是厦门")]))
    call = out.tool_calls[0]
    assert "__raw__" in call.args
    assert out.warnings and "参数解析失败" in out.warnings[0]
    # 关键：不能抛异常，要让 loop 有机会把错误回灌给模型


def test_native_arguments_double_encoded():
    raw = json.dumps({"city": "厦门"}, ensure_ascii=False)
    out = parse_response(resp(tool_calls=[tc("weather", json.dumps(raw))]))
    assert out.tool_calls[0].args == {"city": "厦门"}


# 通道 B：content 里的文本协议
def test_fenced_json_tool_call():
    content = """我来查一下。

```json
{"tool": "weather", "arguments": {"city": "厦门"}}
```
"""
    out = parse_response(resp(content=content))
    assert out.has_tool_call
    assert out.tool_calls[0].name == "weather"
    assert out.tool_calls[0].args == {"city": "厦门"}
    assert "我来查一下" in out.thought


def test_inline_json_tool_call_inside_prose():
    content = '好的，我调用一下。{"name": "calculator", "arguments": {"expression": "1+1"}}'
    out = parse_response(resp(content=content))
    assert out.has_tool_call
    assert out.tool_calls[0].name == "calculator"
    assert out.tool_calls[0].args == {"expression": "1+1"}


def test_wrapped_tool_calls_array():
    content = json.dumps(
        {"tool_calls": [{"name": "todo", "arguments": {"action": "list"}}]},
        ensure_ascii=False,
    )
    out = parse_response(resp(content=content))
    assert out.has_tool_call
    assert out.tool_calls[0].name == "todo"
    assert out.tool_calls[0].args == {"action": "list"}


def test_react_multiline_action_input():
    content = """Thought: 用户问厦门天气
Action: weather
Action Input:
{"city": "厦门", "date": "tomorrow"}
"""
    out = parse_response(resp(content=content))
    assert out.has_tool_call
    assert out.tool_calls[0].name == "weather"
    assert out.tool_calls[0].args == {"city": "厦门", "date": "tomorrow"}
    assert "用户问厦门天气" in out.thought


def test_react_stops_at_observation():
    content = """Action: weather
Action Input: {"city": "厦门"}
Observation: 厦门今天晴
Thought: 已经拿到结果了
"""
    out = parse_response(resp(content=content))
    assert out.tool_calls[0].args == {"city": "厦门"}


def test_react_lowercase_and_chinese_keywords():
    out = parse_response(resp(content="action: calculator\naction input: {\"expression\": \"2*3\"}"))
    assert out.tool_calls[0].name == "calculator"
    assert out.tool_calls[0].args == {"expression": "2*3"}


# 普通回答
def test_plain_text_is_final_answer():
    out = parse_response(resp(content="厦门今天晴，22~28℃。"))
    assert out.is_final
    assert out.final_answer == "厦门今天晴，22~28℃。"
    assert out.tool_calls == []


def test_final_answer_marker_is_stripped():
    content = "Thought: 信息够了\nFinal Answer: 已经把待办记好了。"
    out = parse_response(resp(content=content))
    assert out.final_answer == "已经把待办记好了。"
    assert "信息够了" in out.thought


def test_empty_content_is_final_but_warns():
    out = parse_response(resp(content=""))
    assert out.final_answer == ""
    assert out.is_final
    assert out.warnings


def test_dict_without_name_is_not_a_tool_call():
    out = parse_response(resp(content='{"arguments": {"a": 1}}'))
    assert not out.has_tool_call
    assert out.is_final


# parse_tool_args 单测
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", {}),
        (None, {}),
        ("{}", {}),
        ('{"a": 1}', {"a": 1}),
        ("{'a': 1}", {"a": 1}),  # 单引号 → ast.literal_eval
        ('{"a": 1,}', {"a": 1}),  # 尾逗号
        ('{"a": True, "b": None}', {"a": True, "b": None}),  # Python 字面量
        ('{"a": 1}  ', {"a": 1}),  # 尾部空白
    ],
)
def test_parse_tool_args_lenient(raw, expected):
    args, err = parse_tool_args(raw)
    assert args == expected
    assert err is None


def test_parse_tool_args_handles_dict_input():
    assert parse_tool_args({"a": 1}) == ({"a": 1}, None)


def test_parse_tool_args_reports_failure():
    args, err = parse_tool_args("这不是 JSON")
    assert "__raw__" in args
    assert err is not None


def test_parse_tool_args_unwraps_nested_json_string():
    inner = json.dumps({"city": "厦门"}, ensure_ascii=False)
    outer = json.dumps({"arguments": inner}, ensure_ascii=False)
    args, err = parse_tool_args(outer)
    assert args == {"city": "厦门"}
    assert err is None


def test_parse_tool_args_too_long_rejected():
    args, err = parse_tool_args("{" + "a" * 30000 + "}")
    assert err is not None
    assert args["__raw__"]
