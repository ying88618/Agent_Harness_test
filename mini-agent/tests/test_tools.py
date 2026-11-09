"""工具单测。重点是 todo 的 session 隔离和 calculator 的安全边界。"""

from __future__ import annotations

import json

import pytest

from agent.errors import ToolError
from agent.tools import build_default_registry
from agent.tools.calculator import calculate
from agent.tools.search import search
from agent.tools.todo import todo
from agent.tools.weather import weather
from agent.types import ToolCall, ToolContext


def ctx(session_id: str = "s1") -> ToolContext:
    return ToolContext(session_id=session_id, store={})


# --------------------------------------------------------------------------- #
# calculator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1+2*3", "7"),
        ("(2+3)/2", "2.5"),
        ("sqrt(16)+2", "6"),
        ("2**10", "1024"),
        ("7//2", "3"),
        ("7%2", "1"),
        ("round(3.14159, 2)", "3.14"),
        ("-3*2", "-6"),
        ("pi", "3.141592654"),
    ],
)
def test_calculator_correctness(expr, expected):
    assert calculate({"expression": expr}, ctx()) == f"{expr} = {expected}"


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('dir')",
        "().__class__.__bases__",
        "a.b",
        "[1,2][0]",
        "lambda x: x",
        "[i for i in range(10)]",
        "open('a.txt')",
        "1/0",
        "9**99999",
        "unknown_name + 1",
        "1 +",
    ],
)
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError):
        calculate({"expression": expr}, ctx())


def test_calculator_empty_expression():
    with pytest.raises(ToolError):
        calculate({"expression": "  "}, ctx())


# todo：session 隔离（本题的核心考点）
def test_todo_add_and_list():
    c = ctx("s1")
    out = todo({"action": "add", "text": "给客户回邮件"}, c)
    assert "#1" in out and "给客户回邮件" in out

    listed = todo({"action": "list"}, c)
    assert "给客户回邮件" in listed
    assert "未完成 1 条" in listed


def test_todo_sessions_are_isolated():
    """窗口 1 和窗口 2 各自记待办，互不可见。"""
    window1 = ctx("user-a:window-1")
    window2 = ctx("user-a:window-2")

    todo({"action": "add", "text": "窗口1：查完天气买伞"}, window1)
    todo({"action": "add", "text": "窗口2：写本周周报"}, window2)

    list1 = todo({"action": "list"}, window1)
    list2 = todo({"action": "list"}, window2)

    assert "买伞" in list1 and "周报" not in list1
    assert "周报" in list2 and "买伞" not in list2


def test_todo_done_update_remove():
    c = ctx("s1")
    todo({"action": "add", "text": "任务A"}, c)
    todo({"action": "add", "text": "任务B"}, c)

    assert "已完成待办 #1" in todo({"action": "done", "id": 1}, c)
    assert "[x] #1 任务A" in todo({"action": "list"}, c)

    assert "已更新待办 #2" in todo({"action": "update", "id": 2, "due": "明天"}, c)
    assert "截止：明天" in todo({"action": "list"}, c)

    assert "已删除待办 #2" in todo({"action": "remove", "id": 2}, c)
    assert "#2" not in todo({"action": "list"}, c)


def test_todo_clear_resets_sequence():
    c = ctx("s1")
    todo({"action": "add", "text": "任务A"}, c)
    assert "已清空 1 条待办" in todo({"action": "clear"}, c)

    out = todo({"action": "add", "text": "新任务"}, c)
    assert "#1" in out  # 编号从头开始，不会无限增长


def test_todo_missing_id_raises():
    c = ctx("s1")
    todo({"action": "add", "text": "任务A"}, c)
    with pytest.raises(ToolError, match="id"):
        todo({"action": "done"}, c)
    with pytest.raises(ToolError, match="不存在"):
        todo({"action": "done", "id": 99}, c)


def test_todo_add_without_text_raises():
    with pytest.raises(ToolError):
        todo({"action": "add"}, ctx())


def test_todo_empty_list_message():
    assert "为空" in todo({"action": "list"}, ctx())


def test_todo_render_is_capped():
    """清单过长时只回灌前 N 条。

    每次 add 都会把完整清单塞回 context，不设上限会平方级膨胀，
    几十条待办就能把上下文顶爆、触发没必要的压缩。
    """
    c = ctx("s1")
    for i in range(45):
        todo({"action": "add", "text": f"任务{i}"}, c)

    listed = todo({"action": "list"}, c)

    assert "其余 15 条从略" in listed
    assert len(listed.splitlines()) <= 35   # 30 条 + 头尾


def test_todo_render_unchanged_for_short_list():
    c = ctx("s1")
    for i in range(3):
        todo({"action": "add", "text": f"任务{i}"}, c)
    assert "从略" not in todo({"action": "list"}, c)


def test_todo_state_survives_via_shared_store():
    """同一个 store 被复用（模拟落盘后重新载入）时，待办还在。"""
    store: dict = {}
    c1 = ToolContext(session_id="s1", store=store)
    todo({"action": "add", "text": "持久化任务"}, c1)

    c2 = ToolContext(session_id="s1", store=store)
    assert "持久化任务" in todo({"action": "list"}, c2)


# search
def test_search_returns_relevant_hits():
    hits = json.loads(search({"query": "session 隔离"}, ctx()))
    assert hits and any("会话" in h["title"] or "Session" in h["title"] for h in hits)
    assert all({"title", "snippet", "url", "score"} <= set(h) for h in hits)


def test_search_top_k_is_clamped():
    hits = json.loads(search({"query": "工具", "top_k": 99}, ctx()))
    assert 1 <= len(hits) <= 5


def test_search_no_result_is_friendly():
    out = search({"query": "zzzz-不存在的词-qqqq"}, ctx())
    assert "没有找到" in out


def test_search_empty_query():
    assert "为空" in search({"query": ""}, ctx())


# weather
def test_weather_known_city():
    out = weather({"city": "厦门"}, ctx())
    assert "厦门" in out and "℃" in out


def test_weather_alias():
    assert "厦门" in weather({"city": "Xiamen"}, ctx())


def test_weather_unknown_city_lists_supported():
    out = weather({"city": "火星"}, ctx())
    assert "暂未收录" in out and "厦门" in out


def test_weather_empty_city_raises():
    with pytest.raises(ToolError):
        weather({"city": ""}, ctx())


# 端到端：parser 产出的 ToolCall 直接喂给 registry
def test_parse_then_execute_roundtrip():
    from agent.parser import parse_response
    from agent.types import LLMResponse

    reply = LLMResponse(
        content="",
        reasoning="要算一下",
        tool_calls=[ToolCall(name="calculator", raw_args='{"expression": "(3+4)*2"}', id="c1")],
    )
    parsed = parse_response(reply)

    reg = build_default_registry()
    result = reg.execute(parsed.tool_calls[0], ctx())

    assert result.ok is True
    assert result.content == "(3+4)*2 = 14"
