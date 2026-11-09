"""context 组装与压缩的测试。

重点就一个：压缩不能把"追问要用的信息"压没，也不能把消息配对搞断。
"""

from __future__ import annotations

from agent.context import (
    build_messages,
    build_system_prompt,
    count_chars,
    maybe_compress,
    split_turns,
    summarize_turns,
)
from agent.session import Session
from agent.tools import build_default_registry


def _session(messages: list[dict], summary: str = "") -> Session:
    return Session(session_id="s1", messages=messages, summary=summary)


def test_split_turns_uses_user_boundary():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    turns = split_turns(msgs)
    assert len(turns) == 2
    assert len(turns[0]) == 4
    assert turns[1][0]["content"] == "q2"


def test_summarize_keeps_tool_notes():
    """摘要必须保留工具结果的要点，否则待办 ID 这类信息会永久丢失。"""
    turns = [
        [
            {"role": "user", "content": "记个待办：明天带伞"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "todo"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "已添加待办 #1：明天带伞"},
            {"role": "assistant", "content": "记好了"},
        ]
    ]
    text = summarize_turns(turns)
    assert "明天带伞" in text
    assert "#1" in text
    assert "todo" in text


def test_maybe_compress_noop_under_budget():
    session = _session([{"role": "user", "content": "短"}])
    assert maybe_compress(session, max_chars=10000, keep_recent_turns=1) is False
    assert session.summary == ""


def test_maybe_compress_folds_old_turns():
    messages = []
    for i in range(6):
        messages.append({"role": "user", "content": f"问题{i}" * 20})
        messages.append({"role": "assistant", "content": f"回答{i}" * 20})
    session = _session(messages)

    assert maybe_compress(session, max_chars=200, keep_recent_turns=1) is True
    assert session.summary
    assert len(split_turns(session.messages)) == 1
    assert session.messages[0]["content"].startswith("问题5")


def test_summary_goes_into_system_prompt_not_messages():
    session = _session([{"role": "user", "content": "继续"}], summary="用户之前问过厦门天气")
    messages = build_messages(session, build_default_registry())

    assert messages[0]["role"] == "system"
    assert "厦门天气" in messages[0]["content"]
    # 摘要不能作为一条独立消息混在对话里
    assert all(m.get("role") != "system" for m in messages[1:])


def test_system_prompt_lists_all_tools():
    prompt = build_system_prompt(build_default_registry())
    for name in ("calculator", "search", "todo", "weather"):
        assert name in prompt


def test_count_chars_counts_tool_calls():
    base = [{"role": "assistant", "content": "hi"}]
    with_calls = [
        {
            "role": "assistant",
            "content": "hi",
            "tool_calls": [{"id": "c1", "function": {"name": "todo", "arguments": "{}"}}],
        }
    ]
    assert count_chars(with_calls) > count_chars(base)
