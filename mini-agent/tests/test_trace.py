"""trace 的测试。

trace 的要求很低但很重要：写得进、读得出、出错不能影响主流程。
"""

from __future__ import annotations

import json

from agent.trace import Trace, bind_trace, current_trace, trace_event, unbind_trace


def test_trace_writes_jsonl(tmp_path):
    path = tmp_path / "logs" / "trace.jsonl"
    trace = Trace(path=path, session_id="s1")
    trace.event("user_input", text="你好")
    trace.event("llm_end", turn=1, total_tokens=42)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["event"] == "user_input"
    assert first["session_id"] == "s1"
    assert first["text"] == "你好"
    assert "ts" in first and "run_id" in first

    second = json.loads(lines[1])
    assert second["total_tokens"] == 42


def test_trace_keeps_events_in_memory():
    trace = Trace(path=None, session_id="s1")
    trace.event("a")
    trace.event("b")
    trace.event("a")

    assert len(trace.events) == 3
    assert len(trace.events_of("a")) == 2


def test_trace_drops_none_fields():
    trace = Trace(path=None)
    record = trace.event("tool_end", tool="weather", error=None)
    assert "error" not in record
    assert record["tool"] == "weather"


def test_trace_contextvar_roundtrip():
    trace = Trace(path=None, session_id="s1")
    token = bind_trace(trace)
    try:
        assert current_trace() is trace
        trace_event("from_helper", value=1)
        assert trace.events[-1]["event"] == "from_helper"
    finally:
        unbind_trace(token)

    assert current_trace() is None
    # 解绑之后再调用不能炸
    trace_event("after_unbind")


def test_trace_write_failure_is_swallowed(tmp_path):
    """无法写盘时只能少记日志，绝不能把主流程带崩。"""
    trace = Trace(path=tmp_path / "t.jsonl")
    trace.path = tmp_path  # 指向目录，open 会抛 OSError

    trace.event("still_works")

    assert trace.events[-1]["event"] == "still_works"


def test_trace_serializes_non_json_values():
    trace = Trace(path=None)

    class Weird:
        def __repr__(self):
            return "<weird>"

    record = trace.event("x", obj=Weird())
    assert record["obj"].__class__ is Weird  # 内存里保留原对象
