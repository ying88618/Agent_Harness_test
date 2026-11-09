"""loop 的行为测试。

用 ScriptedLLM 覆盖真实 API 造不出来的场景：
一直调工具不停、参数写错后自我修正、重复调用、纯闲聊。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.errors import LLMError
from agent.loop import AgentLoop
from agent.session import SessionStore
from agent.tools import build_default_registry

from tests.fake_llm import ScriptedLLM, call, reply, tool_reply


def make_loop(llm, tmp_path: Path, **kwargs) -> AgentLoop:
    kwargs.setdefault("trace_path", None)
    return AgentLoop(
        llm=llm,
        registry=build_default_registry(),
        store=SessionStore(tmp_path / "sessions"),
        **kwargs,
    )


# 基本路径
def test_single_tool_then_final(tmp_path):
    llm = ScriptedLLM(
        [
            tool_reply(call("weather", {"city": "厦门"}, call_id="c1")),
            reply("厦门今天晴，22~28℃。"),
        ]
    )
    loop = make_loop(llm, tmp_path)

    result = loop.run("s1", "厦门天气怎么样")

    assert result.stop_reason == "final"
    assert result.turns == 2
    assert result.tool_calls == 1
    assert "晴" in result.answer

    session = loop.store.load_or_create("s1")
    roles = [m["role"] for m in session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_parallel_tool_calls_in_one_turn(tmp_path):
    llm = ScriptedLLM(
        [
            tool_reply(
                call("weather", {"city": "厦门"}, call_id="c1"),
                call("calculator", {"expression": "1+1"}, call_id="c2"),
                call("todo", {"action": "add", "text": "带伞"}, call_id="c3"),
            ),
            reply("都办好了"),
        ]
    )
    loop = make_loop(llm, tmp_path)

    result = loop.run("s1", "查天气、算 1+1、记待办")

    # 三个工具算一轮，不是三轮
    assert result.turns == 2
    assert result.tool_calls == 3

    session = loop.store.load_or_create("s1")
    tool_msgs = [m for m in session.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 3
    assert {m["tool_call_id"] for m in tool_msgs} == {"c1", "c2", "c3"}


def test_pure_chat_returns_immediately(tmp_path):
    llm = ScriptedLLM([reply("你好，我是助手。")])
    loop = make_loop(llm, tmp_path)

    result = loop.run("s1", "你好")

    assert result.stop_reason == "final"
    assert result.turns == 1
    assert result.tool_calls == 0


def test_tool_error_then_self_correct(tmp_path):
    """参数写错 → 工具报错回灌 → 模型改对 → 成功。"""
    llm = ScriptedLLM(
        [
            tool_reply(call("weather", {}, call_id="c1")),  # 缺 city
            tool_reply(call("weather", {"city": "厦门"}, call_id="c2")),
            reply("厦门今天晴。"),
        ]
    )
    loop = make_loop(llm, tmp_path)

    result = loop.run("s1", "查天气")

    assert result.stop_reason == "final"
    assert result.tool_calls == 2

    session = loop.store.load_or_create("s1")
    first_tool = next(m for m in session.messages if m["role"] == "tool")
    assert "参数不合法" in first_tool["content"]
    assert "city" in first_tool["content"]


# --------------------------------------------------------------------------- #
# 重复调用熔断
# --------------------------------------------------------------------------- #
def test_repeat_tool_call_is_blocked(tmp_path):
    def same_call(messages):
        return tool_reply(call("weather", {"city": "厦门"}, call_id=f"c{len(messages)}"))

    llm = ScriptedLLM([same_call], repeat_last=True)
    loop = make_loop(llm, tmp_path, max_turns=3)

    result = loop.run("s1", "一直查同一个天气")

    assert result.stop_reason == "max_turns"
    session = loop.store.load_or_create("s1")
    blocked = [
        m for m in session.messages if m["role"] == "tool" and "连续执行多次" in m["content"]
    ]
    assert blocked, "第二次相同调用必须被短路，且要写入 tool 消息保持配对完整"


# --------------------------------------------------------------------------- #
# 轮次耗尽
# --------------------------------------------------------------------------- #
def _endless_tool_model(messages):
    n = len(messages)
    return tool_reply(call("calculator", {"expression": f"{n}*2"}, call_id=f"c{n}"))


def test_max_turns_keeps_history_and_closes_turn(tmp_path):
    llm = ScriptedLLM([_endless_tool_model], repeat_last=True)
    loop = make_loop(llm, tmp_path, max_turns=3)

    result = loop.run("s1", "给我一直算下去")

    assert result.stop_reason == "max_turns"
    assert "3" in result.answer

    session = loop.store.load_or_create("s1")
    # 用户输入必须还在
    assert session.messages[0] == {"role": "user", "content": "给我一直算下去"}
    # 中间的工具往返必须还在
    assert any(m["role"] == "tool" for m in session.messages)
    # 回合必须收口：末尾是 assistant，而不是悬空的 tool 消息
    assert session.messages[-1]["role"] == "assistant"


def test_max_turns_does_not_lose_history_for_next_run(tmp_path):
    llm = ScriptedLLM([_endless_tool_model], repeat_last=True)
    loop = make_loop(llm, tmp_path, max_turns=2)
    loop.run("s1", "第一个问题：给我算下去")

    llm2 = ScriptedLLM([reply("你刚才问的是继续计算。")])
    loop.llm = llm2
    result = loop.run("s1", "我刚才问你什么了？")

    assert result.stop_reason == "final"
    assert "第一个问题：给我算下去" in llm2.all_message_text()


# 压缩
def test_pure_chat_triggers_compression(tmp_path):
    """纯闲聊不调用工具，也必须压缩——否则长对话会无限膨胀。"""
    llm = ScriptedLLM([reply(f"这是第 {i} 轮回答") for i in range(8)])
    loop = make_loop(llm, tmp_path, max_context_chars=200, keep_recent_turns=1)

    for i in range(8):
        loop.run("chat", f"闲聊第 {i} 句：嗯嗯好的你说得对，我同意你的看法")

    session = loop.store.load_or_create("chat")
    assert session.summary, "纯闲聊场景没有触发压缩"
    assert len(session.messages) < 6


def test_compression_keeps_tool_pairing(tmp_path):
    """压缩后不能出现孤立的 tool 消息，否则下次请求直接 400。"""
    llm = ScriptedLLM(
        [reply(f"回答 {i}") for i in range(10)]
    )
    loop = make_loop(llm, tmp_path, max_context_chars=150, keep_recent_turns=1)

    for i in range(10):
        loop.run("c", f"问题 {i}：请随便聊聊，多说一点内容丰富内容" * 2)

    session = loop.store.load_or_create("c")
    allowed: set[str] = set()
    for msg in session.messages:
        if msg["role"] == "assistant":
            allowed = {tc["id"] for tc in msg.get("tool_calls") or []}
        elif msg["role"] == "tool":
            assert msg["tool_call_id"] in allowed, "出现孤立 tool 消息"


# 异常
def test_llm_error_is_contained(tmp_path):
    class Boom:
        def chat(self, *a, **k):
            raise LLMError("连接超时")

    loop = make_loop(Boom(), tmp_path)
    result = loop.run("s1", "随便说点")

    assert result.stop_reason == "llm_error"
    assert "连接超时" in result.answer


def test_session_isolation_at_loop_level(tmp_path):
    llm = ScriptedLLM(
        [
            tool_reply(call("todo", {"action": "add", "text": "窗口1的事"}, call_id="c1")),
            reply("记好了"),
            tool_reply(call("todo", {"action": "add", "text": "窗口2的事"}, call_id="c2")),
            reply("记好了"),
        ]
    )
    loop = make_loop(llm, tmp_path)

    loop.run("window-1", "记个待办")
    loop.run("window-2", "记个待办")

    s1 = loop.store.load_or_create("window-1")
    s2 = loop.store.load_or_create("window-2")
    assert s1.store["todos"][0]["text"] == "窗口1的事"
    assert s2.store["todos"][0]["text"] == "窗口2的事"
    assert len(s1.messages) == 4 and len(s2.messages) == 4
