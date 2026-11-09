"""session 存储与消息配对修复。

`repair_messages` 是这套代码里"出故障代价最高"的逻辑：
它防的是"进程崩在工具执行中间 → 会话永久 400"。
所以畸形输入的覆盖面要比正常路径更全。
"""

from __future__ import annotations

import json

from agent.session import Session, SessionStore, repair_messages


# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #
def msg(role: str, content: str = "", **extra) -> dict:
    return {"role": role, "content": content, **extra}


def assistant_with_calls(*call_ids: str, content: str = "") -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": "todo", "arguments": "{}"},
            }
            for cid in call_ids
        ],
    }


def tool_msg(call_id: str, content: str = "结果") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def tool_ids(messages: list[dict]) -> list[str]:
    return [m["tool_call_id"] for m in messages if m.get("role") == "tool"]


# --------------------------------------------------------------------------- #
# repair_messages
# --------------------------------------------------------------------------- #
def test_repair_keeps_valid_history_untouched():
    messages = [
        msg("user", "查天气"),
        assistant_with_calls("c1"),
        tool_msg("c1", "晴"),
        msg("assistant", "今天晴"),
    ]
    assert repair_messages(messages) == messages


def test_repair_drops_orphan_tool_message():
    """开头就来一条 tool 消息（上次进程在异常点被 kill）。"""
    messages = [
        tool_msg("c9", "孤立结果"),
        msg("user", "查天气"),
        msg("assistant", "晴"),
    ]
    assert repair_messages(messages) == [msg("user", "查天气"), msg("assistant", "晴")]


def test_repair_fills_missing_tool_results():
    """assistant 发了两个调用，只执行完一个就崩了。"""
    messages = [
        msg("user", "查天气并记待办"),
        assistant_with_calls("c1", "c2"),
        tool_msg("c1", "晴"),
    ]
    out = repair_messages(messages)
    assert tool_ids(out) == ["c1", "c2"]
    assert "缺失" in out[-1]["content"]


def test_repair_drops_unexpected_tool_message():
    messages = [
        msg("user", "q"),
        assistant_with_calls("c1"),
        tool_msg("c1"),
        tool_msg("c9", "多出来的"),
    ]
    assert tool_ids(repair_messages(messages)) == ["c1"]


def test_repair_dedups_duplicate_tool_result():
    messages = [
        msg("user", "q"),
        assistant_with_calls("c1"),
        tool_msg("c1", "第一次"),
        tool_msg("c1", "重复的"),
    ]
    out = repair_messages(messages)
    assert tool_ids(out) == ["c1"]
    assert out[-1]["content"] == "第一次"


def test_repair_keeps_assistant_content_when_filling():
    """补占位消息而不是删 assistant 消息——后者会把思考/正文一起丢掉。"""
    messages = [msg("user", "q"), assistant_with_calls("c1", content="我准备调用工具")]
    out = repair_messages(messages)
    assert out[1]["content"] == "我准备调用工具"
    assert out[1]["tool_calls"][0]["id"] == "c1"


def test_repair_is_idempotent():
    broken = [
        tool_msg("c9"),
        msg("user", "q"),
        assistant_with_calls("c1", "c2"),
        tool_msg("c1"),
        tool_msg("c1", "dup"),
    ]
    once = repair_messages(broken)
    assert repair_messages(once) == once


def test_repair_handles_plain_conversation():
    messages = [msg("user", "你好"), msg("assistant", "你好")]
    assert repair_messages(messages) == messages


# --------------------------------------------------------------------------- #
# 落盘与加载
# --------------------------------------------------------------------------- #
def test_store_save_and_reload(tmp_path):
    store = SessionStore(tmp_path)
    session = store.load_or_create("window-1")
    session.append(msg("user", "你好"))
    session.store["todos"] = [{"id": 1, "text": "带伞", "done": False}]
    session.summary = "用户问过天气"
    store.save(session)

    fresh = SessionStore(tmp_path)  # 模拟新进程
    loaded = fresh.load_or_create("window-1")

    assert loaded.messages[0]["content"] == "你好"
    assert loaded.store["todos"][0]["text"] == "带伞"
    assert loaded.summary == "用户问过天气"


def test_load_or_create_returns_cached_instance(tmp_path):
    store = SessionStore(tmp_path)
    assert store.load_or_create("a") is store.load_or_create("a")


def test_session_id_is_sanitized_into_filename(tmp_path):
    """session_id 里可能有冒号/斜杠，不能直接当文件名。"""
    store = SessionStore(tmp_path)
    store.save(store.load_or_create("user-a:window/1"))

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert ":" not in files[0].name and "/" not in files[0].name
    # 但 session_id 本身必须原样保留，否则跨进程续聊就断了
    assert SessionStore(tmp_path).load_or_create("user-a:window/1").session_id == "user-a:window/1"


def test_long_session_id_is_truncated_with_hash(tmp_path):
    store = SessionStore(tmp_path)
    store.save(store.load_or_create("w" * 200))

    name = list(tmp_path.glob("*.json"))[0].stem
    assert len(name) <= 60


def test_corrupt_file_falls_back_to_empty_session(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    session = SessionStore(tmp_path).load_or_create("broken")

    assert session.session_id == "broken"
    assert session.messages == []


def test_repair_is_applied_on_load(tmp_path):
    (tmp_path / "bad.json").write_text(
        json.dumps(
            {"session_id": "bad", "messages": [tool_msg("c9"), msg("user", "q")]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    session = SessionStore(tmp_path).load_or_create("bad")

    assert all(m["role"] != "tool" for m in session.messages)


def test_delete_clears_file_and_cache(tmp_path):
    store = SessionStore(tmp_path)
    session = store.load_or_create("x")
    session.append(msg("user", "q"))
    store.save(session)

    store.delete("x")

    assert not list(tmp_path.glob("*.json"))
    assert store.load_or_create("x").messages == []


def test_list_sessions(tmp_path):
    store = SessionStore(tmp_path)
    for sid in ("a", "b"):
        store.save(store.load_or_create(sid))

    assert sorted(store.list_sessions()) == ["a", "b"]


def test_list_sessions_skips_corrupt_files(tmp_path):
    store = SessionStore(tmp_path)
    store.save(store.load_or_create("ok"))
    (tmp_path / "bad.json").write_text("oops", encoding="utf-8")

    assert store.list_sessions() == ["ok"]


def test_save_leaves_no_temp_file(tmp_path):
    store = SessionStore(tmp_path)
    store.save(store.load_or_create("t"))
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------------- #
# 锁
# --------------------------------------------------------------------------- #
def test_session_lock_is_reentrant():
    """同一线程内嵌套加锁不能死锁（loop 里 ToolContext 回调用到）。"""
    session = Session(session_id="x")
    with session.lock:
        with session.lock:
            acquired = True
    assert acquired
