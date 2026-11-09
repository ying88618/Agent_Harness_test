"""Session：会话级状态 + 落盘。

容易做错的不是"存消息"，而是三件小事：
1. **隔离粒度**：消息列表和工具状态（待办）都挂在 session 上，不是全局
2. **并发**：同一 session 被两个请求同时写 → 消息顺序错乱，用 per-session 锁
3. **配对完整性**：进程崩在"assistant 发了 tool_calls、工具还没跑完"中间，
   下次加载回来就有一条孤儿 tool 消息 → OpenAI 直接 400，而且是持久性的。
   `repair_messages()` 就是修这个的。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]+")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Session:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    store: dict[str, Any] = field(default_factory=dict)  # 工具私有状态（todo 在这）
    summary: str = ""  # 更早轮次的压缩摘要
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.updated_at = _now()

    def extend(self, messages: list[dict[str, Any]]) -> None:
        self.messages.extend(messages)
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "store": self.store,
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            messages=repair_messages(data.get("messages") or []),
            store=data.get("store") or {},
            summary=data.get("summary") or "",
            created_at=data.get("created_at") or _now(),
            updated_at=data.get("updated_at") or _now(),
        )


# 配对修复
def repair_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保证 assistant.tool_calls 与 role=tool 消息严格一一对应。

    规则：
    - 孤立的 tool 消息（前面没有对应的 tool_calls）→ 丢弃
    - tool_calls 里有 id 没有对应结果 → 补一条占位 tool 消息（而不是删掉 assistant 消息，
      因为那条 assistant 消息可能还带着正文/思考，删了会丢信息）
    """
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            ids = [tc.get("id") for tc in msg["tool_calls"] if isinstance(tc, dict)]
            out.append(msg)
            i += 1

            seen: set[str] = set()
            while i < len(messages) and messages[i].get("role") == "tool":
                tid = messages[i].get("tool_call_id")
                if tid in ids and tid not in seen:
                    seen.add(tid)
                    out.append(messages[i])
                i += 1

            for tid in ids:
                if tid not in seen:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": tid,
                            "content": "（该工具调用此前未执行完成，结果缺失）",
                        }
                    )
            continue

        if role == "tool":
            i += 1  # 孤儿 tool 消息：丢弃
            continue

        out.append(msg)
        i += 1
    return out


# 存储
class SessionStore:
    """文件落盘 + 进程内缓存。
    落盘的意义：用户关掉终端再开，`--session window-1` 还能接着聊。
    """

    def __init__(self, root: str | Path = "sessions") -> None:
        self.root = Path(root)
        self._cache: dict[str, Session] = {}
        self._guard = threading.Lock()

    def _path(self, session_id: str) -> Path:
        safe = _SAFE_RE.sub("_", session_id).strip("._") or "default"
        if len(safe) > 60:
            digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:8]
            safe = safe[:50] + "_" + digest
        return self.root / f"{safe}.json"

    def list_sessions(self) -> list[str]:
        if not self.root.exists():
            return []
        ids = []
        for p in sorted(self.root.glob("*.json")):
            try:
                ids.append(json.loads(p.read_text(encoding="utf-8"))["session_id"])
            except (OSError, ValueError, KeyError):
                continue
        return ids

    def load_or_create(self, session_id: str) -> Session:
        with self._guard:
            if session_id in self._cache:
                return self._cache[session_id]

            path = self._path(session_id)
            session: Session | None = None
            if path.exists():
                try:
                    session = Session.from_dict(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError, KeyError):
                    session = None  # 文件坏了就当新会话，不要让用户卡死

            if session is None:
                session = Session(session_id=session_id)

            self._cache[session_id] = session
            return session

    def save(self, session: Session) -> None:
        path = self._path(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)  # 原子写，避免写一半进程挂掉留个坏文件

    def delete(self, session_id: str) -> None:
        with self._guard:
            self._cache.pop(session_id, None)
            self._path(session_id).unlink(missing_ok=True)
