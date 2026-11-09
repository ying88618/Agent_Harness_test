"""事件流 trace：JSONL 落盘 + 内存环形缓冲。

为什么用 ContextVar 而不是到处传参数：
trace 是横切关注点，每个工具、每次 LLM 调用都想记一笔。传参会污染所有签名。


trace 永远不进入 context（不写进 messages），日志是给人看的。
"""

from __future__ import annotations

import json
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class Trace:
    def __init__(
        self,
        path: Path | str | None = None,
        session_id: str = "",
        run_id: str | None = None,
        keep: int = 500,
    ) -> None:
        self.path = Path(path) if path else None
        self.session_id = session_id
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.keep = keep
        self.events: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, **data: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": _now(),
            "elapsed_ms": round((time.perf_counter() - self._t0) * 1000, 1),
            "session_id": self.session_id,
            "run_id": self.run_id,
            "event": name,
        }
        record.update({k: v for k, v in data.items() if v is not None})

        self.events.append(record)
        if len(self.events) > self.keep:
            del self.events[: len(self.events) - self.keep]

        if self.path is not None:
            # 写盘失败绝不能影响主流程
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass
        return record

    def events_of(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == name]

# ContextVar 绑定
_current: ContextVar[Trace | None] = ContextVar("mini_agent_trace", default=None)


def bind_trace(trace: Trace):
    return _current.set(trace)


def unbind_trace(token) -> None:
    try:
        _current.reset(token)
    except ValueError:  # 跨 context 复位会抛，忽略即可
        pass


def current_trace() -> Trace | None:
    return _current.get()


def trace_event(name: str, **data: Any) -> None:
    trace = _current.get()
    if trace is not None:
        trace.event(name, **data)
