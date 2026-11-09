"""跨模块共享的数据结构。

为什么单独抽一个 types.py：ToolCall 同时被 llm / parser / registry / loop 使用，
塞进它们中的任何一个都会形成循环 import。这里只放纯数据，不放任何逻辑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, MutableMapping


# 模型侧
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCall:
    """归一化后的工具调用。"""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw_args: str = ""
    id: str = ""

    def to_message(self) -> dict[str, Any]:
        """还原成 OpenAI assistant.tool_calls 数组里的元素。

        写回历史消息时必须走这里：后续每一条 role="tool" 的消息都要用同一个 id
        去对应，id 对不上 OpenAI 直接返回 400。
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_args
                or json.dumps(self.args, ensure_ascii=False),
            },
        }


@dataclass
class LLMResponse:
    """llm.py 的唯一出口类型，上层不再接触 SDK 的原始对象。"""

    content: str = ""
    reasoning: str = ""  # DeepSeek-R1 / o1 系的思考过程
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = ""
    raw: Any = None

# 工具侧
@dataclass
class ToolResult:
    ok: bool
    content: str
    tool: str = ""
    error: str = ""
    elapsed_ms: float = 0.0
    truncated: bool = False


@dataclass
class ToolContext:
    """工具执行时能拿到的"环境"。

    关键点：session_id 和 store 在这里，所以像 todo 这种有状态的工具
    才能天然做到按窗口隔离。如果工具签名只是 func(args)，就做不到。
    """

    session_id: str
    store: MutableMapping[str, Any]
    turn: int = 0
    trace: Any = None
    clock: Callable[[], str] | None = None

    def timestamp(self) -> str:
        if self.clock is not None:
            return self.clock()
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def log(self, event: str, **data: Any) -> None:
        if self.trace is not None:
            self.trace.event(event, session_id=self.session_id, turn=self.turn, **data)
