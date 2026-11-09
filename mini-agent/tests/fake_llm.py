"""脚本化假模型。

为什么必须有一个：loop 里真正难测的是"模型不按预期出牌"——
一直调工具不停、参数写错、重复调用同一个工具。
这些用真实 API 造不出来（或者造出来要烧钱且不可复现）。

接口与 `LLMClient.chat` 完全一致，所以 AgentLoop 不用为测试改一行代码。
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable

from agent.types import LLMResponse, ToolCall, Usage


def call(name: str, args: dict[str, Any] | None = None, call_id: str = "call_1") -> ToolCall:
    args = dict(args or {})
    # 真实 API 返回的就是 arguments 字符串，这里必须一致，
    # 否则 parser 走的是"空参数"分支，测出来的行为和线上不一致
    return ToolCall(
        name=name,
        args=args,
        raw_args=json.dumps(args, ensure_ascii=False),
        id=call_id,
    )


def tool_reply(
    *calls: ToolCall,
    content: str = "",
    reasoning: str = "",
    tokens: int = 100,
) -> LLMResponse:
    """模型决定调用工具。"""
    return LLMResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=list(calls),
        usage=Usage(prompt_tokens=tokens, completion_tokens=tokens // 2, total_tokens=tokens),
        model="fake",
        finish_reason="tool_calls",
    )


def reply(content: str, reasoning: str = "", tokens: int = 100) -> LLMResponse:
    """模型给出最终回答。"""
    return LLMResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=[],
        usage=Usage(prompt_tokens=tokens, completion_tokens=tokens // 2, total_tokens=tokens),
        model="fake",
        finish_reason="stop",
    )


class ScriptedLLM:
    """按脚本顺序返回响应。

    - `script` 里可以放 LLMResponse，也可以放 `f(messages) -> LLMResponse` 的函数
      （需要根据第几轮动态生成参数时用后者）
    - `repeat_last=True` 时，脚本耗尽了就重复最后一项，用来测 max_turns
    """

    def __init__(self, script: list[Any], repeat_last: bool = False) -> None:
        self.script = list(script)
        self.repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": tools,
            }
        )

        if not self.script:
            raise AssertionError("脚本已耗尽，模型被调用的次数超出预期")

        if self.repeat_last and len(self.script) == 1:
            item = self.script[0]
        else:
            item = self.script.pop(0)
        return item(messages) if isinstance(item, Callable) else item  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_messages(self) -> list[dict[str, Any]]:
        return self.calls[-1]["messages"]

    def all_message_text(self) -> str:
        chunks: list[str] = []
        for call in self.calls:
            for msg in call["messages"]:
                content = msg.get("content")
                if isinstance(content, str):
                    chunks.append(content)
        return "\n".join(chunks)
