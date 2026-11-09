"""Context 组装与压缩。

两个关键决策
------------
1. **摘要存在 `session.summary`，不进 messages 数组。**
   组装时把它拼到 system prompt 的**末尾**。这样 system 的前缀是稳定的

2. **压缩只在 turn 边界切。**
   一个 turn = 一条 user 消息 + 它之后所有 assistant/tool 消息。
   按 turn 整体裁剪，`assistant.tool_calls` 和对应的 `role=tool` 消息永远同进同出。
   只删 assistant 留 tool（或反过来）→ OpenAI 400，而且是持久性错误：
   存盘之后每次加载都会 400，看起来像"这个会话坏了"。
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from .registry import ToolRegistry
from .session import Session

DEFAULT_MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
DEFAULT_KEEP_RECENT_TURNS = int(os.getenv("KEEP_RECENT_TURNS", "2"))
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "1500"))

_BASE_PROMPT = (
    "你是一个可以调用工具的 AI 助手，运行在命令行终端里。"
    "你的回答会被直接展示给用户，请使用简体中文，简洁、准确、可执行。"
)

_RULES = """\
## 工具使用规则
1. 需要实时信息、外部资料或精确计算时，必须调用工具，不要凭记忆编造。
2. 一轮可以发起多个工具调用；拿到结果后先判断信息是否足够。
3. 信息足够就立刻给出最终回答，不要再调用工具。
4. 工具报错时，根据错误信息里的 schema 修正参数并重试一次；仍然失败就如实告诉用户。
5. 不要用相同的参数重复调用同一个工具，结果不会变化。
6. 数字运算一律用 calculator，不要自己心算。
7. 不要假装调用工具，也不要在回答里伪造工具结果的格式。"""


# 组装
def build_system_prompt(
    registry: ToolRegistry,
    summary: str = "",
    extra: str = "",
) -> str:
    parts = [_BASE_PROMPT]

    tools = registry.describe()
    if tools:
        parts.append("## 可用工具\n" + tools)
    parts.append(_RULES)

    if extra.strip():
        parts.append(extra.strip())

    # 摘要放最后：保持前面所有内容作为稳定前缀
    if summary.strip():
        parts.append("## 更早对话的摘要（不是本轮内容）\n" + summary.strip())

    return "\n\n".join(parts)


def build_messages(session: Session, registry: ToolRegistry, extra_rules: str = "") -> list[dict[str, Any]]:
    system = build_system_prompt(registry, session.summary, extra_rules)
    return [{"role": "system", "content": system}, *session.messages]


def count_chars(messages: list[dict[str, Any]]) -> int:
    """粗估上下文体积。

    不引 tiktoken：中文大约 1 token ≈ 1.5~2 字符，英文 ≈ 4 字符，
    我们只需要一个稳定的、能触发阈值的量级判断，不需要精确计费。
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        if msg.get("tool_calls"):
            total += len(json.dumps(msg["tool_calls"], ensure_ascii=False, default=str))
    return total


# 压缩
def split_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """按 user 消息切 turn。开头的非 user 消息自成一个 turn，保证不丢。"""
    turns: list[list[dict[str, Any]]] = []
    for msg in messages:
        if msg.get("role") == "user" or not turns:
            turns.append([msg])
        else:
            turns[-1].append(msg)
    return turns


def summarize_turns(
    turns: list[list[dict[str, Any]]],
    per_turn: int = 160,
    max_tools_per_turn: int = 2,
) -> str:
    """抽取式摘要：每条 turn 留「用户问了什么 + 调了什么工具 + 助手结论是什么」。

    为什么必须带上工具结果：用户很可能追问「把刚才那个待办改到明天」，
    而待办 ID 只存在于 tool 结果里。摘要若只保留问答对，ID 就永久丢了，
    追问必然失败。所以工具结果要留一条短的要点，而不是抛弃。

    为什么不用 LLM 做摘要：多一次调用、多一份不确定性、还要花钱。
    要求的是"基础的压缩"。真需要生成式摘要时，把 summarizer 参数换掉即可，
    调用方代码不用改。
    """
    lines: list[str] = []
    for turn in turns:
        user_text = ""
        answer_text = ""
        tool_names: dict[str, str] = {}
        tool_notes: list[str] = []

        for msg in turn:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()

            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        tool_names[tc.get("id", "")] = (tc.get("function") or {}).get("name", "")
                if content and not msg.get("tool_calls"):
                    answer_text = content  # 取最后一条纯文本回答
            elif role == "user" and not user_text:
                user_text = content
            elif role == "tool" and content and len(tool_notes) < max_tools_per_turn:
                name = tool_names.get(msg.get("tool_call_id", ""), "tool")
                tool_notes.append(f"{name}→{_clip(content, 80)}")

        if not user_text and not answer_text and not tool_notes:
            continue

        block = [f"- 用户：{_clip(user_text, per_turn)}"]
        if tool_notes:
            block.append("  工具：" + "；".join(tool_notes))
        if answer_text:
            block.append(f"  助手：{_clip(answer_text, per_turn)}")
        lines.append("\n".join(block))
    return "\n".join(lines)


def maybe_compress(
    session: Session,
    max_chars: int | None = None,
    keep_recent_turns: int | None = None,
    summarizer: Callable[[list[list[dict[str, Any]]]], str] | None = None,
) -> bool:
    """超过预算就把较早的 turn 折叠进 summary。返回是否发生了压缩。"""
    budget = DEFAULT_MAX_CONTEXT_CHARS if max_chars is None else max_chars
    keep = max(1, DEFAULT_KEEP_RECENT_TURNS if keep_recent_turns is None else keep_recent_turns)

    # 摘要本身也算体积
    if count_chars(session.messages) + len(session.summary) <= budget:
        return False

    turns = split_turns(session.messages)
    if len(turns) <= keep:
        # 只剩最近几个 turn 还超预算，说明单轮就过长了。
        # 这里不硬切（会破坏配对），留给下次或由工具截断兜底。
        return False

    old, recent = turns[:-keep], turns[-keep:]
    summary = (summarizer or summarize_turns)(old)

    merged = "\n".join(x for x in (session.summary, summary) if x.strip())
    session.summary = merged[-MAX_SUMMARY_CHARS:]  # 摘要也不能无限长
    session.messages = [msg for turn in recent for msg in turn]
    return True


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"