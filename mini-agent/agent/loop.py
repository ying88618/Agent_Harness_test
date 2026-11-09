"""ReAct 主循环 —— 本项目的核心，等价于 create_agent 内部的"图"。

这里把它拉平成一个 for 循环，每一轮做五件事：
    1. 组装 context 并调用模型
    2. 解析输出（thought / tool_calls / final_answer）
    3. 没有 tool_calls → 这就是最终答案，退出
    4. 有 tool_calls → 写回 assistant 消息，逐个执行，结果写回 tool 消息
    5. 检查是否重复调用、是否需要压缩 context

被砍掉的是框架的"壳"（StateGraph / reducer / checkpoint / middleware 协议），
保留的是那些壳存在的理由：轮次上限、重复调用熔断、错误回灌、配对完整性。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import context as context_mod
from .errors import LLMError
from .llm import LLMClient
from .parser import parse_response
from .registry import ToolRegistry
from .session import Session, SessionStore
from .trace import Trace, bind_trace, unbind_trace
from .types import ToolCall, ToolContext

EventHook = Callable[[str, dict[str, Any]], None]


@dataclass
class AgentResult:
    answer: str
    stop_reason: str
    turns: int = 0
    tool_calls: int = 0
    session_id: str = ""
    elapsed_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)


def _noop(name: str, data: dict[str, Any]) -> None:
    return None


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        store: SessionStore | None = None,
        *,
        max_turns: int = 6,
        max_context_chars: int | None = None,
        keep_recent_turns: int | None = None,
        max_repeat: int = 2,
        max_tool_output: int = 4000,
        extra_rules: str = "",
        trace_path: str | Path | None = "logs/trace.jsonl",
        on_event: EventHook | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.store = store or SessionStore()
        self.max_turns = max(1, max_turns)
        self.max_context_chars = max_context_chars
        self.keep_recent_turns = keep_recent_turns
        self.max_repeat = max(2, max_repeat)
        self.max_tool_output = max_tool_output
        self.extra_rules = extra_rules
        self.trace_path = trace_path
        self.on_event = on_event or _noop

    def run(
        self,
        session_id: str,
        user_input: str,
        on_event: EventHook | None = None,
    ) -> AgentResult:
        started = time.perf_counter()
        emit = on_event or self.on_event
        session = self.store.load_or_create(session_id)

        # per-session 锁：同一个窗口被并发写时，消息顺序不会错乱
        with session.lock:
            trace = Trace(path=self.trace_path, session_id=session_id)
            token = bind_trace(trace)
            try:
                result = self._run(session, user_input, trace, emit)
            finally:
                unbind_trace(token)
                self.store.save(session)

        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    def _run(
        self,
        session: Session,
        user_input: str,
        trace: Trace,
        emit: EventHook,
    ) -> AgentResult:
        trace.event("user_input", text=user_input[:1000])
        session.append({"role": "user", "content": user_input})

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_call_count = 0
        last_signature: str | None = None
        repeat = 0
        answer = ""
        stop_reason = "max_turns"
        turn = 0

        try:
            for turn in range(1, self.max_turns + 1):
                # 压缩放在"组装 context 之前"。
                # 之前放在工具执行之后，导致纯闲聊（不走工具分支）永远压不到，
                # 长对话会无限膨胀。现在无论本轮走哪条分支，只要体积超预算都会触发，
                # 而且压完立刻作用于本次请求。
                if context_mod.maybe_compress(
                    session,
                    max_chars=self.max_context_chars,
                    keep_recent_turns=self.keep_recent_turns,
                ):
                    trace.event(
                        "compressed",
                        turn=turn,
                        remaining_turns=len(context_mod.split_turns(session.messages)),
                        summary_chars=len(session.summary),
                    )

                messages = context_mod.build_messages(session, self.registry, self.extra_rules)
                trace.event(
                    "llm_start",
                    turn=turn,
                    messages=len(messages),
                    chars=context_mod.count_chars(messages),
                )
                emit("llm_start", {"turn": turn})

                t0 = time.perf_counter()
                resp = self.llm.chat(messages, tools=self.registry.openai_tools())
                trace.event(
                    "llm_end",
                    turn=turn,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                    tool_calls=len(resp.tool_calls),
                    total_tokens=resp.usage.total_tokens,
                    finish_reason=resp.finish_reason,
                    reasoning=resp.reasoning[:400] or None,
                )
                usage["prompt_tokens"] += resp.usage.prompt_tokens
                usage["completion_tokens"] += resp.usage.completion_tokens
                usage["total_tokens"] += resp.usage.total_tokens

                parsed = parse_response(resp)
                for warning in parsed.warnings:
                    trace.event("parse_warning", turn=turn, detail=warning)

                #情况 1：模型给出了最终答案
                if not parsed.has_tool_call:
                    final = (parsed.final_answer or "").strip()
                    if not final:
                        # 既没调工具也没输出 → 回灌一条提示，给它一次改正机会
                        trace.event("empty_answer", turn=turn)
                        session.append({"role": "assistant", "content": "（本轮无输出）"})
                        session.append(
                            {
                                "role": "user",
                                "content": "你上一轮既没有输出内容也没有调用工具。请直接回答用户的问题，或调用合适的工具。",
                            }
                        )
                        continue

                    answer, stop_reason = final, "final"
                    session.append({"role": "assistant", "content": final})
                    trace.event("final", turn=turn, chars=len(final))
                    emit("final", {"answer": final, "turn": turn})
                    break

                #情况 2：模型要调工具
                tool_call_count += len(parsed.tool_calls)

                # 先把 assistant 消息（含 tool_calls）写回去。
                # 顺序很重要：必须在 tool 消息之前，否则配对断裂。
                session.append(
                    {
                        "role": "assistant",
                        "content": resp.content or "",
                        "tool_calls": [call.to_message() for call in parsed.tool_calls],
                    }
                )

                # 重复调用熔断：同一组 (工具名 + 参数) 连续出现就不再执行
                signature = _signature(parsed.tool_calls)
                if signature == last_signature:
                    repeat += 1
                else:
                    last_signature, repeat = signature, 1

                if repeat >= self.max_repeat:
                    trace.event(
                        "repeat_blocked",
                        turn=turn,
                        repeat=repeat,
                        tool=parsed.tool_calls[0].name,
                    )
                    for call in parsed.tool_calls:
                        session.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": (
                                    "同一组工具调用已经连续执行多次，结果不会变化。"
                                    "请换用不同的参数，或直接基于已有信息回答用户。"
                                ),
                            }
                        )
                    continue

                # 执行工具。注意 ToolContext 里带了 session.store，
                # todo 的数据因此天然按 session 隔离。
                tool_ctx = ToolContext(
                    session_id=session.session_id,
                    store=session.store,
                    turn=turn,
                    trace=trace,
                )
                for call in parsed.tool_calls:
                    trace.event("tool_start", turn=turn, tool=call.name, args=_preview(call.args))
                    emit("tool_call", {"name": call.name, "args": call.args, "id": call.id})

                    result = self.registry.execute(
                        call, tool_ctx, max_output_chars=self.max_tool_output
                    )

                    trace.event(
                        "tool_end",
                        turn=turn,
                        tool=call.name,
                        ok=result.ok,
                        error=result.error or None,
                        elapsed_ms=result.elapsed_ms,
                        chars=len(result.content),
                    )
                    emit(
                        "tool_result",
                        {"name": call.name, "ok": result.ok, "content": result.content},
                    )
                    session.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        }
                    )

            else:
                # for 正常跑完没 break = 用尽了最大轮次
                stop_reason = "max_turns"
                answer = self._max_turns_answer(session, tool_call_count)

                # 关键：把兜底结论写回历史，让这一轮"收口"。
                # 不写的话 session 会停在 assistant(tool_calls) + tool(...) 这种
                # "话说到一半"的状态：用户下一句追问时，模型看到的是一堆悬空的
                # 工具结果，而且不知道自己上一轮已经放弃了，容易原样重来。
                session.append({"role": "assistant", "content": answer})
                trace.event("max_turns", turns=self.max_turns, tool_calls=tool_call_count)

        except LLMError as exc:
            stop_reason = "llm_error"
            answer = f"模型调用失败：{exc}"
            trace.event("llm_error", error=str(exc)[:300])

        return AgentResult(
            answer=answer,
            stop_reason=stop_reason,
            turns=turn,
            tool_calls=tool_call_count,
            session_id=session.session_id,
            usage=usage,
        )

    # ------------------------------------------------------------------ #
    def _max_turns_answer(self, session: Session, tool_calls: int) -> str:
        """轮次耗尽时的兜底回答。

        不能只甩一句"抱歉"——那等于把已经拿到的信息全扔掉，用户还得从头再问。
        产品上正确的做法：说明已经到上限，把掌握到的片段摊开，
        让用户自己决定是补充信息还是把问题拆小。
        """
        gathered = [
            " ".join(str(m.get("content", "")).split())[:120]
            for m in session.messages
            if m.get("role") == "tool" and m.get("content")
        ][-3:]

        lines = [
            f"我已经尝试了 {self.max_turns} 轮、调用了 {tool_calls} 次工具，"
            "仍没能得出确定结论。"
        ]
        if gathered:
            lines.append("目前掌握到的信息：")
            lines.extend(f"- {item}" for item in gathered)
        lines.append("你可以补充更具体的信息，或者把问题拆成几个小问题，我再试一次。")
        return "\n".join(lines)



def _signature(calls: list[ToolCall]) -> str:
    """把一组工具调用归一化成可比较的签名。

    归一化三件事（少一件就会漏检，表现出来就是"死循环没被拦住"）：
      - None / 空串视为同一回事
      - 字符串压掉多余空白
      - key 排序（同一个 dict 的插入顺序可能不同）
    """
    parts: list[str] = []
    for call in calls:
        args: Any = call.args
        if isinstance(args, dict):
            cleaned: dict[str, Any] = {}
            for key, value in args.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    value = " ".join(value.split())
                    if not value:
                        continue
                cleaned[key] = value
            try:
                body = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                body = repr(cleaned)
        else:
            body = repr(args)
        parts.append(f"{call.name}:{body}")
    return "|".join(parts)


def _preview(args: Any, limit: int = 300) -> str:
    try:
        text = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(args)
    return text[:limit]
