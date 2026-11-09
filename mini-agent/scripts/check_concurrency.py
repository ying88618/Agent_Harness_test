#!/usr/bin/env python3
"""并发自测：验证 per-session 锁与 session 隔离在真实线程下的表现。

为什么单独做成脚本而不是 pytest 用例：
并发测试依赖线程调度，写进 CI 容易变成偶发失败（flaky）。
用脚本跑、人工看结果更合适。

    python scripts/check_concurrency.py
    python scripts/check_concurrency.py --threads 50

不需要 API Key —— 用无状态的假模型，只测我们自己的并发逻辑。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows 终端默认可能是 GBK
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from agent.loop import AgentLoop  # noqa: E402
from agent.session import SessionStore  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402
from tests.fake_llm import call, reply, tool_reply  # noqa: E402


class EchoFakeLLM:
    """无状态假模型，线程安全。

    行为：第一轮把用户最后一句话记成待办，第二轮看到 tool 结果就收尾。
    完全没有共享可变状态，所以能安全地被多个线程同时调用。
    """

    def chat(self, messages, tools=None, tool_choice=None, temperature=None):
        last = messages[-1]
        if last.get("role") == "tool":
            return reply("已记录。")
        text = str(last.get("content") or "")[:40]
        return tool_reply(
            call("todo", {"action": "add", "text": text}, call_id=f"c{uuid.uuid4().hex[:8]}")
        )


def run_threads(count: int, target) -> list[str]:
    errors: list[str] = []

    def wrap(i: int) -> None:
        try:
            target(i)
        except Exception as exc:  # 线程里的异常不会自动冒泡，必须自己收
            errors.append(f"thread-{i}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=wrap, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def case_same_session(tmp: Path, count: int) -> tuple[bool, str]:
    """N 个线程写同一个 session：消息不能交错，配对不能断，待办不能丢。"""
    store = SessionStore(tmp / "same")
    # 关掉压缩：本用例要精确数消息条数，压缩会按设计折叠旧轮次，干扰断言
    loop = AgentLoop(
        EchoFakeLLM(),
        build_default_registry(),
        store,
        trace_path=None,
        max_turns=4,
        max_context_chars=10**9,
    )

    started = time.perf_counter()
    errors = run_threads(count, lambda i: loop.run("stress", f"任务-{i}"))
    elapsed = (time.perf_counter() - started) * 1000

    session = store.load_or_create("stress")
    roles = [m.get("role") for m in session.messages]

    problems: list[str] = []
    if errors:
        problems.append(f"{len(errors)} 个线程抛异常，例如 {errors[0]}")

    expected = 4 * count
    if len(roles) != expected:
        problems.append(f"消息数 {len(roles)} != 预期 {expected}")

    # 每次 run 必须是完整连续的一块 [user, assistant, tool, assistant]
    for i in range(0, min(len(roles), expected), 4):
        chunk = roles[i : i + 4]
        if chunk != ["user", "assistant", "tool", "assistant"]:
            problems.append(f"第 {i // 4} 组消息交错：{chunk}")
            break

    # 配对完整性：每条 tool 消息都要有对应的 tool_calls
    allowed: set[str] = set()
    for msg in session.messages:
        if msg.get("role") == "assistant":
            allowed = {tc["id"] for tc in msg.get("tool_calls") or []}
        elif msg.get("role") == "tool" and msg.get("tool_call_id") not in allowed:
            problems.append("出现孤立 tool 消息（配对断裂）")
            break

    todos = session.store.get("todos") or []
    if len(todos) != count:
        problems.append(f"待办数 {len(todos)} != 预期 {count}")
    elif {t["text"] for t in todos} != {f"任务-{i}" for i in range(count)}:
        problems.append("待办内容与提交内容不一致（可能被覆盖）")

    detail = f"{count} 线程 / {len(roles)} 条消息 / {len(todos)} 条待办 / {elapsed:.0f}ms"
    return (not problems), detail + ("；" + "；".join(problems) if problems else "")


def case_multi_session(tmp: Path, count: int) -> tuple[bool, str]:
    """N 个线程写 N 个 session：彼此不能串。"""
    store = SessionStore(tmp / "multi")
    loop = AgentLoop(EchoFakeLLM(), build_default_registry(), store, trace_path=None, max_turns=4)

    errors = run_threads(count, lambda i: loop.run(f"user-{i}", f"任务-{i}"))

    problems: list[str] = []
    if errors:
        problems.append(f"{len(errors)} 个线程抛异常，例如 {errors[0]}")

    for i in range(count):
        session = store.load_or_create(f"user-{i}")
        todos = session.store.get("todos") or []
        if len(todos) != 1:
            problems.append(f"user-{i} 待办数 {len(todos)} != 1")
        elif todos[0]["text"] != f"任务-{i}":
            problems.append(f"user-{i} 拿到了别人的待办：{todos[0]['text']}")

    detail = f"{count} 个 session 并发写"
    return (not problems), detail + ("；" + "；".join(problems[:3]) if problems else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=20)
    args = parser.parse_args()

    tmp = Path("logs/concurrency")
    # 每次从干净状态开始。上一轮的会话文件会被 load_or_create 读回来，
    # 待办数量就会翻倍，看起来像并发 bug（第一次跑 50 线程就栽在这）。
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"并发自测（线程数 {args.threads}）\n")

    results = [("同一 session 并发写", case_same_session(tmp, args.threads)),
               ("多 session 并发隔离", case_multi_session(tmp, args.threads))]

    failed = 0
    for name, (ok, detail) in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         {detail}")
        failed += 0 if ok else 1

    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项失败'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
