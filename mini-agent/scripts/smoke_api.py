#!/usr/bin/env python3
"""真实 API 冒烟：4 个端到端场景，逐条 PASS/FAIL。

    python scripts/smoke_api.py

需要环境变量（或项目根目录的 .env）：
    LLM_API_KEY / LLM_BASE_URL / LLM_MODEL

大约消耗 6~8 次模型调用，用 DeepSeek 级别模型是几分钱。
每次运行使用随机 session id，不会污染你手动测试留下的会话。
"""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows 终端默认可能是 GBK
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from main import _load_dotenv  # noqa: E402

from agent.llm import LLMClient  # noqa: E402
from agent.loop import AgentLoop  # noqa: E402
from agent.session import SessionStore  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def digits_only(text: str) -> str:
    return re.sub(r"[^0-9.]", "", text)


def main() -> int:
    _load_dotenv()

    try:
        llm = LLMClient()
    except Exception as exc:
        print(f"无法初始化模型客户端：{exc}")
        print("请先设置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL，或在项目根目录建 .env")
        return 2

    run = uuid.uuid4().hex[:6]
    sid_a, sid_b = f"smoke-{run}-a", f"smoke-{run}-b"
    store = SessionStore("sessions")
    loop = AgentLoop(
        llm=llm,
        registry=build_default_registry(),
        store=store,
        max_turns=6,
        trace_path="logs/smoke.jsonl",
    )

    print(f"模型 {llm.model} @ {llm.base_url}")
    print(f"会话 {sid_a} / {sid_b}\n")

    # ---------------- 场景 1：单工具 + 精确计算 ----------------
    r1 = loop.run(sid_a, "112358 乘以 2 等于多少？请用计算器算。")
    check(
        "1. 单工具调用（calculator）",
        r1.stop_reason == "final" and r1.tool_calls >= 1 and "224716" in digits_only(r1.answer),
        f"{r1.turns} 轮 / {r1.tool_calls} 次工具",
    )

    # ---------------- 场景 2：一轮内多工具 ----------------
    r2 = loop.run(sid_a, "厦门今天天气怎么样？顺便记一条待办：下午三点带伞。")
    todos = store.load_or_create(sid_a).store.get("todos") or []
    check(
        "2. 多工具（weather + todo）",
        r2.stop_reason == "final" and r2.tool_calls >= 2 and len(todos) >= 1,
        f"{r2.tool_calls} 次工具 / {len(todos)} 条待办",
    )

    # ---------------- 场景 3：同一 session 追问 ----------------
    r3 = loop.run(sid_a, "我刚才让你记的待办是什么？")
    check(
        "3. 同 session 追问",
        r3.stop_reason == "final" and "带伞" in r3.answer,
        r3.answer.replace("\n", " ")[:50],
    )

    # ---------------- 场景 4：另一个 session 不能串 ----------------
    r4 = loop.run(sid_b, "我的待办清单里有什么？")
    todos_b = store.load_or_create(sid_b).store.get("todos") or []
    check(
        "4. session 隔离（窗口 B 看不到窗口 A 的待办）",
        r4.stop_reason == "final" and len(todos_b) == 0,
        f"窗口B 待办数 {len(todos_b)}",
    )

    total_tokens = (r1.usage["total_tokens"] + r2.usage["total_tokens"]
                    + r3.usage["total_tokens"] + r4.usage["total_tokens"])
    elapsed = sum(r.elapsed_ms for r in (r1, r2, r3, r4)) / 1000

    failed = [name for name, ok, _ in _results if not ok]
    print(f"\n合计 {total_tokens} tokens / {elapsed:.1f}s / trace: logs/smoke.jsonl")
    print("全部通过" if not failed else f"{len(failed)} 项失败")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
