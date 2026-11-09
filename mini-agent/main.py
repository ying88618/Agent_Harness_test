#!/usr/bin/env python3
"""CLI 入口：交互式 REPL。

用 `--session` 切换窗口，每个 session 的消息、待办、摘要都是独立的，
正好演示题目要求的"用户 A 两个窗口互不影响"。

    python main.py --session window-1
    python main.py --session window-2
    python main.py --once "厦门今天天气怎么样" --session demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """极简 .env 加载（不引第三方依赖）。已存在的环境变量优先，不覆盖。"""
    file = Path(path)
    if not file.exists():
        return
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _fix_console_encoding() -> None:
    """Windows 终端默认可能是 GBK，中文输出会 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def make_printer(quiet: bool):
    def printer(name: str, data: dict) -> None:
        if quiet:
            return
        if name == "tool_call":
            args = json.dumps(data.get("args", {}), ensure_ascii=False)
            print(f"  -> {data['name']}({args})")
        elif name == "tool_result":
            text = " ".join(str(data.get("content", "")).split())
            mark = "ok" if data.get("ok") else "失败"
            suffix = "..." if len(text) > 220 else ""
            print(f"  <- [{mark}] {text[:220]}{suffix}")

    return printer


def build_loop(args, on_event):
    from agent.llm import LLMClient
    from agent.loop import AgentLoop
    from agent.session import SessionStore
    from agent.tools import build_default_registry

    llm = LLMClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    return AgentLoop(
        llm=llm,
        registry=build_default_registry(),
        store=SessionStore(args.session_dir),
        max_turns=args.max_turns,
        trace_path=None if args.no_trace else args.trace,
        on_event=on_event,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-agent：最小可用的 ReAct Agent")
    p.add_argument("--session", "-s", default="default", help="会话 ID（不同窗口用不同 ID）")
    p.add_argument("--once", "-c", default=None, help="只执行一句用户输入然后退出")
    p.add_argument("--max-turns", type=int, default=6, help="单次对话的最大模型调用轮次")
    p.add_argument("--model", default=None, help="覆盖 LLM_MODEL")
    p.add_argument("--base-url", default=None, help="覆盖 LLM_BASE_URL")
    p.add_argument("--api-key", default=None, help="覆盖 LLM_API_KEY")
    p.add_argument("--session-dir", default="sessions", help="会话落盘目录")
    p.add_argument("--trace", default="logs/trace.jsonl", help="trace 输出路径")
    p.add_argument("--no-trace", action="store_true", help="关闭 trace 落盘")
    p.add_argument("--quiet", "-q", action="store_true", help="不打印工具调用过程")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _fix_console_encoding()
    _load_dotenv()
    args = parse_args(argv)
    printer = make_printer(args.quiet)

    try:
        loop = build_loop(args, printer)
    except Exception as exc:
        print(f"初始化失败：{exc}", file=sys.stderr)
        return 2

    session_id = args.session

    #单次模式
    if args.once:
        result = loop.run(session_id, args.once)
        print(f"\n助手 > {result.answer}")
        print(
            f"[session={result.session_id} 轮次={result.turns} "
            f"工具={result.tool_calls} 用时={result.elapsed_ms / 1000:.1f}s "
            f"stop={result.stop_reason}]"
        )
        return 0 if result.stop_reason == "final" else 1

    #交互模式
    print("mini-agent 就绪。输入 /help 看命令，/exit 退出。")
    print(f"当前会话：{session_id}    模型：{loop.llm.model}")
    print(f"可用工具：{', '.join(loop.registry.names())}")

    while True:
        try:
            line = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            rest = rest.strip()

            if cmd in ("/exit", "/quit", "/q"):
                break
            if cmd == "/help":
                print(
                    "  /session <id>   切换会话（等价于换一个窗口）\n"
                    "  /sessions       列出所有已落盘的会话\n"
                    "  /history        查看当前会话的消息条数\n"
                    "  /reset          清空当前会话（消息 + 待办）\n"
                    "  /exit           退出"
                )
                continue
            if cmd == "/session" and rest:
                session_id = rest
                print(f"已切换到会话：{session_id}")
                continue
            if cmd == "/sessions":
                ids = loop.store.list_sessions()
                print("已落盘的会话：" + (", ".join(ids) if ids else "（无）"))
                continue
            if cmd == "/history":
                s = loop.store.load_or_create(session_id)
                print(f"{session_id}: {len(s.messages)} 条消息，摘要 {len(s.summary)} 字符")
                continue
            if cmd == "/reset":
                loop.store.delete(session_id)
                print(f"已清空会话：{session_id}")
                continue

            print("未知命令，输入 /help 查看")
            continue

        result = loop.run(session_id, line)
        print(f"\n助手 > {result.answer}")
        print(
            f"[{result.turns} 轮 / {result.tool_calls} 次工具 / "
            f"{result.elapsed_ms / 1000:.1f}s / {result.stop_reason}]"
        )

    print("再见。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
