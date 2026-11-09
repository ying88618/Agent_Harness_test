"""待办工具（必须有状态，且必须按 session 隔离）。

如果把待办存成模块级全局变量。
那样"窗口 1 让 Agent 记待办"和"窗口 2 让 Agent 记周报待办"就会串在一起。

这里所有数据都写在 `ctx.store` 里，而 `ctx.store` 是 session 级别的命名空间
（由 session.py 提供，json 落盘）。所以：
    session A 的 store["todos"] 和 session B 的 store["todos"] 是两个列表。
工具代码完全不需要知道 session 的存在，隔离是天然成立的。
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolError
from ..registry import tool
from ..types import ToolContext

_STORE_KEY = "todos"
_SEQ_KEY = "todo_seq"
_MAX_RENDER = 30  # 单次回灌给模型的待办条数上限，防止 context 随清单长度平方增长


@tool(
    name="todo",
    description="管理当前会话的待办清单。action=add 新增、list 列出、done 完成、"
    "update 修改内容、remove 删除、clear 清空。待办只属于当前会话，不会串到别的窗口。",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "done", "update", "remove", "clear"],
                "description": "要执行的操作",
            },
            "text": {"type": "string", "description": "待办内容，action=add / update 时填写"},
            "id": {"type": "integer", "description": "待办编号，done / update / remove 时填写"},
            "due": {"type": "string", "description": "截止时间，自由文本，如「明天下午三点」"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
)
def todo(args: dict[str, Any], ctx: ToolContext) -> str:
    action = str(args.get("action", "")).strip().lower()
    todos: list[dict[str, Any]] = ctx.store.setdefault(_STORE_KEY, [])

    if action == "add":
        text = str(args.get("text", "")).strip()
        if not text:
            raise ToolError("action=add 时必须提供 text")
        seq = int(ctx.store.get(_SEQ_KEY, 0)) + 1
        ctx.store[_SEQ_KEY] = seq
        item = {
            "id": seq,
            "text": text,
            "done": False,
            "due": str(args.get("due", "") or ""),
            "created_at": ctx.timestamp(),
        }
        todos.append(item)
        return f"已添加待办 #{seq}：{text}{_due_suffix(item['due'])}\n\n" + _render_capped(todos)

    if action == "list":
        return _render_capped(todos)

    if action in ("done", "update", "remove"):
        item = _find(todos, args.get("id"))

        if action == "done":
            item["done"] = True
            item["done_at"] = ctx.timestamp()
            msg = f"已完成待办 #{item['id']}：{item['text']}"
        elif action == "update":
            text = str(args.get("text", "")).strip()
            due = str(args.get("due", "") or "")
            if not text and not due:
                raise ToolError("action=update 时 text 和 due 至少要给一个")
            if text:
                item["text"] = text
            if due:
                item["due"] = due
            msg = f"已更新待办 #{item['id']}：{item['text']}{_due_suffix(item['due'])}"
        else:  # remove
            todos.remove(item)
            msg = f"已删除待办 #{item['id']}：{item['text']}"

        return msg + "\n\n" + _render_capped(todos)

    if action == "clear":
        removed = len(todos)
        todos.clear()
        ctx.store[_SEQ_KEY] = 0
        return f"已清空 {removed} 条待办。"

    raise ToolError(f"不支持的 action: {action}")


def _find(todos: list[dict[str, Any]], raw_id: Any) -> dict[str, Any]:
    if raw_id is None or raw_id == "":
        raise ToolError("该操作需要提供 id，可以先 list 查看编号")
    try:
        target = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"id 必须是整数，收到 {raw_id!r}") from exc
    for item in todos:
        if item["id"] == target:
            return item
    existing = [i["id"] for i in todos] or "（空）"
    raise ToolError(f"待办 #{target} 不存在，当前编号：{existing}")


def _render(todos: list[dict[str, Any]]) -> str:
    if not todos:
        return "当前会话的待办为空。"
    pending = sum(1 for t in todos if not t["done"])
    lines = [f"待办清单（共 {len(todos)} 条，未完成 {pending} 条）："]
    for item in todos:
        mark = "x" if item["done"] else " "
        lines.append(f"- [{mark}] #{item['id']} {item['text']}{_due_suffix(item.get('due', ''))}")
    return "\n".join(lines)


def _render_capped(todos: list[dict[str, Any]], limit: int = _MAX_RENDER) -> str:
    """超过上限时只渲染一部分。

    每次 add/done 都会把整份清单回灌进 context，清单越长、每轮的 context 增长越快
    （近似 O(N²)）。50 条待办就足以把上下文顶爆，触发不必要的压缩。
    """
    if len(todos) <= limit:
        return _render(todos)
    head = _render(todos[:limit])
    return f"{head}\n…其余 {len(todos) - limit} 条从略（用 action=list 且 id 定位；共 {len(todos)} 条）"


def _due_suffix(due: str) -> str:
    return f"（截止：{due}）" if due else ""
