"""LLM 输出解析：把模型吐出来的东西变成 (思考, 工具调用, 最终答案)。

这是 Agent 里最容易出 bug、也最容易被写成 `if "Action:" in text` 的地方。
我采用"双通道"：

  通道 A（主）: OpenAI 原生 tool_calls —— 稳定，字段是结构化的
  通道 B（兜底）: 文本协议 —— 模型把 JSON / ReAct 格式写在 content 里时也能救回来

两条通道最终都收敛成同一个 `ParsedOutput`，loop 层只认这一种结构。

另外一大块工作量在 `parse_tool_args()`：模型生成的 JSON 参数经常不合法
（尾逗号、单引号、被 markdown 包裹、套了两层字符串），必须逐级降级修复，
而不是直接抛异常——只要能救回参数，就比让整轮对话失败强。
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from .types import LLMResponse, ToolCall

# 超过这个长度的参数文本不再尝试修复，直接报错（防止正则回溯/内存问题）
_MAX_ARG_CHARS = 20_000

_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.S)
# 工具名必须是合法标识符，避免把正文里的人名/变量名误判成工具
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")

_ACTION_RE = re.compile(
    r"^[ \t]*(?:Action|动作|工具名?|tool)[ \t]*[:：][ \t]*([A-Za-z_][A-Za-z0-9_.\-]*)",
    re.I | re.M,
)
_INPUT_RE = re.compile(
    r"^[ \t]*(?:Action[ \t]*Input|Input|参数|入参|Arguments?)[ \t]*[:：][ \t]*", re.I | re.M
)
_FA_RE = re.compile(r"(?:Final[ \t]*Answer|最终答案|最终回复|答案)[ \t]*[:：][ \t]*", re.I)
_STOP_RE = re.compile(r"\n[ \t]*(?:Observation|观察|工具结果|Thought|思考)[ \t]*[:：]", re.I)
_THOUGHT_PREFIX_RE = re.compile(r"^[ \t]*(?:Thought|思考|推理)[ \t]*[:：][ \t]*", re.I)

_TOOL_NAME_KEYS = ("tool", "name", "tool_name", "function", "action")
_TOOL_ARG_KEYS = ("arguments", "args", "parameters", "params", "input", "action_input")


@dataclass
class ParsedOutput:
    """解析结果。loop 只依赖这个对象，不关心模型用了哪种格式。"""

    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str | None = None
    raw: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def has_tool_call(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final(self) -> bool:
        return not self.tool_calls and self.final_answer is not None


# 对外入口
def parse_response(resp: LLMResponse) -> ParsedOutput:
    """解析一次模型返回。优先级：原生 tool_calls > 文本协议 > 普通回答。"""
    thought = (resp.reasoning or "").strip()
    content = (resp.content or "").strip()
    warnings: list[str] = []

    #通道 A：原生 tool_calls
    if resp.tool_calls:
        calls: list[ToolCall] = []
        for tc in resp.tool_calls:
            if tc.raw_args.strip():
                args, err = parse_tool_args(tc.raw_args)
            elif tc.args:
                # 上游已经解析好参数（自定义 backend / 假模型）时直接用，不要当成空参数
                args, err = dict(tc.args), None
            else:
                args, err = {}, None
            if err:
                warnings.append(f"工具 {tc.name} 的参数解析失败：{err}")
            calls.append(
                ToolCall(
                    name=tc.name,
                    args=args,
                    raw_args=tc.raw_args,
                    id=tc.id or _new_call_id(),
                )
            )
        #思考过程 + 伴随正文合并进 thought，不丢信息
        merged = "\n\n".join(x for x in (thought, content) if x)
        return ParsedOutput(
            thought=merged,
            tool_calls=calls,
            final_answer=None,
            raw=_snip(content),
            warnings=warnings,
        )

    #通道 B：content 里的文本协议
    text_calls, leftover = _parse_text_protocol(content)
    if text_calls:
        merged = "\n\n".join(x for x in (thought, leftover) if x)
        return ParsedOutput(
            thought=merged,
            tool_calls=text_calls,
            final_answer=None,
            raw=_snip(content),
            warnings=warnings,
        )

    #普通回答
    if not content:
        warnings.append("模型既没有调用工具，也没有输出任何内容")

    answer = content
    m = _FA_RE.search(content)
    if m:
        prefix = _clean_thought(content[: m.start()])
        if prefix and not thought:
            thought = prefix
        answer = content[m.end() :].strip()

    return ParsedOutput(
        thought=thought,
        tool_calls=[],
        final_answer=answer,
        raw=_snip(content),
        warnings=warnings,
    )


def parse_tool_args(raw: Any, _depth: int = 0) -> tuple[dict[str, Any], str | None]:
    """把工具参数解析成 dict。

    返回 `(args, error)`，error 非空表示走了兜底，args 里是 `{"__raw__": 原文}`。
    永不抛异常——参数问题交回给模型自我修正，比整轮崩掉划算。
    """
    if isinstance(raw, dict):
        return raw, None
    if raw is None:
        return {}, None

    text = str(raw).strip()
    if not text:
        return {}, None
    if len(text) > _MAX_ARG_CHARS:
        return {"__raw__": text[:2000]}, f"参数过长（{len(text)} 字符）"

    text = _strip_fence(text)

    # 第 1 级：直接 json.loads
    # 第 2 级：正文里抠出平衡的 {...} 再试（模型常加"好的，我来调用"前缀）
    for cand in [text, *_extract_json_objects(text)]:
        val = _loads_lenient(cand)
        if isinstance(val, dict):
            # 参数里再套一层 JSON 字符串（{"arguments": "{...}"}）时拆开
            return _unwrap(val), None
        if isinstance(val, str) and _depth < 2 and val.strip() and val.strip() != text:
            # 整体被二次编码成字符串：'"{\\"a\\": 1}"'
            nested, nested_err = parse_tool_args(val, _depth + 1)
            if nested_err is None:
                return nested, None

    # 第 3 级：修复常见瑕疵后再试
    for cand in _repair_candidates(text):
        val = _loads_lenient(cand)
        if isinstance(val, dict):
            return _unwrap(val), None

    return {"__raw__": text[:2000]}, "参数不是合法的 JSON Object"

# 通道 B 的实现
def _parse_text_protocol(content: str) -> tuple[list[ToolCall], str]:
    """从纯文本里找工具调用。返回 (calls, 剩余的思考文本)。"""
    if not content.strip():
        return [], ""

    # B1: ```json ... ``` 代码块
    for block in _FENCE_RE.findall(content):
        calls = _calls_from_json(block)
        if calls:
            return calls, _clean_thought(content.replace(block, "").replace("```", ""))

    # B2: 裸的 {...} 对象
    for obj in _extract_json_objects(content):
        calls = _calls_from_json(obj)
        if calls:
            return calls, _clean_thought(content.replace(obj, ""))

    # B3: ReAct 文本格式
    m = _ACTION_RE.search(content)
    if m:
        name = m.group(1)
        rest = content[m.end() :]
        raw_args = ""
        im = _INPUT_RE.search(rest)
        if im:
            raw_args = rest[im.end() :]
            stop = _STOP_RE.search(raw_args)
            if stop:
                raw_args = raw_args[: stop.start()]
        args, err = parse_tool_args(raw_args)
        call = ToolCall(name=name, args=args, raw_args=raw_args, id=_new_call_id())
        return [call], _clean_thought(content[: m.start()])

    return [], ""


def _calls_from_json(text: str) -> list[ToolCall]:
    """尝试把一段 JSON 文本解释成工具调用列表。不是工具调用就返回 []。"""
    val = _loads_lenient(_strip_fence(text))
    return _calls_from_obj(val)


def _calls_from_obj(obj: Any) -> list[ToolCall]:
    if isinstance(obj, list):
        out: list[ToolCall] = []
        for item in obj:
            out.extend(_calls_from_obj(item))
        return out
    if not isinstance(obj, dict):
        return []

    # {"tool_calls": [...]} 包一层的形态
    for key in ("tool_calls", "calls", "tool_uses"):
        if isinstance(obj.get(key), list):
            return _calls_from_obj(obj[key])

    name = _pick_name(obj)
    if not name:
        return []

    args_val = _pick_args(obj)
    if isinstance(args_val, str):
        args, _ = parse_tool_args(args_val)
        raw_args = args_val
    elif isinstance(args_val, dict):
        args, raw_args = args_val, json.dumps(args_val, ensure_ascii=False)
    else:
        args, raw_args = {}, ""

    return [ToolCall(name=name, args=args, raw_args=raw_args, id=_new_call_id())]


def _pick_name(obj: dict[str, Any]) -> str:
    for key in _TOOL_NAME_KEYS:
        val = obj.get(key)
        if isinstance(val, dict):  # {"function": {"name": ...}}
            val = val.get("name")
        if isinstance(val, str) and _NAME_RE.match(val.strip()):
            return val.strip()
    return ""


def _pick_args(obj: dict[str, Any]) -> Any:
    for key in _TOOL_ARG_KEYS:
        if key in obj:
            return obj[key]
    return {}


# JSON 容错工具
def _loads_lenient(text: str) -> Any:
    """json.loads 不行就退到 ast.literal_eval（能吃单引号 / True / None）。"""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def _repair_candidates(text: str) -> list[str]:
    fixed = text
    # 中文引号
    fixed = fixed.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # 尾逗号：{"a": 1,}
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    # Python 字面量：True/False/None -> true/false/null
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    # JSON 不允许注释
    fixed = re.sub(r"(?m)^\s*//.*$", "", fixed)
    return [fixed] if fixed != text else []


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        m = _FENCE_RE.search(text)
        if m:
            return m.group(1).strip()
        # 没闭合的围栏（模型被截断）
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text).strip()
    return text


def _extract_json_objects(text: str) -> list[str]:
    """扫描出文本里所有括号平衡的 {...}（能正确跳过字符串内的括号）。"""
    out: list[str] = []
    depth = 0
    start: int | None = None
    in_str = False
    escaped = False

    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start : i + 1])
                start = None
    return out


def _unwrap(val: dict[str, Any]) -> dict[str, Any]:
    """{"arguments": "{...}"} 这种套娃，参数在调用前要摊平。"""
    if len(val) == 1:
        only_val = next(iter(val.values()))
        if isinstance(only_val, str) and only_val.strip().startswith("{"):
            inner = _loads_lenient(only_val)
            if isinstance(inner, dict):
                return inner
    return val


def _clean_thought(text: str) -> str:
    text = _THOUGHT_PREFIX_RE.sub("", text.strip())
    return text.strip()


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _snip(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"
