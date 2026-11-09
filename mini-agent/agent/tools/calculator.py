"""计算器工具：用 AST 白名单求值，不用 eval。

为什么不能用 eval：模型完全可能输出 `__import__("os").system("rm -rf /")`，
或者被用户的 prompt 注入诱导。eval 一旦执行就晚了。
AST 白名单的做法是——只认数字、四则运算、括号和少量数学函数，
其余节点（属性访问 a.b、下标 a[0]、lambda、推导式、调用非白名单函数）一律拒绝。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from ..errors import ToolError
from ..registry import tool
from ..types import ToolContext

_MAX_EXPR_CHARS = 300
_MAX_NODES = 150
_MAX_EXPONENT = 1000
_MAX_RESULT = 1e308

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


@tool(
    name="calculator",
    description="精确计算数学表达式。支持 + - * / // % **、括号和 sqrt/log/sin 等数学函数。"
    "涉及数字运算时必须用它，不要自己心算。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": '要计算的表达式，只写数学本身，例如 "(1+2)*3" 或 "sqrt(16)+2"',
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
)
def calculate(args: dict[str, Any], ctx: ToolContext) -> str:
    expression = str(args.get("expression", "")).strip()
    if not expression:
        raise ToolError("expression 不能为空")
    if len(expression) > _MAX_EXPR_CHARS:
        raise ToolError(f"表达式过长（上限 {_MAX_EXPR_CHARS} 字符）")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"表达式语法错误：{exc.msg}") from exc

    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ToolError("表达式过于复杂")

    value = _eval(tree.body)
    return f"{expression} = {_render(value)}"


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"不支持的字面量：{node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"不支持的运算符：{type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ToolError(f"指数过大（上限 {_MAX_EXPONENT}）")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolError("除数为 0")
        return _guard(op(left, right))

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op(_eval(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ToolError(f"未知标识符 {node.id!r}，只支持常量 {list(_CONSTS)}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ToolError(f"不允许调用 {ast.unparse(node.func)!r}")
        if node.keywords:
            raise ToolError("函数不支持关键字参数")
        try:
            return _guard(_FUNCS[node.func.id](*[_eval(a) for a in node.args]))
        except (ValueError, OverflowError) as exc:
            raise ToolError(f"函数 {node.func.id} 计算出错：{exc}") from exc

    # 兜底：属性访问、下标、lambda、推导式等全部拒绝
    raise ToolError(f"不支持的语法：{type(node).__name__}")


def _guard(value: Any) -> Any:
    if isinstance(value, complex):
        raise ToolError("结果包含虚数，本工具只支持实数")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("计算结果不是有限数（溢出或未定义）")
    if isinstance(value, (int, float)) and abs(value) > _MAX_RESULT:
        raise ToolError("计算结果过大")
    return value


def _render(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)
