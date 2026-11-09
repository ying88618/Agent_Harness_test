"""分层异常。

为什么要单独抽一层：Agent 出问题时，"是网络挂了 / 是模型输出烂 / 是工具炸了"
必须能区分开。上层只捕获 AgentError 就能兜住所有已知失败，
而真正需要暴露给开发者的 bug（比如 TypeError）继续往上抛，不静默吞掉。
"""

from __future__ import annotations


class AgentError(Exception):
    """所有可预期错误的基类。"""


class LLMError(AgentError):
    """调用大模型失败（鉴权、超时、重试耗尽、被拒绝）。"""


class MaxTurnsExceeded(AgentError):
    """超过单轮对话允许的最大循环次数。"""

    def __init__(self, max_turns: int, last_observation: str = "") -> None:
        super().__init__(f"超过最大轮次 {max_turns}，已强制中止")
        self.max_turns = max_turns
        self.last_observation = last_observation


class ToolError(AgentError):
    """工具执行层面的错误基类。"""


class ToolNotFound(ToolError):
    """模型调用了一个不存在的工具（通常是幻觉）。"""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(f"工具 {name!r} 不存在")
        self.name = name
        self.available = available


class ToolArgError(ToolError):
    """工具参数不满足 schema。"""


class ParseError(AgentError):
    """模型输出无法被解析成任何已知格式。"""
