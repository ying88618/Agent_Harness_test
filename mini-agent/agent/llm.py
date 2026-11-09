"""大模型客户端：整个项目唯一直接访问网络的地方。

设计要点
--------
1. 只暴露一个方法 `chat()`，返回归一化的 `LLMResponse`。上层（loop/parser）
   完全不知道底层是 openai SDK、DeepSeek 还是通义，换模型只改环境变量。
2. 重试策略自己写，关掉 SDK 自带的 `max_retries`，这样重试行为是可预测、可测的。
3. 所有网络/协议异常收敛成 `LLMError`，loop 层只需要 except 这一种。
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

from .errors import LLMError
from .types import LLMResponse, ToolCall, Usage

try:  # 允许在没装 openai 的环境下 import 本模块去做纯逻辑单测
    from openai import OpenAI
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
    )
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    APIConnectionError = APIStatusError = APITimeoutError = AuthenticationError = (
        BadRequestError
    ) = RateLimitError = ()  # type: ignore[assignment]

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

# 只有这些异常值得重试：限流、超时、连接抖动
_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        if OpenAI is None:  # pragma: no cover
            raise LLMError("未安装 openai SDK，请先 `pip install -r requirements.txt`")

        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL
        self.temperature = temperature
        self.max_retries = max_retries

        key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError(
                "缺少 API Key。请设置 LLM_API_KEY 环境变量，或用 --api-key 传入。"
            )

        # max_retries=0：重试完全由 _request() 控制
        self._client = OpenAI(
            api_key=key, base_url=self.base_url, timeout=timeout, max_retries=0
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        raw = self._request(payload)
        return self._normalize(raw)

    # 重试
    def _request(self, payload: dict[str, Any]) -> Any:
        last_err: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._client.chat.completions.create(**payload)
            except AuthenticationError as exc:
                # 鉴权错重试没有意义
                raise LLMError(f"鉴权失败，请检查 LLM_API_KEY / LLM_BASE_URL：{exc}") from exc
            except BadRequestError as exc:
                # 参数错（比如 tool_calls 与 tool 消息不配对）重试同样没意义
                raise LLMError(f"请求被服务端拒绝（4xx），通常是消息结构不合法：{exc}") from exc
            except _RETRYABLE as exc:
                last_err = exc
                if attempt >= self.max_retries:
                    break
                delay = min(0.5 * 2**attempt, 8.0)
                log.warning(
                    "LLM 调用失败（%s），%.1fs 后第 %d 次重试", type(exc).__name__, delay, attempt + 1
                )
                time.sleep(delay)
            except APIStatusError as exc:  # 5xx 才值得重试
                last_err = exc
                if exc.status_code < 500 or attempt >= self.max_retries:
                    raise LLMError(f"LLM 服务端错误 {exc.status_code}：{exc}") from exc
                delay = min(0.5 * 2**attempt, 8.0)
                time.sleep(delay)
            except Exception as exc:  # 兜底，避免裸异常穿透到 loop
                raise LLMError(f"LLM 调用出现未知错误：{type(exc).__name__}: {exc}") from exc

        raise LLMError(f"LLM 调用重试 {self.max_retries} 次后仍失败：{last_err}")

    # 归一化：把各家 SDK 的字段差异吃掉
    def _normalize(self, raw: Any) -> LLMResponse:
        try:
            choice = raw.choices[0]
            msg = choice.message
        except (AttributeError, IndexError) as exc:
            raise LLMError(f"LLM 返回结构异常：{raw!r}") from exc

        content = getattr(msg, "content", None) or ""
        reasoning = self._extract_reasoning(msg)

        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            calls.append(
                ToolCall(
                    name=getattr(fn, "name", "") or "",
                    args={},  # 故意留空：参数解析是 parser 的职责
                    raw_args=getattr(fn, "arguments", "") or "",
                    id=getattr(tc, "id", "") or f"call_{uuid.uuid4().hex[:8]}",
                )
            )

        usage = Usage()
        u = getattr(raw, "usage", None)
        if u is not None:
            usage = Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
            usage=usage,
            model=getattr(raw, "model", self.model) or self.model,
            finish_reason=getattr(choice, "finish_reason", "") or "",
            raw=raw,
        )

    @staticmethod
    def _extract_reasoning(msg: Any) -> str:
        """DeepSeek 用 reasoning_content，OpenAI 用 reasoning，第三方还有放 model_extra 的。"""
        for attr in ("reasoning_content", "reasoning"):
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val.strip():
                return val
        extra = getattr(msg, "model_extra", None)
        if isinstance(extra, dict):
            for key in ("reasoning_content", "reasoning"):
                val = extra.get(key)
                if isinstance(val, str) and val.strip():
                    return val
        return ""
