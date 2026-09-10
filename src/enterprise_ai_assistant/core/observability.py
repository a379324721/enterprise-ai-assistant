"""模型用量采集与 HTTP 指标中间件。"""

import time
from typing import Any
from uuid import UUID

import structlog
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from starlette.types import ASGIApp

from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.core.metrics import (
    HTTP_DURATION,
    HTTP_REQUESTS,
    LLM_CALLS,
    LLM_COST,
    LLM_DURATION,
    LLM_TOKENS,
)

logger = structlog.get_logger()


def _usage_from(response: LLMResult) -> tuple[int, int]:
    """兼容两种用量来源：消息上的 usage_metadata 与 provider 的 llm_output。"""
    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if usage:
                return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    raw = (response.llm_output or {}).get("token_usage") or {}
    if isinstance(raw, dict):
        return int(raw.get("prompt_tokens", 0)), int(raw.get("completion_tokens", 0))
    return 0, 0


class LLMUsageTracker(AsyncCallbackHandler):
    """按 agent 维度累计单次请求的模型用量。

    实例是请求级的：图执行时通过 config.callbacks 注入，结束后由调用方读取
    `total_tokens` 写入会话预算。
    """

    raise_error = False

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._runs: dict[UUID, tuple[str, float]] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.calls = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized, messages, kwargs
        agent = str((metadata or {}).get("agent", "unknown"))
        self._runs[run_id] = (agent, time.perf_counter())

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        del kwargs
        agent, started = self._runs.pop(run_id, ("unknown", time.perf_counter()))
        input_tokens, output_tokens = _usage_from(response)
        cost = (
            input_tokens * self._settings.llm_input_cost_per_1k_usd
            + output_tokens * self._settings.llm_output_cost_per_1k_usd
        ) / 1000

        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost

        LLM_CALLS.labels(agent=agent, outcome="success").inc()
        LLM_DURATION.labels(agent=agent).observe(time.perf_counter() - started)
        LLM_TOKENS.labels(agent=agent, kind="input").inc(input_tokens)
        LLM_TOKENS.labels(agent=agent, kind="output").inc(output_tokens)
        LLM_COST.labels(model=self._settings.openai_model).inc(cost)

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        del error, kwargs
        agent, started = self._runs.pop(run_id, ("unknown", time.perf_counter()))
        LLM_CALLS.labels(agent=agent, outcome="error").inc()
        LLM_DURATION.labels(agent=agent).observe(time.perf_counter() - started)

    def log_summary(self, **context: Any) -> None:
        logger.info(
            "llm_usage",
            llm_calls=self.calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            cost_usd=round(self.cost_usd, 6),
            **context,
        )


class MetricsMiddleware:
    """记录请求量与延迟；路由模板作为标签，避免会话 ID 造成标签基数爆炸。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        await _instrumented(self.app, scope, receive, send)


async def _instrumented(app: ASGIApp, scope: Any, receive: Any, send: Any) -> None:
    started = time.perf_counter()
    status_code = 500

    async def send_wrapper(message: Any) -> None:
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = int(message["status"])
        await send(message)

    try:
        await app(scope, receive, send_wrapper)
    finally:
        route = scope.get("route")
        endpoint = getattr(route, "path", None) or scope.get("path", "unknown")
        method = scope.get("method", "GET")
        HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        HTTP_DURATION.labels(method=method, endpoint=endpoint).observe(
            time.perf_counter() - started
        )
