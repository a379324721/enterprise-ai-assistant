"""用量采集、成本护栏与指标端点的测试。"""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from prometheus_client import generate_latest

from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.core.metrics import REGISTRY
from enterprise_ai_assistant.core.observability import LLMUsageTracker
from enterprise_ai_assistant.core.runs import MemoryStreamBridge, RunManager
from enterprise_ai_assistant.core.security import create_access_token
from enterprise_ai_assistant.main import create_app

CONVERSATION_ID = uuid4()


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "openai_api_key": "test-key",
        "openai_model": "test-model",
        "openai_embedding_model": "test-embedding",
        "langsmith_tracing": False,
        "jwt_secret": "observability-secret-" + "z" * 32,
        "app_env": "development",
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


def _result(input_tokens: int, output_tokens: int) -> LLMResult:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


# -- 用量采集 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_tracker_accumulates_tokens_and_cost() -> None:
    settings = _settings(
        llm_input_cost_per_1k_usd=2.0, llm_output_cost_per_1k_usd=10.0
    )
    tracker = LLMUsageTracker(settings)
    run_id = uuid4()

    await tracker.on_chat_model_start({}, [], run_id=run_id, metadata={"agent": "hr"})
    await tracker.on_llm_end(_result(1000, 500), run_id=run_id)

    assert tracker.calls == 1
    assert tracker.input_tokens == 1000
    assert tracker.output_tokens == 500
    assert tracker.total_tokens == 1500
    assert tracker.cost_usd == pytest.approx(2.0 + 5.0)


@pytest.mark.asyncio
async def test_tracker_falls_back_to_provider_token_usage() -> None:
    tracker = LLMUsageTracker(_settings())
    run_id = uuid4()
    response = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="ok"))]],
        llm_output={"token_usage": {"prompt_tokens": 30, "completion_tokens": 12}},
    )

    await tracker.on_chat_model_start({}, [], run_id=run_id, metadata=None)
    await tracker.on_llm_end(response, run_id=run_id)

    assert tracker.total_tokens == 42


@pytest.mark.asyncio
async def test_tracker_survives_missing_usage_metadata() -> None:
    """用量缺失不应让业务请求失败，只是统计为 0。"""
    tracker = LLMUsageTracker(_settings())
    run_id = uuid4()

    await tracker.on_chat_model_start({}, [], run_id=run_id, metadata={"agent": "travel"})
    await tracker.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]),
        run_id=run_id,
    )

    assert tracker.calls == 1
    assert tracker.total_tokens == 0


# -- 成本护栏 -------------------------------------------------------------


class FakeRedis:
    def __init__(self, spent: int | None = None, broken: bool = False) -> None:
        self.spent = spent
        self.broken = broken
        self.values: dict[str, int] = {}
        self.increments: list[int] = []
        self.expired: list[str] = []

    async def get(self, key: str) -> str | None:
        if self.broken:
            raise ConnectionError("redis down")
        value = self.values.get(key, self.spent)
        return None if value is None else str(value)

    async def incrby(self, key: str, amount: int) -> int:
        self.increments.append(amount)
        self.values[key] = self.values.get(key, 0) + amount
        return self.values[key]

    async def expire(self, key: str, seconds: int) -> bool:
        del seconds
        self.expired.append(key)
        return True


def _app(redis: FakeRedis) -> Any:
    logger = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
    return SimpleNamespace(state=SimpleNamespace(redis=redis, logger=logger))


@pytest.mark.asyncio
async def test_budget_guard_rejects_exhausted_conversation() -> None:
    settings = _settings(conversation_token_budget=1000)

    with pytest.raises(HTTPException) as exc:
        await routes._enforce_token_budget(
            _app(FakeRedis(spent=1200)), CONVERSATION_ID, settings
        )

    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_budget_guard_allows_conversation_under_limit() -> None:
    settings = _settings(conversation_token_budget=1000)

    await routes._enforce_token_budget(
        _app(FakeRedis(spent=200)), CONVERSATION_ID, settings
    )


@pytest.mark.asyncio
async def test_budget_guard_is_disabled_by_default() -> None:
    redis = FakeRedis(spent=10**9)

    await routes._enforce_token_budget(_app(redis), CONVERSATION_ID, _settings())


@pytest.mark.asyncio
async def test_budget_guard_fails_open_when_redis_is_down() -> None:
    """指标存储故障不应阻断业务请求。"""
    settings = _settings(conversation_token_budget=1000)

    await routes._enforce_token_budget(
        _app(FakeRedis(broken=True)), CONVERSATION_ID, settings
    )


@pytest.mark.asyncio
async def test_usage_is_written_back_to_the_budget_counter() -> None:
    settings = _settings(conversation_token_budget=1000)
    redis = FakeRedis()
    tracker = LLMUsageTracker(settings)
    run_id = uuid4()
    await tracker.on_chat_model_start({}, [], run_id=run_id, metadata={"agent": "hr"})
    await tracker.on_llm_end(_result(100, 50), run_id=run_id)

    await routes._record_usage(
        _app(redis), CONVERSATION_ID, "u-1", tracker, settings
    )

    assert redis.increments == [150]


@pytest.mark.asyncio
async def test_ip_budget_rejects_an_exhausted_ip_even_in_a_new_conversation() -> None:
    """开新会话能绕过会话预算，按 IP 的额度绕不过。"""
    settings = _settings(ip_token_budget=1000)
    redis = FakeRedis()
    redis.values[routes._ip_budget_key("10.0.0.1")] = 1200

    with pytest.raises(HTTPException) as exc:
        await routes._enforce_token_budget(_app(redis), uuid4(), settings, "10.0.0.1")

    assert exc.value.status_code == 429
    assert exc.value.detail == routes.QUOTA_EXHAUSTED_MESSAGE
    await routes._enforce_token_budget(_app(redis), uuid4(), settings, "10.0.0.2")


@pytest.mark.asyncio
async def test_ip_budget_window_is_not_extended_by_later_usage() -> None:
    """每次用量都续期的话，持续在用的 IP 计数永远不清零。"""
    settings = _settings(ip_token_budget=1000)
    redis = FakeRedis()
    key = routes._ip_budget_key("10.0.0.1")

    for _ in range(2):
        tracker = LLMUsageTracker(settings)
        run_id = uuid4()
        await tracker.on_chat_model_start({}, [], run_id=run_id, metadata={"agent": "hr"})
        await tracker.on_llm_end(_result(100, 50), run_id=run_id)
        await routes._record_usage(
            _app(redis), CONVERSATION_ID, "u-1", tracker, settings, "10.0.0.1"
        )

    assert redis.values[key] == 300
    assert redis.expired == [key]


# -- 指标端点 -------------------------------------------------------------


class EmptyGraph:
    async def aget_state(self, config: dict[str, Any]) -> Any:
        del config
        return SimpleNamespace(values={}, next=(), interrupts=())


@asynccontextmanager
async def _fake_lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.graph = EmptyGraph()
    app.state.logger = SimpleNamespace(info=lambda *a, **k: None)
    app.state.runs = RunManager(MemoryStreamBridge(), app.state.logger)
    yield
    await app.state.runs.aclose()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    settings = _settings()
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: settings)
    with TestClient(create_app(_fake_lifespan)) as test_client:
        yield test_client


def test_metrics_endpoint_is_scrapable_without_a_token(client: TestClient) -> None:
    response = client.get("/api/v1/metrics")

    assert response.status_code == 200
    assert "assistant_http_requests_total" in response.text


def test_http_metrics_use_the_route_template_as_label(client: TestClient) -> None:
    """路径参数必须归一化，否则会话 ID 会让标签基数无限增长。"""
    token, _ = create_access_token("someone", _settings())
    client.get(
        f"/api/v1/conversations/{uuid4()}",
        headers={"Authorization": f"Bearer {token}"},
    )

    exported = generate_latest(REGISTRY).decode()

    assert 'endpoint="/api/v1/conversations/{conversation_id}"' in exported
    assert str(CONVERSATION_ID) not in exported
