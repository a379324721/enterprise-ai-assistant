"""单据清单接口。

右栏要让用户看见自己办过哪些事，数据只能从 workflow_actions 派生——单号的真相
只有那一处。这里盯住三件事：按人隔离、不受长期记忆开关影响、露出的是对外短单号
而不是内部幂等键。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.core.models import RecentAction
from enterprise_ai_assistant.core.security import create_access_token
from enterprise_ai_assistant.main import create_app
from enterprise_ai_assistant.repositories.memories import (
    InMemoryMemoryRepository,
    action_fields,
)

SETTINGS = Settings(
    openai_api_key="test-key",  # type: ignore[arg-type]
    openai_model="test-model",
    openai_embedding_model="test-embedding",
    langsmith_tracing=False,
    jwt_secret="unit-test-secret-" + "x" * 32,  # type: ignore[arg-type]
    app_env="development",
    memory_enabled=False,
)


@asynccontextmanager
async def _noop_lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    yield


def _action(reference_id: str, action_type: str, **fields: str) -> RecentAction:
    return RecentAction(
        reference_id=reference_id,
        action_type=action_type,
        summary=" ".join(f"{name}={value}" for name, value in fields.items()),
        created_at=datetime(2026, 9, 13, tzinfo=UTC),
        fields=fields,
    )


def _client(
    monkeypatch: pytest.MonkeyPatch, repository: Any | None
) -> AsyncClient:
    monkeypatch.setattr(routes, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: SETTINGS)
    app = create_app(_noop_lifespan)
    app.state.memories = repository
    app.state.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _auth(user_id: str) -> dict[str, str]:
    token, _ = create_access_token(user_id, SETTINGS)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_listing_is_available_even_with_memory_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单据是业务事实，不是画像；关掉长期记忆不该让它从界面上消失。"""
    repository = InMemoryMemoryRepository()
    repository.actions["u-1"] = [
        _action("TRV-20260913-EB9750", "travel_application", destination="上海")
    ]

    async with _client(monkeypatch, repository) as client:
        response = await client.get("/api/v1/actions", headers=_auth("u-1"))

    assert SETTINGS.memory_enabled is False
    assert response.status_code == 200
    assert [item["reference_id"] for item in response.json()["actions"]] == [
        "TRV-20260913-EB9750"
    ]


@pytest.mark.asyncio
async def test_listing_only_returns_the_callers_own_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """user_id 取自令牌，不接受调用方指定，否则就是跨用户读取。"""
    repository = InMemoryMemoryRepository()
    repository.actions["u-1"] = [_action("TRV-1", "travel_application", destination="上海")]
    repository.actions["u-2"] = [_action("TRV-2", "travel_application", destination="北京")]

    async with _client(monkeypatch, repository) as client:
        response = await client.get("/api/v1/actions?limit=10", headers=_auth("u-2"))

    assert [item["reference_id"] for item in response.json()["actions"]] == ["TRV-2"]


@pytest.mark.asyncio
async def test_listing_requires_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _client(monkeypatch, InMemoryMemoryRepository()) as client:
        response = await client.get("/api/v1/actions")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_listing_degrades_to_empty_without_a_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """仓储没起来时返回空清单而不是 500：单据面板是旁路信息，不该拖垮页面。"""
    async with _client(monkeypatch, None) as client:
        response = await client.get("/api/v1/actions", headers=_auth("u-1"))

    assert response.status_code == 200
    assert response.json() == {"actions": []}


def test_action_fields_keeps_only_whitelisted_keys_in_order() -> None:
    """界面排版直接用这份字典的顺序，请假原因、票据号一类正文不得进来。"""
    fields = action_fields(
        "leave_request",
        {
            "leave_type": "年假",
            "start_date": "2026-09-18",
            "end_date": "2026-09-18",
            "reason": "家里有事需要处理",
        },
    )

    assert list(fields) == ["leave_type", "start_date", "end_date"]
    assert "家里有事需要处理" not in str(fields)


def test_render_omits_a_missing_document_number() -> None:
    """老数据的 result 里没有单号。这时只能整段省略——幂等键不是备选项。"""
    rendered = RecentAction(
        reference_id="",
        action_type="travel_application",
        summary="destination=上海",
        created_at=datetime(2026, 9, 13, tzinfo=UTC),
    ).render()

    assert rendered == "travel_application（2026-09-13）：destination=上海"
