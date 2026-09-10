"""断线恢复相关的 HTTP 行为。"""

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.core.runs import (
    MemoryStreamBridge,
    Publisher,
    Run,
    RunManager,
    RunStatus,
)
from enterprise_ai_assistant.core.security import create_access_token
from enterprise_ai_assistant.main import create_app

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000002")

SETTINGS = Settings(
    openai_api_key="test-key",  # type: ignore[arg-type]
    openai_model="test-model",
    openai_embedding_model="test-embedding",
    langsmith_tracing=False,
    jwt_secret="unit-test-secret-" + "x" * 32,  # type: ignore[arg-type]
    app_env="development",
)


class FakeGraph:
    async def aget_state(self, config: dict[str, Any]) -> Any:
        del config
        return SimpleNamespace(
            values={
                "user_id": "owner-user",
                "last_answer": "已完成",
                "user_goal": "测试",
                "tasks": [],
                "artifacts": {},
                "tool_results": [],
            },
            next=(),
            interrupts=(),
        )


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(routes, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: SETTINGS)
    application = create_app()
    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    application.state.graph = FakeGraph()
    application.state.logger = logger
    application.state.runs = RunManager(MemoryStreamBridge(), logger)
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client
    await app.state.runs.aclose()


def _auth() -> dict[str, str]:
    token, _ = create_access_token("owner-user", SETTINGS)
    return {"Authorization": f"Bearer {token}"}


async def _start_blocked_run(app: FastAPI, release: asyncio.Event) -> Run:
    async def runner(run: Run, publish: Publisher) -> RunStatus:
        await publish("progress", {"node": "plan", "message": "正在拆解任务"})
        await release.wait()
        return RunStatus.completed

    manager: RunManager = app.state.runs
    return await manager.start(
        conversation_id=CONVERSATION_ID, user_id="owner-user", runner=runner
    )


def _events(body: str) -> list[tuple[str, Any]]:
    parsed: list[tuple[str, Any]] = []
    for frame in body.split("\n\n"):
        lines = [line for line in frame.splitlines() if line and not line.startswith(":")]
        if len(lines) < 2:
            continue
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        parsed.append((event, json.loads(data)))
    return parsed


@pytest.mark.asyncio
async def test_conversation_reports_running_instead_of_a_half_finished_state(
    app: FastAPI, client: AsyncClient
) -> None:
    """执行中途读会话，不能把检查点里的半成品当成失败。"""
    release = asyncio.Event()
    run = await _start_blocked_run(app, release)
    await asyncio.sleep(0)

    response = await client.get(f"/api/v1/conversations/{CONVERSATION_ID}", headers=_auth())

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["run_id"] == run.run_id

    release.set()
    assert run.task is not None
    await run.task


@pytest.mark.asyncio
async def test_attach_endpoint_replays_events_from_the_retention_window(
    app: FastAPI, client: AsyncClient
) -> None:
    """断线期间产生的事件在保留期内仍能补发。"""
    release = asyncio.Event()
    run = await _start_blocked_run(app, release)
    await asyncio.sleep(0)
    release.set()
    assert run.task is not None
    await run.task

    response = await client.get(
        f"/api/v1/conversations/{CONVERSATION_ID}/stream", headers=_auth()
    )

    assert response.status_code == 200
    assert [event for event, _ in _events(response.text)] == ["progress"]


@pytest.mark.asyncio
async def test_attach_without_a_retained_run_returns_a_snapshot(client: AsyncClient) -> None:
    response = await client.get(
        f"/api/v1/conversations/{CONVERSATION_ID}/stream", headers=_auth()
    )

    events = _events(response.text)
    assert [event for event, _ in events] == ["done"]
    assert events[0][1]["answer"] == "已完成"


@pytest.mark.asyncio
async def test_another_user_cannot_attach_to_a_running_conversation(
    app: FastAPI, client: AsyncClient
) -> None:
    release = asyncio.Event()
    run = await _start_blocked_run(app, release)
    intruder, _ = create_access_token("intruder", SETTINGS)

    response = await client.get(
        f"/api/v1/conversations/{CONVERSATION_ID}/stream",
        headers={"Authorization": f"Bearer {intruder}"},
    )

    assert response.status_code == 404

    release.set()
    assert run.task is not None
    await run.task


@pytest.mark.asyncio
async def test_second_turn_is_rejected_while_the_conversation_is_running(
    app: FastAPI, client: AsyncClient
) -> None:
    release = asyncio.Event()
    run = await _start_blocked_run(app, release)

    response = await client.post(
        "/api/v1/chat/stream",
        headers=_auth(),
        json={
            "message": "再帮我查一下",
            "conversation_id": str(CONVERSATION_ID),
            "request_id": str(uuid4()),
        },
    )

    assert response.status_code == 409

    release.set()
    assert run.task is not None
    await run.task
