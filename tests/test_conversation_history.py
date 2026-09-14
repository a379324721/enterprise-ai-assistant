"""会话历史分页。

演示用户长期停在同一个会话里，首屏不该铺开整段历史，所以这个接口默认只给最近
一页，并用 before 游标向前翻。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.core.runs import MemoryStreamBridge, RunManager
from enterprise_ai_assistant.core.security import create_access_token
from enterprise_ai_assistant.main import create_app

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000009")

SETTINGS = Settings(
    openai_api_key="test-key",  # type: ignore[arg-type]
    openai_model="test-model",
    openai_embedding_model="test-embedding",
    langsmith_tracing=False,
    jwt_secret="unit-test-secret-" + "x" * 32,  # type: ignore[arg-type]
    app_env="development",
)


class StubGraph:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    async def aget_state(self, config: dict[str, Any]) -> Any:
        del config
        return SimpleNamespace(values=self.values, next=(), interrupts=())


@asynccontextmanager
async def _noop_lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    yield


def _history(turns: int) -> list[Any]:
    messages: list[Any] = []
    for index in range(1, turns + 1):
        messages.append(HumanMessage(content=f"问题{index}"))
        messages.append(AIMessage(content=f"回答{index}"))
    return messages


def _client(values: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> AsyncClient:
    monkeypatch.setattr(routes, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: SETTINGS)
    app = create_app(_noop_lifespan)
    app.state.graph = StubGraph(values)
    app.state.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _auth(user_id: str = "owner-user") -> dict[str, str]:
    token, _ = create_access_token(user_id, SETTINGS)
    return {"Authorization": f"Bearer {token}"}


def _url(**params: Any) -> str:
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return f"/api/v1/conversations/{CONVERSATION_ID}/messages" + (f"?{query}" if query else "")


@pytest.mark.asyncio
async def test_defaults_to_the_most_recent_page(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"user_id": "owner-user", "messages": _history(10)}

    async with _client(values, monkeypatch) as client:
        response = await client.get(_url(limit=4), headers=_auth())

    body = response.json()
    assert [item["text"] for item in body["messages"]] == ["问题9", "回答9", "问题10", "回答10"]
    assert body["has_more"] is True


@pytest.mark.asyncio
async def test_before_cursor_walks_backwards(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"user_id": "owner-user", "messages": _history(10)}

    async with _client(values, monkeypatch) as client:
        first = (await client.get(_url(limit=4), headers=_auth())).json()
        earlier = (
            await client.get(_url(limit=4, before=first["messages"][0]["index"]), headers=_auth())
        ).json()

    assert [item["text"] for item in earlier["messages"]] == ["问题7", "回答7", "问题8", "回答8"]
    assert earlier["has_more"] is True


@pytest.mark.asyncio
async def test_reaching_the_start_reports_no_more(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"user_id": "owner-user", "messages": _history(2)}

    async with _client(values, monkeypatch) as client:
        body = (await client.get(_url(limit=50), headers=_auth())).json()

    assert len(body["messages"]) == 4
    assert body["has_more"] is False


@pytest.mark.asyncio
async def test_tool_and_empty_messages_are_not_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    """领域子图的内部消息不外泄；空文本占位也不该在界面上留下空气泡。"""
    values = {
        "user_id": "owner-user",
        "messages": [
            HumanMessage(content="问题1"),
            ToolMessage(content="工具原始结果", tool_call_id="call-1"),
            AIMessage(content=""),
            AIMessage(content="回答1"),
        ],
    }

    async with _client(values, monkeypatch) as client:
        body = (await client.get(_url(), headers=_auth())).json()

    assert [(item["role"], item["text"]) for item in body["messages"]] == [
        ("user", "问题1"),
        ("assistant", "回答1"),
    ]


@pytest.mark.asyncio
async def test_answers_bring_back_their_task_title_and_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实时画出来的标题和步骤，刷新后要照原样回来；早先没记下的消息不带。"""
    values = {
        "user_id": "owner-user",
        "messages": [
            HumanMessage(content="请假明天一天，年假"),
            AIMessage(
                content="请假申请已提交。",
                additional_kwargs={
                    "tools_called": ["get_leave_balance", "submit_leave_request"],
                    "task": {"id": "task-1", "title": "提交明天年假申请"},
                    "steps": [
                        {"tool": "get_leave_balance", "success": True},
                        {"tool": "submit_leave_request", "success": False},
                    ],
                },
            ),
            AIMessage(content="旧回答", additional_kwargs={"tools_called": []}),
        ],
    }

    async with _client(values, monkeypatch) as client:
        body = (await client.get(_url(), headers=_auth())).json()

    user, answer, old = body["messages"]
    assert (user["task_id"], user["title"], user["steps"]) == (None, None, [])
    assert (answer["task_id"], answer["title"]) == ("task-1", "提交明天年假申请")
    assert [(step["label"], step["success"]) for step in answer["steps"]] == [
        ("查询假期余额", True),
        ("提交请假申请", False),
    ]
    assert (old["task_id"], old["title"], old["steps"]) == (None, None, [])


@pytest.mark.asyncio
async def test_a_fresh_conversation_returns_an_empty_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """演示用户首次登录时会话还不存在，这不是错误。"""
    async with _client({}, monkeypatch) as client:
        response = await client.get(_url(), headers=_auth())

    assert response.status_code == 200
    assert response.json() == {"messages": [], "has_more": False}


@pytest.mark.asyncio
async def test_another_users_conversation_is_not_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {"user_id": "owner-user", "messages": _history(3)}

    async with _client(values, monkeypatch) as client:
        response = await client.get(_url(), headers=_auth("stranger"))

    assert response.status_code == 404


class RecordingCheckpointer:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


def _client_with_checkpointer(
    values: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[AsyncClient, RecordingCheckpointer]:
    monkeypatch.setattr(routes, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: SETTINGS)
    app = create_app(_noop_lifespan)
    app.state.graph = StubGraph(values)
    app.state.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None
    )
    app.state.runs = RunManager(MemoryStreamBridge(), app.state.logger)
    checkpointer = RecordingCheckpointer()
    app.state.checkpointer = checkpointer
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), checkpointer


@pytest.mark.asyncio
async def test_clearing_drops_the_whole_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"user_id": "owner-user", "messages": _history(3)}

    client, checkpointer = _client_with_checkpointer(values, monkeypatch)
    async with client:
        response = await client.delete(
            f"/api/v1/conversations/{CONVERSATION_ID}", headers=_auth()
        )

    assert response.status_code == 204
    assert checkpointer.deleted == [str(CONVERSATION_ID)]


@pytest.mark.asyncio
async def test_clearing_someone_elses_conversation_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {"user_id": "owner-user", "messages": _history(3)}

    client, checkpointer = _client_with_checkpointer(values, monkeypatch)
    async with client:
        response = await client.delete(
            f"/api/v1/conversations/{CONVERSATION_ID}", headers=_auth("stranger")
        )

    assert response.status_code == 404
    assert checkpointer.deleted == []


@pytest.mark.asyncio
async def test_clearing_an_empty_conversation_still_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """调用方要的是"清空"这个结果，会话本来就不存在时重复调用不该报错。"""
    client, _ = _client_with_checkpointer({}, monkeypatch)
    async with client:
        response = await client.delete(
            f"/api/v1/conversations/{CONVERSATION_ID}", headers=_auth()
        )

    assert response.status_code == 204
