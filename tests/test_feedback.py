"""点赞点踩。

评价的单位是一段话（一条助手消息），不是一轮也不是一个任务：同一次执行、同一个任务里
常有好几段。本地表决定界面上点没点过，LangSmith 上的那份用来筛 badcase。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.core.models import AgentNote, PendingConfirmation
from enterprise_ai_assistant.core.security import create_access_token
from enterprise_ai_assistant.graph.workflow import reply_message
from enterprise_ai_assistant.main import create_app
from enterprise_ai_assistant.repositories.feedback import (
    FeedbackRating,
    FeedbackReason,
    InMemoryFeedbackRepository,
    MessageFeedback,
)
from enterprise_ai_assistant.services.feedback import (
    FeedbackEvent,
    FeedbackSync,
    LangSmithFeedbackSync,
    langsmith_feedback_id,
)

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000021")

SETTINGS = Settings(
    openai_api_key="test-key",  # type: ignore[arg-type]
    openai_model="test-model",
    openai_embedding_model="test-embedding",
    langsmith_tracing=False,
    jwt_secret="unit-test-secret-" + "x" * 32,  # type: ignore[arg-type]
    app_env="development",
)


class StubGraph:
    def __init__(self, values: dict[str, Any], interrupts: tuple[Any, ...] = ()) -> None:
        self.values = values
        self.interrupts = interrupts

    async def aget_state(self, config: dict[str, Any]) -> Any:
        del config
        return SimpleNamespace(values=self.values, next=(), interrupts=self.interrupts)


class RecordingSync(FeedbackSync):
    def __init__(self) -> None:
        self.events: list[FeedbackEvent] = []

    def submit(self, event: FeedbackEvent) -> None:
        self.events.append(event)

    async def aclose(self) -> None:
        return None


@asynccontextmanager
async def _noop_lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    yield


def _app(
    values: dict[str, Any], monkeypatch: pytest.MonkeyPatch, interrupts: tuple[Any, ...] = ()
) -> FastAPI:
    monkeypatch.setattr(routes, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: SETTINGS)
    app = create_app(_noop_lifespan)
    app.state.graph = StubGraph(values, interrupts)
    app.state.feedback = InMemoryFeedbackRepository()
    app.state.feedback_sync = RecordingSync()
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _auth(user_id: str = "owner-user") -> dict[str, str]:
    token, _ = create_access_token(user_id, SETTINGS)
    return {"Authorization": f"Bearer {token}"}


def _values() -> dict[str, Any]:
    return {
        "user_id": "owner-user",
        "messages": [
            HumanMessage(content="我还有多少年假？"),
            reply_message("你的年假还剩 8 天。", [], trace_id="trace-1", message_id="m-said"),
            reply_message("请假申请已提交。", [], trace_id="trace-1", message_id="m-answer"),
            # 取消回复这类固定文案不记 trace。
            reply_message("好的，这件事不办了。", [], message_id="m-template"),
            # 这个功能上线前的老消息。
            AIMessage(content="旧回答", id="m-old"),
        ],
    }


def _url(message_id: str) -> str:
    return f"/api/v1/conversations/{CONVERSATION_ID}/messages/{message_id}/feedback"


@pytest.mark.asyncio
async def test_history_offers_rating_only_for_what_the_model_wrote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(_values(), monkeypatch)
    async with _client(app) as client:
        await client.put(_url("m-answer"), json={"rating": "down"}, headers=_auth())
        response = await client.get(
            f"/api/v1/conversations/{CONVERSATION_ID}/messages", headers=_auth()
        )

    assert [
        (item["text"], item["message_id"], item["feedback"])
        for item in response.json()["messages"]
        if item["role"] == "assistant"
    ] == [
        # 同一个 trace 里的两段话各评各的。
        ("你的年假还剩 8 天。", "m-said", None),
        ("请假申请已提交。", "m-answer", "down"),
        ("好的，这件事不办了。", None, None),
        ("旧回答", None, None),
    ]


@pytest.mark.asyncio
async def test_rating_is_saved_and_changing_it_updates_the_same_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(_values(), monkeypatch)
    async with _client(app) as client:
        first = await client.put(_url("m-answer"), json={"rating": "down"}, headers=_auth())
        second = await client.put(
            _url("m-answer"),
            json={"rating": "down", "reasons": ["fabricated"], "comment": "  我没说过  "},
            headers=_auth(),
        )

    assert (first.status_code, second.status_code) == (204, 204)
    assert app.state.feedback.items[("owner-user", "m-answer")] == MessageFeedback(
        user_id="owner-user",
        conversation_id=CONVERSATION_ID,
        message_id="m-answer",
        trace_id="trace-1",
        rating=FeedbackRating.DOWN,
        reasons=[FeedbackReason.FABRICATED],
        comment="我没说过",
    )
    events = app.state.feedback_sync.events
    # 先点踩、再补理由：第一次在 LangSmith 上建，第二次改同一条。
    assert [(event.first, event.text) for event in events] == [
        (True, "请假申请已提交。"),
        (False, "请假申请已提交。"),
    ]


@pytest.mark.asyncio
async def test_what_was_said_before_a_pending_card_can_be_rated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note = AgentNote(text="你的年假还剩 8 天。", trace_id="trace-ask")
    pending = PendingConfirmation(
        task_id="task-1",
        action="submit_leave_request",
        tool_call_id="call-1",
        title="提交请假申请",
        payload={},
        notes=[note],
    )
    interrupts = (SimpleNamespace(id="i-1", value=pending.model_dump(mode="json")),)
    values = {"user_id": "owner-user", "messages": [HumanMessage(content="请一天年假")]}
    app = _app(values, monkeypatch, interrupts)
    async with _client(app) as client:
        history = await client.get(
            f"/api/v1/conversations/{CONVERSATION_ID}/messages", headers=_auth()
        )
        rated = await client.put(_url(note.id), json={"rating": "up"}, headers=_auth())

    assert history.json()["messages"][-1]["message_id"] == note.id
    assert rated.status_code == 204
    [event] = app.state.feedback_sync.events
    assert (event.feedback.trace_id, event.task_id) == ("trace-ask", "task-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("message_id", ["m-template", "m-old", "missing"])
async def test_only_model_written_messages_can_be_rated(
    message_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(_values(), monkeypatch)
    async with _client(app) as client:
        response = await client.put(_url(message_id), json={"rating": "down"}, headers=_auth())

    assert response.status_code == 404
    assert app.state.feedback.items == {}


@pytest.mark.asyncio
async def test_messages_in_someone_elses_conversation_cannot_be_rated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(_values(), monkeypatch)
    async with _client(app) as client:
        response = await client.put(
            _url("m-answer"), json={"rating": "down"}, headers=_auth("intruder")
        )

    # 不校验归属的话，拿到消息 id 就能往别人的 trace 上刷反馈。
    assert response.status_code == 404
    assert app.state.feedback.items == {}


@pytest.mark.asyncio
async def test_an_upvote_takes_no_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(_values(), monkeypatch)
    async with _client(app) as client:
        response = await client.put(
            _url("m-answer"), json={"rating": "up", "reasons": ["other"]}, headers=_auth()
        )

    assert response.status_code == 422


class FakeLangSmith:
    def __init__(self, *, update_fails: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.update_fails = update_fails

    def read_project(self, *, project_name: str) -> Any:
        self.calls.append(("read_project", {"project_name": project_name}))
        return SimpleNamespace(id=UUID("00000000-0000-0000-0000-0000000000aa"))

    def create_feedback(self, **kwargs: Any) -> None:
        self.calls.append(("create", kwargs))

    def update_feedback(self, feedback_id: UUID, **kwargs: Any) -> None:
        self.calls.append(("update", {"feedback_id": feedback_id, **kwargs}))
        if self.update_fails:
            raise RuntimeError("not found")


def _event(rating: FeedbackRating, *, first: bool, **fields: Any) -> FeedbackEvent:
    return FeedbackEvent(
        feedback=MessageFeedback(
            user_id="owner-user",
            conversation_id=CONVERSATION_ID,
            message_id="m-answer",
            trace_id="trace-1",
            rating=rating,
            **fields,
        ),
        first=first,
        task_id="task-1",
        text="请假申请已提交。",
    )


@pytest.mark.asyncio
async def test_sync_creates_on_the_trace_then_updates_the_same_feedback() -> None:
    client = FakeLangSmith()
    sync = LangSmithFeedbackSync(client, "enterprise-ai-assistant")

    sync.submit(_event(FeedbackRating.DOWN, first=True, reasons=[FeedbackReason.WRONG_FIELDS]))
    sync.submit(_event(FeedbackRating.UP, first=False))
    await sync.aclose()

    feedback_id = langsmith_feedback_id("owner-user", "m-answer")
    assert [name for name, _ in client.calls] == ["read_project", "create", "update"]
    created, updated = client.calls[1][1], client.calls[2][1]
    assert created["run_id"] == "trace-1"
    assert created["feedback_id"] == feedback_id
    assert (created["key"], created["score"], created["value"]) == ("user_score", 0, "wrong_fields")
    assert created["session_id"] == UUID("00000000-0000-0000-0000-0000000000aa")
    assert created["extra"]["text"] == "请假申请已提交。"
    # 踩改成赞：旧理由要清掉，传 None 的话 update_feedback 会跳过这一项。
    assert updated == {"feedback_id": feedback_id, "score": 1, "value": "", "comment": ""}


@pytest.mark.asyncio
async def test_sync_creates_when_there_is_nothing_to_update() -> None:
    client = FakeLangSmith(update_fails=True)
    sync = LangSmithFeedbackSync(client, "enterprise-ai-assistant")

    sync.submit(_event(FeedbackRating.DOWN, first=False))
    await sync.aclose()

    # 第一次创建时 LangSmith 不可达，这次改不到，补建一条。
    assert [name for name, _ in client.calls] == ["update", "read_project", "create"]
