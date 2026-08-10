from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from langchain_core.messages import HumanMessage

from enterprise_ai_assistant.api.routes import (
    _conversation_title,
    _response,
    delete_conversation,
    list_conversations,
)
from enterprise_ai_assistant.core.models import PendingConfirmation, PlannedTask, TaskStatus


def test_conversation_title_normalizes_whitespace_and_truncates() -> None:
    assert _conversation_title("  下周去上海\n出差  ") == "下周去上海 出差"
    assert _conversation_title("这是一条很长的会话消息", max_length=5) == "这是一条很…"


@pytest.mark.asyncio
async def test_conversation_list_is_loaded_for_current_user() -> None:
    conversation_id = uuid4()
    now = datetime.now(UTC)

    class Repository:
        async def list_for_user(self, user_id: str) -> list[dict[str, object]]:
            assert user_id == "u-1"
            return [
                {
                    "conversation_id": conversation_id,
                    "title": "上海差旅申请",
                    "created_at": now,
                    "updated_at": now,
                }
            ]

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(conversations=Repository()))
    )

    result = await list_conversations(request, "u-1")  # type: ignore[arg-type]

    assert result[0].conversation_id == conversation_id
    assert result[0].title == "上海差旅申请"


@pytest.mark.asyncio
async def test_response_hides_confirmation_without_a_real_interrupt() -> None:
    conversation_id = uuid4()
    pending = PendingConfirmation(
        task_id="task-1",
        action="travel.application.submit",
        summary="提交差旅申请",
        payload={"destination": "北京"},
    )
    snapshot = SimpleNamespace(
        next=("select_task",),
        values={
            "user_id": "u-1",
            "messages": [HumanMessage(content="明天出差")],
            "user_goal": "创建差旅申请",
            "tasks": [
                PlannedTask(
                    id="task-1",
                    title="创建差旅申请",
                    operation="submit",
                    required_capabilities=["travel.application.write"],
                    status=TaskStatus.RUNNING,
                )
            ],
            "pending_confirmation": pending,
        },
    )

    class Graph:
        async def aget_state(self, config: object) -> object:
            return snapshot

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(graph=Graph())))

    response = await _response(request, conversation_id, "u-1")  # type: ignore[arg-type]

    assert response.status == "failed"
    assert response.pending_confirmation is None
    assert response.tasks[0].status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_delete_conversation_removes_checkpoint_and_index() -> None:
    conversation_id = uuid4()

    class Repository:
        deleted = False

        async def belongs_to_user(self, **kwargs: object) -> bool:
            assert kwargs == {"conversation_id": conversation_id, "user_id": "u-1"}
            return True

        async def delete(self, **kwargs: object) -> bool:
            assert kwargs == {"conversation_id": conversation_id, "user_id": "u-1"}
            self.deleted = True
            return True

    class Checkpointer:
        deleted_thread: str | None = None

        async def adelete_thread(self, thread_id: str) -> None:
            self.deleted_thread = thread_id

    repository = Repository()
    checkpointer = Checkpointer()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                conversations=repository,
                graph=SimpleNamespace(checkpointer=checkpointer),
            )
        )
    )

    await delete_conversation(conversation_id, request, "u-1")  # type: ignore[arg-type]

    assert checkpointer.deleted_thread == str(conversation_id)
    assert repository.deleted is True


@pytest.mark.asyncio
async def test_delete_conversation_rejects_another_users_conversation() -> None:
    conversation_id = uuid4()

    class Repository:
        async def belongs_to_user(self, **kwargs: object) -> bool:
            return False

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                conversations=Repository(),
                graph=SimpleNamespace(checkpointer=None),
            )
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_conversation(conversation_id, request, "u-2")  # type: ignore[arg-type]

    assert exc_info.value.status_code == 404
