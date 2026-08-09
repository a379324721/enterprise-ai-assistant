from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from enterprise_ai_assistant.api.routes import _conversation_title, list_conversations


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
