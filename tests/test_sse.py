import json
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from langchain_core.messages import AIMessageChunk

from enterprise_ai_assistant.api.routes import _encode_sse, _stream_graph


def test_sse_event_is_typed_utf8_json() -> None:
    encoded = _encode_sse("token", {"content": "差\n旅"})

    assert encoded.startswith("event: token\ndata: ")
    assert encoded.endswith("\n\n")
    payload = encoded.split("data: ", maxsplit=1)[1].strip()
    assert json.loads(payload) == {"content": "差\n旅"}


class FakeGraph:
    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content="内部规划", id="internal-1"),
                {"tags": ["domain-internal"], "task_id": "task-1"},
            ),
        }
        for content in ("真", "流式"):
            yield {
                "type": "messages",
                "data": (
                    AIMessageChunk(content=content, id="answer-1"),
                    {
                        "tags": ["user-visible"],
                        "agent": "expense",
                        "task_id": "task-1",
                    },
                ),
            }
        yield {"type": "updates", "data": {"domain_respond": {}}}

    async def aget_state(self, config: dict[str, Any]) -> Any:
        del config
        return SimpleNamespace(
            values={
                "user_id": "u-1",
                "last_answer": "真流式",
                "user_goal": "测试流式",
                "tasks": [],
                "artifacts": {},
                "tool_results": [],
                "pending_confirmation": None,
            },
            next=(),
        )


class FakeRequest:
    def __init__(self) -> None:
        logger = SimpleNamespace(info=lambda *args, **kwargs: None, exception=lambda *args, **kwargs: None)
        self.app = SimpleNamespace(state=SimpleNamespace(graph=FakeGraph(), logger=logger))

    async def is_disconnected(self) -> bool:
        return False


def decode_event(encoded: str) -> tuple[str, dict[str, Any]]:
    lines = encoded.strip().splitlines()
    return lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))


@pytest.mark.asyncio
async def test_graph_stream_forwards_only_native_user_visible_chunks() -> None:
    events = [
        decode_event(item)
        async for item in _stream_graph(
            FakeRequest(),  # type: ignore[arg-type]
            {},
            UUID("00000000-0000-0000-0000-000000000001"),
            "u-1",
        )
    ]

    assert [data["content"] for event, data in events if event == "token"] == ["真", "流式"]
    assert sum(event == "answer_start" for event, _ in events) == 1
    assert all(data.get("content") != "内部规划" for _, data in events)
    assert events[-1][0] == "done"
