import asyncio
import json
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessageChunk

from enterprise_ai_assistant.api.routes import (
    QUOTA_EXHAUSTED_MESSAGE,
    _AnswerRelay,
    _encode_sse,
    _execute_run,
    _failure_message,
    _pending_confirmation,
    _subscribe_sse,
)
from enterprise_ai_assistant.core.models import ConfirmationField, PendingConfirmation
from enterprise_ai_assistant.core.runs import (
    END_SENTINEL,
    HEARTBEAT_SENTINEL,
    DisconnectMode,
    MemoryStreamBridge,
    Publisher,
    Run,
    RunConflictError,
    RunManager,
    RunStatus,
    StreamEvent,
    StreamGap,
)

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000001")


def test_sse_event_is_typed_utf8_json() -> None:
    encoded = _encode_sse("token", {"content": "差\n旅"})

    assert encoded.startswith("event: token\ndata: ")
    assert encoded.endswith("\n\n")
    payload = encoded.split("data: ", maxsplit=1)[1].strip()
    assert json.loads(payload) == {"content": "差\n旅"}


def test_sse_event_carries_id_for_reconnection() -> None:
    encoded = _encode_sse("token", {"content": "差旅"}, event_id=7)

    assert encoded.startswith("id: 7\nevent: token\n")


def test_pending_confirmation_comes_from_interrupt_payload() -> None:
    pending = PendingConfirmation(
        task_id="task-1",
        action="submit_leave_request",
        tool_call_id="call-1",
        title="提交请假申请",
        fields=[ConfirmationField(name="leave_type", label="假期类型", value="annual")],
        payload={"leave_type": "annual"},
    )
    snapshot = SimpleNamespace(
        next=("domain_task",),
        interrupts=(SimpleNamespace(value=pending.model_dump(mode="json")),),
    )

    assert _pending_confirmation(snapshot) == pending


# -- 事件桥 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_replays_only_events_after_the_client_cursor() -> None:
    bridge = MemoryStreamBridge()
    for index in range(3):
        await bridge.publish("run-1", "token", {"content": str(index)})
    await bridge.publish_end("run-1")

    resumed = [item async for item in bridge.subscribe("run-1", last_seq=0)]

    assert [item.data["content"] for item in resumed if isinstance(item, StreamEvent) and item.data]
    assert [
        item.seq for item in resumed if isinstance(item, StreamEvent) and item is not END_SENTINEL
    ] == [1, 2]
    assert resumed[-1] is END_SENTINEL


@pytest.mark.asyncio
async def test_bridge_reports_a_gap_when_the_cursor_fell_out_of_the_buffer() -> None:
    """缓冲滚过客户端游标时必须显式报缺口，而不是假装回放完整。"""
    bridge = MemoryStreamBridge(buffer_size=2)
    for index in range(5):
        await bridge.publish("run-1", "token", {"content": str(index)})
    await bridge.publish_end("run-1")

    items = [item async for item in bridge.subscribe("run-1", last_seq=0)]

    assert len(items) == 1
    gap = items[0]
    assert isinstance(gap, StreamGap)
    assert gap.requested_seq == 1
    assert gap.earliest_available_seq == 3


@pytest.mark.asyncio
async def test_bridge_emits_heartbeats_while_idle() -> None:
    bridge = MemoryStreamBridge(heartbeat_interval=0.01)
    stream = bridge.subscribe("run-1")

    assert await anext(stream) is HEARTBEAT_SENTINEL

    await bridge.publish("run-1", "token", {"content": "差旅"})
    event = await anext(stream)

    assert isinstance(event, StreamEvent)
    assert event.data == {"content": "差旅"}
    await stream.aclose()


# -- 后台运行 -------------------------------------------------------------


def _manager(**kwargs: Any) -> RunManager:
    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    return RunManager(MemoryStreamBridge(**kwargs), logger)


@pytest.mark.asyncio
async def test_run_keeps_executing_after_every_subscriber_is_gone() -> None:
    """SSE 断线不再中断执行：这是整个改造的核心不变量。"""
    manager = _manager()
    released = asyncio.Event()
    finished = asyncio.Event()

    async def runner(run: Run, publish: Publisher) -> RunStatus:
        await publish("token", {"content": "第一段"})
        await released.wait()
        await publish("token", {"content": "第二段"})
        finished.set()
        return RunStatus.completed

    run = await manager.start(conversation_id=CONVERSATION_ID, user_id="u-1", runner=runner)
    stream = manager.subscribe(run)
    assert isinstance(await anext(stream), StreamEvent)
    await stream.aclose()  # 客户端关掉标签页

    released.set()
    await asyncio.wait_for(finished.wait(), timeout=1)
    assert run.task is not None
    await run.task
    assert run.status is RunStatus.completed

    # 重新 attach 能拿到断线期间产生的全部事件。
    resumed = [item async for item in manager.subscribe(run, last_seq=0)]
    contents = [item.data["content"] for item in resumed if isinstance(item, StreamEvent) and item.data]
    assert contents == ["第二段"]


@pytest.mark.asyncio
async def test_concurrent_run_on_the_same_conversation_is_rejected() -> None:
    manager = _manager()
    blocked = asyncio.Event()

    async def runner(run: Run, publish: Publisher) -> RunStatus:
        await blocked.wait()
        return RunStatus.completed

    request_id = uuid4()
    run = await manager.start(
        conversation_id=CONVERSATION_ID, user_id="u-1", runner=runner, request_id=request_id
    )

    # 同一个 request_id 是客户端重试，应当拿回同一次运行而不是再跑一遍。
    assert (
        await manager.start(
            conversation_id=CONVERSATION_ID, user_id="u-1", runner=runner, request_id=request_id
        )
        is run
    )
    with pytest.raises(RunConflictError):
        await manager.start(
            conversation_id=CONVERSATION_ID, user_id="u-1", runner=runner, request_id=uuid4()
        )

    blocked.set()
    assert run.task is not None
    await run.task


@pytest.mark.asyncio
async def test_manager_close_cancels_running_work() -> None:
    manager = _manager()

    async def runner(run: Run, publish: Publisher) -> RunStatus:
        await asyncio.Event().wait()
        return RunStatus.completed

    run = await manager.start(conversation_id=CONVERSATION_ID, user_id="u-1", runner=runner)
    await manager.aclose()

    assert run.task is not None
    assert run.task.done()
    assert run.status is RunStatus.cancelled


@pytest.mark.asyncio
async def test_failed_run_publishes_an_error_event_and_ends_the_stream() -> None:
    manager = _manager()

    async def runner(run: Run, publish: Publisher) -> RunStatus:
        raise RuntimeError("boom")

    run = await manager.start(conversation_id=CONVERSATION_ID, user_id="u-1", runner=runner)
    assert run.task is not None
    await run.task

    items = [item async for item in manager.subscribe(run)]

    assert run.status is RunStatus.failed
    assert isinstance(items[0], StreamEvent)
    assert items[0].event == "error"
    assert items[-1] is END_SENTINEL


# -- 图执行与 SSE 转发 ----------------------------------------------------


class FakeGraph:
    def __init__(self) -> None:
        self.stream_kwargs: dict[str, Any] = {}

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        del args
        self.stream_kwargs = kwargs
        # 任务开始先于回答增量到达，进度文案才对得上正在发生的事。
        yield {
            "type": "tasks",
            "data": {"id": "t-1", "name": "respond", "input": {}, "triggers": ()},
        }
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
        # 结束事件带 result，不该再发一次"正在生成回答"。
        yield {
            "type": "tasks",
            "data": {"id": "t-1", "name": "respond", "error": None, "result": [], "interrupts": ()},
        }

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
            interrupts=(),
        )


def _app(manager: RunManager, graph: FakeGraph | None = None) -> Any:
    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    return SimpleNamespace(
        state=SimpleNamespace(graph=graph or FakeGraph(), logger=logger, runs=manager)
    )


class FakeRequest:
    """最小 SSE 客户端：可以设定重连游标和断连时机。"""

    def __init__(self, app: Any, *, last_event_id: str | None = None, disconnect_after: int | None = None) -> None:
        self.app = app
        self.headers = {"Last-Event-ID": last_event_id} if last_event_id else {}
        self.query_params: dict[str, str] = {}
        self._disconnect_after = disconnect_after
        self.checks = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self._disconnect_after is not None and self.checks > self._disconnect_after


def decode_event(encoded: str) -> tuple[str, dict[str, Any]]:
    lines = [line for line in encoded.strip().splitlines() if not line.startswith("id: ")]
    return lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))


async def _run_to_completion(manager: RunManager, app: Any) -> Run:
    async def runner(run: Run, publish: Publisher) -> RunStatus:
        return await _execute_run(app, run, publish, {}, CONVERSATION_ID, "u-1")

    run = await manager.start(conversation_id=CONVERSATION_ID, user_id="u-1", runner=runner)
    assert run.task is not None
    await run.task
    return run


@pytest.mark.asyncio
async def test_graph_stream_forwards_only_native_user_visible_chunks() -> None:
    manager = _manager()
    graph = FakeGraph()
    app = _app(manager, graph)

    run = await _run_to_completion(manager, app)
    request = FakeRequest(app)
    events = [
        decode_event(frame)
        async for frame in _subscribe_sse(request, run, apply_on_disconnect=False)  # type: ignore[arg-type]
    ]

    # 回答太短，攒不够放行字数，在模型调用结束（下一个任务事件）时整段放行。
    assert [data["content"] for event, data in events if event == "token"] == ["真流式"]
    # 进度只在任务开始时发一次，且排在回答增量之前——反过来就意味着回答已经流完
    # 才显示"正在生成回答"。
    names = [event for event, _ in events]
    assert names.count("progress") == 1
    assert names.index("progress") < names.index("token")
    assert sum(event == "answer_start" for event, _ in events) == 1
    assert all(data.get("content") != "内部规划" for _, data in events)
    assert events[0][0] == "metadata"
    assert events[0][1]["run_id"] == run.run_id
    assert events[-1][0] == "done"
    assert graph.stream_kwargs["subgraphs"] is True
    assert run.status is RunStatus.completed


def _recording_relay() -> tuple[_AnswerRelay, list[tuple[str, dict[str, Any]]]]:
    published: list[tuple[str, dict[str, Any]]] = []

    async def publish(event: str, data: dict[str, Any]) -> None:
        published.append((event, data))

    return _AnswerRelay(publish), published  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_relay_never_leaks_text_from_a_tool_calling_decision() -> None:
    relay, published = _recording_relay()
    metadata = {"tags": ["user-visible"], "agent": "travel", "task_id": "task-1"}

    # 决策调用先吐了几个字，随后才出现工具调用：这几个字不是回答。
    await relay.chunk(AIMessageChunk(content="我先查一下", id="m-1"), metadata)
    await relay.chunk(
        AIMessageChunk(
            content="",
            id="m-1",
            tool_call_chunks=[{"name": "search_travel_policy", "args": "{}", "id": "c-1", "index": 0}],
        ),
        metadata,
    )
    await relay.chunk(AIMessageChunk(content="还有", id="m-1"), metadata)
    await relay.flush()

    assert published == []


@pytest.mark.asyncio
async def test_relay_streams_a_long_answer_once_the_hold_is_exceeded() -> None:
    relay, published = _recording_relay()
    metadata = {"tags": ["user-visible"], "agent": "hr", "task_id": "task-1"}
    head = "已为你提交请假申请，单号 LV-1234，"

    await relay.chunk(AIMessageChunk(content=head, id="m-1"), metadata)
    await relay.chunk(AIMessageChunk(content="审批结果以查询为准。", id="m-1"), metadata)

    assert [event for event, _ in published] == ["answer_start", "token", "token"]
    assert "".join(data["content"] for event, data in published if event == "token") == (
        head + "审批结果以查询为准。"
    )


@pytest.mark.asyncio
async def test_relay_publishes_answers_that_bypassed_the_model() -> None:
    relay, published = _recording_relay()

    await relay.whole("预计哪天返回？", {"answer": "预计哪天返回？", "agent": "travel", "task_id": "task-2"})

    assert published[0] == (
        "answer_start",
        {"message_id": "answer:task-2", "agent": "travel", "task_id": "task-2"},
    )
    assert published[1][1]["content"] == "预计哪天返回？"


@pytest.mark.asyncio
async def test_reconnect_replays_only_the_missing_events() -> None:
    manager = _manager()
    app = _app(manager)
    run = await _run_to_completion(manager, app)

    first = FakeRequest(app)
    frames = [frame async for frame in _subscribe_sse(first, run, apply_on_disconnect=False)]  # type: ignore[arg-type]
    cutoff = frames[1].splitlines()[0].removeprefix("id: ")

    resumed = FakeRequest(app, last_event_id=cutoff)
    replayed = [
        decode_event(frame)
        async for frame in _subscribe_sse(resumed, run, apply_on_disconnect=False)  # type: ignore[arg-type]
    ]

    assert [event for event, _ in replayed] == [event for event, _ in map(decode_event, frames)][2:]


@pytest.mark.asyncio
async def test_reconnect_beyond_the_buffer_window_reports_a_gap() -> None:
    manager = _manager(buffer_size=2)
    app = _app(manager)

    async def runner(run: Run, publish: Publisher) -> RunStatus:
        for index in range(5):
            await publish("token", {"content": str(index)})
        return RunStatus.completed

    run = await manager.start(conversation_id=CONVERSATION_ID, user_id="u-1", runner=runner)
    assert run.task is not None
    await run.task

    request = FakeRequest(app, last_event_id="0")
    frames = [
        decode_event(frame)
        async for frame in _subscribe_sse(request, run, apply_on_disconnect=False)  # type: ignore[arg-type]
    ]

    assert [event for event, _ in frames] == ["gap"]
    assert frames[0][1]["recovery"] == "reload_conversation"


@pytest.mark.asyncio
async def test_disconnect_cancels_the_run_only_when_the_creator_asked_for_it() -> None:
    manager = _manager()
    blocked = asyncio.Event()

    async def runner(run: Run, publish: Publisher) -> RunStatus:
        await publish("token", {"content": "第一段"})
        await blocked.wait()
        return RunStatus.completed

    run = await manager.start(
        conversation_id=CONVERSATION_ID,
        user_id="u-1",
        runner=runner,
        on_disconnect=DisconnectMode.cancel,
    )
    app = _app(manager)

    # 只读旁观者断开不得终止别人的执行。
    observer = FakeRequest(app, disconnect_after=0)
    async for _ in _subscribe_sse(observer, run, apply_on_disconnect=False):  # type: ignore[arg-type]
        pass
    assert run.status is RunStatus.running

    creator = FakeRequest(app, disconnect_after=0)
    async for _ in _subscribe_sse(creator, run, apply_on_disconnect=True):  # type: ignore[arg-type]
        pass
    assert run.task is not None
    await asyncio.gather(run.task, return_exceptions=True)
    assert run.status is RunStatus.cancelled


class _QuotaError(Exception):
    """模拟 openai SDK 的状态码异常：服务端返回的错误码挂在 code 上。"""

    def __init__(self, code: str) -> None:
        super().__init__(f"Error code: 403 - {code}")
        self.code = code


def test_quota_exhaustion_gets_its_own_message() -> None:
    assert _failure_message(_QuotaError("AllocationQuota.FreeTierOnly")) == QUOTA_EXHAUSTED_MESSAGE
    assert _failure_message(_QuotaError("insufficient_quota")) == QUOTA_EXHAUSTED_MESSAGE


def test_quota_error_is_found_under_a_wrapping_exception() -> None:
    try:
        try:
            raise _QuotaError("Arrearage")
        except _QuotaError as inner:
            raise RuntimeError("graph failed") from inner
    except RuntimeError as outer:
        assert _failure_message(outer) == QUOTA_EXHAUSTED_MESSAGE


def test_other_failures_keep_the_generic_message() -> None:
    assert _failure_message(_QuotaError("InvalidParameter")) != QUOTA_EXHAUSTED_MESSAGE
    assert _failure_message(RuntimeError("boom")) != QUOTA_EXHAUSTED_MESSAGE
