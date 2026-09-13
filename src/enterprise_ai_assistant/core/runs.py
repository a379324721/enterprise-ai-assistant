"""后台运行注册表与 SSE 事件桥。

执行与传输在这里解耦：图执行跑在后台任务里，SSE 连接只是事件的订阅者。
客户端断开只是退订，不再中断执行；重连时凭 `Last-Event-ID` 从缓冲续读。

`StreamBridge` 是可替换的抽象：单进程用 `MemoryStreamBridge`，多副本部署
可以换成 Redis Streams 实现而不动路由层。
"""

import asyncio
import contextlib
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

DEFAULT_HEARTBEAT_SECONDS = 15.0


@dataclass(frozen=True)
class StreamEvent:
    """一条流事件；`seq` 单调递增，作为 SSE 的 `id:` 字段供断线续传使用。"""

    seq: int
    event: str
    data: Any


@dataclass(frozen=True)
class StreamGap:
    """订阅者的游标已经落在保留窗口之外，无法完整回放。

    客户端收到后应重新拉取会话快照，而不是把残缺的回放当成完整流。
    """

    requested_seq: int
    earliest_available_seq: int | None
    latest_available_seq: int | None


# 用哨兵事件而不是独立类型，订阅者一次 isinstance 判断即可分流。
HEARTBEAT_SENTINEL = StreamEvent(seq=-1, event="__heartbeat__", data=None)
END_SENTINEL = StreamEvent(seq=-1, event="__end__", data=None)

StreamItem = StreamEvent | StreamGap


class StreamBridge(ABC):
    """生产者（后台运行）与消费者（SSE 连接）之间的事件通道。"""

    #: 实现是否能跨进程订阅；内存实现只在启动运行的那个进程内可见。
    supports_cross_process: bool = False

    def __init__(self, *, heartbeat_interval: float = DEFAULT_HEARTBEAT_SECONDS) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        self._heartbeat_interval = heartbeat_interval

    @property
    def heartbeat_interval(self) -> float:
        return self._heartbeat_interval

    @abstractmethod
    async def publish(self, run_id: str, event: str, data: Any) -> None:
        """追加一条事件。"""

    @abstractmethod
    async def publish_end(self, run_id: str) -> None:
        """标记该运行不会再有新事件。"""

    @abstractmethod
    def subscribe(
        self,
        run_id: str,
        *,
        last_seq: int | None = None,
        heartbeat_interval: float | None = None,
    ) -> AsyncIterator[StreamItem]:
        """订阅事件流。

        `last_seq` 表示客户端已收到的最后一个序号，从它的下一条开始回放。
        空闲超过心跳间隔会产出 `HEARTBEAT_SENTINEL`；生产者结束后产出
        `END_SENTINEL`；游标落在保留窗口之外则产出 `StreamGap` 并结束。
        """

    @abstractmethod
    async def cleanup(self, run_id: str) -> None:
        """释放该运行占用的缓冲。"""

    async def close(self) -> None:
        """释放后端资源；无外部连接的实现无需覆写。"""
        return None


@dataclass
class _RunStream:
    events: list[StreamEvent] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    ended: bool = False
    next_seq: int = 0
    #: 已被裁剪掉的事件数量，即当前保留的最小序号。
    start_seq: int = 0


class MemoryStreamBridge(StreamBridge):
    """进程内实现：每个运行一份有界事件日志。

    事件在缓冲里保留到运行被清理为止，因此断线重连能补齐窗口内的增量；
    超出窗口的游标会得到 `StreamGap`，由客户端改用会话快照恢复。
    """

    def __init__(
        self,
        *,
        buffer_size: int = 512,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        super().__init__(heartbeat_interval=heartbeat_interval)
        self._buffer_size = max(1, buffer_size)
        self._streams: dict[str, _RunStream] = {}

    def _stream(self, run_id: str) -> _RunStream:
        stream = self._streams.get(run_id)
        if stream is None:
            stream = _RunStream()
            self._streams[run_id] = stream
        return stream

    @staticmethod
    def _gap(stream: _RunStream, requested_seq: int) -> StreamGap:
        return StreamGap(
            requested_seq=requested_seq,
            earliest_available_seq=stream.events[0].seq if stream.events else None,
            latest_available_seq=stream.events[-1].seq if stream.events else None,
        )

    async def publish(self, run_id: str, event: str, data: Any) -> None:
        stream = self._stream(run_id)
        async with stream.condition:
            if stream.ended:
                return
            stream.events.append(StreamEvent(seq=stream.next_seq, event=event, data=data))
            stream.next_seq += 1
            overflow = len(stream.events) - self._buffer_size
            if overflow > 0:
                del stream.events[:overflow]
                stream.start_seq = stream.events[0].seq
            stream.condition.notify_all()

    async def publish_end(self, run_id: str) -> None:
        stream = self._stream(run_id)
        async with stream.condition:
            stream.ended = True
            stream.condition.notify_all()

    async def subscribe(
        self,
        run_id: str,
        *,
        last_seq: int | None = None,
        heartbeat_interval: float | None = None,
    ) -> AsyncIterator[StreamItem]:
        interval = heartbeat_interval or self._heartbeat_interval
        stream = self._stream(run_id)
        cursor = 0 if last_seq is None else last_seq + 1
        while True:
            pending: list[StreamEvent] = []
            gap: StreamGap | None = None
            ended = False
            heartbeat = False
            async with stream.condition:
                if cursor < stream.start_seq:
                    gap = self._gap(stream, cursor)
                elif cursor < stream.next_seq:
                    offset = cursor - stream.start_seq
                    pending = stream.events[offset:]
                    cursor = stream.next_seq
                elif stream.ended:
                    ended = True
                else:
                    # 条件变量的等待会释放锁；超时即产出心跳，让反向代理和
                    # 前端都能确认连接仍然存活。
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(stream.condition.wait(), interval)
                        continue
                    heartbeat = True
            if gap is not None:
                yield gap
                return
            for item in pending:
                yield item
            if heartbeat:
                yield HEARTBEAT_SENTINEL
            if ended:
                yield END_SENTINEL
                return

    async def cleanup(self, run_id: str) -> None:
        self._streams.pop(run_id, None)

    async def close(self) -> None:
        self._streams.clear()


class RunStatus(StrEnum):
    """一次运行的生命周期状态。"""

    running = "running"
    completed = "completed"
    #: 图停在人工确认中断上，等待 confirm 接口继续。
    interrupted = "interrupted"
    failed = "failed"
    cancelled = "cancelled"


class DisconnectMode(StrEnum):
    """SSE 订阅者断开时对后台执行的处置策略。"""

    cancel = "cancel"
    continue_ = "continue"


@dataclass
class Run:
    """一次后台图执行。"""

    run_id: str
    conversation_id: UUID
    user_id: str
    request_id: UUID | None = None
    on_disconnect: DisconnectMode = DisconnectMode.continue_
    status: RunStatus = RunStatus.running
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    # 失败时给用户看的一句话；执行体没有设置时用通用文案。
    error_message: str | None = None


class RunConflictError(RuntimeError):
    """同一会话已有正在执行的运行。"""


#: 执行体用来发事件的回调，屏蔽掉具体的 bridge 实现。
Publisher = Callable[[str, Any], Awaitable[None]]
#: 执行体：拿到运行句柄和 publisher，跑完返回最终状态。
Runner = Callable[["Run", Publisher], Awaitable[RunStatus]]


class RunManager:
    """后台运行的注册表：负责启动、查找、取消和回收。"""

    def __init__(
        self,
        bridge: StreamBridge,
        logger: Any,
        *,
        retention_seconds: float = 300.0,
    ) -> None:
        self._bridge = bridge
        self._logger = logger
        self._retention = retention_seconds
        self._runs: dict[str, Run] = {}
        # 每个会话只记最近一次运行：既用于并发准入，也用于断线重连时找回它。
        self._by_conversation: dict[UUID, str] = {}
        self._cleanups: set[asyncio.Task[None]] = set()
        self._closed = False

    @property
    def bridge(self) -> StreamBridge:
        return self._bridge

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def latest(self, conversation_id: UUID) -> Run | None:
        """返回该会话最近一次仍在保留期内的运行。"""
        run_id = self._by_conversation.get(conversation_id)
        return self._runs.get(run_id) if run_id is not None else None

    def active(self, conversation_id: UUID) -> Run | None:
        run = self.latest(conversation_id)
        return run if run is not None and run.status is RunStatus.running else None

    async def start(
        self,
        *,
        conversation_id: UUID,
        user_id: str,
        runner: Runner,
        request_id: UUID | None = None,
        on_disconnect: DisconnectMode = DisconnectMode.continue_,
    ) -> Run:
        """启动一次后台运行；同一会话已有运行在跑时拒绝。

        重复提交同一个 `request_id` 视为断线重连，直接返回已有运行，
        避免客户端重试把同一轮对话跑两遍。
        """
        if self._closed:
            raise RunConflictError("服务正在停止，请稍后重试")
        current = self.active(conversation_id)
        if current is not None:
            if request_id is not None and current.request_id == request_id:
                return current
            raise RunConflictError("当前会话仍在执行中")
        run = Run(
            run_id=uuid4().hex,
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
            on_disconnect=on_disconnect,
        )
        self._runs[run.run_id] = run
        self._by_conversation[conversation_id] = run.run_id
        run.task = asyncio.create_task(self._supervise(run, runner), name=f"run-{run.run_id}")
        return run

    async def _supervise(self, run: Run, runner: Runner) -> None:
        async def publish(event: str, data: Any) -> None:
            await self._bridge.publish(run.run_id, event, data)

        try:
            run.status = await runner(run, publish)
        except asyncio.CancelledError:
            run.status = RunStatus.cancelled
            await self._publish_quietly(run, "cancelled", {"run_id": run.run_id})
            raise
        except Exception:
            run.status = RunStatus.failed
            self._logger.exception(
                "run_failed", run_id=run.run_id, conversation_id=str(run.conversation_id)
            )
            await self._publish_quietly(
                run, "error", {"message": "智能助手执行失败，请稍后重试"}
            )
        finally:
            run.finished_at = time.monotonic()
            with contextlib.suppress(Exception):
                await self._bridge.publish_end(run.run_id)
            self._schedule_cleanup(run)

    async def _publish_quietly(self, run: Run, event: str, data: Any) -> None:
        """收尾事件不能因为通道异常而掩盖真正的失败原因。"""
        with contextlib.suppress(Exception):
            await self._bridge.publish(run.run_id, event, data)

    def _schedule_cleanup(self, run: Run) -> None:
        task = asyncio.create_task(self._cleanup_later(run), name=f"run-cleanup-{run.run_id}")
        self._cleanups.add(task)
        task.add_done_callback(self._cleanups.discard)

    async def _cleanup_later(self, run: Run) -> None:
        # 运行结束后再保留一段时间，让断线的客户端回来仍能读到 done 事件。
        try:
            await asyncio.sleep(self._retention)
        except asyncio.CancelledError:
            return
        self._runs.pop(run.run_id, None)
        if self._by_conversation.get(run.conversation_id) == run.run_id:
            self._by_conversation.pop(run.conversation_id, None)
        await self._bridge.cleanup(run.run_id)

    def subscribe(
        self, run: Run, *, last_seq: int | None = None
    ) -> AsyncIterator[StreamItem]:
        return self._bridge.subscribe(run.run_id, last_seq=last_seq)

    def request_cancel(self, run_id: str) -> bool:
        """请求取消但不等待收尾。

        SSE 订阅者断开后的清理跑在异步生成器的 `finally` 里，那里不能长时间
        阻塞，因此只投递取消信号，由 `_supervise` 负责状态流转和收尾事件。
        """
        run = self._runs.get(run_id)
        if run is None or run.task is None or run.task.done():
            return False
        run.task.cancel()
        return True

    async def cancel(self, run_id: str) -> bool:
        """取消并等待运行真正收尾。"""
        if not self.request_cancel(run_id):
            return False
        task = self._runs[run_id].task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        return True

    async def aclose(self) -> None:
        """停止所有后台运行；进程关停时由 lifespan 调用。"""
        self._closed = True
        running = [
            run.task for run in self._runs.values() if run.task is not None and not run.task.done()
        ]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        for run in self._runs.values():
            # 任务在真正进入 _supervise 之前就被取消时，状态还停在 running。
            if run.status is RunStatus.running and run.task is not None and run.task.done():
                run.status = RunStatus.cancelled
        pending_cleanups = list(self._cleanups)
        for task in pending_cleanups:
            task.cancel()
        if pending_cleanups:
            await asyncio.gather(*pending_cleanups, return_exceptions=True)
        await self._bridge.close()
