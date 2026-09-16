"""把用户的点赞点踩同步到 LangSmith 的 trace 上，用于筛 badcase。

同步在后台队列里串行做，不拖慢接口：LangSmith 客户端是同步的，trace 刚跑完时可能还没
入库，创建反馈会按 404 重试好几轮。串行还保证了同一条消息先创建、后修改——点踩后紧接着
补理由是常态，两次并发发出去，修改可能先到，找不到要改的那条。
"""

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

import structlog
from langsmith import Client

from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.repositories.feedback import FeedbackRating, MessageFeedback

logger = structlog.get_logger()

#: LangSmith 上反馈的名字，按它筛：score 为 0 的就是被点踩的 trace。
FEEDBACK_KEY = "user_score"

_FEEDBACK_NAMESPACE = "urn:enterprise-ai-assistant:message-feedback"


def langsmith_feedback_id(user_id: str, message_id: str) -> UUID:
    """同一个人对同一条消息的反馈在 LangSmith 上固定一个 id，改主意时改的是同一条。"""
    return uuid5(NAMESPACE_URL, f"{_FEEDBACK_NAMESPACE}:{user_id}:{message_id}")


@dataclass(frozen=True)
class FeedbackEvent:
    feedback: MessageFeedback
    # 本地第一次记这条反馈。之后的提交在 LangSmith 上改原来那条，不另建。
    first: bool
    task_id: str | None
    # 被评价的那段话。一个 trace 里常有好几段，复核时要知道踩的是哪一段。
    text: str


class FeedbackSync(Protocol):
    def submit(self, event: FeedbackEvent) -> None: ...

    async def aclose(self) -> None: ...


class LangSmithFeedbackSync(FeedbackSync):
    TEXT_LIMIT = 500
    CLOSE_TIMEOUT_SECONDS = 10.0

    def __init__(self, client: Any, project_name: str) -> None:
        self._client = client
        self._project_name = project_name
        self._project_id: UUID | None = None
        self._queue: asyncio.Queue[FeedbackEvent] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    def submit(self, event: FeedbackEvent) -> None:
        self._queue.put_nowait(event)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._work(), name="langsmith-feedback")

    async def aclose(self) -> None:
        """进程关停时尽量把排队的反馈发完，超时就放弃：本地表里已经有了。"""
        if self._worker is None:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._queue.join(), self.CLOSE_TIMEOUT_SECONDS)
        self._worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker

    async def _work(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await asyncio.to_thread(self._send, event)
            except Exception:
                logger.exception(
                    "langsmith_feedback_failed",
                    message_id=event.feedback.message_id,
                    trace_id=event.feedback.trace_id,
                )
            finally:
                self._queue.task_done()

    def _send(self, event: FeedbackEvent) -> None:
        feedback = event.feedback
        feedback_id = langsmith_feedback_id(feedback.user_id, feedback.message_id)
        score = 1 if feedback.rating == FeedbackRating.UP else 0
        # 空串而不是 None：update_feedback 跳过 None，踩改成赞时旧理由会留在上面。
        value = ",".join(reason.value for reason in feedback.reasons)
        comment = feedback.comment or ""
        if not event.first:
            try:
                self._client.update_feedback(
                    feedback_id, score=score, value=value, comment=comment
                )
                return
            except Exception:
                # 第一次创建没成功（比如当时 LangSmith 不可达），那边没有可改的，补建一条。
                logger.warning("langsmith_feedback_update_missed", feedback_id=str(feedback_id))
        # 只给 run_id 不给 trace_id：给了 trace_id，SDK 会把反馈丢进它自己的后台批量队列，
        # 失败只在那边打日志，这里的重试和补建就无从谈起。反馈本来就挂在根 run 上，两者相同。
        self._client.create_feedback(
            run_id=feedback.trace_id,
            key=FEEDBACK_KEY,
            score=score,
            value=value,
            comment=comment,
            feedback_id=feedback_id,
            # 不带项目 id 时 SDK 会告警，部分部署直接拒绝：服务端要按项目定位 run。
            session_id=self._project(),
            extra={
                "message_id": feedback.message_id,
                "conversation_id": str(feedback.conversation_id),
                "task_id": event.task_id,
                "text": event.text[: self.TEXT_LIMIT],
            },
        )

    def _project(self) -> UUID:
        if self._project_id is None:
            self._project_id = self._client.read_project(project_name=self._project_name).id
        return self._project_id


def build_feedback_sync(settings: Settings) -> LangSmithFeedbackSync | None:
    """没开 tracing 或没配 key 时不同步：没有 trace 可挂，反馈只留在本地表里。"""
    if not settings.langsmith_tracing or settings.langsmith_api_key is None:
        return None
    client = Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        api_url=settings.langsmith_endpoint,
    )
    return LangSmithFeedbackSync(client, settings.langsmith_project)
