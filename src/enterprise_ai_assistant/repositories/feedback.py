"""用户对助手回答的点赞点踩。

界面显示"点过没有"以这张表为准，不回头去 LangSmith 查：那边可能没配、写失败、
或者 trace 已过保留期，查不到时界面分不清是没点过还是点过但丢了。LangSmith 上的
那份只用于筛 badcase。
"""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol
from uuid import UUID

import asyncpg


class FeedbackRating(StrEnum):
    UP = "up"
    DOWN = "down"


class FeedbackReason(StrEnum):
    """点踩的理由，按领域 Agent 的原则划分，而不是按具体 badcase 列。"""

    # 说了工具没查到的事实、谎称办过。
    FABRICATED = "fabricated"
    # 没理解用户要办什么。
    MISUNDERSTOOD = "misunderstood"
    # 字段取值不是用户说的。
    WRONG_FIELDS = "wrong_fields"
    # 做了用户没要求的事，或者超出了当前任务。
    OVERREACH = "overreach"
    OTHER = "other"


@dataclass(frozen=True)
class MessageFeedback:
    user_id: str
    conversation_id: UUID
    message_id: str
    trace_id: str
    rating: FeedbackRating
    reasons: list[FeedbackReason] = field(default_factory=list)
    comment: str | None = None


class FeedbackRepository(Protocol):
    async def save(self, feedback: MessageFeedback) -> bool:
        """覆盖写同一个用户对同一条消息的反馈；返回 True 表示这是第一次评价。"""
        ...

    async def ratings(self, user_id: str, message_ids: list[str]) -> dict[str, FeedbackRating]: ...


class PostgresFeedbackRepository(FeedbackRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, feedback: MessageFeedback) -> bool:
        async with self._pool.acquire() as connection:
            # xmax = 0 说明这一行是本次插入的，不是冲突后更新的：用一条语句区分首次评价和
            # 改主意，先查再写在并发点击时两边都会以为是第一次。
            inserted = await connection.fetchval(
                """
                INSERT INTO message_feedback
                    (user_id, message_id, conversation_id, trace_id, rating, reasons, comment)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                ON CONFLICT (user_id, message_id) DO UPDATE
                  SET rating = EXCLUDED.rating,
                      reasons = EXCLUDED.reasons,
                      comment = EXCLUDED.comment,
                      updated_at = now()
                RETURNING xmax = 0
                """,
                feedback.user_id,
                feedback.message_id,
                feedback.conversation_id,
                feedback.trace_id,
                feedback.rating.value,
                json.dumps([reason.value for reason in feedback.reasons]),
                feedback.comment,
            )
        return bool(inserted)

    async def ratings(self, user_id: str, message_ids: list[str]) -> dict[str, FeedbackRating]:
        if not message_ids:
            return {}
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT message_id, rating FROM message_feedback
                WHERE user_id = $1 AND message_id = ANY($2::text[])
                """,
                user_id,
                message_ids,
            )
        return {row["message_id"]: FeedbackRating(row["rating"]) for row in rows}


class InMemoryFeedbackRepository(FeedbackRepository):
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], MessageFeedback] = {}

    async def save(self, feedback: MessageFeedback) -> bool:
        key = (feedback.user_id, feedback.message_id)
        first = key not in self.items
        self.items[key] = feedback
        return first

    async def ratings(self, user_id: str, message_ids: list[str]) -> dict[str, FeedbackRating]:
        return {
            message_id: self.items[(user_id, message_id)].rating
            for message_id in message_ids
            if (user_id, message_id) in self.items
        }
