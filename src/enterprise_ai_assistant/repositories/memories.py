"""跨会话记忆的读写边界。

记忆分成两层，来源不同：

- `profile` / `preference` 存在 `user_memories` 表，是会话里说出来、但业务系统里没有
  的稳定属性（常驻城市、成本中心、交通偏好）。同一 `(user_id, kind, key)` 覆盖写。
- 近期业务事实（差旅单号、报销单号）不复制到记忆表，直接从 `workflow_actions` 派生。
  那张表已经是写操作的幂等与审计记录，单号的真相只应该有一处；复制一份会在单据
  作废或改期后留下无法失效的旧值。

派生摘要走字段白名单：请假原因、票据号、备注正文这类一次性或敏感内容不进摘要，
它们对后续任务没有复用价值，却会被原样送进模型上下文。
"""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

import asyncpg

from enterprise_ai_assistant.core.models import (
    MemoryCandidate,
    MemoryKind,
    MemoryRecord,
    RecentAction,
)

#: 每类写操作允许进入摘要的字段。未列出的字段一律丢弃。
_ACTION_SUMMARY_FIELDS: dict[str, tuple[str, ...]] = {
    "travel_application": ("destination", "start_date", "end_date"),
    "expense_claim": ("expense_type", "amount", "currency"),
    "leave_request": ("leave_type", "start_date", "end_date"),
}


def _decode_json_object(value: Any) -> dict[str, Any]:
    decoded = json.loads(value) if isinstance(value, str) else value
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def summarize_action(action_type: str, payload: Mapping[str, Any]) -> str:
    """按白名单把写操作载荷压成一行摘要。"""
    fields = _ACTION_SUMMARY_FIELDS.get(action_type)
    if not fields:
        return ""
    parts = [
        f"{name}={payload[name]}"
        for name in fields
        if payload.get(name) not in (None, "", [])
    ]
    return " ".join(parts)


class MemoryRepository(Protocol):
    async def list_memories(self, user_id: str, limit: int) -> list[MemoryRecord]: ...

    async def recent_actions(self, user_id: str, limit: int) -> list[RecentAction]: ...

    async def upsert(
        self,
        user_id: str,
        candidates: Sequence[MemoryCandidate],
        *,
        source_conversation_id: UUID | None = None,
    ) -> None: ...

    async def delete(self, user_id: str, memory_id: UUID) -> bool: ...


class PostgresMemoryRepository:
    """用 Postgres 持久化用户画像，并从写操作审计记录派生近期业务事实。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def list_memories(self, user_id: str, limit: int) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT id, kind, key, value, source_conversation_id, updated_at
                FROM user_memories
                WHERE user_id = $1
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
        return [MemoryRecord.model_validate(dict(row)) for row in rows]

    async def recent_actions(self, user_id: str, limit: int) -> list[RecentAction]:
        if limit <= 0:
            return []
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT idempotency_key, action_type, payload, created_at
                FROM workflow_actions
                WHERE user_id = $1 AND action_type = ANY($2::text[])
                ORDER BY created_at DESC
                LIMIT $3
                """,
                user_id,
                list(_ACTION_SUMMARY_FIELDS),
                limit,
            )
        actions: list[RecentAction] = []
        for row in rows:
            summary = summarize_action(
                row["action_type"], _decode_json_object(row["payload"])
            )
            if not summary:
                continue
            actions.append(
                RecentAction(
                    reference_id=row["idempotency_key"],
                    action_type=row["action_type"],
                    summary=summary,
                    created_at=row["created_at"],
                )
            )
        return actions

    async def upsert(
        self,
        user_id: str,
        candidates: Sequence[MemoryCandidate],
        *,
        source_conversation_id: UUID | None = None,
    ) -> None:
        if not candidates:
            return
        async with self._pool.acquire() as connection:
            await connection.executemany(
                """
                INSERT INTO user_memories
                    (id, user_id, kind, key, value, source_conversation_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, kind, key) DO UPDATE
                  SET value = EXCLUDED.value,
                      source_conversation_id = EXCLUDED.source_conversation_id,
                      updated_at = now()
                """,
                [
                    (
                        uuid4(),
                        user_id,
                        candidate.kind.value,
                        candidate.key,
                        candidate.value,
                        source_conversation_id,
                    )
                    for candidate in candidates
                ],
            )

    async def delete(self, user_id: str, memory_id: UUID) -> bool:
        async with self._pool.acquire() as connection:
            result: str = await connection.execute(
                "DELETE FROM user_memories WHERE user_id = $1 AND id = $2",
                user_id,
                memory_id,
            )
        # asyncpg 的 DELETE 返回 "DELETE <行数>"；0 行说明该记忆不属于此用户或已删除。
        return result.endswith(" 1")


class InMemoryMemoryRepository:
    """测试与本地评测用的等价实现。"""

    def __init__(self) -> None:
        self.records: dict[tuple[str, MemoryKind, str], MemoryRecord] = {}
        self.actions: dict[str, list[RecentAction]] = {}

    async def list_memories(self, user_id: str, limit: int) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        owned = [
            record for (owner, _, _), record in self.records.items() if owner == user_id
        ]
        owned.sort(key=lambda record: record.updated_at, reverse=True)
        return owned[:limit]

    async def recent_actions(self, user_id: str, limit: int) -> list[RecentAction]:
        if limit <= 0:
            return []
        return self.actions.get(user_id, [])[:limit]

    async def upsert(
        self,
        user_id: str,
        candidates: Sequence[MemoryCandidate],
        *,
        source_conversation_id: UUID | None = None,
    ) -> None:
        for candidate in candidates:
            key = (user_id, candidate.kind, candidate.key)
            existing = self.records.get(key)
            self.records[key] = MemoryRecord(
                id=existing.id if existing else uuid4(),
                kind=candidate.kind,
                key=candidate.key,
                value=candidate.value,
                source_conversation_id=source_conversation_id,
                updated_at=datetime.now().astimezone(),
            )

    async def delete(self, user_id: str, memory_id: UUID) -> bool:
        for key, record in list(self.records.items()):
            if key[0] == user_id and record.id == memory_id:
                del self.records[key]
                return True
        return False
