from typing import Protocol
from uuid import UUID

import asyncpg


class ConversationRepository(Protocol):
    async def upsert(
        self, *, conversation_id: UUID, user_id: str, title: str
    ) -> None: ...

    async def touch(self, *, conversation_id: UUID, user_id: str) -> None: ...

    async def ensure(
        self, *, conversation_id: UUID, user_id: str, title: str
    ) -> None: ...

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict[str, object]]: ...

    async def belongs_to_user(self, *, conversation_id: UUID, user_id: str) -> bool: ...

    async def delete(self, *, conversation_id: UUID, user_id: str) -> bool: ...


class PostgresConversationRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert(self, *, conversation_id: UUID, user_id: str, title: str) -> None:
        await self._pool.execute(
            """
            INSERT INTO assistant_conversations (conversation_id, user_id, title)
            VALUES ($1, $2, $3)
            ON CONFLICT (conversation_id) DO UPDATE
              SET updated_at = now()
            WHERE assistant_conversations.user_id = EXCLUDED.user_id
            """,
            conversation_id,
            user_id,
            title,
        )

    async def touch(self, *, conversation_id: UUID, user_id: str) -> None:
        await self._pool.execute(
            """
            UPDATE assistant_conversations
               SET updated_at = now()
             WHERE conversation_id = $1 AND user_id = $2
            """,
            conversation_id,
            user_id,
        )

    async def ensure(self, *, conversation_id: UUID, user_id: str, title: str) -> None:
        await self._pool.execute(
            """
            INSERT INTO assistant_conversations (conversation_id, user_id, title)
            VALUES ($1, $2, $3)
            ON CONFLICT (conversation_id) DO NOTHING
            """,
            conversation_id,
            user_id,
            title,
        )

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict[str, object]]:
        rows = await self._pool.fetch(
            """
            SELECT conversation_id, title, created_at, updated_at
              FROM assistant_conversations
             WHERE user_id = $1
             ORDER BY updated_at DESC
             LIMIT $2
            """,
            user_id,
            limit,
        )
        return [dict(row) for row in rows]

    async def belongs_to_user(self, *, conversation_id: UUID, user_id: str) -> bool:
        return bool(
            await self._pool.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                      FROM assistant_conversations
                     WHERE conversation_id = $1 AND user_id = $2
                )
                """,
                conversation_id,
                user_id,
            )
        )

    async def delete(self, *, conversation_id: UUID, user_id: str) -> bool:
        result = await self._pool.execute(
            """
            DELETE FROM assistant_conversations
             WHERE conversation_id = $1 AND user_id = $2
            """,
            conversation_id,
            user_id,
        )
        return result == "DELETE 1"
