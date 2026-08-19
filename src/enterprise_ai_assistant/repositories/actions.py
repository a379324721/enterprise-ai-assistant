import json
from collections.abc import Mapping
from typing import Any, Protocol

import asyncpg


def _decode_json_object(value: Any) -> dict[str, Any]:
    """兼容 asyncpg 默认返回的 JSON 字符串和自定义 codec 返回的映射。"""
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Mapping):
        raise TypeError("database JSON result must be an object")
    return dict(decoded)


class ActionRepository(Protocol):
    async def execute_once(
        self, *, idempotency_key: str, action_type: str, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...


class PostgresActionRepository:
    """为企业写操作提供持久且幂等的执行边界。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def execute_once(
        self, *, idempotency_key: str, action_type: str, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        # 该表只记录适配器调用并提供幂等性；它不代表外部企业系统已成功受理。
        result = {"reference_id": idempotency_key, "status": "recorded", **payload}
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO workflow_actions (idempotency_key, action_type, user_id, payload, result)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                ON CONFLICT (idempotency_key) DO UPDATE
                  SET idempotency_key = EXCLUDED.idempotency_key
                RETURNING result
                """,
                idempotency_key,
                action_type,
                user_id,
                json.dumps(payload, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
            )
        return _decode_json_object(row["result"])


class InMemoryActionRepository:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    async def execute_once(
        self, *, idempotency_key: str, action_type: str, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if idempotency_key not in self.records:
            self.records[idempotency_key] = {
                "reference_id": idempotency_key,
                "status": "recorded",
                "action_type": action_type,
                "user_id": user_id,
                **payload,
            }
        return self.records[idempotency_key]
