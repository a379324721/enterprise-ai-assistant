import json
from hashlib import sha256
from typing import Any, Protocol

import asyncpg


class ActionRepository(Protocol):
    async def execute_once(
        self, *, idempotency_key: str, action_type: str, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...


def _reference_id(action_type: str, idempotency_key: str) -> str:
    prefix = {"travel": "TRAVEL", "expense": "EXPENSE", "leave": "LEAVE"}.get(
        action_type, "ACTION"
    )
    digest = sha256(idempotency_key.encode()).hexdigest()[:12].upper()
    return f"{prefix}-{digest}"


def _json_object(value: object, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"workflow action {field} must be a JSON object")
    return value


class PostgresActionRepository:
    """为企业写操作提供持久且幂等的执行边界。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def execute_once(
        self, *, idempotency_key: str, action_type: str, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        result = {
            "reference_id": _reference_id(action_type, idempotency_key),
            "status": "submitted",
            **payload,
        }
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO workflow_actions (idempotency_key, action_type, user_id, payload, result)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                ON CONFLICT (idempotency_key) DO UPDATE
                  SET idempotency_key = EXCLUDED.idempotency_key
                RETURNING action_type, user_id, payload, result
                """,
                idempotency_key,
                action_type,
                user_id,
                json.dumps(payload, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
            )
        stored_payload = _json_object(row["payload"], "payload")
        if (
            row["action_type"] != action_type
            or row["user_id"] != user_id
            or stored_payload != payload
        ):
            raise RuntimeError("idempotency key was already used for a different action")
        stored_result = _json_object(row["result"], "result")
        return stored_result


class InMemoryActionRepository:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    async def execute_once(
        self, *, idempotency_key: str, action_type: str, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if idempotency_key not in self.records:
            self.records[idempotency_key] = {
                "reference_id": _reference_id(action_type, idempotency_key),
                "status": "submitted",
                "action_type": action_type,
                "user_id": user_id,
                **payload,
            }
        stored = self.records[idempotency_key]
        stored_payload = {
            key: value
            for key, value in stored.items()
            if key not in {"reference_id", "status", "action_type", "user_id"}
        }
        if (
            stored["action_type"] != action_type
            or stored["user_id"] != user_id
            or stored_payload != payload
        ):
            raise RuntimeError("idempotency key was already used for a different action")
        return stored
