import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

import asyncpg
from pydantic import BaseModel


def _decode_json_object(value: Any) -> dict[str, Any]:
    """兼容 asyncpg 默认返回的 JSON 字符串和自定义 codec 返回的映射。"""
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Mapping):
        raise TypeError("database JSON result must be an object")
    return dict(decoded)


#: 单据号前缀。未登记的写操作退回通用前缀，不会因为漏配就暴露内部键。
_REFERENCE_PREFIXES = {
    "travel_application": "TRV",
    "expense_claim": "EXP",
    "leave_request": "LVE",
    "meeting_booking": "MTG",
}


def build_reference_id(action_type: str, idempotency_key: str) -> str:
    """由幂等键派生对外可见的短单据号。

    幂等键本身是 会话:请求:任务:工具 拼成的，直接当单号给用户看既冗长，又把内部
    结构泄漏出去。这里取它的哈希前缀，同一个幂等键在同一天总得到同一个单号。

    单号只在首次写入时生成并随 result 一起持久化，重放取回的是存下来的那一个，
    所以跨天重试也不会换号。反过来说，不要在别处重新调用本函数去"重算"某张单据的
    号——日期部分会变。要用就从 result 里读。
    """
    digest = hashlib.sha256(idempotency_key.encode()).hexdigest()[:6].upper()
    prefix = _REFERENCE_PREFIXES.get(action_type, "ACT")
    return f"{prefix}-{datetime.now(UTC):%Y%m%d}-{digest}"


class SubmittedAction(BaseModel):
    """一张已提交的单据。只带对外单号，不带幂等键。"""

    reference_id: str
    action_type: str
    payload: dict[str, Any]
    created_at: datetime
    # 最近一次修改的时间；从未改过为 None。
    revised_at: datetime | None = None
    # 撤销时间。撤销只打标记不删行：单号仍然查得到，用户问起时能如实说"已撤销"。
    revoked_at: datetime | None = None


class ActionRepository(Protocol):
    async def execute_once(
        self, *, idempotency_key: str, action_type: str, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def list_submitted(
        self,
        *,
        user_id: str,
        action_type: str,
        reference_id: str | None = None,
        limit: int = 5,
    ) -> list[SubmittedAction]: ...

    async def update_submitted(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        action_type: str,
        reference_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None: ...

    async def revoke_submitted(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        action_type: str,
        reference_id: str,
    ) -> dict[str, Any] | None: ...


class PostgresActionRepository:
    """为企业写操作提供持久且幂等的执行边界。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def execute_once(
        self, *, idempotency_key: str, action_type: str, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        # 该表只记录适配器调用并提供幂等性；它不代表外部企业系统已成功受理。
        result = {
            "reference_id": build_reference_id(action_type, idempotency_key),
            "status": "recorded",
            **payload,
        }
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

    async def list_submitted(
        self,
        *,
        user_id: str,
        action_type: str,
        reference_id: str | None = None,
        limit: int = 5,
    ) -> list[SubmittedAction]:
        # 老数据的 result 里可能没有单号，这些行直接跳过：用户拿不到单号就无从指认，
        # 而退回幂等键会把 会话:请求:任务:工具 的内部结构泄漏出去。
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT result->>'reference_id' AS reference_id, payload, created_at,
                       (result->>'revised_at')::timestamptz AS revised_at,
                       (result->>'revoked_at')::timestamptz AS revoked_at
                FROM workflow_actions
                WHERE user_id = $1 AND action_type = $2
                  AND result ? 'reference_id'
                  AND ($3::text IS NULL OR result->>'reference_id' = $3)
                ORDER BY created_at DESC
                LIMIT $4
                """,
                user_id,
                action_type,
                reference_id,
                limit,
            )
        return [
            SubmittedAction(
                reference_id=row["reference_id"],
                action_type=action_type,
                payload=_decode_json_object(row["payload"]),
                created_at=row["created_at"],
                revised_at=row["revised_at"],
                revoked_at=row["revoked_at"],
            )
            for row in rows
        ]

    async def update_submitted(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        action_type: str,
        reference_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """用完整的新载荷覆盖一张已提交的单据；单据不存在或不属于该用户时返回 None。

        修改本身也是写操作，同样按幂等键记一行 `<action_type>_update`：确认后重放不会
        改两次，审计上也看得出这张单被谁在哪一轮改过。原单据行原地更新而不是另起一行，
        单号的真相只有一处，"我的单据"和记忆派生读到的自然是改后的值。
        """
        async with self._pool.acquire() as connection, connection.transaction():
            replayed = await connection.fetchrow(
                "SELECT result FROM workflow_actions WHERE idempotency_key = $1",
                idempotency_key,
            )
            if replayed is not None:
                return _decode_json_object(replayed["result"])
            original = await connection.fetchrow(
                """
                SELECT idempotency_key, result FROM workflow_actions
                WHERE user_id = $1 AND action_type = $2 AND result->>'reference_id' = $3
                FOR UPDATE
                """,
                user_id,
                action_type,
                reference_id,
            )
            if original is None:
                return None
            previous = _decode_json_object(original["result"])
            result = {
                "reference_id": reference_id,
                "status": previous.get("status", "recorded"),
                **payload,
                "revised_at": datetime.now(UTC).isoformat(),
            }
            await connection.execute(
                """
                UPDATE workflow_actions SET payload = $2::jsonb, result = $3::jsonb
                WHERE idempotency_key = $1
                """,
                original["idempotency_key"],
                json.dumps(payload, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
            )
            await connection.execute(
                """
                INSERT INTO workflow_actions (idempotency_key, action_type, user_id, payload, result)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                """,
                idempotency_key,
                f"{action_type}_update",
                user_id,
                json.dumps({"reference_id": reference_id, **payload}, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
            )
        return result

    async def revoke_submitted(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        action_type: str,
        reference_id: str,
    ) -> dict[str, Any] | None:
        """给单据打上撤销标记；单据不存在或不属于该用户时返回 None。

        和修改一样按幂等键另记一行 `<action_type>_revoke`。已经撤销过的单据原样返回，
        不刷新撤销时间：重复撤销不是新的业务事实。
        """
        async with self._pool.acquire() as connection, connection.transaction():
            replayed = await connection.fetchrow(
                "SELECT result FROM workflow_actions WHERE idempotency_key = $1",
                idempotency_key,
            )
            if replayed is not None:
                return _decode_json_object(replayed["result"])
            original = await connection.fetchrow(
                """
                SELECT idempotency_key, result FROM workflow_actions
                WHERE user_id = $1 AND action_type = $2 AND result->>'reference_id' = $3
                FOR UPDATE
                """,
                user_id,
                action_type,
                reference_id,
            )
            if original is None:
                return None
            result = _decode_json_object(original["result"])
            if "revoked_at" in result:
                return result
            result["revoked_at"] = datetime.now(UTC).isoformat()
            await connection.execute(
                "UPDATE workflow_actions SET result = $2::jsonb WHERE idempotency_key = $1",
                original["idempotency_key"],
                json.dumps(result, ensure_ascii=False),
            )
            await connection.execute(
                """
                INSERT INTO workflow_actions (idempotency_key, action_type, user_id, payload, result)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                """,
                idempotency_key,
                f"{action_type}_revoke",
                user_id,
                json.dumps({"reference_id": reference_id}, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
            )
        return result


#: 内存实现把元数据和载荷摊平在同一个字典里，取载荷时要剔掉这些键。
_RECORD_METADATA = frozenset(
    {"reference_id", "status", "action_type", "user_id", "revised_at", "revoked_at"}
)


class InMemoryActionRepository:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.created_at: dict[str, datetime] = {}
        self.updates: dict[str, dict[str, Any]] = {}

    async def execute_once(
        self, *, idempotency_key: str, action_type: str, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if idempotency_key not in self.records:
            self.records[idempotency_key] = {
                "reference_id": build_reference_id(action_type, idempotency_key),
                "status": "recorded",
                "action_type": action_type,
                "user_id": user_id,
                **payload,
            }
            self.created_at[idempotency_key] = datetime.now(UTC)
        return self.records[idempotency_key]

    async def list_submitted(
        self,
        *,
        user_id: str,
        action_type: str,
        reference_id: str | None = None,
        limit: int = 5,
    ) -> list[SubmittedAction]:
        matched = [
            SubmittedAction(
                reference_id=str(record["reference_id"]),
                action_type=action_type,
                payload={
                    name: value
                    for name, value in record.items()
                    if name not in _RECORD_METADATA
                },
                created_at=self.created_at[key],
                revised_at=record.get("revised_at"),
                revoked_at=record.get("revoked_at"),
            )
            for key, record in self.records.items()
            if record["user_id"] == user_id
            and record["action_type"] == action_type
            and reference_id in (None, record["reference_id"])
        ]
        matched.sort(key=lambda item: item.created_at, reverse=True)
        return matched[:limit]

    async def update_submitted(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        action_type: str,
        reference_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        if idempotency_key in self.updates:
            return self.updates[idempotency_key]
        record = self._find(user_id, action_type, reference_id)
        if record is None:
            return None
        metadata = {
            name: record[name]
            for name in ("reference_id", "status", "action_type", "user_id", "revoked_at")
            if name in record
        }
        record.clear()
        record.update({**metadata, **payload, "revised_at": datetime.now(UTC)})
        result = self._result(record)
        self.updates[idempotency_key] = result
        return result

    def _find(self, user_id: str, action_type: str, reference_id: str) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self.records.values()
                if item["user_id"] == user_id
                and item["action_type"] == action_type
                and item["reference_id"] == reference_id
            ),
            None,
        )

    @staticmethod
    def _result(record: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value.isoformat() if isinstance(value, datetime) else value
            for name, value in record.items()
            if name not in {"action_type", "user_id"}
        }

    async def revoke_submitted(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        action_type: str,
        reference_id: str,
    ) -> dict[str, Any] | None:
        if idempotency_key in self.updates:
            return self.updates[idempotency_key]
        record = self._find(user_id, action_type, reference_id)
        if record is None:
            return None
        record.setdefault("revoked_at", datetime.now(UTC))
        result = self._result(record)
        self.updates[idempotency_key] = result
        return result
