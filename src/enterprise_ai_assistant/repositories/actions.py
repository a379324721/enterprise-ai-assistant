import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

import asyncpg


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


class InMemoryActionRepository:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

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
        return self.records[idempotency_key]
