import json

import pytest

from enterprise_ai_assistant.repositories.actions import (
    InMemoryActionRepository,
    PostgresActionRepository,
)


class FakeConnection:
    def __init__(
        self,
        stored_result: object,
        *,
        action_type: str = "travel",
        user_id: str = "u-1",
        payload: object = '{"destination": "北京"}',
    ) -> None:
        self.stored_result = stored_result
        self.action_type = action_type
        self.user_id = user_id
        self.payload = payload

    async def fetchrow(self, query: str, *args: object) -> dict[str, object]:
        assert "INSERT INTO workflow_actions" in query
        return {
            "action_type": self.action_type,
            "user_id": self.user_id,
            "payload": self.payload,
            "result": self.stored_result,
        }


class FakeAcquire:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, *args: object) -> None:
        return None


class FakePool:
    def __init__(self, stored_result: object, **kwargs: object) -> None:
        self.connection = FakeConnection(stored_result, **kwargs)

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self.connection)


@pytest.mark.parametrize(
    "stored_result",
    [
        {"reference_id": "u-1:task-1", "status": "submitted"},
        json.dumps({"reference_id": "u-1:task-1", "status": "submitted"}),
    ],
)
@pytest.mark.asyncio
async def test_postgres_action_repository_decodes_json_result(stored_result: object) -> None:
    repository = PostgresActionRepository(FakePool(stored_result))  # type: ignore[arg-type]

    result = await repository.execute_once(
        idempotency_key="u-1:task-1",
        action_type="travel",
        user_id="u-1",
        payload={"destination": "北京"},
    )

    assert result == {"reference_id": "u-1:task-1", "status": "submitted"}


@pytest.mark.asyncio
async def test_postgres_action_repository_rejects_idempotency_collision() -> None:
    repository = PostgresActionRepository(
        FakePool(
            {"reference_id": "old", "status": "submitted"},
            action_type="travel",
            payload={"destination": "上海"},
        )
    )  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="different action"):
        await repository.execute_once(
            idempotency_key="same-key",
            action_type="leave",
            user_id="u-1",
            payload={"leave_start": "2026-08-21"},
        )


@pytest.mark.asyncio
async def test_public_reference_does_not_expose_idempotency_key() -> None:
    repository = InMemoryActionRepository()

    result = await repository.execute_once(
        idempotency_key="hr.leave.submit:confirmation-secret",
        action_type="leave",
        user_id="demo-user",
        payload={"leave_start": "2026-08-21"},
    )

    assert result["reference_id"].startswith("LEAVE-")
    assert "demo-user" not in result["reference_id"]
    assert "confirmation-secret" not in result["reference_id"]
