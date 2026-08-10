import json

import pytest

from enterprise_ai_assistant.repositories.actions import PostgresActionRepository


class FakeConnection:
    def __init__(self, stored_result: object) -> None:
        self.stored_result = stored_result

    async def fetchrow(self, query: str, *args: object) -> dict[str, object]:
        assert "INSERT INTO workflow_actions" in query
        return {"result": self.stored_result}


class FakeAcquire:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, *args: object) -> None:
        return None


class FakePool:
    def __init__(self, stored_result: object) -> None:
        self.connection = FakeConnection(stored_result)

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
