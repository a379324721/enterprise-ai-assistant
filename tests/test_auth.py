"""HTTP 层的身份校验测试。"""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.core.runs import MemoryStreamBridge, RunManager
from enterprise_ai_assistant.core.security import create_access_token
from enterprise_ai_assistant.main import create_app

CONVERSATION_ID = uuid4()


def _settings(**overrides: Any) -> Settings:
    # 显式写死每一项被断言的配置：Settings 未显式传参的字段会回落到 .env，
    # 开发机上 DEV_LOGIN_ENABLED=true 会让"默认关闭"的断言在本地失败、CI 通过。
    defaults: dict[str, Any] = {
        "openai_api_key": "test-key",
        "openai_model": "test-model",
        "openai_embedding_model": "test-embedding",
        "langsmith_tracing": False,
        "jwt_secret": "unit-test-secret-" + "x" * 32,
        "app_env": "development",
        "dev_login_enabled": False,
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


SETTINGS = _settings()


class FakeGraph:
    """只回放一个属于 owner-user 的既有会话。"""

    async def aget_state(self, config: dict[str, Any]) -> Any:
        del config
        return SimpleNamespace(
            values={
                "user_id": "owner-user",
                "last_answer": "已完成",
                "user_goal": "测试",
                "tasks": [],
                "artifacts": {},
                "tool_results": [],
            },
            next=(),
            interrupts=(),
        )


@asynccontextmanager
async def _fake_lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.graph = FakeGraph()
    app.state.logger = SimpleNamespace(
        info=lambda *a, **k: None, exception=lambda *a, **k: None
    )
    app.state.runs = RunManager(MemoryStreamBridge(), app.state.logger)
    yield
    await app.state.runs.aclose()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(routes, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: SETTINGS)
    with TestClient(create_app(_fake_lifespan)) as test_client:
        yield test_client


def _auth(user_id: str, **kwargs: Any) -> dict[str, str]:
    token, _ = create_access_token(user_id, SETTINGS, **kwargs)
    return {"Authorization": f"Bearer {token}"}


def test_request_without_token_is_rejected(client: TestClient) -> None:
    response = client.get(f"/api/v1/conversations/{CONVERSATION_ID}")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_spoofed_user_header_is_ignored(client: TestClient) -> None:
    """旧的 X-User-ID 头不再具备任何权限含义。"""
    response = client.get(
        f"/api/v1/conversations/{CONVERSATION_ID}", headers={"X-User-ID": "owner-user"}
    )

    assert response.status_code == 401


def test_token_signed_with_another_key_is_rejected(client: TestClient) -> None:
    foreign, _ = create_access_token("owner-user", _settings(jwt_secret="another-secret-" + "y" * 32))

    response = client.get(
        f"/api/v1/conversations/{CONVERSATION_ID}",
        headers={"Authorization": f"Bearer {foreign}"},
    )

    assert response.status_code == 401


def test_expired_token_is_rejected(client: TestClient) -> None:
    response = client.get(
        f"/api/v1/conversations/{CONVERSATION_ID}",
        headers=_auth("owner-user", ttl=timedelta(seconds=-1)),
    )

    assert response.status_code == 401


def test_malformed_authorization_scheme_is_rejected(client: TestClient) -> None:
    token, _ = create_access_token("owner-user", SETTINGS)

    response = client.get(
        f"/api/v1/conversations/{CONVERSATION_ID}",
        headers={"Authorization": f"Basic {token}"},
    )

    assert response.status_code == 401


def test_valid_token_grants_access_to_own_conversation(client: TestClient) -> None:
    response = client.get(
        f"/api/v1/conversations/{CONVERSATION_ID}", headers=_auth("owner-user")
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "已完成"


def test_valid_token_cannot_read_another_users_conversation(client: TestClient) -> None:
    response = client.get(
        f"/api/v1/conversations/{CONVERSATION_ID}", headers=_auth("intruder")
    )

    assert response.status_code == 404


def test_health_endpoint_stays_public(client: TestClient) -> None:
    """健康检查供编排器探活，不应要求令牌。"""
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_dev_token_endpoint_is_hidden_by_default(client: TestClient) -> None:
    response = client.post("/api/v1/auth/dev-token", json={"user_id": "demo-user"})

    assert response.status_code == 404


def test_dev_token_endpoint_issues_usable_token(monkeypatch: pytest.MonkeyPatch) -> None:
    enabled = _settings(dev_login_enabled=True)
    monkeypatch.setattr(routes, "get_settings", lambda: enabled)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: enabled)

    with TestClient(create_app(_fake_lifespan)) as client:
        issued = client.post("/api/v1/auth/dev-token", json={"user_id": "owner-user"})
        assert issued.status_code == 200
        body = issued.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0

        response = client.get(
            f"/api/v1/conversations/{CONVERSATION_ID}",
            headers={"Authorization": f"Bearer {body['access_token']}"},
        )

    assert response.status_code == 200


def test_dev_login_cannot_be_enabled_outside_development() -> None:
    with pytest.raises(ValueError, match="DEV_LOGIN_ENABLED"):
        _settings(app_env="production", dev_login_enabled=True)


def test_jwt_secret_is_required_outside_development() -> None:
    with pytest.raises(ValueError, match="JWT_SECRET"):
        _settings(app_env="production", jwt_secret=None)
