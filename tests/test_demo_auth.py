"""演示用的名字注册/登录。

这套接口没有凭据，输入名字即取得该身份，所以测试的重点是它在非开发环境和未开启
开关时必须彻底消失，以及名字规范化不会把同一个人拆成两个身份。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.core.security import decode_identity
from enterprise_ai_assistant.main import create_app
from enterprise_ai_assistant.repositories.users import (
    InMemoryDemoUserRepository,
    conversation_id_for,
    normalize_name,
)


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "openai_api_key": "test-key",
        "openai_model": "test-model",
        "openai_embedding_model": "test-embedding",
        "langsmith_tracing": False,
        "jwt_secret": "unit-test-secret-" + "x" * 32,
        "app_env": "development",
        "dev_login_enabled": False,
        "demo_login_enabled": True,
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


SETTINGS = _settings()


@asynccontextmanager
async def _noop_lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    yield


def _app(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: settings)
    application = create_app(_noop_lifespan)
    # ASGITransport 不会执行 lifespan，依赖项直接挂到 app.state 上。
    application.state.demo_users = InMemoryDemoUserRepository()
    application.state.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None
    )
    return application


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    app = _app(SETTINGS, monkeypatch)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        yield async_client


def test_name_normalization_folds_incidental_whitespace() -> None:
    """否则"张三"和"张 三 "会各自持有互不可见的会话与记忆。"""
    assert normalize_name("  张三  ") == "张三"
    assert normalize_name("Wang  Ning") == "Wang Ning"


def test_conversation_id_is_derived_deterministically() -> None:
    """演示用户固定一个会话：换设备或清了本地存储也要回到同一个 thread。"""
    assert conversation_id_for("张三") == conversation_id_for("张三")
    assert conversation_id_for("张三") != conversation_id_for("李四")


@pytest.mark.asyncio
async def test_register_then_login_returns_the_same_identity(client: AsyncClient) -> None:
    registered = await client.post("/api/v1/auth/register", json={"name": "  王宁 "})
    assert registered.status_code == 201
    first = registered.json()
    assert first["created"] is True
    assert first["user_id"] == "王宁"

    again = await client.post("/api/v1/auth/login", json={"name": "王宁"})
    assert again.status_code == 200
    second = again.json()
    assert second["created"] is False
    assert second["conversation_id"] == first["conversation_id"]


@pytest.mark.asyncio
async def test_token_carries_the_display_name(client: AsyncClient) -> None:
    """称呼走令牌而不是长期记忆：姓名是"永远相关"的身份信息，不该被相关性筛选掉。"""
    response = await client.post("/api/v1/auth/register", json={"name": "王宁"})

    identity = decode_identity(response.json()["access_token"], SETTINGS)

    assert identity.user_id == "王宁"
    assert identity.display_name == "王宁"


@pytest.mark.asyncio
async def test_duplicate_registration_is_rejected(client: AsyncClient) -> None:
    await client.post("/api/v1/auth/register", json={"name": "王宁"})

    duplicate = await client.post("/api/v1/auth/register", json={"name": "王宁"})

    assert duplicate.status_code == 409


@pytest.mark.asyncio
async def test_login_before_registering_is_rejected(client: AsyncClient) -> None:
    response = await client.post("/api/v1/auth/login", json={"name": "查无此人"})

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_endpoints_disappear_when_the_switch_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关掉开关后不能只是拒绝，要连接口存在的事实一起藏起来。"""
    app = _app(_settings(demo_login_enabled=False), monkeypatch)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        registered = await client.post("/api/v1/auth/register", json={"name": "王宁"})
        logged_in = await client.post("/api/v1/auth/login", json={"name": "王宁"})

    assert registered.status_code == 404
    assert logged_in.status_code == 404


def test_demo_login_cannot_be_enabled_outside_development() -> None:
    """配置层就要拦住：这套接口一旦上生产，等于开放任意身份冒用。"""
    with pytest.raises(ValueError, match="DEMO_LOGIN_ENABLED"):
        _settings(app_env="production")
