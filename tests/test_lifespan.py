"""应用生命周期的资源管理测试。"""

from typing import Any

import pytest
from fastapi import FastAPI

from enterprise_ai_assistant import main
from enterprise_ai_assistant.core.config import Settings


class FakePool:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeRedis:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeMilvus:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _settings() -> Settings:
    return Settings(
        openai_api_key="test-key",  # type: ignore[arg-type]
        openai_model="test-model",
        openai_embedding_model="test-embedding",
        langsmith_tracing=False,
    )


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> tuple[FakePool, FakeRedis, FakeMilvus]:
    pool, redis, milvus = FakePool(), FakeRedis(), FakeMilvus()

    async def fake_create_pool(dsn: str) -> FakePool:
        del dsn
        return pool

    monkeypatch.setattr(main, "get_settings", _settings)
    monkeypatch.setattr(main, "create_pool", fake_create_pool)
    monkeypatch.setattr(main.Redis, "from_url", classmethod(lambda cls, *a, **k: redis))
    monkeypatch.setattr(main, "MilvusClient", lambda **kwargs: milvus)
    monkeypatch.setattr(main, "build_embeddings", lambda settings: object())
    # 模型客户端与规划链的构造会真实建连/编译 prompt，这里替换成占位对象。
    monkeypatch.setattr(main, "build_chat_model", lambda: object())
    monkeypatch.setattr(main, "LLMPlanningService", lambda model: object())
    return pool, redis, milvus


@pytest.mark.asyncio
async def test_startup_failure_releases_already_opened_resources(
    monkeypatch: pytest.MonkeyPatch, wired: tuple[FakePool, FakeRedis, FakeMilvus]
) -> None:
    """Milvus 初始化失败时，先建立的连接池和 Redis 客户端必须被关闭。"""
    pool, redis, milvus = wired

    async def exploding_bootstrap(client: Any, embeddings: Any) -> None:
        del client, embeddings
        raise RuntimeError("milvus unreachable")

    monkeypatch.setattr(main, "bootstrap_policy_collection", exploding_bootstrap)

    with pytest.raises(RuntimeError, match="milvus unreachable"):
        async with main.lifespan(FastAPI()):
            pass  # pragma: no cover - 启动必然失败

    assert pool.closed is True
    assert redis.closed is True
    assert milvus.closed is True


@pytest.mark.asyncio
async def test_normal_shutdown_releases_all_resources(
    monkeypatch: pytest.MonkeyPatch, wired: tuple[FakePool, FakeRedis, FakeMilvus]
) -> None:
    pool, redis, milvus = wired
    saver_closed = False

    class FakeCheckpointer:
        async def setup(self) -> None:
            return None

    class FakeSaverContext:
        async def __aenter__(self) -> FakeCheckpointer:
            return FakeCheckpointer()

        async def __aexit__(self, *exc: Any) -> None:
            nonlocal saver_closed
            saver_closed = True

    async def noop_bootstrap(client: Any, embeddings: Any) -> None:
        del client, embeddings

    monkeypatch.setattr(main, "bootstrap_policy_collection", noop_bootstrap)
    monkeypatch.setattr(
        main.AsyncPostgresSaver, "from_conn_string", classmethod(lambda cls, dsn: FakeSaverContext())
    )
    monkeypatch.setattr(main, "build_graph", lambda *args: object())

    app = FastAPI()
    async with main.lifespan(app):
        assert app.state.db_pool is pool
        assert pool.closed is False

    assert saver_closed is True
    assert pool.closed is True
    assert redis.closed is True
    assert milvus.closed is True
