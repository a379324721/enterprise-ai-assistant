"""应用生命周期的资源管理测试。"""

from types import SimpleNamespace
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


async def _setup() -> None:
    return None


class FakeCheckpointPool:
    def __init__(self) -> None:
        self.opened = False
        self.closed = False

    async def open(self, wait: bool = False) -> None:
        del wait
        self.opened = True

    async def close(self) -> None:
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
    monkeypatch.setattr(main, "build_chat_model", lambda role="domain": object())
    monkeypatch.setattr(main, "LLMPlanningService", lambda model: object())
    return pool, redis, milvus


async def _noop_bootstrap(client: Any, embeddings: Any) -> None:
    del client, embeddings


@pytest.mark.asyncio
async def test_startup_failure_releases_already_opened_resources(
    monkeypatch: pytest.MonkeyPatch, wired: tuple[FakePool, FakeRedis, FakeMilvus]
) -> None:
    """检查点连接池建立失败时，先建立的连接池和 Redis 客户端必须被关闭。"""
    pool, redis, milvus = wired

    class ExplodingCheckpointPool(FakeCheckpointPool):
        async def open(self, wait: bool = False) -> None:
            raise RuntimeError("checkpoint database unreachable")

    monkeypatch.setattr(main, "bootstrap_policy_collection", _noop_bootstrap)
    monkeypatch.setattr(main, "AsyncConnectionPool", lambda *a, **k: ExplodingCheckpointPool())

    with pytest.raises(RuntimeError, match="checkpoint database unreachable"):
        async with main.lifespan(FastAPI()):
            pass  # pragma: no cover - 启动必然失败

    assert pool.closed is True
    assert redis.closed is True
    assert milvus.closed is True


@pytest.mark.asyncio
async def test_policy_bootstrap_failure_does_not_block_startup(
    monkeypatch: pytest.MonkeyPatch, wired: tuple[FakePool, FakeRedis, FakeMilvus]
) -> None:
    """制度语料初始化失败只降级检索能力，不应让整个服务起不来。"""
    checkpoint_pool = FakeCheckpointPool()

    async def exploding_bootstrap(client: Any, embeddings: Any) -> None:
        del client, embeddings
        raise RuntimeError("milvus unreachable")

    monkeypatch.setattr(main, "bootstrap_policy_collection", exploding_bootstrap)
    monkeypatch.setattr(main, "AsyncConnectionPool", lambda *a, **k: checkpoint_pool)
    monkeypatch.setattr(main, "AsyncPostgresSaver", lambda conn, **_: SimpleNamespace(setup=_setup))
    monkeypatch.setattr(main, "build_graph", lambda *args: object())

    app = FastAPI()
    async with main.lifespan(app):
        assert app.state.graph is not None

    assert checkpoint_pool.closed is True


@pytest.mark.asyncio
async def test_normal_shutdown_releases_all_resources(
    monkeypatch: pytest.MonkeyPatch, wired: tuple[FakePool, FakeRedis, FakeMilvus]
) -> None:
    pool, redis, milvus = wired
    checkpoint_pool = FakeCheckpointPool()

    class FakeCheckpointer:
        def __init__(self, conn: Any, **kwargs: Any) -> None:
            self.conn = conn

        async def setup(self) -> None:
            return None

    monkeypatch.setattr(main, "bootstrap_policy_collection", _noop_bootstrap)
    monkeypatch.setattr(main, "AsyncConnectionPool", lambda *a, **k: checkpoint_pool)
    monkeypatch.setattr(main, "AsyncPostgresSaver", FakeCheckpointer)
    monkeypatch.setattr(main, "build_graph", lambda *args: object())

    app = FastAPI()
    async with main.lifespan(app):
        assert app.state.db_pool is pool
        assert checkpoint_pool.opened is True
        assert pool.closed is False

    assert checkpoint_pool.closed is True
    assert pool.closed is True
    assert redis.closed is True
    assert milvus.closed is True
