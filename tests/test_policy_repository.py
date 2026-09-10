"""制度检索的缓存降级测试。"""

import json
from typing import Any

import pytest

from enterprise_ai_assistant.repositories.policies import CachedMilvusPolicyRepository

HIT = {"title": "差旅制度", "content": "国内差旅须事前审批。", "domain": "travel"}


class FakeRedis:
    def __init__(self, value: str | None = None, broken: bool = False) -> None:
        self.value = value
        self.broken = broken
        self.writes: list[tuple[str, str]] = []

    async def get(self, key: str) -> str | None:
        del key
        if self.broken:
            raise ConnectionError("redis down")
        return self.value

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        del ttl
        if self.broken:
            raise ConnectionError("redis down")
        self.writes.append((key, value))
        return True


class FakeMilvus:
    def __init__(self) -> None:
        self.searches = 0

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        del kwargs
        self.searches += 1
        return [[{"entity": dict(HIT)}]]


class FakeEmbeddings:
    async def aembed_query(self, text: str) -> list[float]:
        del text
        return [0.1, 0.2, 0.3]


def _repository(redis: FakeRedis, milvus: FakeMilvus) -> CachedMilvusPolicyRepository:
    return CachedMilvusPolicyRepository(
        milvus,  # type: ignore[arg-type]
        redis,  # type: ignore[arg-type]
        FakeEmbeddings(),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_cache_hit_skips_vector_search() -> None:
    milvus = FakeMilvus()
    redis = FakeRedis(value=json.dumps([HIT], ensure_ascii=False))

    results = await _repository(redis, milvus).search("住宿标准", "travel")

    assert results == [HIT]
    assert milvus.searches == 0


@pytest.mark.asyncio
async def test_search_falls_back_to_milvus_when_redis_is_down() -> None:
    """Redis 只是加速手段，它不可用时检索必须继续可用，而不是整体失败。"""
    milvus = FakeMilvus()

    results = await _repository(FakeRedis(broken=True), milvus).search("住宿标准", "travel")

    assert results == [HIT]
    assert milvus.searches == 1


@pytest.mark.asyncio
async def test_corrupted_cache_entry_is_ignored() -> None:
    milvus = FakeMilvus()

    results = await _repository(FakeRedis(value="{not-json"), milvus).search("住宿标准", "travel")

    assert results == [HIT]
    assert milvus.searches == 1


@pytest.mark.asyncio
async def test_results_are_cached_after_a_miss() -> None:
    redis = FakeRedis()

    await _repository(redis, FakeMilvus()).search("住宿标准", "travel")

    assert len(redis.writes) == 1
    assert json.loads(redis.writes[0][1]) == [HIT]
