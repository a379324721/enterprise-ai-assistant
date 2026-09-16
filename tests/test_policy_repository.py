"""制度检索的缓存降级，以及启动时语料同步进 Milvus。"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pymilvus import MilvusClient

from enterprise_ai_assistant.repositories.policies import (
    CachedMilvusPolicyRepository,
    InMemoryPolicyRepository,
    PolicyDocument,
    bootstrap_policy_collection,
)

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


async def test_in_memory_search_ranks_the_clause_that_matches_the_question() -> None:
    # 语料按条款拆开后领域里不止一条，评测问住宿标准时要拿到住宿那条，而不是领域的前几条。
    results = await InMemoryPolicyRepository().search("出差住宿每晚标准", "travel", limit=1)

    assert results[0]["title"] == "差旅制度·住宿标准"


class CountingEmbeddings:
    """按文本给出确定的向量，并记下每次被要求向量化的文本。"""

    def __init__(self, model: str = "embed-a", dimension: int = 4) -> None:
        self.model = model
        self.dimension = dimension
        self.embedded: list[str] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[float(len(text) % 7 + 1)] + [0.5] * (self.dimension - 1) for text in texts]


def _document(id: int, content: str) -> PolicyDocument:
    return PolicyDocument(id, "travel", f"条款{id}", content)


def _stored(client: MilvusClient) -> dict[int, str]:
    rows = client.query(
        collection_name="policies", filter="id >= 0", output_fields=["content"], limit=100
    )
    return {int(row["id"]): row["content"] for row in rows}


async def _sync(
    client: MilvusClient,
    embeddings: CountingEmbeddings,
    documents: tuple[PolicyDocument, ...],
    rebuild: bool = False,
) -> None:
    await bootstrap_policy_collection(
        client,
        embeddings,  # type: ignore[arg-type]
        "policies",
        documents=documents,
        rebuild=rebuild,
    )


@pytest.fixture
def milvus(tmp_path: Path) -> Iterator[MilvusClient]:
    client = MilvusClient(uri=str(tmp_path / "milvus.db"))
    yield client
    client.close()


async def test_sync_only_embeds_added_and_changed_clauses(milvus: MilvusClient) -> None:
    embeddings = CountingEmbeddings()
    await _sync(milvus, embeddings, (_document(1, "甲"), _document(2, "乙"), _document(3, "丙")))

    embeddings.embedded.clear()
    # 改了 2、删了 3、加了 4，1 没动。
    await _sync(milvus, embeddings, (_document(1, "甲"), _document(2, "乙改"), _document(4, "丁")))

    assert embeddings.embedded == ["条款2\n乙改", "条款4\n丁"]
    assert _stored(milvus) == {1: "甲", 2: "乙改", 4: "丁"}


async def test_unchanged_corpus_does_not_call_the_embedding_service(milvus: MilvusClient) -> None:
    documents = (_document(1, "甲"), _document(2, "乙"))
    await _sync(milvus, CountingEmbeddings(), documents)

    embeddings = CountingEmbeddings()
    await _sync(milvus, embeddings, documents)

    assert embeddings.embedded == []


async def test_switching_model_reembeds_every_clause(milvus: MilvusClient) -> None:
    documents = (_document(1, "甲"), _document(2, "乙"))
    await _sync(milvus, CountingEmbeddings(model="embed-a"), documents)

    # 维度相同也要全部重算：两个模型的向量不可比，混在一个库里相似度没有意义。
    embeddings = CountingEmbeddings(model="embed-b")
    await _sync(milvus, embeddings, documents)

    assert len(embeddings.embedded) == 2


async def test_switching_to_a_model_of_another_dimension_rebuilds_the_collection(
    milvus: MilvusClient,
) -> None:
    documents = (_document(1, "甲"), _document(2, "乙"))
    await _sync(milvus, CountingEmbeddings(model="embed-a", dimension=4), documents)

    await _sync(milvus, CountingEmbeddings(model="embed-b", dimension=8), documents)

    assert _stored(milvus) == {1: "甲", 2: "乙"}
    fields = milvus.describe_collection(collection_name="policies")["fields"]
    assert next(item for item in fields if item["name"] == "vector")["params"]["dim"] == 8


async def test_rebuild_reembeds_everything_even_when_nothing_changed(
    milvus: MilvusClient,
) -> None:
    documents = (_document(1, "甲"), _document(2, "乙"))
    await _sync(milvus, CountingEmbeddings(), documents)

    embeddings = CountingEmbeddings()
    await _sync(milvus, embeddings, documents, rebuild=True)

    assert len(embeddings.embedded) == 2
    assert _stored(milvus) == {1: "甲", 2: "乙"}
