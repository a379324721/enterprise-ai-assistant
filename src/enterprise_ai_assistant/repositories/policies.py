import asyncio
import hashlib
import json
from typing import Any, Protocol, cast

from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
from redis.asyncio import Redis


class PolicyRepository(Protocol):
    async def search(self, query: str, domain: str, limit: int = 3) -> list[dict[str, str]]: ...


class CachedMilvusPolicyRepository:
    """Semantic policy retrieval with Redis query cache and Milvus vector storage."""

    def __init__(
        self,
        client: MilvusClient,
        redis: Redis,
        embeddings: OpenAIEmbeddings,
        collection: str = "enterprise_policies",
    ) -> None:
        self._client = client
        self._redis = redis
        self._embeddings = embeddings
        self._collection = collection

    async def search(self, query: str, domain: str, limit: int = 3) -> list[dict[str, str]]:
        key = "policy:" + hashlib.sha256(f"{domain}:{query}".encode()).hexdigest()
        cached = await self._redis.get(key)
        if cached:
            return cast(list[dict[str, str]], json.loads(cached))
        vector = await self._embeddings.aembed_query(query)
        rows = await asyncio.to_thread(
            self._client.search,
            collection_name=self._collection,
            data=[vector],
            filter=f'domain == "{domain}"',
            limit=limit,
            output_fields=["title", "content", "domain"],
        )
        raw_results: list[Any] = rows[0]
        results = cast(list[dict[str, str]], [dict(hit["entity"]) for hit in raw_results])
        await self._redis.setex(key, 300, json.dumps(results, ensure_ascii=False))
        return results


async def bootstrap_policy_collection(
    client: MilvusClient, embeddings: OpenAIEmbeddings, collection: str = "enterprise_policies"
) -> None:
    """Create a minimal searchable corpus. Replace these records with the policy ingestion pipeline."""
    exists = await asyncio.to_thread(client.has_collection, collection_name=collection)
    if exists:
        return
    documents = [
        (1, "travel", "差旅制度", "国内差旅必须事前审批；交通和住宿应遵守员工职级标准。"),
        (2, "expense", "报销制度", "差旅结束后30日内提交报销，并附发票及已审批差旅单。"),
        (3, "hr", "休假制度", "年假应提前申请；可用余额以HR系统记录为准。"),
        (4, "general", "信息安全制度", "企业敏感信息不得输入未经批准的外部系统。"),
    ]
    vectors = await embeddings.aembed_documents([item[3] for item in documents])
    await asyncio.to_thread(
        client.create_collection,
        collection_name=collection,
        dimension=len(vectors[0]),
        metric_type="COSINE",
        auto_id=False,
        enable_dynamic_field=True,
    )
    data = [
        {"id": item[0], "vector": vector, "domain": item[1], "title": item[2], "content": item[3]}
        for item, vector in zip(documents, vectors, strict=True)
    ]
    await asyncio.to_thread(client.insert, collection_name=collection, data=data)


class InMemoryPolicyRepository:
    POLICIES = {
        "travel": [
            {
                "title": "差旅制度",
                "content": "国内差旅须事前审批；住宿和交通按职级标准执行。",
                "domain": "travel",
            }
        ],
        "expense": [
            {
                "title": "报销制度",
                "content": "差旅结束后30日内提交发票和审批单。",
                "domain": "expense",
            }
        ],
        "hr": [
            {"title": "休假制度", "content": "年假须提前申请，余额以HR系统为准。", "domain": "hr"}
        ],
        "general": [
            {"title": "企业制度", "content": "请提供要查询的制度主题。", "domain": "general"}
        ],
    }

    async def search(self, query: str, domain: str, limit: int = 3) -> list[dict[str, str]]:
        del query
        return self.POLICIES.get(domain, self.POLICIES["general"])[:limit]
