import asyncio
import hashlib
import json
from typing import Any, Protocol, cast

import structlog
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
from redis.asyncio import Redis

logger = structlog.get_logger()


class PolicyRepository(Protocol):
    async def search(self, query: str, domain: str, limit: int = 3) -> list[dict[str, str]]: ...


class CachedMilvusPolicyRepository:
    """使用 Redis 查询缓存和 Milvus 向量存储进行制度语义检索。"""

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
        cached = await self._read_cache(key)
        if cached is not None:
            return cached
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
        await self._write_cache(key, results)
        return results

    async def _read_cache(self, key: str) -> list[dict[str, str]] | None:
        """缓存只用于加速。Redis 故障或数据损坏时回退到向量检索，而不是让查询失败。"""
        try:
            cached = await self._redis.get(key)
        except Exception:
            logger.warning("policy_cache_read_failed", cache_key=key)
            return None
        if not cached:
            return None
        try:
            return cast(list[dict[str, str]], json.loads(cached))
        except json.JSONDecodeError:
            logger.warning("policy_cache_corrupted", cache_key=key)
            return None

    async def _write_cache(self, key: str, results: list[dict[str, str]]) -> None:
        try:
            await self._redis.setex(key, 300, json.dumps(results, ensure_ascii=False))
        except Exception:
            logger.warning("policy_cache_write_failed", cache_key=key)


async def bootstrap_policy_collection(
    client: MilvusClient, embeddings: OpenAIEmbeddings, collection: str = "enterprise_policies"
) -> None:
    """写入最小可检索语料库；生产环境应通过制度摄取管道替换这些记录。

    用固定 id 做 upsert 而不是"已存在就跳过"：后者会让新增领域的制度永远进不去
    已建好的 collection，新领域上线后检索一直落空，还得手工删库才能生效。
    """
    documents = [
        (1, "travel", "差旅制度", "国内差旅必须事前审批；交通和住宿应遵守员工职级标准。"),
        (2, "expense", "报销制度", "差旅结束后30日内提交报销，并附发票及已审批差旅单。"),
        (3, "hr", "休假制度", "年假应提前申请；可用余额以HR系统记录为准。"),
        (4, "general", "信息安全制度", "企业敏感信息不得输入未经批准的外部系统。"),
        (
            5,
            "meeting",
            "会议室管理制度",
            "会议室须提前预订，单次预订不超过4小时；预订后30分钟内无人到场自动释放。",
        ),
    ]
    vectors = await embeddings.aembed_documents([item[3] for item in documents])
    exists = await asyncio.to_thread(client.has_collection, collection_name=collection)
    if not exists:
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
    await asyncio.to_thread(client.upsert, collection_name=collection, data=data)


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
        "meeting": [
            {
                "title": "会议室管理制度",
                "content": "会议室须提前预订，单次不超过4小时；30分钟无人到场自动释放。",
                "domain": "meeting",
            }
        ],
        "general": [
            {"title": "企业制度", "content": "请提供要查询的制度主题。", "domain": "general"}
        ],
    }

    async def search(self, query: str, domain: str, limit: int = 3) -> list[dict[str, str]]:
        del query
        return self.POLICIES.get(domain, self.POLICIES["general"])[:limit]
