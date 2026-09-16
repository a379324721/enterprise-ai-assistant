import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, cast

import structlog
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
from redis.asyncio import Redis

logger = structlog.get_logger()


class PolicyRepository(Protocol):
    async def search(self, query: str, domain: str, limit: int = 3) -> list[dict[str, str]]: ...


class CachedMilvusPolicyRepository(PolicyRepository):
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


@dataclass(frozen=True)
class PolicyDocument:
    id: int
    domain: str
    title: str
    content: str

    def as_hit(self) -> dict[str, str]:
        return {"title": self.title, "content": self.content, "domain": self.domain}


#: 演示用的制度语料，Milvus 和内存实现共用一份，免得评测看到的条款和线上不一样。
#: 按条款拆成短文档而不是整篇入库：一篇几千字的制度向量化后，住宿限额这类细节会被
#: 总则稀释，检索回来的也是整篇，模型还得自己在里面找。
#: id 是 upsert 的主键，改内容沿用原 id；删掉的条目不会自动从已建好的 collection 里消失。
POLICY_DOCUMENTS: tuple[PolicyDocument, ...] = (
    PolicyDocument(
        1,
        "travel",
        "差旅制度·总则",
        "国内差旅必须事前在系统提交出差申请并获审批，未经审批产生的差旅费用不予报销。"
        "交通和住宿按员工职级标准执行，超出标准的部分由个人承担。",
    ),
    PolicyDocument(
        6,
        "travel",
        "差旅制度·住宿标准",
        "住宿费按职级和城市分级设每晚上限（含税）。P1–P5：一线城市（北京、上海、广州、深圳）"
        "500元，二线城市400元，其他城市300元。P6–P8：一线城市700元，二线城市550元，"
        "其他城市450元。P9及以上：各城市每晚不超过1200元。合住按一间房的标准计算。",
    ),
    PolicyDocument(
        7,
        "travel",
        "差旅制度·交通标准",
        "飞机乘坐经济舱，P9及以上可乘坐商务舱；高铁动车乘坐二等座，P7及以上可乘坐一等座。"
        "单程4小时以内有高铁的线路优先乘坐高铁。目的地市内交通优先公共交通，"
        "打车须在报销时注明起止地点和事由。",
    ),
    PolicyDocument(
        8,
        "travel",
        "差旅制度·出差补助",
        "出差伙食补助按自然日发放，含出发和返回当天：一线城市每天120元，其他城市每天100元。"
        "接待方已提供全天餐食的日期不发放。领取补助的日期不再另行报销个人餐费。",
    ),
    PolicyDocument(
        9,
        "travel",
        "差旅制度·行程变更",
        "行程变更或取消须在出发前修改或撤销原出差申请。因工作原因产生的改签、退票费用可以报销，"
        "因个人原因产生的由个人承担。",
    ),
    PolicyDocument(
        2,
        "expense",
        "报销制度·时限与材料",
        "差旅结束后30日内提交报销，并附发票及已审批的差旅单；超过90日的费用不再受理。"
        "非差旅费用在费用发生后30日内提交。",
    ),
    PolicyDocument(
        10,
        "expense",
        "报销制度·发票要求",
        "发票抬头须为公司全称，税号须正确；电子发票上传原始PDF文件，不接受截图。"
        "个人抬头发票、收据、白条不予报销。同一张发票不得重复报销。",
    ),
    PolicyDocument(
        11,
        "expense",
        "报销制度·餐费",
        "业务招待餐费须注明招待对象和人数，人均不超过200元，须取得餐饮类增值税发票。"
        "加班至20:00以后可报销加班餐，每人每餐不超过40元。差旅期间已领取伙食补助的不再报销餐费。",
    ),
    PolicyDocument(
        12,
        "expense",
        "报销制度·市内交通",
        "因公外出打车可以报销，须注明起止地点和事由；加班至21:00以后打车回家可以报销。"
        "上下班通勤交通费不予报销。",
    ),
    PolicyDocument(
        13,
        "expense",
        "报销制度·审批与付款",
        "单笔5000元以下由直属领导审批；5000元及以上增加部门负责人审批；"
        "20000元及以上再增加财务负责人审批。审批通过后10个工作日内付款到员工工资卡。",
    ),
    PolicyDocument(
        3,
        "hr",
        "休假制度·年假",
        "年假应提前申请：1–2天至少提前3个工作日，3天及以上至少提前2周。"
        "可用余额以HR系统记录为准。当年未休完的年假可顺延至次年3月31日，逾期作废。",
    ),
    PolicyDocument(
        14,
        "hr",
        "休假制度·病假",
        "病假应在当天上班前告知直属领导；连续病假2天及以上须在返岗后3个工作日内补交医院证明。",
    ),
    PolicyDocument(
        15,
        "hr",
        "休假制度·事假与其他假期",
        "事假不计薪，每年累计不超过15天。婚假、产假、陪产假、丧假按国家及当地规定执行，"
        "申请时须提交相应证明材料。",
    ),
    PolicyDocument(
        16,
        "hr",
        "休假制度·审批",
        "请假3天以内由直属领导审批，3天以上增加部门负责人审批。审批通过前请假不生效。",
    ),
    PolicyDocument(
        5,
        "meeting",
        "会议室管理制度·预订",
        "会议室须提前预订，单次预订不超过4小时；预订后30分钟内无人到场自动释放。",
    ),
    PolicyDocument(
        17,
        "meeting",
        "会议室管理制度·使用规范",
        "10人以上的大会议室优先安排跨部门会议。会后关闭投影等设备并清理白板。"
        "有外部访客参会的，须在预订时注明，以便前台登记。",
    ),
    PolicyDocument(
        18,
        "meeting",
        "会议室管理制度·取消与爽约",
        "不再使用的会议室须提前取消预订。一个自然月内累计3次预订后未到场且未取消的，"
        "暂停预订权限两周。",
    ),
    PolicyDocument(
        4,
        "general",
        "信息安全制度",
        "企业敏感信息不得输入未经批准的外部系统，包括公共AI工具和个人网盘。"
        "离开工位须锁屏；工作账号不得借给他人使用。",
    ),
    PolicyDocument(
        19,
        "general",
        "考勤制度",
        "标准工作时间为每天8小时，弹性上班时间为9:00–10:00。每天上下班各打卡一次；"
        "漏打卡每月可补卡3次，须在3个工作日内提交补卡申请。",
    ),
)


#: Milvus 单次 query 返回条数的上限。语料远小于它；超过时要改成分页读，否则多出来的
#: 条目读不到哈希，每次启动都会被当成新增重算一遍。
_QUERY_LIMIT = 16384


def _content_hash(item: PolicyDocument, model: str) -> str:
    """模型名算进哈希：换模型后旧向量和新查询向量不可比，每条都得重算。"""
    payload = json.dumps([model, item.domain, item.title, item.content], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


async def bootstrap_policy_collection(
    client: MilvusClient,
    embeddings: OpenAIEmbeddings,
    collection: str = "enterprise_policies",
    *,
    documents: tuple[PolicyDocument, ...] = POLICY_DOCUMENTS,
    rebuild: bool = False,
) -> None:
    """把语料同步进 Milvus：只重算新增和改过的条目，删掉语料里已经没有的；生产环境应通过
    制度摄取管道替换这些记录。

    不每次整批重算：每次启动（开发时每次保存代码）都调 embedding 接口，语料一多既慢又花钱。
    也不"集合存在就跳过"：新增、改过的条款会永远进不了库，检索一直落空却没有任何报错。
    """
    model = embeddings.model
    exists = await asyncio.to_thread(client.has_collection, collection_name=collection)
    if exists and rebuild:
        await asyncio.to_thread(client.drop_collection, collection_name=collection)
        exists = False

    stored: dict[int, str | None] = {}
    if exists:
        rows = await asyncio.to_thread(
            client.query,
            collection_name=collection,
            filter="id >= 0",
            output_fields=["content_hash"],
            limit=_QUERY_LIMIT,
        )
        # 早先写入的条目没有 content_hash，取到 None，会被当作改过重算一遍。
        stored = {int(row["id"]): row.get("content_hash") for row in rows}

    hashes = {item.id: _content_hash(item, model) for item in documents}
    changed = [item for item in documents if stored.get(item.id) != hashes[item.id]]
    removed = sorted(set(stored) - set(hashes))

    vectors = await _embed(embeddings, changed)
    if exists and vectors:
        dimension = await _dimension(client, collection)
        if dimension != len(vectors[0]):
            # 换了维度不同的模型：旧集合写不进新向量，只能整个重建。不重建的话写入失败只记
            # 一条日志，之后每次检索都失败。维度不同必然是换了模型，哈希已经全变，
            # changed 就是全部语料。
            logger.warning(
                "policy_collection_dimension_changed",
                collection=collection,
                stored=dimension,
                embedding=len(vectors[0]),
            )
            await asyncio.to_thread(client.drop_collection, collection_name=collection)
            exists = False
            removed = []
    if not exists:
        await asyncio.to_thread(
            client.create_collection,
            collection_name=collection,
            dimension=len(vectors[0]),
            metric_type="COSINE",
            auto_id=False,
            enable_dynamic_field=True,
        )
    if changed:
        data = [
            {"id": item.id, "vector": vector, "content_hash": hashes[item.id], **item.as_hit()}
            for item, vector in zip(changed, vectors, strict=True)
        ]
        await asyncio.to_thread(client.upsert, collection_name=collection, data=data)
    if removed:
        await asyncio.to_thread(client.delete, collection_name=collection, ids=removed)
    logger.info(
        "policy_collection_synced",
        collection=collection,
        model=model,
        embedded=len(changed),
        removed=len(removed),
        total=len(documents),
    )


async def _embed(
    embeddings: OpenAIEmbeddings, documents: list[PolicyDocument]
) -> list[list[float]]:
    if not documents:
        return []
    # 标题一起向量化：条款正文里常常不出现"住宿标准"这类用户会问的说法，标题里有。
    return await embeddings.aembed_documents(
        [f"{item.title}\n{item.content}" for item in documents]
    )


async def _dimension(client: MilvusClient, collection: str) -> int:
    description = await asyncio.to_thread(client.describe_collection, collection_name=collection)
    field = next(item for item in description["fields"] if item["name"] == "vector")
    return int(field["params"]["dim"])


class InMemoryPolicyRepository(PolicyRepository):
    async def search(self, query: str, domain: str, limit: int = 3) -> list[dict[str, str]]:
        candidates = [item for item in POLICY_DOCUMENTS if item.domain == domain] or [
            item for item in POLICY_DOCUMENTS if item.domain == "general"
        ]
        # 按字的二元组重合数粗排，标题里的重合算两倍：总则里常顺带提到住宿、标准这些词，
        # 只数正文会排在真正讲住宿标准的条款前面。不排序的话条目一多，评测里永远只拿到
        # 领域的前几条，和线上向量检索的表现差太远。
        wanted = _bigrams(query)
        ranked = sorted(
            candidates,
            key=lambda item: (
                -(2 * len(wanted & _bigrams(item.title)) + len(wanted & _bigrams(item.content)))
            ),
        )
        return [item.as_hit() for item in ranked[:limit]]


def _bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)}
