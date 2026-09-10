from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from pymilvus import MilvusClient
from redis.asyncio import Redis

from enterprise_ai_assistant.agents.domain_runtime import DomainRuntimeFactory
from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.api.routes import router
from enterprise_ai_assistant.core.config import get_settings
from enterprise_ai_assistant.core.logging import configure_logging
from enterprise_ai_assistant.core.observability import MetricsMiddleware
from enterprise_ai_assistant.core.runs import MemoryStreamBridge, RunManager
from enterprise_ai_assistant.db.postgres import create_pool
from enterprise_ai_assistant.graph.domain import DomainTaskWorkflow
from enterprise_ai_assistant.graph.workflow import Workflow, build_graph
from enterprise_ai_assistant.repositories.actions import PostgresActionRepository
from enterprise_ai_assistant.repositories.policies import (
    CachedMilvusPolicyRepository,
    bootstrap_policy_collection,
)
from enterprise_ai_assistant.services.llm import build_chat_model, build_embeddings
from enterprise_ai_assistant.services.planning import LLMPlanningService
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider
from enterprise_ai_assistant.tools.registry import DomainToolRegistry

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger = structlog.get_logger()
    # 用 AsyncExitStack 逐个登记资源：启动过程中任何一步失败（例如 Milvus 不可达），
    # 已经建立的连接都会按逆序释放，而不是把连接池泄漏到崩溃的进程里。
    async with AsyncExitStack() as stack:
        db_pool = await create_pool(settings.postgres_dsn)
        stack.push_async_callback(db_pool.close)
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        stack.push_async_callback(redis.aclose)
        milvus = MilvusClient(uri=settings.milvus_uri)
        stack.callback(milvus.close)

        embeddings = build_embeddings(settings)
        try:
            await bootstrap_policy_collection(milvus, embeddings)
        except Exception:
            # 语料初始化失败只影响制度检索，工具会返回明确的失败结果而不是编造内容；
            # 让进程继续启动，其余领域能力和健康检查仍然可用。
            logger.exception("policy_bootstrap_failed", milvus_uri=settings.milvus_uri)
        policies = CachedMilvusPolicyRepository(milvus, redis, embeddings)
        actions = PostgresActionRepository(db_pool)
        model = build_chat_model()
        supervisor = SupervisorAgent(LLMPlanningService(model))
        provider = LocalEnterpriseToolProvider(actions, policies)
        workflow = Workflow(supervisor)
        domain_workflow = DomainTaskWorkflow(
            DomainRuntimeFactory(model, DomainToolRegistry(provider))
        )
        # from_conn_string 只建立单条连接，检查点写入会被 saver 内部的锁串行化。
        # 这里显式使用连接池，让并发会话的状态读写可以并行。
        checkpoint_pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
            settings.postgres_dsn,
            min_size=2,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False,
        )
        await checkpoint_pool.open(wait=True)
        stack.push_async_callback(checkpoint_pool.close)
        checkpointer = AsyncPostgresSaver(checkpoint_pool)
        await checkpointer.setup()

        app.state.graph = build_graph(workflow, domain_workflow, checkpointer)
        # 图执行跑在后台运行里，SSE 连接只是订阅者：客户端断开不再中断执行。
        # 单进程用内存事件桥；多副本部署时换成跨进程实现即可，路由层不用动。
        bridge = MemoryStreamBridge(
            buffer_size=settings.run_event_buffer_size,
            heartbeat_interval=settings.sse_heartbeat_seconds,
        )
        runs = RunManager(bridge, logger, retention_seconds=settings.run_retention_seconds)
        # 最后登记意味着最先释放：进程关停时先停掉后台运行，再拆连接池。
        stack.push_async_callback(runs.aclose)
        app.state.runs = runs
        app.state.db_pool = db_pool
        app.state.redis = redis
        app.state.milvus = milvus
        app.state.logger = logger
        logger.info("application_started", environment=settings.app_env)
        yield
        logger.info("application_stopping", environment=settings.app_env)


def create_app(lifespan_handler: Any = lifespan) -> FastAPI:
    settings = get_settings()
    application = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan_handler)
    application.add_middleware(MetricsMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
    application.include_router(router)
    return application


app = create_app()
