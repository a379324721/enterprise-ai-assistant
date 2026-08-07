from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pymilvus import MilvusClient
from redis.asyncio import Redis

from enterprise_ai_assistant.agents.domain_agents import (
    ExpenseAgent,
    HRAgent,
    PolicyAgent,
    TravelAgent,
)
from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.api.routes import router
from enterprise_ai_assistant.core.config import get_settings
from enterprise_ai_assistant.core.logging import configure_logging
from enterprise_ai_assistant.db.postgres import create_pool
from enterprise_ai_assistant.graph.workflow import Workflow, build_graph
from enterprise_ai_assistant.repositories.actions import PostgresActionRepository
from enterprise_ai_assistant.repositories.policies import (
    CachedMilvusPolicyRepository,
    bootstrap_policy_collection,
)
from enterprise_ai_assistant.services.capabilities import CapabilityRegistry
from enterprise_ai_assistant.services.llm import build_chat_model, build_embeddings
from enterprise_ai_assistant.services.planning import LLMPlanningService

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger = structlog.get_logger()
    db_pool = await create_pool(settings.postgres_dsn)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    milvus = MilvusClient(uri=settings.milvus_uri)
    embeddings = build_embeddings(settings)
    await bootstrap_policy_collection(milvus, embeddings)
    policies = CachedMilvusPolicyRepository(milvus, redis, embeddings)
    actions = PostgresActionRepository(db_pool)
    supervisor = SupervisorAgent(LLMPlanningService(build_chat_model()), CapabilityRegistry())
    workflow = Workflow(
        supervisor,
        TravelAgent(policies, actions),
        ExpenseAgent(policies, actions),
        HRAgent(policies, actions),
        PolicyAgent(policies),
    )
    async with AsyncPostgresSaver.from_conn_string(settings.postgres_dsn) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(workflow, checkpointer)
        app.state.db_pool = db_pool
        app.state.redis = redis
        app.state.milvus = milvus
        app.state.logger = logger
        logger.info("application_started", environment=settings.app_env)
        yield
    await redis.aclose()
    await db_pool.close()
    milvus.close()


def create_app(lifespan_handler: Any = lifespan) -> FastAPI:
    settings = get_settings()
    application = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan_handler)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-User-ID"],
    )
    application.include_router(router)
    return application


app = create_app()
