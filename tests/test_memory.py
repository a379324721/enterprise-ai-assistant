"""跨会话长期记忆的召回、写入、隔离与降级行为。"""

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.core.models import (
    AgentName,
    ContextResolution,
    MemoryCandidate,
    MemoryExtraction,
    MemoryKind,
    MemoryRecord,
    PlannedTask,
    RecentAction,
    TaskPlan,
    TaskStatus,
)
from enterprise_ai_assistant.core.security import create_access_token
from enterprise_ai_assistant.graph.workflow import Workflow
from enterprise_ai_assistant.main import create_app
from enterprise_ai_assistant.repositories.memories import (
    InMemoryMemoryRepository,
    summarize_action,
)

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000001")
REQUEST_ID = UUID("00000000-0000-0000-0000-000000000002")


class MemoryPlanningService:
    """按脚本返回抽取结果，并记录送进抽取器的输入。"""

    def __init__(
        self,
        extraction: MemoryExtraction | None = None,
        relevant_keys: Sequence[str] = (),
    ) -> None:
        self.extraction = extraction or MemoryExtraction()
        self.seen: list[tuple[list[dict[str, str]], list[str]]] = []
        self.seen_keys: list[list[str]] = []
        self.seen_direct: list[tuple[list[str], list[str]]] = []
        self._relevant = list(relevant_keys)

    async def resolve_context(
        self, conversation: list[dict[str, str]], memory_keys: Sequence[str] = ()
    ) -> ContextResolution:
        del conversation
        self.seen_keys.append(list(memory_keys))
        return ContextResolution(
            standalone_request="创建差旅申请",
            intent_summary="差旅",
            requires_task_planning=True,
            relevant_memory_keys=self._relevant,
        )

    async def plan(self, context: ContextResolution) -> TaskPlan:
        return TaskPlan(
            user_goal=context.standalone_request,
            tasks=[
                PlannedTask(
                    id="task-1",
                    title="创建差旅申请",
                    domain=AgentName.TRAVEL,
                    objective="创建差旅申请",
                )
            ],
        )

    async def respond_direct(
        self,
        context: ContextResolution,
        memories: Sequence[str] = (),
        recent_actions: Sequence[str] = (),
    ) -> AIMessage:
        del context
        self.seen_direct.append((list(memories), list(recent_actions)))
        return AIMessage(content="好的")

    async def extract_memories(
        self, conversation: list[dict[str, str]], known: list[str]
    ) -> MemoryExtraction:
        self.seen.append((conversation, known))
        return self.extraction


def _record(key: str, value: str, kind: MemoryKind = MemoryKind.PROFILE) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(), kind=kind, key=key, value=value, updated_at=datetime.now(UTC)
    )


def _state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="下周去上海出差")],
        "user_id": "u-1",
        "conversation_id": CONVERSATION_ID,
        "request_id": REQUEST_ID,
        "user_goal": "创建差旅申请",
        "tasks": [],
        "artifacts": {},
        "tool_results": [],
        "current_agent": None,
        "active_task_id": None,
        "last_answer": "",
    }
    state.update(overrides)
    return state


# --- 派生摘要的字段白名单 -------------------------------------------------


def test_leave_summary_drops_the_reason_text() -> None:
    """请假原因常含健康或家庭信息，不进入任何长期可读的摘要。"""
    summary = summarize_action(
        "leave_request",
        {
            "leave_type": "annual",
            "start_date": "2026-10-01",
            "end_date": "2026-10-03",
            "reason": "家人住院需要陪护",
        },
    )

    assert summary == "leave_type=annual start_date=2026-10-01 end_date=2026-10-03"
    assert "住院" not in summary


def test_expense_summary_drops_receipt_refs() -> None:
    summary = summarize_action(
        "expense_claim",
        {
            "expense_type": "交通",
            "amount": "480.00",
            "currency": "CNY",
            "receipt_refs": ["INV-99887766"],
        },
    )

    assert summary == "expense_type=交通 amount=480.00 currency=CNY"
    assert "INV-99887766" not in summary


def test_unknown_action_type_yields_no_summary() -> None:
    """新增写操作若没登记白名单，宁可不召回，也不要把整个载荷倒进上下文。"""
    assert summarize_action("salary_adjustment", {"amount": "100000"}) == ""


# --- 仓储语义 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_key_is_overwritten_rather_than_accumulated() -> None:
    repository = InMemoryMemoryRepository()
    candidate = MemoryCandidate(kind=MemoryKind.PROFILE, key="cost_center", value="RD-02")

    await repository.upsert("u-1", [candidate])
    first = await repository.list_memories("u-1", 10)
    await repository.upsert(
        "u-1",
        [MemoryCandidate(kind=MemoryKind.PROFILE, key="cost_center", value="RD-07")],
    )
    second = await repository.list_memories("u-1", 10)

    assert [item.value for item in second] == ["RD-07"]
    # 覆盖写保留同一条记录的 id，用户先前拿到的删除链接不会失效。
    assert second[0].id == first[0].id


@pytest.mark.asyncio
async def test_memories_are_isolated_per_user() -> None:
    repository = InMemoryMemoryRepository()
    await repository.upsert(
        "u-1", [MemoryCandidate(kind=MemoryKind.PROFILE, key="home_city", value="杭州")]
    )
    owned = await repository.list_memories("u-1", 10)

    assert await repository.list_memories("u-2", 10) == []
    assert await repository.delete("u-2", owned[0].id) is False
    assert await repository.delete("u-1", owned[0].id) is True


# --- recall / remember 节点 ----------------------------------------------


@pytest.mark.asyncio
def _understanding(relevant: Sequence[str]) -> dict[str, Any]:
    return ContextResolution(
        standalone_request="创建差旅申请",
        intent_summary="差旅",
        requires_task_planning=True,
        relevant_memory_keys=list(relevant),
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_only_relevant_memories_reach_the_domain_subgraph() -> None:
    """全量下发的成本是 记忆条数 × 任务数，且无关档案会成为领域模型的噪音。"""
    repository = InMemoryMemoryRepository()
    await repository.upsert(
        "u-1",
        [
            MemoryCandidate(
                kind=MemoryKind.PREFERENCE, key="preferred_transport", value="高铁"
            ),
            MemoryCandidate(kind=MemoryKind.PROFILE, key="job_level", value="P6"),
        ],
    )
    repository.actions["u-1"] = [
        RecentAction(
            reference_id="TRV-8821",
            action_type="travel_application",
            summary="destination=北京",
            created_at=datetime.now(UTC),
        )
    ]
    workflow = Workflow(SupervisorAgent(MemoryPlanningService()), memories=repository)

    recalled = await workflow.recall(_state())
    request = (
        await workflow.select_task(
            _state(
                tasks=[
                    PlannedTask(
                        id="task-1", title="差旅", domain=AgentName.TRAVEL, objective="创建"
                    )
                ],
                understanding=_understanding(["preferred_transport"]),
                **recalled,
            )
        )
    )["domain_request"]

    # job_level 没被理解阶段选中，不该进入子图。
    assert request.memories == ["preferred_transport=高铁"]
    assert [item.reference_id for item in request.recent_actions] == ["TRV-8821"]


@pytest.mark.asyncio
async def test_understand_sends_keys_without_values() -> None:
    """Supervisor 只做相关性筛选；拿不到 value 就无从用记忆补写领域字段。"""
    repository = InMemoryMemoryRepository()
    await repository.upsert(
        "u-1",
        [MemoryCandidate(kind=MemoryKind.PROFILE, key="cost_center", value="RD-02")],
    )
    planning = MemoryPlanningService()
    workflow = Workflow(SupervisorAgent(planning), memories=repository)

    recalled = await workflow.recall(_state())
    await workflow.understand(_state(**recalled))

    assert planning.seen_keys == [["cost_center"]]


@pytest.mark.asyncio
async def test_unselected_memories_are_hidden_from_small_talk() -> None:
    """闲聊节点同样只拿筛选后的档案，不接触全量记忆。"""
    repository = InMemoryMemoryRepository()
    await repository.upsert(
        "u-1",
        [
            MemoryCandidate(kind=MemoryKind.PROFILE, key="home_city", value="杭州"),
            MemoryCandidate(kind=MemoryKind.PROFILE, key="job_level", value="P6"),
        ],
    )
    workflow = Workflow(SupervisorAgent(MemoryPlanningService()), memories=repository)
    recalled = await workflow.recall(_state())

    relevant = workflow._relevant_memories(
        _state(understanding=_understanding(["home_city"]), **recalled)
    )

    assert relevant == ["home_city=杭州"]


@pytest.mark.asyncio
async def test_recall_degrades_to_empty_when_the_repository_fails() -> None:
    """记忆是增益而非执行前提，仓储故障应退化成无记忆行为而不是中断本轮。"""

    class BrokenRepository(InMemoryMemoryRepository):
        async def list_memories(self, user_id: str, limit: int) -> list[MemoryRecord]:
            raise RuntimeError("postgres down")

    workflow = Workflow(SupervisorAgent(MemoryPlanningService()), memories=BrokenRepository())

    assert await workflow.recall(_state()) == {"memories": [], "recent_actions": []}


@pytest.mark.asyncio
async def test_remember_writes_extracted_memories_with_known_ones_passed_in() -> None:
    repository = InMemoryMemoryRepository()
    planning = MemoryPlanningService(
        MemoryExtraction(
            memories=[
                MemoryCandidate(kind=MemoryKind.PROFILE, key="home_city", value="杭州")
            ]
        )
    )
    workflow = Workflow(SupervisorAgent(planning), memories=repository)

    await workflow.remember(_state(memories=[_record("cost_center", "RD-02")]))

    stored = await repository.list_memories("u-1", 10)
    assert [(item.key, item.value) for item in stored] == [("home_city", "杭州")]
    assert stored[0].source_conversation_id == CONVERSATION_ID
    # 已知记忆随输入送进抽取器，避免每轮重复输出同一条事实。
    assert planning.seen[0][1] == ["cost_center=RD-02"]


@pytest.mark.asyncio
async def test_remember_skips_turns_that_are_waiting_for_user_input() -> None:
    """字段还没谈定就写画像，下一轮会拿着半成品去预填。"""
    repository = InMemoryMemoryRepository()
    planning = MemoryPlanningService(
        MemoryExtraction(
            memories=[MemoryCandidate(kind=MemoryKind.PROFILE, key="home_city", value="杭州")]
        )
    )
    workflow = Workflow(SupervisorAgent(planning), memories=repository)
    waiting = PlannedTask(
        id="task-1",
        title="差旅",
        domain=AgentName.TRAVEL,
        objective="创建",
        status=TaskStatus.WAITING_INPUT,
    )

    await workflow.remember(_state(tasks=[waiting]))

    assert planning.seen == []
    assert await repository.list_memories("u-1", 10) == []


@pytest.mark.asyncio
async def test_remember_survives_a_write_failure() -> None:
    class BrokenRepository(InMemoryMemoryRepository):
        async def upsert(
            self,
            user_id: str,
            candidates: Sequence[MemoryCandidate],
            *,
            source_conversation_id: UUID | None = None,
        ) -> None:
            raise RuntimeError("postgres down")

    workflow = Workflow(
        SupervisorAgent(
            MemoryPlanningService(
                MemoryExtraction(
                    memories=[
                        MemoryCandidate(kind=MemoryKind.PROFILE, key="home_city", value="杭州")
                    ]
                )
            )
        ),
        memories=BrokenRepository(),
    )

    assert await workflow.remember(_state()) == {}


@pytest.mark.asyncio
async def test_disabled_memory_never_touches_the_repository() -> None:
    """开关关闭时不建仓储连接，也不产生额外的抽取调用。"""
    planning = MemoryPlanningService()
    workflow = Workflow(SupervisorAgent(planning), memories=None)

    assert await workflow.recall(_state()) == {"memories": [], "recent_actions": []}
    assert await workflow.remember(_state()) == {}
    assert planning.seen == []


# --- 管理接口 -------------------------------------------------------------

SETTINGS = Settings(
    openai_api_key="test-key",  # type: ignore[arg-type]
    openai_model="test-model",
    openai_embedding_model="test-embedding",
    langsmith_tracing=False,
    jwt_secret="unit-test-secret-" + "x" * 32,  # type: ignore[arg-type]
    app_env="development",
    memory_enabled=True,
)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(routes, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: SETTINGS)
    application = create_app()
    application.state.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    application.state.memories = InMemoryMemoryRepository()
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        yield async_client


def _auth(user_id: str) -> dict[str, str]:
    token, _ = create_access_token(user_id, SETTINGS)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_listing_memories_is_scoped_to_the_token_owner(
    app: FastAPI, client: AsyncClient
) -> None:
    await app.state.memories.upsert(
        "owner-user",
        [MemoryCandidate(kind=MemoryKind.PROFILE, key="home_city", value="杭州")],
    )

    mine = await client.get("/api/v1/memories", headers=_auth("owner-user"))
    other = await client.get("/api/v1/memories", headers=_auth("stranger"))

    assert [item["value"] for item in mine.json()["memories"]] == ["杭州"]
    assert other.json()["memories"] == []


@pytest.mark.asyncio
async def test_a_user_cannot_delete_another_users_memory(
    app: FastAPI, client: AsyncClient
) -> None:
    await app.state.memories.upsert(
        "owner-user",
        [MemoryCandidate(kind=MemoryKind.PROFILE, key="home_city", value="杭州")],
    )
    stored = await app.state.memories.list_memories("owner-user", 10)
    memory_id = str(stored[0].id)

    denied = await client.delete(f"/api/v1/memories/{memory_id}", headers=_auth("stranger"))
    allowed = await client.delete(f"/api/v1/memories/{memory_id}", headers=_auth("owner-user"))

    assert denied.status_code == 404
    assert allowed.status_code == 204
    assert await app.state.memories.list_memories("owner-user", 10) == []


@pytest.mark.asyncio
async def test_small_talk_receives_filtered_profile_and_recent_actions() -> None:
    """闲聊节点带上档案和单据，用于称呼贴合与待办提示。"""
    repository = InMemoryMemoryRepository()
    await repository.upsert(
        "u-1",
        [
            MemoryCandidate(kind=MemoryKind.PROFILE, key="home_city", value="杭州"),
            MemoryCandidate(kind=MemoryKind.PROFILE, key="job_level", value="P6"),
        ],
    )
    repository.actions["u-1"] = [
        RecentAction(
            reference_id="TRV-8821",
            action_type="travel_application",
            summary="destination=北京",
            created_at=datetime.now(UTC),
        )
    ]
    planning = MemoryPlanningService()
    workflow = Workflow(SupervisorAgent(planning), memories=repository)
    recalled = await workflow.recall(_state())

    await workflow.direct_respond(
        _state(understanding=_understanding(["home_city"]), **recalled)
    )

    memories, actions = planning.seen_direct[0]
    # 闲聊拿到的同样是筛选后的档案，job_level 不在其中。
    assert memories == ["home_city=杭州"]
    today = datetime.now(UTC).date().isoformat()
    assert actions == [f"travel_application TRV-8821（{today}）：destination=北京"]


@pytest.mark.asyncio
async def test_small_talk_never_reads_raw_messages() -> None:
    """执行链路上的节点只吃理解阶段的输出；闲聊节点也不例外。"""
    planning = MemoryPlanningService()
    workflow = Workflow(SupervisorAgent(planning), memories=InMemoryMemoryRepository())
    state = _state(understanding=_understanding([]))
    state["messages"] = [HumanMessage(content="这句原话不该被闲聊节点看到")]

    await workflow.direct_respond(state)

    # respond_direct 的入参里没有会话，stub 只收到 context 与档案。
    assert planning.seen == []
    assert planning.seen_direct == [([], [])]
