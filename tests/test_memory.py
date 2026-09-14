
"""跨会话长期记忆的召回、写入、隔离与降级行为。"""

import asyncio
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
    OpenTask,
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
        self.seen_actions: list[list[str]] = []
        self.seen_names: list[str] = []
        self._relevant = list(relevant_keys)

    async def resolve_context(
        self,
        conversation: list[dict[str, str]],
        memory_keys: Sequence[str] = (),
        open_tasks: Sequence[OpenTask] = (),
        recent_actions: Sequence[str] = (),
        user_name: str = "",
    ) -> ContextResolution:
        del conversation
        self.seen_keys.append(list(memory_keys))
        self.seen_actions.append(list(recent_actions))
        self.seen_names.append(user_name)
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
    )["domain_batch"][0]

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
async def test_unselected_memories_are_hidden_from_domain_agents() -> None:
    """领域 Agent 只拿筛选后的档案，不接触全量记忆。"""
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
    await workflow.drain_background()

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
    await workflow.drain_background()


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
async def test_supervisor_receives_recent_actions_but_only_memory_keys() -> None:
    """单据清单用来指认"第一条"是哪张；档案只给 key，值留给领域 Agent 当建议默认值。"""
    repository = InMemoryMemoryRepository()
    await repository.upsert(
        "u-1", [MemoryCandidate(kind=MemoryKind.PROFILE, key="home_city", value="杭州")]
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

    await workflow.understand(_state(**recalled))  # type: ignore[arg-type]

    today = datetime.now(UTC).date().isoformat()
    assert planning.seen_actions == [[f"[travel] travel_application TRV-8821（{today}）：destination=北京"]]
    assert planning.seen_keys == [["home_city"]]


@pytest.mark.asyncio
async def test_display_name_reaches_the_supervisor_without_going_through_memory() -> None:
    """称呼来自令牌而不是画像：它永远相关，交给相关性筛选会在问候上被丢掉。"""
    repository = InMemoryMemoryRepository()
    planning = MemoryPlanningService()
    workflow = Workflow(SupervisorAgent(planning), memories=repository)

    await workflow.understand(_state(user_name="王宁"))  # type: ignore[arg-type]

    assert planning.seen_names == ["王宁"]
    # 画像里没有任何一条姓名记录，记忆表的职责没有被扩大。
    assert await repository.list_memories("u-1", 10) == []


@pytest.mark.asyncio
async def test_display_name_reaches_the_domain_subgraph() -> None:
    workflow = Workflow(SupervisorAgent(MemoryPlanningService()))

    update = await workflow.select_task(
        _state(
            user_name="王宁",
            understanding=_understanding([]),
            tasks=[
                PlannedTask(id="task-1", title="差旅", domain=AgentName.TRAVEL, objective="创建")
            ],
        )
    )

    assert update["domain_batch"][0].user_name == "王宁"


@pytest.mark.asyncio
async def test_a_turn_without_a_name_still_works() -> None:
    """早于该字段的检查点没有 user_name，不能因此报错。"""
    planning = MemoryPlanningService()
    workflow = Workflow(SupervisorAgent(planning))

    await workflow.understand(_state())  # type: ignore[arg-type]

    assert planning.seen_names == [""]


@pytest.mark.asyncio
async def test_extraction_only_sees_what_the_user_said() -> None:
    """助手的回答里有会议室名、目的地和称呼，模型会把它们当成用户的稳定属性。

    实测出现过把出差地"上海分部"记成常驻办公地、把称呼记成姓名，所以抽取输入
    在代码层面就只保留用户消息，不依赖 prompt 里的"不得推断"。
    """
    planning = MemoryPlanningService()
    workflow = Workflow(SupervisorAgent(planning), memories=InMemoryMemoryRepository())
    state = _state(understanding=_understanding([]))
    state["messages"] = [
        HumanMessage(content="记一下，我常驻杭州"),
        AIMessage(content="好的，演示小王。已为你预订上海分部 301 讨论室。"),
    ]

    await workflow.remember(state)
    await workflow.drain_background()

    conversation, _ = planning.seen[0]
    assert [turn["content"] for turn in conversation] == ["记一下，我常驻杭州"]
    assert all("上海分部" not in turn["content"] for turn in conversation)


@pytest.mark.asyncio
async def test_remember_does_not_wait_for_the_extraction() -> None:
    """抽取是一次完整的模型调用；同步等它，这一轮的 done 事件和执行锁都要跟着拖。"""
    release = asyncio.Event()

    class SlowPlanningService(MemoryPlanningService):
        async def extract_memories(
            self, conversation: list[dict[str, str]], known: list[str]
        ) -> MemoryExtraction:
            await release.wait()
            return MemoryExtraction(
                memories=[MemoryCandidate(kind=MemoryKind.PROFILE, key="home_city", value="杭州")]
            )

    repository = InMemoryMemoryRepository()
    workflow = Workflow(SupervisorAgent(SlowPlanningService()), memories=repository)

    assert await asyncio.wait_for(workflow.remember(_state()), timeout=1) == {}
    assert await repository.list_memories("u-1", 10) == []

    release.set()
    await workflow.drain_background()

    assert [item.key for item in await repository.list_memories("u-1", 10)] == ["home_city"]


@pytest.mark.asyncio
async def test_domain_agents_read_only_the_most_recent_turns_verbatim() -> None:
    """领域 Agent 读最近几条原文，含本轮前面任务刚写的回答；不带改写过的早先摘要。"""
    workflow = Workflow(
        SupervisorAgent(MemoryPlanningService()), history_window=4, domain_window=5
    )
    state = _state(
        understanding=_understanding([]),
        history_digest=["早先问过报销制度"],
        messages=[
            HumanMessage(content="报销制度是什么"),
            AIMessage(content="报销需在 30 天内提交。"),
            HumanMessage(content="下周去上海出差，顺便订个会议室"),
            AIMessage(content="请问返程日期是哪天？"),
            HumanMessage(content="当天往返"),
            # 本轮排在前面的差旅任务刚写进 messages 的回答。
            AIMessage(content="差旅申请已提交，单号 TRV-1。"),
        ],
        tasks=[
            PlannedTask(id="task-2", title="会议室", domain=AgentName.MEETING, objective="预订")
        ],
    )

    update = await workflow.select_task(state)  # type: ignore[arg-type]

    assert [
        (turn.role, turn.content) for turn in update["domain_batch"][0].recent_messages
    ] == [
        # 窗口里排在最前的是摘要条目，窗口放得下也不给。
        ("user", "下周去上海出差，顺便订个会议室"),
        ("assistant", "请问返程日期是哪天？"),
        ("user", "当天往返"),
        ("assistant", "差旅申请已提交，单号 TRV-1。"),
    ]


@pytest.mark.asyncio
async def test_domain_window_of_zero_gives_no_conversation() -> None:
    workflow = Workflow(SupervisorAgent(MemoryPlanningService()), domain_window=0)
    state = _state(
        understanding=_understanding([]),
        messages=[HumanMessage(content="下周去上海出差")],
        tasks=[PlannedTask(id="task-1", title="出差", domain=AgentName.TRAVEL, objective="申请")],
    )

    update = await workflow.select_task(state)  # type: ignore[arg-type]

    assert update["domain_batch"][0].recent_messages == []
