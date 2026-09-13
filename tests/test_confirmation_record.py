"""确认决定在会话历史里的留痕。

卡片一关，对话里就必须留下"确认了"还是"取消了"，否则回头看只剩一句没头没尾的
回答。这条记录随恢复命令一起写进检查点，因此要盯两件事：它不能破坏人工确认的恢复
链路，也不能溜进模型上下文——它不是用户说的话。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from enterprise_ai_assistant.api import routes
from enterprise_ai_assistant.api.schemas import ConfirmationRequest
from enterprise_ai_assistant.core.config import Settings
from enterprise_ai_assistant.core.models import PendingConfirmation
from enterprise_ai_assistant.core.security import create_access_token
from enterprise_ai_assistant.main import create_app

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000011")

SETTINGS = Settings(
    openai_api_key="test-key",  # type: ignore[arg-type]
    openai_model="test-model",
    openai_embedding_model="test-embedding",
    langsmith_tracing=False,
    jwt_secret="unit-test-secret-" + "x" * 32,  # type: ignore[arg-type]
    app_env="development",
)


class StubGraph:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    async def aget_state(self, config: dict[str, Any]) -> Any:
        del config
        return SimpleNamespace(values=self.values, next=(), interrupts=())


@asynccontextmanager
async def _noop_lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    yield


def _client(values: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> AsyncClient:
    monkeypatch.setattr(routes, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr("enterprise_ai_assistant.core.security.get_settings", lambda: SETTINGS)
    app = create_app(_noop_lifespan)
    app.state.graph = StubGraph(values)
    app.state.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _auth(user_id: str = "owner-user") -> dict[str, str]:
    token, _ = create_access_token(user_id, SETTINGS)
    return {"Authorization": f"Bearer {token}"}


_PENDING = PendingConfirmation(
    task_id="task-1",
    action="create_travel_application",
    tool_call_id="call-1",
    title="提交差旅申请",
    payload={},
)


def test_resume_command_carries_the_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    interrupt = SimpleNamespace(id="interrupt-1")
    approved = routes._resume_command(
        ConfirmationRequest(confirmation_id=CONVERSATION_ID, approved=True), interrupt, _PENDING
    )
    rejected = routes._resume_command(
        ConfirmationRequest(confirmation_id=CONVERSATION_ID, approved=False), interrupt, _PENDING
    )

    # 按中断 id 指明恢复哪一个：并行任务同时停在确认卡上时，给单个值 LangGraph 会报错。
    assert approved.resume["interrupt-1"]["approved"] is True  # type: ignore[index]
    # 记录里写明是哪个操作：同一轮可能先后确认好几次，只写"确认了"回头看分不清。
    assert approved.update["messages"][0].content == "你确认了：提交差旅申请"  # type: ignore[index]
    assert rejected.update["messages"][0].content == "你取消了：提交差旅申请"  # type: ignore[index]


def test_the_decision_is_not_a_user_turn() -> None:
    """SystemMessage 才进不了 prompt：`_conversation()` 只挑 human/ai。

    用 HumanMessage 记这条会让下一轮的 Context Supervisor 把它当成用户的新输入。
    """
    message = routes._decision_message(True, "提交差旅申请")

    assert message.type == "system"
    assert message.additional_kwargs == {"kind": "decision"}


@pytest.mark.asyncio
async def test_history_exposes_the_decision_as_its_own_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "user_id": "owner-user",
        "messages": [
            HumanMessage(content="帮我申请明天去上海出差"),
            SystemMessage(content="你确认了：提交差旅申请", additional_kwargs={"kind": "decision"}),
            AIMessage(content="差旅申请已提交"),
        ],
    }

    async with _client(values, monkeypatch) as client:
        response = await client.get(
            f"/api/v1/conversations/{CONVERSATION_ID}/messages", headers=_auth()
        )

    assert [(item["role"], item["text"]) for item in response.json()["messages"]] == [
        ("user", "帮我申请明天去上海出差"),
        ("decision", "你确认了：提交差旅申请"),
        ("assistant", "差旅申请已提交"),
    ]


class _Child(TypedDict):
    decided: bool


class _Parent(TypedDict):
    messages: Annotated[list[Any], add_messages]


@pytest.mark.asyncio
async def test_resuming_with_an_update_still_resumes_the_interrupt() -> None:
    """盯住框架行为：`Command` 同时带 resume 和 update 时两者都要生效。

    人工确认是这个系统的安全边界，恢复链路不能因为顺手塞了一条消息就断掉。这里用
    真实的 LangGraph 跑一遍父图 + 领域子图的中断结构，而不是信任文档。
    """

    def ask(state: _Child) -> dict[str, Any]:
        del state
        return {"decided": bool(interrupt({"need": "confirm"})["approved"])}

    child = StateGraph(_Child)
    child.add_node("ask", ask)
    child.add_edge(START, "ask")
    child.add_edge("ask", END)
    subgraph = child.compile()

    async def run_child(state: _Parent) -> dict[str, Any]:
        del state
        result = await subgraph.ainvoke({"decided": False})
        return {"messages": [AIMessage(content=f"decided={result['decided']}")]}

    parent = StateGraph(_Parent)
    parent.add_node("run_child", run_child)
    parent.add_edge(START, "run_child")
    parent.add_edge("run_child", END)
    graph = parent.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t-1"}}
    async for _ in graph.astream(
        {"messages": [HumanMessage(content="hi")]}, config, subgraphs=True
    ):
        pass
    assert (await graph.aget_state(config)).next == ("run_child",)

    command: Command[Any] = Command(
        resume={"approved": True},
        update={"messages": [routes._decision_message(True, "提交差旅申请")]},
    )
    async for _ in graph.astream(command, config, subgraphs=True):
        pass

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()
    assert [
        (message.type, str(message.content)) for message in snapshot.values["messages"]
    ] == [
        ("human", "hi"),
        ("system", "你确认了：提交差旅申请"),
        ("ai", "decided=True"),
    ]
