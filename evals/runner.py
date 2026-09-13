"""执行回归评测并输出指标报告。

用法：
    uv run python -m evals.runner                       # 跑全部
    uv run python -m evals.runner --suite planning      # 只跑规划
    uv run python -m evals.runner --json report.json    # 导出报告

评测直接复用生产代码路径（同一套 prompt、同一套工具注册表），
仅把 Postgres/Milvus 换成内存实现，因此不需要启动任何外部依赖，
只需要一个可用的模型服务配置。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from enterprise_ai_assistant.agents.domain_runtime import (
    DomainRuntime,
    DomainRuntimeFactory,
    DomainRuntimeProvider,
)
from enterprise_ai_assistant.agents.supervisor import SupervisorAgent
from enterprise_ai_assistant.core.models import (
    AgentName,
    PlannedTask,
)
from enterprise_ai_assistant.repositories.actions import InMemoryActionRepository
from enterprise_ai_assistant.repositories.policies import InMemoryPolicyRepository
from enterprise_ai_assistant.services.llm import build_chat_model
from enterprise_ai_assistant.services.planning import LLMPlanningService, PlanningService
from enterprise_ai_assistant.tools import LocalEnterpriseToolProvider, ToolContext, ToolRisk
from enterprise_ai_assistant.tools.registry import DomainToolRegistry
from evals.dataset import (
    ContextCase,
    ConversationTurn,
    DomainAnswerCase,
    EvalDataset,
    GuardrailCase,
    PlanningCase,
    SmallTalkCase,
    ToolChoiceCase,
    load_dataset,
)

SUITES = (
    "context",
    "planning",
    "tool_choice",
    "guardrail",
    "small_talk",
    "domain_answer",
)


@dataclass(frozen=True)
class CaseResult:
    suite: str
    case_id: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "case_id": self.case_id,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class SuiteReport:
    suite: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for item in self.results if item.passed)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def failures(self) -> list[CaseResult]:
        return [item for item in self.results if not item.passed]


class EvalHarness:
    """把评测用例映射到生产组件上。

    默认使用真实模型；两个依赖都可注入，以便用桩件对判定逻辑本身做单元测试。
    """

    def __init__(
        self,
        planning: PlanningService | None = None,
        domains: DomainRuntimeProvider | None = None,
    ) -> None:
        if planning is None or domains is None:
            provider = LocalEnterpriseToolProvider(
                InMemoryActionRepository(), InMemoryPolicyRepository()
            )
            planning = planning or LLMPlanningService(build_chat_model("supervisor"))
            domains = domains or DomainRuntimeFactory(
                build_chat_model("domain"), DomainToolRegistry(provider)
            )
        self._planning = planning
        self._domains = domains
        self._conversation_id = uuid4()

    # -- context ---------------------------------------------------------

    async def run_context_case(self, case: ContextCase) -> CaseResult:
        conversation = [turn.model_dump() for turn in case.conversation]
        resolution = await self._planning.resolve_context(
            conversation, open_tasks=case.open_tasks
        )
        problems: list[str] = []
        if resolution.requires_task_planning != case.expect_task_planning:
            problems.append(
                f"requires_task_planning={resolution.requires_task_planning}"
                f"，期望 {case.expect_task_planning}"
            )
        if (
            case.expect_turn_relation is not None
            and resolution.turn_relation != case.expect_turn_relation
        ):
            problems.append(
                f"turn_relation={resolution.turn_relation.value}"
                f"，期望 {case.expect_turn_relation.value}"
            )
        if (
            case.expect_target_plan_id is not None
            and resolution.target_plan_id != case.expect_target_plan_id
        ):
            problems.append(
                f"target_plan_id={resolution.target_plan_id}，期望 {case.expect_target_plan_id}"
            )
        if case.expect_domains and set(resolution.domains) != set(case.expect_domains):
            problems.append(
                f"domains={[item.value for item in resolution.domains]}"
                f"，期望 {[item.value for item in case.expect_domains]}"
            )
        if resolution.target_plan_id in case.forbid_target_plan_ids:
            problems.append(f"target_plan_id 指向了不该恢复的事项 {resolution.target_plan_id}")
        missing = [
            keyword
            for keyword in case.expect_keywords
            if keyword not in resolution.standalone_request
        ]
        if missing:
            problems.append(
                f"改写后的请求缺少关键词 {missing}：{resolution.standalone_request!r}"
            )
        if case.expect_any_keywords and not any(
            keyword in resolution.standalone_request for keyword in case.expect_any_keywords
        ):
            problems.append(
                f"改写后的请求未命中任何指代 {case.expect_any_keywords}："
                f"{resolution.standalone_request!r}"
            )
        return CaseResult("context", case.id, not problems, "；".join(problems))

    # -- planning --------------------------------------------------------

    async def run_planning_case(self, case: PlanningCase) -> CaseResult:
        resolution = await self._planning.resolve_context(
            [{"role": "user", "content": case.request}]
        )
        # 与生产路径一致：任务由 Supervisor 在理解结果里给出，没给出时才退回 Planner。
        plan = await SupervisorAgent(self._planning).plan(resolution)
        problems: list[str] = []
        domains = [task.domain for task in plan.tasks]
        if set(domains) != set(case.expect_domains):
            problems.append(
                f"领域路由为 {[item.value for item in domains]}"
                f"，期望 {[item.value for item in case.expect_domains]}"
            )
        if len(plan.tasks) != case.expect_task_count:
            problems.append(f"拆出 {len(plan.tasks)} 个任务，期望 {case.expect_task_count} 个")
        has_dependency = any(task.depends_on for task in plan.tasks)
        if has_dependency != case.expect_dependency:
            problems.append(
                f"依赖关系存在={has_dependency}，期望 {case.expect_dependency}"
            )
        return CaseResult("planning", case.id, not problems, "；".join(problems))

    # -- domain agent ----------------------------------------------------

    def _runtime(self, domain: AgentName, task_id: str) -> DomainRuntime:
        return self._domains.create(
            domain,
            ToolContext(
                user_id="eval-user",
                conversation_id=self._conversation_id,
                request_id=uuid4(),
                task_id=task_id,
            ),
        )

    async def _decide(
        self,
        domain: AgentName,
        objective: str,
        user_goal: str,
        case_id: str,
        memories: Sequence[str] = (),
        recent_messages: Sequence[ConversationTurn] = (),
    ) -> tuple[DomainRuntime, AIMessage]:
        """复刻 DomainTaskWorkflow.initialize 构造的首轮输入。"""
        task = PlannedTask(id=case_id, title=case_id, domain=domain, objective=objective)
        payload: dict[str, Any] = {
            "standalone_request": user_goal,
            "task": task.model_dump(mode="json"),
            "dependency_results": {},
        }
        if memories:
            payload["user_memory"] = list(memories)
        if recent_messages:
            payload["recent_messages"] = [turn.model_dump() for turn in recent_messages]
        runtime = self._runtime(domain, case_id)
        response = await runtime.decide(
            objective,
            [HumanMessage(content=json.dumps(payload, ensure_ascii=False))],
            task_id=case_id,
        )
        return runtime, response

    async def run_tool_choice_case(self, case: ToolChoiceCase) -> CaseResult:
        _, response = await self._decide(
            case.domain,
            case.objective,
            case.user_goal,
            case.id,
            case.memories,
            case.recent_messages,
        )
        if not response.tool_calls:
            return CaseResult(
                "tool_choice", case.id, False, f"未调用任何工具，期望 {case.expect_tool}"
            )
        if len(response.tool_calls) > 1:
            names = [str(call["name"]) for call in response.tool_calls]
            return CaseResult("tool_choice", case.id, False, f"一次调用了多个工具 {names}")
        chosen = str(response.tool_calls[0]["name"])
        passed = chosen == case.expect_tool or chosen in case.also_accept
        detail = "" if passed else f"选中 {chosen}，期望 {[case.expect_tool, *case.also_accept]}"
        return CaseResult("tool_choice", case.id, passed, detail)

    async def run_guardrail_case(self, case: GuardrailCase) -> CaseResult:
        runtime, response = await self._decide(
            case.domain,
            case.objective,
            case.user_goal,
            case.id,
            case.memories,
            case.recent_messages,
        )
        problems: list[str] = []
        names = [str(call["name"]) for call in response.tool_calls]
        if case.forbid_write:
            written = [
                name
                for name in names
                if _risk_of(runtime, name) == ToolRisk.WRITE
            ]
            if written:
                problems.append(f"触发了写操作 {written}")
        if case.expect_information_request and "request_information" not in names:
            problems.append(f"未调用 request_information，实际调用 {names or '无'}")
        if case.forbid_information_request and "request_information" in names:
            problems.append("为本领域办不到的事项索要字段，等于承诺办理")
        if case.forbid_handoff and "handoff_task" in names:
            problems.append("把不该转交的任务转交给了其他领域")
        return CaseResult("guardrail", case.id, not problems, "；".join(problems))


    async def run_domain_answer_case(self, case: DomainAnswerCase) -> CaseResult:
        """喂进既定的工具结果，只考察最终回答的措辞。"""
        runtime = self._runtime(case.domain, case.id)
        messages: list[Any] = [HumanMessage(content=case.user_goal)]
        for index, result in enumerate(case.tool_results):
            call_id = f"{case.id}-tool-{index}"
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": "recorded", "args": {}, "id": call_id, "type": "tool_call"}],
                )
            )
            messages.append(ToolMessage(content=result, tool_call_id=call_id))
        # 和 DomainTaskWorkflow 走同一条路：看完工具结果的决策调用直接作答，
        # 它给不出文字时才落到兜底回答调用。只测兜底调用会漏掉线上真正在说话的那一次。
        response = await runtime.decide(
            case.objective, messages, task_id=case.id, answering=True
        )
        if response.tool_calls or not str(response.content).strip():
            response = await runtime.respond(case.objective, messages, task_id=case.id)
        answer = str(response.content)
        leaked = [phrase for phrase in case.forbid_phrases if phrase in answer]
        detail = f"回答越出真实能力 {leaked}：{answer}" if leaked else ""
        return CaseResult("domain_answer", case.id, not leaked, detail)

    async def run_small_talk_case(self, case: SmallTalkCase) -> CaseResult:
        resolution = await self._planning.resolve_context(
            [turn.model_dump() for turn in case.conversation],
            recent_actions=case.recent_actions,
            user_name=case.user_name,
        )
        if resolution.requires_task_planning:
            return CaseResult(
                "small_talk", case.id, False, f"被判为需要执行：{resolution.standalone_request}"
            )
        answer = resolution.reply
        leaked = [phrase for phrase in case.forbid_phrases if phrase in answer]
        detail = f"回复中出现了不应出现的说法 {leaked}：{answer}" if leaked else ""
        return CaseResult("small_talk", case.id, not leaked, detail)


def _risk_of(runtime: DomainRuntime, name: str) -> ToolRisk | None:
    """白名单外的工具名不算写操作——它在 decide 阶段就会被拒绝。"""
    try:
        return runtime.tool(name).risk
    except (KeyError, ValueError):
        return None


async def _gather(
    tasks: Sequence[Callable[[], Awaitable[CaseResult]]], concurrency: int
) -> list[CaseResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(factory: Callable[[], Awaitable[CaseResult]]) -> CaseResult:
        async with semaphore:
            try:
                return await factory()
            except Exception as exc:  # 单条用例异常不应中断整轮评测
                return CaseResult("unknown", "unknown", False, f"用例执行异常：{exc!r}")

    return list(await asyncio.gather(*(guarded(item) for item in tasks)))


async def run_suites(
    dataset: EvalDataset,
    suites: Sequence[str],
    concurrency: int,
    harness: EvalHarness | None = None,
) -> list[SuiteReport]:
    harness = harness or EvalHarness()
    reports: list[SuiteReport] = []
    plans: dict[str, list[Callable[[], Awaitable[CaseResult]]]] = {
        "context": [
            partial(harness.run_context_case, case) for case in dataset.context_cases
        ],
        "planning": [
            partial(harness.run_planning_case, case) for case in dataset.planning_cases
        ],
        "tool_choice": [
            partial(harness.run_tool_choice_case, case)
            for case in dataset.tool_choice_cases
        ],
        "guardrail": [
            partial(harness.run_guardrail_case, case) for case in dataset.guardrail_cases
        ],
        "small_talk": [
            partial(harness.run_small_talk_case, case)
            for case in dataset.small_talk_cases
        ],
        "domain_answer": [
            partial(harness.run_domain_answer_case, case)
            for case in dataset.domain_answer_cases
        ],
    }
    for suite in suites:
        factories = plans[suite]
        if not factories:
            continue
        results = await _gather(factories, concurrency)
        reports.append(SuiteReport(suite, results))
    return reports


def format_report(reports: Sequence[SuiteReport]) -> str:
    lines = ["", f"{'suite':<14}{'passed':>10}{'total':>8}{'accuracy':>11}", "-" * 43]
    total = sum(report.total for report in reports)
    passed = sum(report.passed for report in reports)
    for report in reports:
        lines.append(
            f"{report.suite:<14}{report.passed:>10}{report.total:>8}"
            f"{report.accuracy:>10.1%}"
        )
    lines.append("-" * 43)
    overall = passed / total if total else 0.0
    lines.append(f"{'overall':<14}{passed:>10}{total:>8}{overall:>10.1%}")

    failures = [item for report in reports for item in report.failures]
    if failures:
        lines.extend(["", f"失败用例（{len(failures)}）："])
        lines.extend(
            f"  [{item.suite}] {item.case_id}: {item.detail}" for item in failures
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="企业助手回归评测")
    parser.add_argument(
        "--suite",
        choices=(*SUITES, "all"),
        default="all",
        help="只运行指定评测集，默认全部",
    )
    parser.add_argument("--json", type=Path, default=None, help="把明细报告写入 JSON 文件")
    parser.add_argument("--concurrency", type=int, default=4, help="并发请求数")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="整体准确率低于该阈值时以非零码退出，便于 CI 卡门槛",
    )
    args = parser.parse_args(argv)

    dataset = load_dataset()
    suites = SUITES if args.suite == "all" else (args.suite,)
    reports = asyncio.run(run_suites(dataset, suites, max(1, args.concurrency)))
    print(format_report(reports))

    total = sum(report.total for report in reports)
    passed = sum(report.passed for report in reports)
    overall = passed / total if total else 0.0
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "overall_accuracy": overall,
                    "passed": passed,
                    "total": total,
                    "suites": {
                        report.suite: {
                            "passed": report.passed,
                            "total": report.total,
                            "accuracy": report.accuracy,
                            "results": [item.as_dict() for item in report.results],
                        }
                        for report in reports
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0 if overall >= args.min_accuracy else 1


if __name__ == "__main__":
    raise SystemExit(main())
