"""右栏"进行中"事项的投影。

事项没有自己的存储：计划、任务状态、草稿、搁置计划都在父图检查点里，这里只是把它们
翻译成界面上的卡片。所有规则集中在 project_matters 一个函数里，快照（done、页面加载、
GET 接口）和执行中途的检查点事件用的都是它。

原先快照和执行中途各有一套算法，中途那套的输入是手工塞进 task_done 事件的几个字段，
前端再把两边的结果合并去重。两套一有出入就是 bug：卡片在两个任务之间消失、搁置的事项
在换话题那一轮不见、失败后一直显示处理中。不要再为某个时机另写一条投影路径。
"""

from collections.abc import Mapping
from typing import Any

from enterprise_ai_assistant.api.schemas import Matter, MatterStatus, MatterTask
from enterprise_ai_assistant.core.models import (
    DraftField,
    PendingConfirmation,
    PlannedTask,
    ShelvedPlan,
    TaskDraft,
    TaskStatus,
    recover_interrupted,
)

_STUCK = {TaskStatus.WAITING_INPUT, TaskStatus.WAITING_CONFIRMATION}


def project_matters(
    values: Mapping[str, Any],
    *,
    running: bool,
    tasks: list[PlannedTask] | None = None,
    pending: PendingConfirmation | None = None,
) -> list[Matter]:
    """把父图状态投影成事项卡，当前计划在前，搁置计划按最近搁置的在前。

    running 表示这个会话此刻有没有运行在执行，由调用方从运行管理器得知，不从任务状态猜：
    - 在执行时，当前计划没有卡住的任务、但还有任务排队或在跑，显示为处理中。
    - 没在执行时，停在 RUNNING 的任务是失败、重启或取消留下的残留，按 recover_interrupted
      解读成它真实所处的状态，不会显示成一件永远处理不完的事。

    tasks 和 pending 只有快照调用方给：快照里能读到中断和已跑完未归并的分支，调用方据此
    修正任务状态；检查点事件里没有这两样，执行中途也不会停在确认卡上。
    """
    drafts = _drafts(values.get("drafts"))
    current_tasks = [PlannedTask.model_validate(item) for item in values.get("tasks") or []]
    if tasks is not None:
        current_tasks = tasks
    if not running:
        current_tasks = recover_interrupted(current_tasks, drafts)
    current = _card(
        str(values.get("plan_id", "")),
        current_tasks,
        drafts,
        pending=pending,
        shelved=False,
        running=running,
    )
    # 最近搁置的排在前面：用户最可能想接着办的是刚放下的那件。
    parked = [ShelvedPlan.model_validate(item) for item in values.get("shelved_plans") or []]
    cards = [
        _card(plan.plan_id, plan.tasks, plan.drafts, pending=None, shelved=True, running=False)
        for plan in reversed(parked)
    ]
    return [card for card in (current, *cards) if card is not None]


def _card(
    plan_id: str,
    tasks: list[PlannedTask],
    drafts: Mapping[str, TaskDraft],
    *,
    pending: PendingConfirmation | None,
    shelved: bool,
    running: bool,
) -> Matter | None:
    stuck = next((task for task in tasks if task.status in _STUCK), None)
    if stuck is not None:
        status: MatterStatus = (
            "shelved"
            if shelved
            else "waiting_confirmation"
            if stuck.status == TaskStatus.WAITING_CONFIRMATION
            else "waiting_input"
        )
        if pending is not None and pending.task_id == stuck.id:
            return _matter(plan_id, status, stuck, _confirmation_draft(pending), tasks)
        return _matter(plan_id, status, stuck, drafts.get(stuck.id, TaskDraft()), tasks)
    if shelved or not running:
        return None
    # 先取在跑的，再取排队的：标题要对得上此刻真正在做的那件事。
    working = next((task for task in tasks if task.status == TaskStatus.RUNNING), None) or next(
        (task for task in tasks if task.status == TaskStatus.PENDING), None
    )
    if working is None:
        return None
    # 续跑的任务带着上一轮的草稿重新排队，已知字段仍然成立；缺失字段用户这一轮刚补过，
    # 列出来就是过期的"待补充"。
    known = drafts[working.id].known_fields if working.id in drafts else []
    return _matter(plan_id, "in_progress", working, TaskDraft(known_fields=known), tasks)


def _confirmation_draft(pending: PendingConfirmation) -> TaskDraft:
    # 草稿只在追问时写入、归并时才清掉。用户补完字段后任务直接停到确认卡上，草稿还是
    # 追问那一刻的，卡片会在"待确认"下面列出一排早已补齐的"待补充"。待确认时字段
    # 以确认卡上即将提交的参数为准。
    return TaskDraft(
        known_fields=[
            # 确认卡的字段没有长度约束，事由这类自由文本可能超出草稿字段的上限。
            DraftField(name=item.name[:64], label=item.label[:64], value=item.value[:500])
            for item in pending.fields
            if item.value.strip()
        ]
    )


def _drafts(raw: Any) -> dict[str, TaskDraft]:
    return {task_id: TaskDraft.model_validate(draft) for task_id, draft in (raw or {}).items()}


def _matter(
    plan_id: str,
    status: MatterStatus,
    focus: PlannedTask,
    draft: TaskDraft,
    tasks: list[PlannedTask],
) -> Matter:
    return Matter(
        plan_id=plan_id,
        status=status,
        task_id=focus.id,
        title=focus.title,
        known_fields=draft.known_fields,
        missing_fields=draft.missing_fields,
        tasks=[
            MatterTask(id=task.id, title=task.title, domain=task.domain, status=task.status)
            for task in tasks
        ],
    )
