"""未办完的事项：计划在用户眼里的样子。

事项没有自己的存储：计划、任务状态、草稿、搁置计划都在父图检查点里，这里只是把它们
翻译成事项。哪些计划算未办完、卡在哪个任务上，规则集中在 project_matters 一个函数里，
两个读者都用它：

- 右栏"进行中"：快照（done、页面加载、GET 接口）和执行中途的检查点事件。
- Context Supervisor：每轮开头经 open_matters 拿到同一份事项，去掉字段值。

两边各有一套算法时，模型看到的和用户看到的会对不上：右栏列着排队的会议室任务，
模型却只知道正在追问的差旅任务，用户提到会议室时它认不出是哪件事。右栏自己也吃过这个亏，
快照和执行中途各算一套，卡片在两个任务之间消失、失败后一直显示处理中。不要再为某个读者
或某个时机另写一条投影路径。
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from enterprise_ai_assistant.core.models import (
    AgentName,
    DraftField,
    PendingConfirmation,
    Plan,
    PlannedTask,
    TaskDraft,
    TaskStatus,
    recover_interrupted,
)

_STUCK = {TaskStatus.WAITING_INPUT, TaskStatus.WAITING_CONFIRMATION}


class MatterTask(BaseModel):
    id: str
    title: str
    domain: AgentName
    status: TaskStatus


#: in_progress 只在运行执行期间出现：当前计划没有卡住的任务，但还有任务排队或在跑。
MatterStatus = Literal["waiting_input", "waiting_confirmation", "in_progress", "shelved"]


class Matter(BaseModel):
    """右栏的一张事项卡：一件还没办完的事。

    卡在待补充、待确认上的计划，以及执行中途还有任务排队或在跑的当前计划，才会成为事项。
    办完的计划不出现在这里——提交过的单据已经在"我的单据"里，纯查询也没有需要跟进的状态。
    """

    plan_id: str
    status: MatterStatus
    # 卡住或正在处理的那个任务；卡片标题和字段都来自它，同一计划里的其他任务作为子项列出。
    task_id: str
    title: str
    known_fields: list[DraftField] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    tasks: list[MatterTask] = Field(default_factory=list)


class OpenMatterTask(BaseModel):
    task_id: str
    title: str
    domain: AgentName
    status: TaskStatus
    # 只有等用户补充的任务有：Supervisor 要知道"刚才在问什么"，才能认出"当天往返""1"
    # 这种短回复在补充谁。
    missing_fields: list[str] = Field(default_factory=list)


class OpenMatter(BaseModel):
    """交给 Context Supervisor 的一件未办完的事。

    和右栏是同一份事项，只是去掉了字段值：拿到值，Supervisor 就有了补写领域字段的材料。
    """

    plan_id: str
    # false 是当前事项，true 是用户换话题时被搁置的事项。
    shelved: bool = False
    # 计划里的全部任务，按计划顺序，和右栏卡片上列出的一样。
    tasks: list[OpenMatterTask] = Field(default_factory=list)


def project_matters(
    current: Plan,
    shelved: Sequence[Plan],
    *,
    running: bool,
    pending: PendingConfirmation | None = None,
) -> list[Matter]:
    """把计划投影成事项，当前计划在前，搁置计划按最近搁置的在前。

    running 表示这个会话此刻有没有运行在执行，由调用方从运行管理器得知，不从任务状态猜：
    - 在执行时，当前计划没有卡住的任务、但还有任务排队或在跑，显示为处理中。
    - 没在执行时，停在 RUNNING 的任务是失败、重启或取消留下的残留，按 recover_interrupted
      解读成它真实所处的状态，不会显示成一件永远处理不完的事。

    pending 只有快照调用方给：执行中途和每轮开头都不会停在确认卡上。
    """
    if not running:
        current = current.model_copy(
            update={"tasks": recover_interrupted(current.tasks, current.drafts)}
        )
    cards = [
        _card(current, pending=pending, shelved=False, running=running),
        # 最近搁置的排在前面：用户最可能想接着办的是刚放下的那件。
        *(_card(plan, pending=None, shelved=True, running=False) for plan in reversed(shelved)),
    ]
    return [card for card in cards if card is not None]


def open_matters(current: Plan, shelved: Sequence[Plan]) -> list[OpenMatter]:
    """每轮开头交给 Supervisor 的事项，和上一轮结束后右栏显示的一样。"""
    # 任务和状态取卡片上的，不回头读计划：停在 RUNNING 的残留在投影里已经按恢复后的状态
    # 解读过，两边各读一份就会出现右栏写着待补充、模型看到的却是执行中。
    drafts = {plan.plan_id: plan.drafts for plan in (current, *shelved)}
    return [
        OpenMatter(
            plan_id=card.plan_id,
            shelved=card.status == "shelved",
            tasks=[
                OpenMatterTask(
                    task_id=task.id,
                    title=task.title,
                    domain=task.domain,
                    status=task.status,
                    missing_fields=(
                        drafts[card.plan_id][task.id].missing_fields
                        if task.status == TaskStatus.WAITING_INPUT
                        and task.id in drafts[card.plan_id]
                        else []
                    ),
                )
                for task in card.tasks
            ],
        )
        for card in project_matters(current, shelved, running=False)
    ]


def _card(
    plan: Plan,
    *,
    pending: PendingConfirmation | None,
    shelved: bool,
    running: bool,
) -> Matter | None:
    tasks, drafts = plan.tasks, plan.drafts
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
            return _matter(plan, status, stuck, _confirmation_draft(pending))
        return _matter(plan, status, stuck, drafts.get(stuck.id, TaskDraft()))
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
    return _matter(plan, "in_progress", working, TaskDraft(known_fields=known))


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


def _matter(plan: Plan, status: MatterStatus, focus: PlannedTask, draft: TaskDraft) -> Matter:
    return Matter(
        plan_id=plan.plan_id,
        status=status,
        task_id=focus.id,
        title=focus.title,
        known_fields=draft.known_fields,
        missing_fields=draft.missing_fields,
        tasks=[
            MatterTask(id=task.id, title=task.title, domain=task.domain, status=task.status)
            for task in plan.tasks
        ],
    )
