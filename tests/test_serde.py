from enterprise_ai_assistant.core import models
from enterprise_ai_assistant.core.models import AgentName, Plan, PlannedTask
from enterprise_ai_assistant.graph.serde import checkpoint_serializer


def test_plans_stored_under_the_old_class_name_still_load() -> None:
    # 改名前的检查点里，搁置计划按 ShelvedPlan 这个类名序列化。造一个同名同模块的类来
    # 编码，模拟线上已有的数据；序列化器只放行登记过的类型，没登记旧名字就读不出来。
    legacy = type("ShelvedPlan", (Plan,), {"__module__": models.__name__})
    stored = legacy(
        plan_id="p-1",
        user_goal="去上海出差",
        tasks=[PlannedTask(id="task-1", title="提交差旅申请", domain=AgentName.TRAVEL, objective="x")],
    )
    serializer = checkpoint_serializer()

    loaded = serializer.loads_typed(serializer.dumps_typed(stored))

    assert type(loaded) is Plan
    assert loaded.plan_id == "p-1"
    assert loaded.tasks[0].id == "task-1"
