"""检查点序列化器：显式登记状态里会出现的本项目类型。

LangGraph 默认放行未登记的类型，只打一条"将来会被拦截"的警告。升级到拦截的版本后，
所有存着 PlannedTask、TaskStatus 这类对象的检查点都读不出来，停在待补充、待确认上的
会话会直接恢复失败。
"""

import inspect
from enum import Enum

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from enterprise_ai_assistant.core import models


def _state_types() -> list[tuple[str, str]]:
    # 按模块自动收集，而不是手写清单：新增一个进状态的模型时忘了登记，只有在严格模式下
    # 恢复会话才会暴露，测试和评测默认都不开严格模式。
    # 按模块里的名字登记而不是类自己的 __name__：改名后留下的别名（ShelvedPlan）是旧检查点
    # 里写着的类名，只登记新名字，严格模式下旧检查点就读不出来。
    return [
        (models.__name__, name)
        for name, value in inspect.getmembers(models, inspect.isclass)
        if value.__module__ == models.__name__ and issubclass(value, (BaseModel, Enum))
    ]


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_state_types())
