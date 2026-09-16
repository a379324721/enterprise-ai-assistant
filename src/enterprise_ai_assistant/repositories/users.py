"""演示用户名册。

这不是身份系统：没有凭据，输入名字即可取得该名字的全部会话与记忆。它的存在只是
为了让演示对象用自己的名字进来，接口层必须靠 APP_ENV 和 DEMO_LOGIN_ENABLED 双重
开关把它挡在生产之外，正式部署由企业 SSO 取代。

没有独立的注册动作：在没有凭据的前提下，注册和登录是同一件事，把名字占用做成
冲突错误挡不住任何冒用，只会让演示现场打错字的人多点一次按钮。
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

import asyncpg

#: 会话 ID 的派生命名空间。演示用户固定一个会话，从名字确定性推出会话 ID，
#: 客户端换设备或清了本地存储后仍能回到同一个会话，不需要额外存一张映射表。
_CONVERSATION_NAMESPACE = "urn:enterprise-ai-assistant:demo-conversation"

_WHITESPACE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """把展示名折叠成稳定标识。

    去掉首尾空白并把内部连续空白压成一个空格，避免"张三"和"张 三 "被当成两个人——
    它们会各自持有一份互不可见的会话和记忆，在演示现场很难解释。
    """
    return _WHITESPACE.sub(" ", name).strip()


def conversation_id_for(user_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"{_CONVERSATION_NAMESPACE}:{user_id}")


@dataclass(frozen=True)
class DemoUser:
    user_id: str
    display_name: str
    created_at: datetime


class DemoUserRepository(Protocol):
    async def get_or_create(self, user_id: str, display_name: str) -> tuple[DemoUser, bool]: ...


class PostgresDemoUserRepository(DemoUserRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_or_create(self, user_id: str, display_name: str) -> tuple[DemoUser, bool]:
        """按名字取用户，不存在就建一个；第二个返回值表示本次是否新建。

        先插再查而不是先查再插：并发提交同一个名字时，先查后插会让两个请求都认为
        自己是首个注册者。冲突时 RETURNING 不出行，此时那一行必定已经存在，回查即可。
        """
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO demo_users (user_id, display_name)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO NOTHING
                RETURNING user_id, display_name, created_at
                """,
                user_id,
                display_name,
            )
            if row is not None:
                return DemoUser(**dict(row)), True
            existing = await connection.fetchrow(
                "SELECT user_id, display_name, created_at FROM demo_users WHERE user_id = $1",
                user_id,
            )
        return DemoUser(**dict(existing)), False


class InMemoryDemoUserRepository(DemoUserRepository):
    def __init__(self) -> None:
        self.users: dict[str, DemoUser] = {}

    async def get_or_create(self, user_id: str, display_name: str) -> tuple[DemoUser, bool]:
        existing = self.users.get(user_id)
        if existing is not None:
            return existing, False
        user = DemoUser(user_id, display_name, datetime.now().astimezone())
        self.users[user_id] = user
        return user, True
