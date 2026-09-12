"""基于 JWT 的调用方身份校验。

用户身份必须来自签名令牌而不是请求头：`X-User-ID` 这类裸传字段可以被任意
伪造，任何人改一个值就能读写他人的会话和业务单据。
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from uuid import uuid4

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from enterprise_ai_assistant.core.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False, description="企业 SSO 颁发的访问令牌")

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="缺少或无效的访问令牌",
    headers={"WWW-Authenticate": "Bearer"},
)


@dataclass(frozen=True)
class Identity:
    """令牌携带的调用方身份。

    display_name 只用于展示和称呼，绝不参与会话归属或幂等键——那些一律取 user_id。
    """

    user_id: str
    display_name: str

    @property
    def name(self) -> str:
        return self.display_name or self.user_id


def create_access_token(
    user_id: str,
    settings: Settings | None = None,
    ttl: timedelta | None = None,
    display_name: str | None = None,
) -> tuple[str, int]:
    """签发访问令牌，返回 (token, 有效期秒数)。"""
    config = settings or get_settings()
    lifetime = ttl or timedelta(minutes=config.access_token_ttl_minutes)
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": user_id,
        "iss": config.jwt_issuer,
        "aud": config.jwt_audience,
        "iat": now,
        "exp": now + lifetime,
        "jti": str(uuid4()),
    }
    # 姓名走令牌而不是长期记忆：用户不会对助手自报姓名，抽取不到；而称呼是"永远相关"
    # 的身份信息，交给按相关性筛选的记忆链路会在问候这类输入上被筛掉。
    if display_name:
        payload["name"] = display_name
    token = jwt.encode(payload, config.signing_key, algorithm=config.jwt_algorithm)
    return token, int(lifetime.total_seconds())


def decode_identity(token: str, settings: Settings | None = None) -> Identity:
    """校验令牌并返回其中的调用方身份。"""
    config = settings or get_settings()
    try:
        payload = jwt.decode(
            token,
            config.signing_key,
            algorithms=[config.jwt_algorithm],
            audience=config.jwt_audience,
            issuer=config.jwt_issuer,
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except jwt.InvalidTokenError as exc:
        raise _UNAUTHORIZED from exc
    subject = payload.get("sub")
    # sub 直接参与会话归属和幂等键构造，长度和类型必须收敛。
    if not isinstance(subject, str) or not 1 <= len(subject) <= 128:
        raise _UNAUTHORIZED
    claimed_name = payload.get("name")
    # 展示名只影响文案，取不到就退回标识，不因为它缺失或超长而拒绝整个令牌。
    display_name = (
        claimed_name if isinstance(claimed_name, str) and 1 <= len(claimed_name) <= 128 else subject
    )
    return Identity(user_id=subject, display_name=display_name)


def decode_access_token(token: str, settings: Settings | None = None) -> str:
    """校验令牌并返回其中的用户标识。"""
    return decode_identity(token, settings).user_id


async def get_current_identity(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Identity:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _UNAUTHORIZED
    return decode_identity(credentials.credentials)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    return (await get_current_identity(credentials)).user_id


CurrentUser = Annotated[str, Depends(get_current_user)]
CurrentIdentity = Annotated[Identity, Depends(get_current_identity)]
