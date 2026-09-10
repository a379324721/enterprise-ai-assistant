"""基于 JWT 的调用方身份校验。

用户身份必须来自签名令牌而不是请求头：`X-User-ID` 这类裸传字段可以被任意
伪造，任何人改一个值就能读写他人的会话和业务单据。
"""

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


def create_access_token(
    user_id: str, settings: Settings | None = None, ttl: timedelta | None = None
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
    token = jwt.encode(payload, config.signing_key, algorithm=config.jwt_algorithm)
    return token, int(lifetime.total_seconds())


def decode_access_token(token: str, settings: Settings | None = None) -> str:
    """校验令牌并返回其中的用户标识。"""
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
    return subject


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _UNAUTHORIZED
    return decode_access_token(credentials.credentials)


CurrentUser = Annotated[str, Depends(get_current_user)]
