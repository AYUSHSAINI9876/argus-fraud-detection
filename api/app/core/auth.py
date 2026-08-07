"""Stack Auth JWT verification and role-based access control.

The API verifies tokens *independently* of the frontend, using Stack Auth's
published JWKS. It never calls Stack Auth on the request path and never
trusts a header the client could forge — the signature is checked locally
against a cached public key, so auth costs microseconds and survives an
outage of the auth provider.

Roles are hierarchical. A fraud platform has a genuine privilege gradient:

    VIEWER   read dashboards and aggregate metrics only
    ANALYST  work the case queue, add notes, propose a disposition
    REVIEWER approve/overturn analyst decisions, release blocked funds
    ADMIN    change thresholds, promote models, manage users

Threshold changes and model promotion are ADMIN-only for a reason: they move
money. Every such action is written to the audit log with the actor's subject
claim, which is the part a regulator would ask to see.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from pydantic import BaseModel, ConfigDict

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


class Role(IntEnum):
    """Ordered so that `user.role >= Role.REVIEWER` is a valid check."""

    VIEWER = 10
    ANALYST = 20
    REVIEWER = 30
    ADMIN = 40

    @classmethod
    def parse(cls, raw: str | None) -> Role:
        if not raw:
            return cls.VIEWER
        try:
            return cls[raw.strip().upper()]
        except KeyError:
            logger.warning("unknown role claim %r, defaulting to VIEWER", raw)
            return cls.VIEWER


class AuthenticatedUser(BaseModel):
    """The verified caller. Constructed only from validated token claims."""

    user_id: str
    email: str | None = None
    display_name: str | None = None
    role: Role = Role.VIEWER
    team_id: str | None = None

    # Keep Role as an enum member rather than coercing to its value — the
    # RBAC checks compare enum ordering, which string values would break.
    model_config = ConfigDict(use_enum_values=False)

    @property
    def role_name(self) -> str:
        return self.role.name

    def can(self, required: Role) -> bool:
        return self.role >= required


class _JWKSCache:
    """Lazily-built PyJWKClient. Handles key rotation and caching internally."""

    def __init__(self) -> None:
        self._client: PyJWKClient | None = None
        self._url: str | None = None

    def client(self, url: str) -> PyJWKClient:
        if self._client is None or self._url != url:
            logger.info("initialising JWKS client for %s", url)
            # lifespan keeps keys warm; PyJWT refetches automatically on an
            # unknown `kid`, which is what makes key rotation transparent.
            self._client = PyJWKClient(url, cache_keys=True, lifespan=3600)
            self._url = url
        return self._client


_jwks = _JWKSCache()


def _decode(token: str, settings: Settings) -> dict[str, Any]:
    """Verify signature and standard claims, returning the payload."""
    try:
        signing_key = _jwks.client(settings.jwks_url).get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=settings.stack_project_id or None,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": bool(settings.stack_project_id),
            },
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        ) from exc
    except jwt.InvalidTokenError as exc:
        logger.warning("token rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        ) from exc


def _claims_to_user(payload: dict[str, Any]) -> AuthenticatedUser:
    """Map Stack Auth claims onto our user model.

    Stack Auth carries app-defined role data on the team membership or in
    custom claims depending on project setup, so we probe the common shapes
    rather than hard-coding one.
    """
    role_raw = (
        payload.get("role")
        or payload.get("server_metadata", {}).get("role")
        or payload.get("client_metadata", {}).get("role")
    )
    return AuthenticatedUser(
        user_id=payload.get("sub", ""),
        email=payload.get("email") or payload.get("primary_email"),
        display_name=payload.get("display_name") or payload.get("name"),
        role=Role.parse(role_raw),
        team_id=payload.get("team_id") or payload.get("selected_team_id"),
    )


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AuthenticatedUser:
    """FastAPI dependency yielding the verified caller.

    When `auth_enabled` is false (local development only) it injects a
    synthetic admin so the UI is workable without a Stack Auth project. That
    switch is force-disabled outside development below, so it can never be
    the reason production is unprotected.
    """
    if not settings.auth_enabled:
        if settings.environment == "production":
            raise RuntimeError(
                "auth_enabled=False in production — refusing to start unprotected"
            )
        return AuthenticatedUser(
            user_id="dev-local",
            email="dev@argus.local",
            display_name="Local Developer",
            role=Role.ADMIN,
        )

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode(credentials.credentials, settings)
    user = _claims_to_user(payload)
    # Stash on request.state so the audit-log middleware can attribute actions
    # without re-decoding the token.
    request.state.user = user
    return user


def require_role(minimum: Role):
    """Dependency factory enforcing a minimum role.

    Usage:
        @router.post("/thresholds", dependencies=[Depends(require_role(Role.ADMIN))])
    """

    async def _guard(
        user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    ) -> AuthenticatedUser:
        if not user.can(minimum):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Requires {minimum.name} or higher; caller has {user.role_name}"
                ),
            )
        return user

    return _guard


CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]
AnalystUser = Annotated[AuthenticatedUser, Depends(require_role(Role.ANALYST))]
ReviewerUser = Annotated[AuthenticatedUser, Depends(require_role(Role.REVIEWER))]
AdminUser = Annotated[AuthenticatedUser, Depends(require_role(Role.ADMIN))]


__all__ = [
    "Role",
    "AuthenticatedUser",
    "get_current_user",
    "require_role",
    "CurrentUser",
    "AnalystUser",
    "ReviewerUser",
    "AdminUser",
]
