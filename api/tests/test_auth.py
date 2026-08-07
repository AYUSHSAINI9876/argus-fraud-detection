"""Tests for role-based access control.

The role hierarchy decides who can release blocked funds and who can change
thresholds that move money for every customer. These tests pin the ordering
and the fail-safe defaults — the two things that would be quietly catastrophic
to get wrong.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.auth import AuthenticatedUser, Role, _claims_to_user


class TestRoleHierarchy:
    def test_roles_are_strictly_ordered(self):
        assert Role.VIEWER < Role.ANALYST < Role.REVIEWER < Role.ADMIN

    def test_can_is_inclusive_upward(self):
        admin = AuthenticatedUser(user_id="u", role=Role.ADMIN)
        assert admin.can(Role.VIEWER)
        assert admin.can(Role.ANALYST)
        assert admin.can(Role.REVIEWER)
        assert admin.can(Role.ADMIN)

    def test_analyst_cannot_reach_reviewer_actions(self):
        analyst = AuthenticatedUser(user_id="u", role=Role.ANALYST)
        assert analyst.can(Role.ANALYST)
        assert not analyst.can(Role.REVIEWER)
        assert not analyst.can(Role.ADMIN)

    def test_reviewer_cannot_change_policy(self):
        """Releasing one block is REVIEWER; changing the threshold for every
        future transaction is ADMIN. These are different blast radii."""
        reviewer = AuthenticatedUser(user_id="u", role=Role.REVIEWER)
        assert reviewer.can(Role.REVIEWER)
        assert not reviewer.can(Role.ADMIN)


class TestRoleParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ADMIN", Role.ADMIN),
            ("admin", Role.ADMIN),
            ("  Reviewer  ", Role.REVIEWER),
            ("analyst", Role.ANALYST),
        ],
    )
    def test_parses_case_and_whitespace_insensitively(self, raw, expected):
        assert Role.parse(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "superuser", "root", "ADMIN2"])
    def test_unknown_and_missing_roles_fail_closed(self, raw):
        """An unrecognised claim must never escalate. VIEWER is the floor."""
        assert Role.parse(raw) == Role.VIEWER


class TestClaimMapping:
    def test_reads_role_from_top_level_claim(self):
        user = _claims_to_user({"sub": "u1", "email": "a@b.c", "role": "REVIEWER"})
        assert user.user_id == "u1"
        assert user.role == Role.REVIEWER

    def test_reads_role_from_server_metadata(self):
        user = _claims_to_user({"sub": "u1", "server_metadata": {"role": "ADMIN"}})
        assert user.role == Role.ADMIN

    def test_top_level_claim_wins_over_metadata(self):
        user = _claims_to_user(
            {"sub": "u1", "role": "ANALYST", "server_metadata": {"role": "ADMIN"}}
        )
        assert user.role == Role.ANALYST

    def test_missing_role_defaults_to_viewer(self):
        """A user who signed up but has not been granted anything reads only."""
        user = _claims_to_user({"sub": "u1", "email": "new@user.com"})
        assert user.role == Role.VIEWER

    def test_client_metadata_is_accepted_but_ranks_last(self):
        """Client metadata is user-writable in some Stack Auth configurations,
        so it must never outrank a server-set claim."""
        user = _claims_to_user(
            {"sub": "u1", "server_metadata": {"role": "VIEWER"},
             "client_metadata": {"role": "ADMIN"}}
        )
        assert user.role == Role.VIEWER


class TestProductionSafety:
    @pytest.mark.asyncio
    async def test_auth_disabled_in_production_is_refused(self, monkeypatch):
        """The dev-mode bypass must not be able to leave production open."""
        from app.core.auth import get_current_user
        from app.core.config import Settings

        settings = Settings(auth_enabled=False, environment="production")

        class _Req:
            state = type("S", (), {})()

        with pytest.raises(RuntimeError, match="refusing to start unprotected"):
            await get_current_user(_Req(), None, settings)

    @pytest.mark.asyncio
    async def test_auth_disabled_in_development_yields_local_admin(self):
        from app.core.auth import get_current_user
        from app.core.config import Settings

        settings = Settings(auth_enabled=False, environment="development")

        class _Req:
            state = type("S", (), {})()

        user = await get_current_user(_Req(), None, settings)
        assert user.role == Role.ADMIN
        assert user.user_id == "dev-local"

    @pytest.mark.asyncio
    async def test_missing_bearer_token_is_401(self):
        from app.core.auth import get_current_user
        from app.core.config import Settings

        settings = Settings(auth_enabled=True, environment="development")

        class _Req:
            state = type("S", (), {})()

        with pytest.raises(HTTPException) as exc:
            await get_current_user(_Req(), None, settings)
        assert exc.value.status_code == 401
