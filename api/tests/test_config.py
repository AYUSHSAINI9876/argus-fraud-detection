"""Tests for runtime configuration.

The database-URL normalisation is production-critical and easy to regress:
it only ever matters on a managed host, so a local test suite that skipped it
would stay green while every cloud deploy failed at startup.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings

ASYNC_PREFIX = "postgresql+asyncpg://"


class TestDatabaseUrlNormalisation:
    @pytest.mark.parametrize(
        "raw",
        [
            # Render, Heroku and Railway all emit the bare `postgres://` form,
            # which SQLAlchemy 2 removed support for entirely.
            "postgres://argus:pw@db.internal:5432/argus",
            # `postgresql://` parses, but selects the *sync* psycopg driver and
            # then fails at runtime under create_async_engine.
            "postgresql://argus:pw@db.internal:5432/argus",
            # Already correct — must pass through unchanged.
            "postgresql+asyncpg://argus:pw@db.internal:5432/argus",
        ],
    )
    def test_all_forms_resolve_to_the_async_driver(self, raw):
        assert Settings(database_url=raw).database_url.startswith(ASYNC_PREFIX)

    def test_credentials_and_path_survive_the_rewrite(self):
        s = Settings(database_url="postgres://argus:s3cret@db.internal:5432/argus")
        assert s.database_url == f"{ASYNC_PREFIX}argus:s3cret@db.internal:5432/argus"

    def test_sslmode_is_stripped(self):
        """libpq understands ?sslmode=require; asyncpg raises on it.

        asyncpg negotiates TLS on its own, so dropping the parameter is safe
        and is the difference between a booting service and a crash loop.
        """
        s = Settings(database_url="postgres://u:p@host/db?sslmode=require")
        assert "sslmode" not in s.database_url
        assert s.database_url == f"{ASYNC_PREFIX}u:p@host/db"

    def test_idempotent(self):
        once = Settings(database_url="postgres://u:p@h/db").database_url
        twice = Settings(database_url=once).database_url
        assert once == twice


class TestProductionSafety:
    def test_jwks_url_derives_from_project_id(self):
        s = Settings(stack_project_id="proj_abc123")
        assert s.jwks_url.endswith("/proj_abc123/.well-known/jwks.json")

    def test_explicit_jwks_url_wins(self):
        s = Settings(stack_project_id="proj_abc", stack_jwks_url="https://self.hosted/jwks")
        assert s.jwks_url == "https://self.hosted/jwks"

    def test_no_secret_has_a_default(self):
        """A secret with a default is a secret that ships in the image."""
        s = Settings()
        assert s.stack_secret_server_key == ""
        assert s.anthropic_api_key == ""
        assert s.resend_api_key == ""
