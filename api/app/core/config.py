"""Runtime configuration, loaded from environment with sane local defaults.

Every value here is overridable by env var so the same image runs unchanged
in local Docker, CI and production. Nothing secret has a default.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # -- service ----------------------------------------------------------
    app_name: str = "Argus Risk Engine"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000"]

    # -- artefacts --------------------------------------------------------
    artifacts_dir: Path = Path(__file__).resolve().parents[3] / "ml" / "artifacts"
    champion_model: str = "xgboost_risk"
    challenger_model: str | None = "logistic_baseline"
    anomaly_model: str = "gaussian_anomaly"
    # Fraction of traffic mirrored to the challenger. Shadow only — the
    # challenger's score is logged for comparison but never acted on.
    challenger_traffic_pct: float = 1.0

    # -- decision policy --------------------------------------------------
    # Overridden at startup by the thresholds fitted during training.
    review_threshold: float = 0.35
    block_threshold: float = 0.85
    anomaly_review_threshold: float = 0.97

    # -- stores -----------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+asyncpg://argus:argus@localhost:5432/argus"
    state_ttl_seconds: int = 60 * 60 * 24 * 8  # one day beyond the 7d window

    # -- auth (Stack Auth) ------------------------------------------------
    stack_project_id: str = ""
    stack_publishable_client_key: str = ""
    stack_secret_server_key: str = ""
    stack_jwks_url: str = ""
    auth_enabled: bool = True

    # -- email (provider-agnostic; Resend by default) ---------------------
    email_provider: str = "resend"          # "resend" | "sendgrid" | "console"
    resend_api_key: str = ""
    sendgrid_api_key: str = ""
    email_from: str = "Argus Alerts <alerts@argus.local>"

    # -- LLM copilot ------------------------------------------------------
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5"
    # Effort is the primary cost/latency lever. Case narratives are short and
    # highly structured, so "low" is sufficient and keeps the copilot cheap
    # enough to run on every review-queue case.
    llm_effort: str = "low"
    copilot_enabled: bool = True
    copilot_max_similar_cases: int = 5

    @field_validator("database_url")
    @classmethod
    def _normalise_database_url(cls, v: str) -> str:
        """Coerce a managed-provider URL into the async driver form.

        Render, Heroku, Railway and Neon all hand out `postgres://...`.
        SQLAlchemy 2 removed support for that scheme outright, and even
        `postgresql://` would select the *sync* psycopg driver — which fails
        at runtime under `create_async_engine`. Rewriting here means the same
        image boots unchanged whether the URL came from a managed provider,
        docker-compose, or a local dev shell.
        """
        if v.startswith("postgres://"):
            v = "postgresql+asyncpg://" + v[len("postgres://"):]
        elif v.startswith("postgresql://"):
            v = "postgresql+asyncpg://" + v[len("postgresql://"):]

        # Managed providers often append `?sslmode=require`. libpq understands
        # that; asyncpg does not and raises on the unknown keyword. asyncpg
        # negotiates TLS automatically, so dropping it is safe.
        if "?sslmode=" in v:
            v = v.split("?sslmode=")[0]
        return v

    @property
    def jwks_url(self) -> str:
        """Stack Auth publishes per-project JWKS; allow a full override."""
        if self.stack_jwks_url:
            return self.stack_jwks_url
        return (
            "https://api.stack-auth.com/api/v1/projects/"
            f"{self.stack_project_id}/.well-known/jwks.json"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
