"""Argus Risk Engine — FastAPI application entrypoint.

Startup is deliberately fail-fast. The service loads model artefacts,
asserts the feature contract, connects to Redis and warms customer state
*before* accepting traffic. If any of that is wrong we would rather the
container never become healthy than have it serve confident nonsense.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.db import create_all, init_engine
from app.core.state_store import StateStore
from app.routers import admin, cases, copilot, health, metrics, score
from app.services.copilot import AnalystCopilot
from app.services.policy import PolicyConfig
from app.services.scoring import ModelBundle, ScoringService

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("argus")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("starting %s in %s mode", settings.app_name, settings.environment)

    if settings.environment == "production" and not settings.auth_enabled:
        raise RuntimeError("refusing to start: auth disabled in production")

    bundle = ModelBundle(settings).load()

    init_engine(settings.database_url, echo=False)
    # Alembic owns the schema in production; this is a convenience so a fresh
    # `docker compose up` has working tables without a manual migration step.
    if settings.environment != "production":
        await create_all()

    store = StateStore(settings.redis_url, settings.state_ttl_seconds)
    await store.connect()

    policy = PolicyConfig()
    app.state.settings = settings
    app.state.bundle = bundle
    app.state.store = store
    app.state.policy = policy
    app.state.scorer = ScoringService(bundle, store, policy)
    app.state.copilot = AnalystCopilot(settings)
    app.state.started_at = time.time()

    if settings.copilot_enabled and not settings.anthropic_api_key:
        logger.warning(
            "copilot enabled but ANTHROPIC_API_KEY is unset — "
            "draft requests will return 503 until it is configured"
        )

    logger.info("ready")
    try:
        yield
    finally:
        await store.close()
        logger.info("shutdown complete")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Real-time transaction risk scoring with explainable decisions, "
        "shadow-mode challenger evaluation and full decision audit."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request ID and emit a structured access log.

    The request ID is echoed in the response header and written into every
    audit record produced during the request, so a decision in the UI can be
    traced back to the exact call that produced it.
    """
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception(
            "request failed | id=%s %s %s %.1fms",
            request_id, request.method, request.url.path, elapsed,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"x-request-id": request_id},
        )

    elapsed = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{elapsed:.2f}"

    user = getattr(request.state, "user", None)
    logger.info(
        "%s %s -> %d | %.1fms | id=%s actor=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        request_id,
        getattr(user, "user_id", "anonymous"),
    )
    return response


app.include_router(health.router, tags=["health"])
app.include_router(score.router, prefix=settings.api_prefix, tags=["scoring"])
app.include_router(cases.router, prefix=settings.api_prefix, tags=["cases"])
app.include_router(copilot.router, prefix=settings.api_prefix, tags=["copilot"])
app.include_router(metrics.router, prefix=settings.api_prefix, tags=["metrics"])
app.include_router(admin.router, prefix=settings.api_prefix, tags=["admin"])


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
