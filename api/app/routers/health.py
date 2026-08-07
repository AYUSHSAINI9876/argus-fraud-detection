"""Liveness and readiness probes.

Split deliberately. `/health` answers "is the process alive" and is what a
container orchestrator restarts on. `/ready` answers "can this instance
serve a correct score" — it checks the model bundle and Redis, so a pod with
a stale or missing artefact is pulled from the load balancer instead of
serving wrong decisions.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    started = getattr(request.app.state, "started_at", time.time())
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - started, 1),
        "environment": request.app.state.settings.environment,
    }


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict:
    checks: dict[str, str] = {}
    ok = True

    bundle = getattr(request.app.state, "bundle", None)
    if bundle is not None and bundle.champion is not None:
        checks["model"] = f"loaded ({len(bundle.feature_names)} features)"
    else:
        checks["model"] = "missing"
        ok = False

    store = getattr(request.app.state, "store", None)
    try:
        if store is not None:
            await store.client.ping()
            checks["redis"] = "connected"
        else:
            checks["redis"] = "not initialised"
            ok = False
    except Exception as exc:  # noqa: BLE001 - report, don't crash the probe
        checks["redis"] = f"unreachable: {type(exc).__name__}"
        ok = False

    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {"ready": ok, "checks": checks}
