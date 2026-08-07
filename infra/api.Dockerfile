# Multi-stage build. The ML package is installed as a real dependency of the
# API rather than copied in, so the container physically cannot drift from the
# code that trained the models.

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY ml/pyproject.toml ml/pyproject.toml
COPY ml/argus_ml ml/argus_ml
COPY api/pyproject.toml api/pyproject.toml
COPY api/app api/app

RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache ./ml ./api


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# libgomp is required by XGBoost at runtime; without it the import fails with
# an opaque shared-object error.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 argus

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=argus:argus api/app /app/app
COPY --chown=argus:argus ml/argus_ml /app/argus_ml

USER argus
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
