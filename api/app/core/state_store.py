"""Redis-backed persistence for rolling customer state.

This is the online half of the feature store. The offline pipeline builds
`CustomerState` by replaying history in memory; the API reconstitutes the
same object from Redis, computes features with the *same function*, then
writes the mutated state back.

The state is bounded by the 7-day feature window, so each customer's payload
stays small (a few KB) no matter how long they have been a customer — which
is what makes per-request round trips viable at low latency.

Read-modify-write is done under a short per-customer lock. Without it, two
concurrent authorisations on the same card race, and the velocity counters
under-count exactly when a burst attack is in progress — the moment they
matter most.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime
from typing import Any

import redis.asyncio as redis
from argus_ml.features.engineering import CustomerState

logger = logging.getLogger(__name__)

_KEY = "argus:state:{customer_id}"
_LOCK = "argus:lock:{customer_id}"


def _serialise(state: CustomerState) -> str:
    return json.dumps(
        {
            "customer_id": state.customer_id,
            "events": [
                [ts.isoformat(), amt, mid, country, cat]
                for ts, amt, mid, country, cat in state.events
            ],
            "seen_devices": sorted(state.seen_devices),
            "seen_countries": sorted(state.seen_countries),
            "seen_categories": sorted(state.seen_categories),
            "n_lifetime": state.n_lifetime,
            "mean_amount": state.mean_amount,
            "m2_amount": state.m2_amount,
            "last_timestamp": state.last_timestamp.isoformat() if state.last_timestamp else None,
            "last_lat": state.last_lat,
            "last_lon": state.last_lon,
            "first_seen": state.first_seen.isoformat() if state.first_seen else None,
        },
        separators=(",", ":"),
    )


def _deserialise(raw: str | bytes) -> CustomerState:
    d: dict[str, Any] = json.loads(raw)
    return CustomerState(
        customer_id=d["customer_id"],
        events=deque(
            (datetime.fromisoformat(ts), amt, mid, country, cat)
            for ts, amt, mid, country, cat in d.get("events", [])
        ),
        seen_devices=set(d.get("seen_devices", [])),
        seen_countries=set(d.get("seen_countries", [])),
        seen_categories=set(d.get("seen_categories", [])),
        n_lifetime=d.get("n_lifetime", 0),
        mean_amount=d.get("mean_amount", 0.0),
        m2_amount=d.get("m2_amount", 0.0),
        last_timestamp=(
            datetime.fromisoformat(d["last_timestamp"]) if d.get("last_timestamp") else None
        ),
        last_lat=d.get("last_lat"),
        last_lon=d.get("last_lon"),
        first_seen=(
            datetime.fromisoformat(d["first_seen"]) if d.get("first_seen") else None
        ),
    )


class StateStore:
    """Async Redis client for customer feature state."""

    def __init__(self, url: str, ttl_seconds: int) -> None:
        self._url = url
        self._ttl = ttl_seconds
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        self._client = redis.from_url(
            self._url, encoding="utf-8", decode_responses=True
        )
        await self._client.ping()
        logger.info("connected to redis at %s", self._url)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("StateStore.connect() was not awaited")
        return self._client

    async def get(self, customer_id: str) -> CustomerState:
        """Load state, or return a fresh one for an unseen customer."""
        raw = await self.client.get(_KEY.format(customer_id=customer_id))
        if raw is None:
            return CustomerState(customer_id=customer_id)
        try:
            return _deserialise(raw)
        except (ValueError, KeyError, TypeError):
            # A malformed payload must not take down scoring. Cold-start the
            # customer and log it — the drift monitor will surface a spike in
            # `is_first_transaction` if this ever becomes systemic.
            logger.exception("corrupt state for %s, cold-starting", customer_id)
            return CustomerState(customer_id=customer_id)

    async def put(self, state: CustomerState) -> None:
        await self.client.set(
            _KEY.format(customer_id=state.customer_id),
            _serialise(state),
            ex=self._ttl,
        )

    async def acquire_lock(self, customer_id: str, timeout_ms: int = 2_000) -> bool:
        """Best-effort per-customer lock; scoring proceeds either way.

        Correctness here is a trade: blocking an authorisation because Redis
        is contended is worse than a slightly stale velocity counter, so a
        failed acquisition logs and continues rather than raising.
        """
        return bool(
            await self.client.set(
                _LOCK.format(customer_id=customer_id), "1", nx=True, px=timeout_ms
            )
        )

    async def release_lock(self, customer_id: str) -> None:
        await self.client.delete(_LOCK.format(customer_id=customer_id))

    async def warm(self, states: dict[str, CustomerState]) -> int:
        """Bulk-load terminal offline states so the API starts warm.

        Without this every customer looks brand new on first request and the
        velocity features are all zero — the model would score a live burst
        attack as a first transaction.
        """
        pipe = self.client.pipeline()
        for cid, st in states.items():
            pipe.set(_KEY.format(customer_id=cid), _serialise(st), ex=self._ttl)
        await pipe.execute()
        logger.info("warmed %d customer states into redis", len(states))
        return len(states)


__all__ = ["StateStore"]
