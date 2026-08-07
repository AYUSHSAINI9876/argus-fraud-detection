"""Replay driver — streams held-out transactions into the live scoring API.

Without this the console is an empty shell: the dashboard shows no traffic and
the case queue has nothing to work. The replay pushes the *test-window*
transactions (the ones no model was trained on) through the real HTTP path, so
what the UI displays is genuine end-to-end behaviour rather than seeded rows.

Run:
    python -m argus_ml.replay --rate 20 --limit 5000

`--rate` is transactions per second. Real card portfolios are bursty, so the
inter-arrival times are drawn from an exponential distribution around that
mean rather than being evenly spaced — a constant rate would make the latency
percentiles on the dashboard look better than they should.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from pathlib import Path

import httpx
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

_WIRE_FIELDS = [
    "transaction_id", "timestamp", "customer_id", "card_id", "merchant_id",
    "merchant_category", "amount", "currency", "channel", "entry_mode",
    "device_id", "ip_country", "lat", "lon", "is_recurring",
]


def _to_payload(row: dict) -> dict:
    payload = {k: row[k] for k in _WIRE_FIELDS}
    payload["timestamp"] = pd.Timestamp(payload["timestamp"]).isoformat()
    payload["amount"] = float(payload["amount"])
    payload["lat"] = float(payload["lat"])
    payload["lon"] = float(payload["lon"])
    payload["is_recurring"] = bool(payload["is_recurring"])
    return payload


async def replay(
    api_url: str,
    token: str | None,
    rate: float,
    limit: int,
    artifacts: Path,
) -> None:
    path = artifacts / "recent_transactions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m argus_ml.train` first"
        )

    df = pd.read_parquet(path).tail(limit)
    print(f"replaying {len(df):,} transactions at ~{rate}/s -> {api_url}")

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    stats = {"allow": 0, "review": 0, "block": 0, "error": 0}
    latencies: list[float] = []
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        for i, row in enumerate(df.to_dict("records")):
            try:
                resp = await client.post(f"{api_url}/score", json=_to_payload(row))
                if resp.status_code == 200:
                    body = resp.json()
                    stats[body["decision"]] += 1
                    latencies.append(body["latency_ms"])
                else:
                    stats["error"] += 1
                    if stats["error"] <= 3:
                        print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            except httpx.HTTPError as exc:
                stats["error"] += 1
                if stats["error"] <= 3:
                    print(f"  transport error: {exc}")

            if (i + 1) % 500 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  {i + 1:,}/{len(df):,} | {(i + 1) / elapsed:.1f}/s | "
                    f"allow={stats['allow']:,} review={stats['review']:,} "
                    f"block={stats['block']:,} err={stats['error']:,}"
                )

            # Exponential inter-arrival — bursty, like real traffic.
            if rate > 0:
                await asyncio.sleep(random.expovariate(rate))

    elapsed = time.perf_counter() - t0
    total = sum(stats.values())
    print(f"\ndone in {elapsed:.1f}s ({total / elapsed:.1f}/s)")
    for k, v in stats.items():
        print(f"  {k:<8} {v:>8,}  ({v / total:.2%})" if total else f"  {k}: {v}")

    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        print(f"\nserver-side scoring latency: p50 {p50:.1f}ms  p95 {p95:.1f}ms  p99 {p99:.1f}ms")


def main() -> None:
    p = argparse.ArgumentParser(description="Replay transactions into the Argus API")
    p.add_argument("--api-url", default="http://localhost:8000/api/v1")
    p.add_argument("--token", default=None, help="Bearer token (omit if AUTH_ENABLED=false)")
    p.add_argument("--rate", type=float, default=20.0, help="transactions per second")
    p.add_argument("--limit", type=int, default=5_000)
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = p.parse_args()

    asyncio.run(
        replay(args.api_url, args.token, args.rate, args.limit, args.artifacts)
    )


if __name__ == "__main__":
    main()
