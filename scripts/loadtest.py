"""Closed-loop load test against a running API (async httpx).

Measures, per endpoint and concurrency level: throughput, p50/p95/p99 latency,
error rate, and the server's resident memory (from its own /metrics) before
and after. Questions are drawn from the golden set so caching cannot flatter
the numbers. Results are written to reports/loadtest/<timestamp>.json.

Example:
    python scripts/loadtest.py --base-url http://127.0.0.1:18001 --endpoint search --concurrency 1 4 8 --duration 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]


def rss_bytes(base: str) -> float | None:
    try:
        text = httpx.get(f"{base}/metrics", timeout=5).text
        m = re.search(r"^process_resident_memory_bytes ([0-9.e+]+)$", text, re.M)
        return float(m.group(1)) if m else None
    except httpx.HTTPError:
        return None


async def worker(
    client: httpx.AsyncClient, url: str, questions: list[str], stop: float, lat: list[float], codes: list[int], rnd: random.Random
) -> None:
    while time.perf_counter() < stop:
        q = rnd.choice(questions)
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json={"question": q} if url.endswith("/ask") else {"question": q, "top_k": 5})
            codes.append(r.status_code)
        except httpx.HTTPError:
            codes.append(-1)
        lat.append(time.perf_counter() - t0)


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


async def run_level(base: str, endpoint: str, conc: int, duration: float, questions: list[str], ok_codes: set[int]) -> dict:
    url = f"{base}/api/v1/{endpoint}"
    lat: list[float] = []
    codes: list[int] = []
    rnd = random.Random(conc)
    async with httpx.AsyncClient(timeout=60, headers={"X-Request-ID": "loadtest"}) as client:
        stop = time.perf_counter() + duration
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client, url, questions, stop, lat, codes, rnd) for _ in range(conc)))
        elapsed = time.perf_counter() - t0
    errors = sum(1 for c in codes if c not in ok_codes)
    return {
        "endpoint": endpoint,
        "concurrency": conc,
        "requests": len(codes),
        "duration_s": round(elapsed, 2),
        "throughput_rps": round(len(codes) / elapsed, 2),
        "error_rate": round(errors / max(1, len(codes)), 4),
        "status_codes": {str(c): codes.count(c) for c in sorted(set(codes))},
        "p50_ms": round(1000 * pct(lat, 0.50), 1),
        "p95_ms": round(1000 * pct(lat, 0.95), 1),
        "p99_ms": round(1000 * pct(lat, 0.99), 1),
        "mean_ms": round(1000 * statistics.fmean(lat), 1) if lat else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--endpoint", choices=["search", "ask"], default="search")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--duration", type=float, default=20)
    p.add_argument("--ok-codes", type=int, nargs="+", default=[200])
    p.add_argument("--label", default="")
    args = p.parse_args()
    items = yaml.safe_load((ROOT / "data/evaluation/golden_v1.yaml").read_text(encoding="utf-8"))["items"]
    questions = [i["question"] for i in items]
    rss_before = rss_bytes(args.base_url)
    levels = [
        asyncio.run(run_level(args.base_url, args.endpoint, c, args.duration, questions, set(args.ok_codes))) for c in args.concurrency
    ]
    rss_after = rss_bytes(args.base_url)
    out = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": args.label,
        "base_url": args.base_url,
        "client_machine": {"platform": platform.platform(), "python": platform.python_version()},
        "server_rss_bytes_before": rss_before,
        "server_rss_bytes_after": rss_after,
        "levels": levels,
    }
    d = ROOT / "reports/loadtest"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{out['run_at'][:19].replace(':', '')}_{args.endpoint}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
