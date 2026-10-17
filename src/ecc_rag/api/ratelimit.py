"""Small in-process token-bucket rate limiter (per client key).

Suitable for a single replica. Behind several replicas, enforce limits at the
gateway / ingress instead (documented in docs/DEPLOYMENT.md).
"""

from __future__ import annotations

import threading
import time


class TokenBucketLimiter:
    def __init__(self, per_minute: int, burst: int | None = None, max_keys: int = 10_000):
        self.rate = per_minute / 60.0
        self.capacity = float(burst or max(1, per_minute))
        self.max_keys = max_keys
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.rate <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            if len(self._buckets) > self.max_keys:  # bounded memory
                oldest = sorted(self._buckets.items(), key=lambda kv: kv[1][1])[: len(self._buckets) // 2]
                for k, _ in oldest:
                    self._buckets.pop(k, None)
            return True
