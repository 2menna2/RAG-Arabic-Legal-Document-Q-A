"""Qdrant vector store wrapper.

* Embedded mode (``QDRANT_PATH``) for local development, tests and CI — same
  client API, no server.
* Server mode (``QDRANT_URL``) for docker-compose / production.

Each index version lives in its own collection (``ecc_<index_version>``) so a
new corpus or embedder never overwrites the collection a running API is
serving; switching is a config change and rollback is pointing back.
"""

from __future__ import annotations

import atexit
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    pass


# Embedded Qdrant storage may be opened by only one client per process: share it.
_EMBEDDED: dict[str, tuple[object, threading.Lock]] = {}
_EMBEDDED_GUARD = threading.Lock()


@atexit.register
def _close_embedded() -> None:
    for client, _lock in list(_EMBEDDED.values()):
        try:
            client.close()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — interpreter shutdown, best effort
            logger.debug("qdrant close at exit failed: %s", exc)
    _EMBEDDED.clear()


def release_embedded(path: str | Path) -> None:
    """Close and forget this process's embedded client for `path` (before deleting/rebuilding that folder)."""
    key = str(Path(path).resolve())
    with _EMBEDDED_GUARD:
        entry = _EMBEDDED.pop(key, None)
    if entry is not None:
        try:
            entry[0].close()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — best effort, the folder is rebuilt next
            logger.debug("closing embedded Qdrant at %s failed: %s", key, exc)


class QdrantStore:
    def __init__(self, url: str | None = None, path: Path | None = None, api_key: str | None = None, timeout_s: float = 5.0):
        from qdrant_client import QdrantClient

        # The embedded (local) client is not designed for concurrent access: serialise calls.
        self._lock = threading.Lock()
        try:
            if url:
                self.client = QdrantClient(url=url, api_key=api_key, timeout=int(timeout_s))
                self.mode = "server"
            elif path:
                key = str(Path(path).resolve())
                with _EMBEDDED_GUARD:
                    if key not in _EMBEDDED:
                        Path(path).mkdir(parents=True, exist_ok=True)
                        _EMBEDDED[key] = (QdrantClient(path=key), threading.Lock())
                    self.client, self._lock = _EMBEDDED[key]  # type: ignore[assignment]
                self._key = key
                self.mode = "embedded"
            else:
                raise VectorStoreError("either QDRANT_URL or QDRANT_PATH must be configured")
        except VectorStoreError:
            raise
        except Exception as exc:
            if "already accessed" in str(exc):
                raise VectorStoreError(
                    "embedded Qdrant index is locked by another process — stop the other process using it "
                    "(e.g. the API server) or run a Qdrant server and set QDRANT_URL"
                ) from exc
            raise VectorStoreError(f"cannot open Qdrant ({type(exc).__name__})") from exc

    def recreate(self, collection: str, dim: int) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        if self.mode == "server":  # payload indexes are a no-op in embedded mode
            for field, kind in (
                ("article_number", PayloadSchemaType.INTEGER),
                ("legal_status", PayloadSchemaType.KEYWORD),
                ("book", PayloadSchemaType.KEYWORD),
                ("language", PayloadSchemaType.KEYWORD),
            ):
                self.client.create_payload_index(collection, field, kind)

    def upsert(self, collection: str, vectors: np.ndarray, payloads: list[dict[str, Any]], batch: int = 256) -> None:
        from qdrant_client.models import PointStruct

        if len(vectors) != len(payloads):
            raise VectorStoreError("vectors/payloads length mismatch")
        for i in range(0, len(payloads), batch):
            pts = [
                PointStruct(id=i + j, vector=vectors[i + j].tolist(), payload=payloads[i + j]) for j in range(min(batch, len(payloads) - i))
            ]
            self.client.upsert(collection, points=pts, wait=True)

    def info(self, collection: str) -> dict[str, Any]:
        try:
            with self._lock:
                if not self.client.collection_exists(collection):
                    raise VectorStoreError(f"collection {collection} does not exist")
                c = self.client.get_collection(collection)
                count = self.client.count(collection, exact=True).count
            size = c.config.params.vectors.size  # type: ignore[union-attr]
            return {"points": int(count), "dim": int(size)}
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Qdrant unavailable ({type(exc).__name__})") from exc

    def search(
        self, collection: str, vector: np.ndarray, limit: int, filters: dict[str, Any] | None = None
    ) -> list[tuple[int, float, dict[str, Any]]]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        flt = None
        if filters:
            conds: list[Any] = []
            for k, v in filters.items():
                conds.append(FieldCondition(key=k, match=MatchAny(any=v) if isinstance(v, list) else MatchValue(value=v)))
            flt = Filter(must=conds)
        try:
            with self._lock:
                res = self.client.query_points(collection, query=vector.tolist(), limit=limit, query_filter=flt, with_payload=True)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed ({type(exc).__name__})") from exc
        return [(int(p.id), float(p.score), dict(p.payload or {})) for p in res.points]

    def close(self) -> None:
        if self.mode == "embedded":
            return  # shared per process; released at interpreter exit
        try:
            self.client.close()
        except Exception as exc:  # noqa: BLE001 — closing is best effort
            logger.debug("qdrant close failed: %s", exc)
