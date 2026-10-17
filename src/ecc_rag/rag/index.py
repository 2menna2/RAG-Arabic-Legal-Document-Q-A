"""Build / load the retrieval index and enforce corpus ↔ index compatibility.

``build_index`` (a DVC stage) embeds the chunks, writes them to a fresh Qdrant
collection and records a manifest::

    corpus_hash, chunks_hash, chunking config, embedder identity, dim,
    n_points, collection, git commit, library versions, created_at

``load_index`` refuses to serve when *any* of these disagree with the chunks
file and configuration currently deployed (``IndexIncompatibleError``), so a
stale index can never be silently mixed with a new corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ecc_rag.config import Settings, config_hash, get_settings, load_rag_config
from ecc_rag.rag.embeddings import Embedder, LsaEmbedder, make_embedder
from ecc_rag.rag.vector_store import QdrantStore

logger = logging.getLogger(__name__)

PAYLOAD_KEYS = (
    "chunk_id",
    "article_id",
    "article_number",
    "chunk_index",
    "kind",
    "language",
    "legal_status",
    "source_page",
    "book",
    "chapter",
    "section",
)


class IndexIncompatibleError(RuntimeError):
    pass


def load_chunks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunks_hash(chunks: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["chunk_id"].encode())
        h.update(c["content_hash"].encode())
        h.update(hashlib.sha256(c["search_text"].encode("utf-8")).digest())
    return h.hexdigest()


def corpus_hash_of(articles_path: Path) -> str:
    hashes = [json.loads(line)["content_hash"] for line in articles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _lib_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for pkg in ("qdrant-client", "scikit-learn", "rank-bm25", "numpy", "pymupdf", "google-genai"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def collection_name(manifest_core: dict[str, Any]) -> str:
    return "ecc_" + config_hash(manifest_core)[:12]


def build_index(chunks_path: Path, articles_path: Path, index_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    chunks = load_chunks(chunks_path)
    texts = [c["search_text"] for c in chunks]
    emb_cfg = cfg["embedding"]
    embedder: Embedder
    if emb_cfg["provider"] == "lsa":
        embedder = make_embedder(emb_cfg, fit_texts=texts)
        assert isinstance(embedder, LsaEmbedder)
        embedder.save(index_dir / "lsa.joblib")
    else:
        embedder = make_embedder(emb_cfg)
    vectors = embedder.embed_documents(texts)
    if vectors.shape != (len(chunks), embedder.dim):
        raise IndexIncompatibleError(f"embedding shape {vectors.shape} != ({len(chunks)}, {embedder.dim})")
    core = {
        "corpus_hash": corpus_hash_of(articles_path),
        "chunks_hash": chunks_hash(chunks),
        "chunking": cfg["chunking"],
        "embedder": embedder.identity(),
    }
    coll = collection_name(core)
    s = get_settings()
    store = QdrantStore(
        url=s.qdrant_url,
        path=None if s.qdrant_url else (s.qdrant_path or index_dir / "qdrant"),
        api_key=s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None,
        timeout_s=s.qdrant_timeout_s,
    )
    store.recreate(coll, embedder.dim)
    store.upsert(coll, vectors, [{k: c.get(k) for k in PAYLOAD_KEYS} for c in chunks])
    info = store.info(coll)
    store.close()
    manifest = {
        **core,
        "collection": coll,
        "dim": embedder.dim,
        "n_points": info["points"],
        "n_chunks": len(chunks),
        "store_mode": store.mode,
        "git_commit": git_commit(),
        "libraries": _lib_versions(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if info["points"] != len(chunks):
        raise IndexIncompatibleError(f"Qdrant has {info['points']} points, expected {len(chunks)}")
    (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("index built: collection=%s points=%d dim=%d", coll, info["points"], embedder.dim)
    return manifest


@dataclass
class LoadedIndex:
    manifest: dict[str, Any]
    chunks: list[dict[str, Any]]
    embedder: Embedder
    store: QdrantStore


def load_index(
    chunks_path: Path,
    articles_path: Path,
    index_dir: Path,
    cfg: dict[str, Any],
    settings: Settings | None = None,
    qdrant_path: Path | None = None,
) -> LoadedIndex:
    mpath = index_dir / "manifest.json"
    if not mpath.exists():
        raise IndexIncompatibleError(f"index manifest missing at {mpath}; run `make index`")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    chunks = load_chunks(chunks_path)
    problems = []
    if manifest["corpus_hash"] != corpus_hash_of(articles_path):
        problems.append("corpus_hash differs (corpus changed since the index was built)")
    if manifest["chunks_hash"] != chunks_hash(chunks):
        problems.append("chunks_hash differs (chunks changed since the index was built)")
    if manifest["chunking"] != cfg["chunking"]:
        problems.append("chunking config differs from configs/rag.yaml")
    embedder = make_embedder(cfg["embedding"], index_dir=index_dir)
    if embedder.identity() != manifest["embedder"]:
        problems.append(f"embedder identity differs: index={manifest['embedder']} config={embedder.identity()}")
    s = settings or get_settings()
    store = QdrantStore(
        url=s.qdrant_url,
        path=None if s.qdrant_url else (qdrant_path or s.qdrant_path or index_dir / "qdrant"),
        api_key=s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None,
        timeout_s=s.qdrant_timeout_s,
    )
    info = store.info(manifest["collection"])
    if info["points"] != len(chunks):
        problems.append(f"vector count {info['points']} != chunk count {len(chunks)}")
    if info["dim"] != embedder.dim:
        problems.append(f"vector dim {info['dim']} != embedder dim {embedder.dim}")
    if problems:
        store.close()
        raise IndexIncompatibleError("; ".join(problems))
    return LoadedIndex(manifest, chunks, embedder, store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the vector index")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args.index_dir.mkdir(parents=True, exist_ok=True)
    m = build_index(args.chunks, args.articles, args.index_dir, load_rag_config())
    print(json.dumps({k: m[k] for k in ("collection", "n_points", "dim", "corpus_hash")}, indent=2))


if __name__ == "__main__":
    main()
