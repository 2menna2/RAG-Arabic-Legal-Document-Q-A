"""Embedding back-ends behind one small interface.

* ``lsa``  — TF-IDF over character n-grams reduced with truncated SVD (latent
  semantic analysis).  Fully offline and deterministic; used in CI and in any
  environment without network access to a model hub or embedding API.  It is a
  *classical* dense representation, not a neural one, and is reported as such.
* ``gemini`` — Google Gemini embedding API (multilingual, supports Arabic).
* ``sentence_transformers`` — any local Hugging Face model (e.g.
  ``intfloat/multilingual-e5-base`` or ``BAAI/bge-m3``) for self-hosted setups.

The embedder's identity (name, model, version, dimension, params) is recorded
in the index manifest; an index built with one embedder is refused at load
time by another.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from ecc_rag.config import config_hash
from ecc_rag.data_pipeline.arabic import normalize_for_search

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str
    dim: int

    def identity(self) -> dict[str, Any]: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class LsaEmbedder:
    """Character n-gram TF-IDF + truncated SVD (fit on the corpus)."""

    name = "lsa"

    def __init__(self, n_components: int = 256, ngram_range: tuple[int, int] = (2, 5), min_df: int = 2, seed: int = 42):
        self.n_components = n_components
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.seed = seed
        self._vec: Any = None
        self._svd: Any = None
        self._w: np.ndarray | None = None
        self.dim = n_components

    def identity(self) -> dict[str, Any]:
        import sklearn

        params = {"n_components": self.n_components, "ngram_range": list(self.ngram_range), "min_df": self.min_df, "seed": self.seed}
        return {
            "name": self.name,
            "model": "char-tfidf+truncated-svd",
            "version": f"sklearn-{sklearn.__version__}",
            "dim": self.dim,
            "params": params,
            "params_hash": config_hash(params),
        }

    def fit(self, texts: list[str]) -> LsaEmbedder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.ngram_range, min_df=self.min_df, sublinear_tf=True, preprocessor=normalize_for_search
        )
        x = self._vec.fit_transform(texts)
        k = min(self.n_components, x.shape[1] - 1, x.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=k, random_state=self.seed, algorithm="randomized", n_iter=7)
        self._svd.fit(x)
        self.dim = k
        return self

    def _ensure(self) -> None:
        if self._vec is None or self._svd is None:
            raise EmbeddingError("LSA embedder is not fitted/loaded")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        return _l2(self._svd.transform(self._vec.transform(texts)).astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        """Project one query without materialising a copy of the (n_features x k) SVD matrix.

        ``TruncatedSVD.transform`` on a sparse row copies ``components_.T`` (~70 MB here) on every
        call — under 16 concurrent requests that OOM-killed a 1 GiB container in the load test.
        Gathering only the rows of the non-zero features gives the same result (max abs diff ~1e-17)
        with ~0.2 MB of temporary memory.
        """
        self._ensure()
        if self._w is None:
            self._w = np.ascontiguousarray(self._svd.components_.T)
        x = self._vec.transform([text]).tocsr()
        v = (x.data @ self._w[x.indices]) if x.nnz else np.zeros(self._w.shape[1])
        return _l2(np.asarray(v, dtype=np.float32)[None, :])[0]

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vec": self._vec, "svd": self._svd, "identity": self.identity()}, path)

    @classmethod
    def load(cls, path: Path) -> LsaEmbedder:
        import joblib

        # joblib/pickle: only load artefacts produced by our own build_index stage
        obj = joblib.load(path)
        p = obj["identity"]["params"]
        emb = cls(p["n_components"], tuple(p["ngram_range"]), p["min_df"], p["seed"])
        emb._vec, emb._svd, emb.dim = obj["vec"], obj["svd"], obj["identity"]["dim"]
        return emb


class GeminiEmbedder:
    name = "gemini"

    def __init__(self, model: str, api_key: str, dim: int = 768, timeout_s: float = 30.0, max_retries: int = 2, batch: int = 50):
        self.model = model
        self.dim = dim
        self.batch = batch
        self.max_retries = max_retries
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def identity(self) -> dict[str, Any]:
        params = {"output_dimensionality": self.dim, "task_types": ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]}
        return {
            "name": self.name,
            "model": self.model,
            "version": self.model,
            "dim": self.dim,
            "params": params,
            "params_hash": config_hash(params),
        }

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            part = texts[i : i + self.batch]
            for attempt in range(self.max_retries + 1):
                try:
                    res = self._client.models.embed_content(
                        model=self.model,
                        contents=cast(Any, list(part)),  # SDK's ContentListUnion varies by version
                        config=self._types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
                    )
                    out.extend(list(e.values or []) for e in (res.embeddings or []))
                    break
                except Exception as exc:
                    if attempt >= self.max_retries:
                        raise EmbeddingError(f"Gemini embedding failed: {type(exc).__name__}") from exc
                    time.sleep(0.5 * 2**attempt)
        return _l2(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class SentenceTransformerEmbedder:
    name = "sentence_transformers"

    def __init__(self, model: str, query_prefix: str = "", doc_prefix: str = "", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self._m = SentenceTransformer(model, device=device)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def identity(self) -> dict[str, Any]:
        import sentence_transformers

        params = {"query_prefix": self.query_prefix, "doc_prefix": self.doc_prefix}
        return {
            "name": self.name,
            "model": self.model_name,
            "version": f"st-{sentence_transformers.__version__}",
            "dim": self.dim,
            "params": params,
            "params_hash": config_hash(params),
        }

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        v = self._m.encode([self.doc_prefix + t for t in texts], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        v = self._m.encode([self.query_prefix + text], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)[0]


def make_embedder(cfg: dict[str, Any], index_dir: Path | None = None, fit_texts: list[str] | None = None) -> Embedder:
    """Create an embedder from ``configs/rag.yaml``'s ``embedding`` section."""
    from ecc_rag.config import get_settings

    kind = cfg["provider"]
    if kind == "lsa":
        p = cfg.get("lsa", {})
        if fit_texts is not None:
            return LsaEmbedder(p.get("n_components", 256), tuple(p.get("ngram_range", (2, 5))), p.get("min_df", 2)).fit(fit_texts)
        if index_dir is None:
            raise EmbeddingError("LSA embedder needs index_dir to load its fitted state")
        return LsaEmbedder.load(index_dir / "lsa.joblib")
    if kind == "gemini":
        s = get_settings()
        if s.gemini_api_key is None:
            raise EmbeddingError("GEMINI_API_KEY is not set; cannot use the gemini embedder")
        g = cfg.get("gemini", {})
        return GeminiEmbedder(
            g.get("model", s.gemini_embedding_model),
            s.gemini_api_key.get_secret_value(),
            dim=g.get("dim", 768),
            timeout_s=s.llm_timeout_s,
            max_retries=s.llm_max_retries,
        )
    if kind == "sentence_transformers":
        st = cfg.get("sentence_transformers", {})
        return SentenceTransformerEmbedder(st["model"], st.get("query_prefix", ""), st.get("doc_prefix", ""))
    raise EmbeddingError(f"unknown embedding provider: {kind}")
