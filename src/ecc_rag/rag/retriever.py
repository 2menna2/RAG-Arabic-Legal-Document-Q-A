"""Retrieval: BM25, char-TF-IDF, dense (Qdrant) and hybrid (reciprocal-rank fusion).

All methods return *article-level* hits: chunks of the same article are
merged (best chunk wins, all matched chunks kept), so the same article never
fills several slots of the context window.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ecc_rag.data_pipeline.arabic import normalize_for_search, tokenize_for_search
from ecc_rag.rag.embeddings import Embedder
from ecc_rag.rag.query import AnalyzedQuery
from ecc_rag.rag.vector_store import QdrantStore

logger = logging.getLogger(__name__)

METHODS = ("bm25", "tfidf", "dense", "hybrid")


class RetrievalError(RuntimeError):
    pass


@dataclass
class Hit:
    article_number: int
    chunk: dict[str, Any]
    score: float
    rank: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    matched_chunks: list[str] = field(default_factory=list)
    explicit_reference: bool = False
    rerank_score: float | None = None


class LexicalIndex:
    """BM25 over light-stemmed tokens + cosine over char n-gram TF-IDF."""

    def __init__(self, chunks: list[dict[str, Any]], bm25_k1: float = 1.5, bm25_b: float = 0.75):
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        texts = [c["search_text"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize_for_search(t) or ["∅"] for t in texts], k1=bm25_k1, b=bm25_b)
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, preprocessor=normalize_for_search)
        self.matrix = self.tfidf.fit_transform(texts)

    def bm25_scores(self, query: str) -> np.ndarray:
        toks = tokenize_for_search(query)
        if not toks:
            return np.zeros(len(self.chunks))
        return np.asarray(self.bm25.get_scores(toks), dtype=np.float64)

    def tfidf_scores(self, query: str) -> np.ndarray:
        q = self.tfidf.transform([query])
        return np.asarray((self.matrix @ q.T).toarray()).ravel()


class Retriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embedder: Embedder | None,
        store: QdrantStore | None,
        collection: str | None,
        cfg: dict[str, Any],
    ):
        self.chunks = chunks
        self.cfg = cfg
        self.embedder, self.store, self.collection = embedder, store, collection
        self.lex = LexicalIndex(chunks, cfg.get("bm25_k1", 1.5), cfg.get("bm25_b", 0.75))
        self.by_article: dict[int, list[int]] = {}
        for i, c in enumerate(chunks):
            self.by_article.setdefault(c["article_number"], []).append(i)
        self.last_timings: dict[str, float] = {}

    # ---- per-method ranked chunk lists: list[(chunk_idx, score)]
    def _ranked(self, scores: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    def _definition(self, q: AnalyzedQuery, k: int) -> list[tuple[int, float]]:
        """Statutory definitions open with the defined term ("التأمين عقد ...", "الشفعة رخصة ...").
        Score chunks whose first words contain the query's content terms."""
        terms = [t for t in tokenize_for_search(q.content) if len(t) > 2]
        if not terms:
            return []
        scored = []
        for i, c in enumerate(self.chunks):
            head = set(tokenize_for_search(" ".join(c["display_text"].split()[:8])))
            hit = sum(1 for t in terms if t in head)
            if hit:
                scored.append((i, hit / len(terms)))
        scored.sort(key=lambda x: (-x[1], self.chunks[x[0]]["article_number"]))
        return scored[:k]

    def _dense(self, query: str, k: int, filters: dict[str, Any] | None) -> list[tuple[int, float]]:
        if self.embedder is None or self.store is None or self.collection is None:
            raise RetrievalError("dense retrieval requested but no vector index is loaded")
        vec = self.embedder.embed_query(query)
        return [(pid, score) for pid, score, _ in self.store.search(self.collection, vec, k, filters)]

    def method_lists(
        self, q: AnalyzedQuery, methods: list[str], k: int, filters: dict[str, Any] | None = None
    ) -> dict[str, list[tuple[int, float]]]:
        out: dict[str, list[tuple[int, float]]] = {}
        base = q.content if self.cfg.get("strip_question_frame", True) else q.normalized
        text = (base + " " + " ".join(q.expansions)) if (self.cfg.get("use_dialect_expansion", True) and q.expansions) else base
        for m in methods:
            t0 = time.perf_counter()
            if m == "bm25":
                out[m] = self._ranked(self.lex.bm25_scores(text), k)
            elif m == "tfidf":
                out[m] = self._ranked(self.lex.tfidf_scores(text), k)
            elif m == "definition":
                out[m] = self._definition(q, k) if q.is_definition else []
            elif m == "dense":
                out[m] = self._dense(q.original if not q.expansions else f"{q.original} {' '.join(q.expansions)}", k, filters)
            else:
                raise ValueError(f"unknown retrieval method {m}")
            self.last_timings[m] = time.perf_counter() - t0
            if filters and m != "dense":
                out[m] = [(i, s) for i, s in out[m] if all(self.chunks[i].get(fk) == fv for fk, fv in filters.items())]
        return out

    def retrieve(self, q: AnalyzedQuery, method: str = "hybrid", top_k: int = 5, filters: dict[str, Any] | None = None) -> list[Hit]:
        cand_k = int(self.cfg.get("candidate_k", 50))
        if method == "hybrid":
            components = list(self.cfg.get("hybrid_components", ["bm25", "tfidf", "dense"]))
            if self.embedder is None and "dense" in components:
                components.remove("dense")
        else:
            components = [method]
        lists = self.method_lists(q, components, cand_k, filters)
        weights = self.cfg.get("rrf_weights", {})
        rrf_k = float(self.cfg.get("rrf_k", 60))

        agg: dict[int, Hit] = {}
        for m, ranked in lists.items():
            seen_articles: set[int] = set()
            art_rank = 0
            for idx, score in ranked:
                ch = self.chunks[idx]
                n = ch["article_number"]
                hit = agg.get(n)
                if hit is None:
                    hit = agg[n] = Hit(article_number=n, chunk=ch, score=0.0)
                if ch["chunk_id"] not in hit.matched_chunks:
                    hit.matched_chunks.append(ch["chunk_id"])
                if n in seen_articles:
                    continue  # article-level rank: only the best chunk of an article counts
                seen_articles.add(n)
                art_rank += 1
                hit.scores[m] = score
                hit.ranks[m] = art_rank
                if len(components) == 1:
                    if score > hit.score:
                        hit.score, hit.chunk = score, ch
                else:
                    hit.score += float(weights.get(m, 1.0)) / (rrf_k + art_rank)
                    if art_rank == 1 or hit.chunk is None:
                        hit.chunk = ch
        hits = sorted(agg.values(), key=lambda h: (-h.score, h.article_number))

        # explicit "المادة N" references are honoured first (exact lookup, not similarity)
        if self.cfg.get("honor_explicit_refs", True) and q.article_refs:
            explicit: list[Hit] = []
            for n in q.article_refs[:top_k]:
                if n in self.by_article:
                    h = agg.get(n) or Hit(article_number=n, chunk=self.chunks[self.by_article[n][0]], score=0.0)
                    h.explicit_reference = True
                    explicit.append(h)
            rest = [h for h in hits if not h.explicit_reference]
            hits = explicit + rest
        for i, h in enumerate(hits, start=1):
            h.rank = i
        return hits[:top_k]

    def article_chunks(self, article_number: int) -> list[dict[str, Any]]:
        return [self.chunks[i] for i in self.by_article.get(article_number, [])]


class CrossEncoderReranker:
    """Optional multilingual cross-encoder (needs sentence-transformers + model download)."""

    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model
        self._m = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._m.predict([(query, h.chunk["display_text"]) for h in hits])
        for h, s in zip(hits, scores, strict=True):
            h.rerank_score = float(s)
        out = sorted(hits, key=lambda h: (not h.explicit_reference, -(h.rerank_score or 0.0)))[:top_n]
        for i, h in enumerate(out, start=1):
            h.rank = i
        return out
