"""Evidence sufficiency — decide *before* generation whether retrieval found enough support.

No single similarity score is treated as proof.  The gate combines:

* ``top_tfidf``   — best char-TF-IDF cosine among the retrieved articles
* ``coverage``    — share of the question's content terms (light-stemmed) that
                    occur in the top-N retrieved articles
* ``explicit``    — the user named an article number that exists in the corpus

and the LLM keeps the right to answer ``insufficient_evidence`` after reading
the documents (second gate), and the citation validator is the third gate.
Thresholds live in ``configs/rag.yaml`` and are calibrated on the dev split
(``scripts/calibrate_thresholds.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ecc_rag.data_pipeline.arabic import tokenize_for_search
from ecc_rag.rag.query import QUESTION_FRAME_WORDS, AnalyzedQuery
from ecc_rag.rag.retriever import Hit, Retriever


@dataclass
class EvidenceAssessment:
    sufficient: bool
    top_tfidf: float
    coverage: float
    explicit_found: list[int] = field(default_factory=list)
    explicit_missing: list[str] = field(default_factory=list)
    reason: str = ""


def coverage_of(q: AnalyzedQuery, hits: list[Hit], top_n: int) -> float:
    terms = {t for t in tokenize_for_search(q.content) if t not in QUESTION_FRAME_WORDS}
    if not terms:
        return 0.0
    doc_terms: set[str] = set()
    for h in hits[:top_n]:
        doc_terms |= set(tokenize_for_search(h.chunk["search_text"]))
    return len(terms & doc_terms) / len(terms)


def top_tfidf_of(retriever: Retriever, q: AnalyzedQuery, hits: list[Hit]) -> float:
    if not hits:
        return 0.0
    scores = retriever.lex.tfidf_scores(q.content)
    best = 0.0
    for h in hits:
        for i in retriever.by_article.get(h.article_number, []):
            best = max(best, float(scores[i]))
    return best


def assess(retriever: Retriever, q: AnalyzedQuery, hits: list[Hit], cfg: dict[str, float]) -> EvidenceAssessment:
    explicit_found = [n for n in q.article_refs if n in retriever.by_article]
    explicit_missing = [str(n) for n in q.article_refs if n not in retriever.by_article]
    tf = top_tfidf_of(retriever, q, hits)
    cov = coverage_of(q, hits, int(cfg.get("coverage_top_n", 3)))
    if q.bis_refs:  # "المادة 147 مكرر" — amended/inserted articles are not in this corpus
        return EvidenceAssessment(False, tf, cov, explicit_found, [f"{n} مكرر" for n in q.bis_refs], "article_not_in_corpus")
    if q.article_refs and not explicit_found:
        return EvidenceAssessment(False, tf, cov, explicit_found, explicit_missing, "article_not_in_corpus")
    if explicit_found:
        return EvidenceAssessment(True, tf, cov, explicit_found, explicit_missing, "explicit_reference")
    if not hits:
        return EvidenceAssessment(False, tf, cov, reason="no_retrieval_results")
    ok = tf >= cfg["min_top_tfidf"] and cov >= cfg["min_query_coverage"]
    return EvidenceAssessment(ok, tf, cov, reason="thresholds_met" if ok else "below_thresholds")
