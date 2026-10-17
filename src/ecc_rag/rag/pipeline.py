"""End-to-end RAG pipeline with an explicit, non-overlapping outcome taxonomy.

=====================  =====================================================
status                 meaning
=====================  =====================================================
answered               grounded answer whose citations passed validation
abstained              in scope, but evidence insufficient / citations failed /
                       requested article not in corpus / model declined
out_of_scope           other statute or non-legal question
needs_clarification    question too vague to answer
rejected               invalid input or prompt-injection attempt
error                  dependency failure (LLM, vector store, index) — never
                       disguised as an answer
=====================  =====================================================
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ecc_rag.config import PROJECT_ROOT
from ecc_rag.rag import citations, evidence, guardrails, scope
from ecc_rag.rag.llm import GenerationOutput, LLMClient, LLMError, LLMResult
from ecc_rag.rag.query import AnalyzedQuery, analyze_query
from ecc_rag.rag.retriever import CrossEncoderReranker, Hit, RetrievalError, Retriever
from ecc_rag.rag.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

DISCLAIMER_AR = (
    "تنبيه: هذه أداة مساعدة للبحث في نصوص القانون المدني المصري كما وردت في المصدر المتاح للنظام، "
    "وليست استشارة قانونية ولا بديلاً عن محامٍ. لم يتم التحقق من سريان النصوص أو تعديلاتها اللاحقة."
)
DISCLAIMER_EN = (
    "Notice: research aid over the Egyptian Civil Code text available to the system; not legal advice and not a "
    "substitute for a lawyer. Current validity/amendments of the texts are not verified."
)

ABSTAIN_MESSAGES = {
    "insufficient_evidence": "لم تسمح النصوص المتاحة في قاعدة البيانات بإجابة موثقة عن هذا السؤال.",
    "model_insufficient_evidence": "بعد مراجعة المواد المسترجعة، لا تكفي النصوص المتاحة لإجابة موثقة.",
    "citation_validation_failed": "تعذّر إنتاج إجابة تجتاز التحقق الآلي من الاستشهادات، لذلك لم تُعرض إجابة.",
    "article_not_in_corpus": "رقم المادة المطلوب غير موجود في نص القانون المدني المتاح (المواد من 1 إلى 1149).",
    "context_rejected": "تم استبعاد المستندات المسترجعة لاحتوائها على تعليمات مشبوهة، ولا توجد أدلة كافية بدونها.",
}


@dataclass
class Source:
    article_number: int
    citation: str
    source_page: int | None
    legal_status: str
    text: str
    rank: int
    scores: dict[str, float]
    status_note: str | None = None


@dataclass
class Citation:
    statement: str
    articles: list[int]
    quotes: list[str]
    kind: str


@dataclass
class RAGResult:
    status: str
    answer: str | None = None
    reason: str | None = None
    message: str | None = None
    citations: list[Citation] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    clarification_question: str | None = None
    notices: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Observer = Callable[[str, dict[str, Any]], None]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        articles: dict[int, dict[str, Any]],
        cfg: dict[str, Any],
        llm: LLMClient | None,
        prompts: dict[str, str] | None = None,
        reranker: CrossEncoderReranker | None = None,
        max_question_chars: int = 1000,
        observer: Observer | None = None,
    ):
        self.retriever = retriever
        self.articles = articles
        self.cfg = cfg
        self.llm = llm
        self.reranker = reranker
        self.prompts = prompts or load_prompts(cfg["generation"]["prompt_version"])
        self.max_question_chars = max_question_chars
        self.observer = observer or (lambda _e, _d: None)

    # ------------------------------------------------------------------ helpers
    def _doc_text(self, n: int, english: bool) -> str:
        a = self.articles[n]
        if a["legal_status"] == "repealed_per_source":
            ev = a.get("status_evidence") or {}
            return f"[ملغاة وفقاً لملاحظة المصدر، صفحة {ev.get('source_page')}] {ev.get('note_ar') or ''} {ev.get('note_en') or ''}".strip()
        text = a["article_text"] or ""
        if english or not text:
            text = (text + ("\n[English translation] " + a["text_en"] if a.get("text_en") else "")).strip()
        return text[: int(self.cfg["generation"].get("max_article_chars", 2500))]

    def _sources(self, hits: list[Hit], english: bool) -> list[Source]:
        out = []
        for h in hits:
            a = self.articles[h.article_number]
            ev = a.get("status_evidence") or {}
            out.append(
                Source(
                    article_number=h.article_number,
                    citation=a["citation"],
                    source_page=a["source_page"],
                    legal_status=a["legal_status"],
                    text=self._doc_text(h.article_number, english),
                    rank=h.rank,
                    scores={k: round(v, 4) for k, v in h.scores.items()},
                    status_note=ev.get("note_ar") if ev else None,
                )
            )
        return out

    def _build_user_prompt(self, question: str, sources: list[Source]) -> str:
        docs = "\n".join(
            self.prompts["document_template"].format(
                article_number=s.article_number,
                source_page=s.source_page,
                legal_status=s.legal_status,
                text=guardrails.escape_document(s.text),
            )
            for s in sources
        )
        return self.prompts["user_template"].format(documents=docs, question=guardrails.escape_document(question))

    def _notices(self, english: bool, sources: list[Source]) -> list[str]:
        notes = [DISCLAIMER_EN if english else DISCLAIMER_AR]
        if any(s.legal_status == "repealed_per_source" for s in sources):
            notes.append("بعض المواد المسترجعة ملغاة وفقاً لملاحظة واردة في المصدر نفسه.")
        flagged = [s.article_number for s in sources if self.articles[s.article_number].get("quality_flags")]
        if flagged:
            notes.append(f"المواد {flagged} عليها ملاحظات جودة استخراج من المصدر؛ راجع النص الأصلي.")
        return notes

    def specificity(self, question: str) -> float:
        from ecc_rag.data_pipeline.arabic import tokenize_for_search

        idf = self.retriever.lex.bm25.idf
        vals = [idf.get(t, 0.0) for t in tokenize_for_search(question) if t in idf]
        return max(vals) if vals else 0.0

    # ------------------------------------------------------------------ main
    def search(self, question: str, top_k: int | None = None) -> tuple[AnalyzedQuery, list[Hit]]:
        q = analyze_query(question)
        k = top_k or int(self.cfg["retrieval"]["top_k"])
        method = self.cfg["retrieval"]["method"]
        if self.reranker is not None:
            cands = self.retriever.retrieve(q, method, int(self.cfg["reranker"]["candidates"]))
            return q, self.reranker.rerank(question, cands, k)
        return q, self.retriever.retrieve(q, method, k)

    def answer(self, question: str, top_k: int | None = None) -> RAGResult:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        diag: dict[str, Any] = {"timings_ms": timings}

        def finish(res: RAGResult) -> RAGResult:
            timings["total"] = round(1000 * (time.perf_counter() - t_start), 2)
            res.diagnostics = {**diag, **res.diagnostics}
            self.observer("result", {"status": res.status, "reason": res.reason})
            return res

        chk = guardrails.check_input(question, self.max_question_chars)
        if not chk.ok:
            return finish(RAGResult("rejected", reason=f"invalid_input:{chk.code}", message=chk.message))
        inj = guardrails.detect_injection(question)
        if inj:
            self.observer("guardrail", {"kind": "prompt_injection", "signals": inj})
            return finish(
                RAGResult(
                    "rejected",
                    reason="prompt_injection",
                    message="تم رفض الطلب لاحتوائه على تعليمات تحاول تغيير سلوك النظام.",
                    diagnostics={"injection_signals": inj},
                )
            )
        english = sum(c.isascii() and c.isalpha() for c in question) > sum("؀" <= c <= "ۿ" for c in question)

        t0 = time.perf_counter()
        try:
            q, hits = self.search(question, top_k)
        except (VectorStoreError, RetrievalError) as exc:
            logger.error("retrieval failure: %s", exc)
            return finish(RAGResult("error", reason="retrieval_unavailable", message="خدمة الاسترجاع غير متاحة حالياً."))
        timings["retrieval"] = round(1000 * (time.perf_counter() - t0), 2)
        self.observer("retrieval", {"seconds": time.perf_counter() - t0})

        ev_cfg = {**self.cfg["evidence"]}
        ass = evidence.assess(self.retriever, q, hits, ev_cfg)
        diag.update(
            {
                "retrieved_articles": [h.article_number for h in hits],
                "top_tfidf": round(ass.top_tfidf, 4),
                "coverage": round(ass.coverage, 4),
                "evidence": ass.reason,
            }
        )
        sc = scope.classify(
            question, ass.top_tfidf, ass.coverage, bool(q.article_refs) or bool(q.bis_refs), self.cfg["scope"], self.specificity(question)
        )
        if not sc.in_scope:
            status = "needs_clarification" if sc.reason == "needs_clarification" else "out_of_scope"
            self.observer("guardrail", {"kind": sc.reason})
            return finish(
                RAGResult(
                    status,
                    reason=sc.reason,
                    message=sc.detail,
                    clarification_question="ما المسألة القانونية أو نوع العقد أو المادة التي تقصدها تحديداً؟"
                    if status == "needs_clarification"
                    else None,
                )
            )
        if not ass.sufficient:
            reason = ass.reason if ass.reason == "article_not_in_corpus" else "insufficient_evidence"
            self.observer("abstain", {"reason": reason})
            return finish(
                RAGResult(
                    "abstained",
                    reason=reason,
                    message=ABSTAIN_MESSAGES[reason],
                    sources=self._sources(hits, english) if reason != "article_not_in_corpus" else [],
                )
            )

        sources = self._sources(hits[: int(self.cfg["generation"]["max_context_articles"])], english)
        flagged = guardrails.scan_documents([s.text for s in sources])
        dropped = [s.article_number for s, f in zip(sources, flagged, strict=True) if f]
        if dropped:
            self.observer("guardrail", {"kind": "document_injection"})
            diag["dropped_documents"] = dropped
            sources = [s for s, f in zip(sources, flagged, strict=True) if not f]
            if not sources:
                return finish(RAGResult("abstained", reason="context_rejected", message=ABSTAIN_MESSAGES["context_rejected"]))

        if self.llm is None:
            return finish(
                RAGResult("error", reason="llm_unavailable", message="خدمة توليد الإجابات غير مهيأة (لا يوجد LLM).", sources=sources)
            )
        context = {s.article_number: {"text": s.text, "legal_status": s.legal_status} for s in sources}
        user_prompt = self._build_user_prompt(question, sources)
        history: list[dict[str, str]] = []
        result: LLMResult | None = None
        validation = None
        t0 = time.perf_counter()
        try:
            for attempt in range(int(self.cfg["generation"].get("max_validation_retries", 1)) + 1):
                result = self.llm.generate(self.prompts["system"], user_prompt, history)
                self.observer(
                    "llm",
                    {
                        "model": result.model,
                        "seconds": result.latency_s,
                        "prompt_tokens": result.prompt_tokens,
                        "output_tokens": result.output_tokens,
                    },
                )
                validation = citations.validate(result.output, context, self.cfg["generation"].get("require_citations", True))
                if validation.valid or result.output.status != "answered":
                    break
                diag.setdefault("validation_attempts", []).append(validation.issues)
                history = [
                    {"role": "user", "text": user_prompt},
                    {"role": "model", "text": result.raw},
                ]
                user_prompt = self.prompts["repair_template"].format(issues="\n".join(f"- {i}" for i in validation.issues))
                if attempt == 0:
                    self.observer("citation_retry", {})
        except LLMError as exc:
            logger.error("llm failure: %s", exc)
            self.observer("llm_error", {"code": exc.code})
            return finish(
                RAGResult("error", reason=exc.code, message="تعذّر الحصول على إجابة من نموذج اللغة. لم يتم توليد أي إجابة.", sources=sources)
            )
        timings["generation"] = round(1000 * (time.perf_counter() - t0), 2)
        assert result is not None and validation is not None
        out: GenerationOutput = result.output
        diag.update(
            {
                "llm_model": result.model,
                "prompt_version": self.prompts.get("version"),
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens,
                "citation_warnings": validation.warnings,
                "quotes_checked": validation.quotes_checked,
                "quotes_valid": validation.quotes_valid,
            }
        )
        if out.status == "insufficient_evidence":
            return finish(
                RAGResult(
                    "abstained",
                    reason="model_insufficient_evidence",
                    message=ABSTAIN_MESSAGES["model_insufficient_evidence"]
                    + (f" ({out.missing_information})" if out.missing_information else ""),
                    sources=sources,
                )
            )
        if out.status == "needs_clarification":
            return finish(
                RAGResult(
                    "needs_clarification",
                    reason="model_needs_clarification",
                    clarification_question=out.clarification_question,
                    sources=sources,
                )
            )
        if out.status == "out_of_scope":
            return finish(RAGResult("out_of_scope", reason="model_out_of_scope", message="السؤال خارج نطاق القانون المدني المتاح."))
        if not validation.valid:
            self.observer("abstain", {"reason": "citation_validation_failed"})
            diag["validation_issues"] = validation.issues
            return finish(
                RAGResult(
                    "abstained",
                    reason="citation_validation_failed",
                    message=ABSTAIN_MESSAGES["citation_validation_failed"],
                    sources=sources,
                )
            )
        cits = [Citation(c.statement, c.articles, c.quotes, c.kind) for c in out.claims]
        return finish(RAGResult("answered", answer=out.answer, citations=cits, sources=sources, notices=self._notices(english, sources)))


def load_prompts(version: str) -> dict[str, str]:
    data = yaml.safe_load((PROJECT_ROOT / "configs" / "prompts" / f"{version}.yaml").read_text(encoding="utf-8"))
    return data


def load_articles_map(path: Path) -> dict[int, dict[str, Any]]:
    return {r["article_number"]: r for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}
