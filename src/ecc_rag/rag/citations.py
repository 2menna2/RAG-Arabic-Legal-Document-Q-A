"""Citation validation — every legal claim must be traceable to retrieved text.

Checks (all deterministic):

``unknown_article``       a claim cites an article that is not in the context
``hallucinated_reference`` the answer text mentions "المادة N"/"[م N]"/"Article N"
                          for an N that is not in the context
``quote_mismatch``        a quote is not a (normalised) substring of the cited article
``missing_citation``      a claim of kind "stated" has no article
``empty_answer``          status answered but no answer / no claims
``repealed_without_notice`` a repealed article is cited but the answer does not
                          say it is repealed
Warnings (not blocking, reported for evaluation):
``weak_support``          an *inferred* claim shares < 20 % of its content
                          terms with the cited article(s) — heuristic,
                          labelled as such, not a semantic entailment check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ecc_rag.data_pipeline.arabic import normalize_for_search, to_western_digits, tokenize_for_search
from ecc_rag.rag.llm import GenerationOutput

ANSWER_REF_RE = re.compile(r"(?:\[\s*م\s*\.?\s*|(?:ال)?ماد[ةه]\s*(?:رقم\s*)?\(?\s*|\barticles?\s+|\bart\.\s*)([0-9٠-٩]{1,4})", re.I)
ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…)\s*")


@dataclass
class ValidationResult:
    valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cited_articles: list[int] = field(default_factory=list)
    quotes_checked: int = 0
    quotes_valid: int = 0


def _norm(s: str) -> str:
    return " ".join(normalize_for_search(s).split())


def quote_in_text(quote: str, text: str) -> bool:
    q = quote.strip().strip("\"«»“”'")
    if not q:
        return False
    nt = _norm(text)
    for part in ELLIPSIS_RE.split(q):
        p = _norm(part)
        if len(p) < 3:
            continue
        if p not in nt:
            return False
    return True


def answer_article_mentions(answer: str) -> list[int]:
    return sorted({int(to_western_digits(m.group(1))) for m in ANSWER_REF_RE.finditer(answer)})


def validate(output: GenerationOutput, context: dict[int, dict[str, Any]], require_citations: bool = True) -> ValidationResult:
    """``context`` maps article_number -> {"text": verbatim text, "legal_status": ...}."""
    res = ValidationResult(valid=True)
    if output.status != "answered":
        return res
    if not output.answer.strip():
        res.issues.append("empty_answer")
    if require_citations and not output.claims:
        res.issues.append("empty_answer: no claims with citations")
    cited: set[int] = set()
    for i, claim in enumerate(output.claims):
        if claim.kind == "stated" and not claim.articles:
            res.issues.append(f"missing_citation: claim {i}")
        for a in claim.articles:
            if a not in context:
                res.issues.append(f"unknown_article: claim {i} cites article {a} which is not in the retrieved context")
            else:
                cited.add(a)
        for qt in claim.quotes:
            res.quotes_checked += 1
            sources = [context[a]["text"] for a in claim.articles if a in context]
            if any(quote_in_text(qt, s) for s in sources):
                res.quotes_valid += 1
            else:
                res.issues.append(f"quote_mismatch: claim {i} quote not found verbatim in cited article(s)")
        if claim.kind == "inferred" and claim.articles:
            terms = set(tokenize_for_search(claim.statement))
            src = set()
            for a in claim.articles:
                if a in context:
                    src |= set(tokenize_for_search(context[a]["text"]))
            if terms and len(terms & src) / len(terms) < 0.2:
                res.warnings.append(f"weak_support: claim {i} (heuristic lexical overlap < 20%)")
    for n in answer_article_mentions(output.answer):
        if n not in context:
            res.issues.append(f"hallucinated_reference: answer mentions article {n} not in the retrieved context")
    for a in cited:
        if context[a].get("legal_status") == "repealed_per_source" and not re.search(r"ملغ|ألغي|الغي|repeal", output.answer, re.I):
            res.issues.append(f"repealed_without_notice: article {a} is repealed per source")
    res.cited_articles = sorted(cited)
    res.valid = not res.issues
    return res
