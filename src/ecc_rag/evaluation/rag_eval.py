"""End-to-end RAG evaluation WITH the real LLM (Gemini). Requires GEMINI_API_KEY.

Metrics (all computed from this run's outputs):

Rows that ended in ``error`` (provider unavailable / rate-limited / timeout) are NOT quality judgements:
they are counted in ``n_errors``/``error_rate`` and excluded from every quality metric below. A metric whose
denominator is empty is ``null`` ("not measurable in this run"), never 0. A run with errors is marked
``complete: false`` and the quality gate rejects it when LLM floors are configured.

* ``outcome_accuracy``          status == expected behaviour (acceptable alternatives allowed)
* ``answered_rate``             share of answerable questions that end ``answered``
* ``abstention_recall``         share of non-answerable questions NOT answered
* ``expected_article_cited``    recall of expected articles among citations of answered items
* ``first_pass_validation_rate`` answers that passed citation validation without a repair round
* ``citation_failure_rate``     answerable questions abstained because citations failed validation
* ``heuristic_weak_support_rate`` inferred claims with < 20 % lexical overlap with the cited article
                                (HEURISTIC, not RAGAS faithfulness)
* optional ``judge_key_point_coverage`` — LLM-as-a-judge (Gemini) share of reference key points
  covered; labelled as LLM-judged, not human-validated
* optional RAGAS metrics (``--ragas``) via the ``ragas`` library, if installed and compatible.

A stratified sample is exported for human review (docs/EVALUATION.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ecc_rag.config import PROJECT_ROOT, get_settings, load_rag_config
from ecc_rag.evaluation.dataset import GoldenItem, dataset_hash, load_golden
from ecc_rag.evaluation.gates_eval import build_offline_pipeline
from ecc_rag.rag.index import git_commit
from ecc_rag.rag.llm import make_llm_client

logger = logging.getLogger(__name__)
STATUS_FOR = {
    "answer": "answered",
    "abstain": "abstained",
    "out_of_scope": "out_of_scope",
    "clarify": "needs_clarification",
    "reject": "rejected",
}

JUDGE_PROMPT = """You are grading a legal research assistant against reference key points.
Question: {question}
Reference key points (from the statute text):
{key_points}
System answer:
{answer}
For each key point answer 1 if the system answer conveys it correctly, else 0. Return JSON: {{"scores": [0 or 1, ...]}}"""


def run(
    split: str,
    limit: int | None,
    judge: bool,
    ragas: bool,
    sample_n: int,
    seed: int,
    min_interval_s: float = 0.0,
    max_retries: int | None = None,
    max_retry_wait_s: float | None = None,
) -> dict[str, Any]:
    if get_settings().gemini_api_key is None:
        raise SystemExit("GEMINI_API_KEY is not set — LLM evaluation cannot run (nothing is faked).")
    cfg = load_rag_config()
    version, items = load_golden()
    items = [i for i in items if split == "all" or i.split == split][: limit or None]
    pipe = build_offline_pipeline()
    llm = make_llm_client()
    if llm is None:
        raise SystemExit("LLM_PROVIDER=none — set LLM_PROVIDER=gemini to run the LLM evaluation.")
    # batch runs may wait longer for rate limits than an interactive API request would
    if hasattr(llm, "min_interval_s"):
        llm.min_interval_s = min_interval_s
    if max_retries is not None and hasattr(llm, "max_retries"):
        llm.max_retries = max_retries
    if max_retry_wait_s is not None and hasattr(llm, "max_retry_wait_s"):
        llm.max_retry_wait_s = max_retry_wait_s
    pipe.llm = llm
    rows = []
    for it in items:
        t0 = time.perf_counter()
        res = pipe.answer(it.question)
        ok = {STATUS_FOR[it.expected_behavior], *(STATUS_FOR[b] for b in it.acceptable_behaviors)}
        cited = sorted({a for c in res.citations for a in c.articles})
        d = res.diagnostics
        rows.append(
            {
                "id": it.id,
                "split": it.split,
                "style": it.style,
                "type": it.type,
                "expected_behavior": it.expected_behavior,
                "status": res.status,
                "reason": res.reason,
                "correct_outcome": res.status in ok,
                "answer": res.answer,
                "cited_articles": cited,
                "expected_articles": it.expected_articles,
                "retrieved_articles": d.get("retrieved_articles", []),
                "first_pass_valid": res.status == "answered" and not d.get("validation_attempts"),
                "citation_warnings": d.get("citation_warnings", []),
                "n_claims": len(res.citations),
                "latency_s": round(time.perf_counter() - t0, 3),
                "prompt_tokens": d.get("prompt_tokens"),
                "output_tokens": d.get("output_tokens"),
                "key_points": it.key_points,
                "question": it.question,
                "contexts": [s.text for s in res.sources],
            }
        )
        if res.status == "error":
            logger.error("%s: error %s", it.id, res.reason)
    metrics = compute_metrics(rows, items)
    if judge:
        metrics["judge_key_point_coverage (LLM-judged)"] = judge_key_points(llm, rows)
    if ragas:
        metrics.update(run_ragas(rows))
    out = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset": version,
        "dataset_hash": dataset_hash(),
        "split": split,
        "llm_model": llm.model,
        "limit": limit,
        "complete": metrics["n_errors"] == 0 and limit is None,
        "prompt_version": cfg["generation"]["prompt_version"],
        "metrics": metrics,
        "rows": rows,
    }
    reports = PROJECT_ROOT / "reports"
    (reports / "rag_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    export_review_sheet(rows, reports / "human_review" / f"review_sheet_{out['run_at'][:10]}.csv", sample_n, seed)
    return out


def _ratio(num: float, den: int) -> float | None:
    return round(num / den, 4) if den else None


def compute_metrics(rows: list[dict[str, Any]], items: list[GoldenItem]) -> dict[str, Any]:
    errors = [r for r in rows if r["status"] == "error"]
    scored = [r for r in rows if r["status"] != "error"]
    ans = [r for r in scored if r["expected_behavior"] == "answer"]
    neg = [r for r in scored if r["expected_behavior"] != "answer"]
    answered = [r for r in ans if r["status"] == "answered"]
    exp_recall = [
        len(set(r["cited_articles"]) & set(r["expected_articles"])) / len(r["expected_articles"])
        for r in answered
        if r["expected_articles"]
    ]
    claims = sum(r["n_claims"] for r in answered)
    weak = sum(sum(1 for w in r["citation_warnings"] if w.startswith("weak_support")) for r in answered)
    lat = sorted(r["latency_s"] for r in scored)
    return {
        "n": len(rows),
        "n_scored": len(scored),
        "n_errors": len(errors),
        "error_rate": _ratio(len(errors), len(rows)),
        "error_reasons": dict(sorted(Counter(r["reason"] for r in errors).items())),
        "n_answerable_scored": len(ans),
        "n_negative_scored": len(neg),
        "outcome_accuracy": _ratio(sum(r["correct_outcome"] for r in scored), len(scored)),
        "answered_rate": _ratio(len(answered), len(ans)),
        "abstention_recall": _ratio(sum(r["status"] != "answered" for r in neg), len(neg)),
        "expected_article_cited": round(statistics.fmean(exp_recall), 4) if exp_recall else None,
        "first_pass_validation_rate": _ratio(sum(r["first_pass_valid"] for r in answered), len(answered)),
        "citation_failure_rate": _ratio(sum(r["reason"] == "citation_validation_failed" for r in ans), len(ans)),
        "heuristic_weak_support_rate": _ratio(weak, claims),
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
        "prompt_tokens_total": sum(r["prompt_tokens"] or 0 for r in rows),
        "output_tokens_total": sum(r["output_tokens"] or 0 for r in rows),
    }


def judge_key_points(llm: Any, rows: list[dict[str, Any]]) -> float:
    scores = []
    for r in rows:
        if r["status"] != "answered" or not r["key_points"]:
            continue
        prompt = JUDGE_PROMPT.format(question=r["question"], key_points="\n".join(f"- {k}" for k in r["key_points"]), answer=r["answer"])
        try:
            raw = llm.complete_text(prompt)
            s = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])["scores"]
            scores.append(sum(s) / max(1, len(s)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge failed on %s: %s", r["id"], exc)
    return round(statistics.fmean(scores), 4) if scores else 0.0


def run_ragas(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Official RAGAS metrics (needs `pip install .[ragas]` and GEMINI_API_KEY). Untested offline."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    except ImportError as exc:
        logger.warning("ragas not available: %s", exc)
        return {"ragas_status": "not_installed"}  # type: ignore[dict-item]
    s = get_settings()
    key = s.gemini_api_key.get_secret_value() if s.gemini_api_key else None
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=s.gemini_model, google_api_key=key, temperature=0))
    emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model=f"models/{s.gemini_embedding_model}", google_api_key=key))
    samples = [
        {"user_input": r["question"], "response": r["answer"], "retrieved_contexts": r["contexts"], "reference": " ".join(r["key_points"])}
        for r in rows
        if r["status"] == "answered"
    ]
    ds = EvaluationDataset.from_list(samples)
    res = evaluate(
        ds, metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()], llm=llm, embeddings=emb
    )
    df = res.to_pandas()
    return {f"ragas_{c}": round(float(df[c].mean()), 4) for c in df.columns if df[c].dtype.kind == "f"}


def export_review_sheet(rows: list[dict[str, Any]], path: Path, n: int, seed: int) -> None:
    rnd = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    sample: list[dict[str, Any]] = []
    while len(sample) < min(n, len(rows)):
        for t in sorted(by_type):
            if by_type[t] and len(sample) < n:
                sample.append(by_type[t].pop(rnd.randrange(len(by_type[t]))))
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "id",
        "question",
        "expected_behavior",
        "status",
        "answer",
        "cited_articles",
        "expected_articles",
        "key_points",
        "correctness_0_2",
        "citation_correct_0_1",
        "evidence_complete_0_2",
        "unsupported_claim_0_1",
        "abstention_quality_0_2_or_NA",
        "clarity_0_2",
        "reviewer",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sample:
            w.writerow({k: (json.dumps(r[k], ensure_ascii=False) if isinstance(r.get(k), list) else r.get(k, "")) for k in cols})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge", action="store_true", help="add LLM-as-a-judge key-point coverage")
    p.add_argument("--ragas", action="store_true", help="add official RAGAS metrics (pip install .[ragas])")
    p.add_argument("--review-sample", type=int, default=40)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--min-interval", type=float, default=0.0, help="seconds between LLM calls (pace free-tier rate limits, e.g. 7)")
    p.add_argument("--max-retries", type=int, default=5, help="retries per LLM call on 429/5xx (batch default 5)")
    p.add_argument("--max-retry-wait", type=float, default=90.0, help="longest provider retry-after (s) to honour before giving up")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    for noisy in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = run(
        args.split,
        args.limit,
        args.judge,
        args.ragas,
        args.review_sample,
        args.seed,
        min_interval_s=args.min_interval,
        max_retries=args.max_retries,
        max_retry_wait_s=args.max_retry_wait,
    )
    print(json.dumps(out["metrics"], ensure_ascii=False, indent=2))
    if not out["complete"]:
        print(
            f"NOTE: incomplete run (errors={out['metrics']['n_errors']}, limit={out['limit']}) — "
            "fine for a smoke check, not valid as a baseline or for the quality gate."
        )


if __name__ == "__main__":
    main()
