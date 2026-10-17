"""Evaluate (and calibrate) the deterministic decision layer — no LLM involved.

For each golden item the pipeline is run with ``llm=None`` and the
*pre-generation* outcome is compared with the expected behaviour:

==================  =========================================
expected_behavior   correct pre-generation outcome
==================  =========================================
answer              proceed (reaches the generator)
abstain             abstained
out_of_scope        out_of_scope
clarify             needs_clarification
reject              rejected
==================  =========================================

``--calibrate`` grid-searches the scope/evidence thresholds on the **dev**
split only (objective: balanced accuracy across behaviours subject to
≥ 90 % of answerable dev questions proceeding) and writes the result to
``reports/threshold_calibration.json``.  Test numbers are reported for the
chosen thresholds but never used to choose them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ecc_rag.config import PROJECT_ROOT, load_rag_config
from ecc_rag.evaluation.dataset import GoldenItem, dataset_hash, load_golden
from ecc_rag.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

OUTCOME_FOR = {
    "answer": "proceed",
    "abstain": "abstained",
    "out_of_scope": "out_of_scope",
    "clarify": "needs_clarification",
    "reject": "rejected",
}


def predicted_outcome(pipe: RAGPipeline, question: str) -> tuple[str, str | None, dict[str, Any]]:
    res = pipe.answer(question)
    if res.status == "error" and res.reason == "llm_unavailable":
        return "proceed", None, res.diagnostics
    return res.status, res.reason, res.diagnostics


def evaluate(pipe: RAGPipeline, items: list[GoldenItem]) -> list[dict[str, Any]]:
    rows = []
    for it in items:
        out, reason, diag = predicted_outcome(pipe, it.question)
        ok = {OUTCOME_FOR[it.expected_behavior], *(OUTCOME_FOR[b] for b in it.acceptable_behaviors)}
        rows.append(
            {
                "id": it.id,
                "split": it.split,
                "style": it.style,
                "type": it.type,
                "expected": OUTCOME_FOR[it.expected_behavior],
                "predicted": out,
                "reason": reason,
                "correct": out in ok,
                "top_tfidf": diag.get("top_tfidf"),
                "coverage": diag.get("coverage"),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    by_exp: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_exp[r["expected"]].append(r["correct"])
    per_class = {k: round(sum(v) / len(v), 4) for k, v in by_exp.items()}
    confusion = Counter((r["expected"], r["predicted"]) for r in rows)
    return {
        "n": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "balanced_accuracy": round(sum(per_class.values()) / len(per_class), 4),
        "per_expected_class_accuracy": per_class,
        "class_counts": {k: len(v) for k, v in by_exp.items()},
        "false_proceed_rate": round(
            sum(1 for r in rows if r["expected"] != "proceed" and r["predicted"] == "proceed")
            / max(1, sum(1 for r in rows if r["expected"] != "proceed")),
            4,
        ),
        "false_block_rate": round(
            sum(1 for r in rows if r["expected"] == "proceed" and r["predicted"] != "proceed")
            / max(1, sum(1 for r in rows if r["expected"] == "proceed")),
            4,
        ),
        "confusion": {f"{e}->{p}": c for (e, p), c in sorted(confusion.items())},
    }


def _features(pipe: RAGPipeline, it: GoldenItem) -> dict[str, Any]:
    """Threshold-independent features of one question (computed once)."""
    from ecc_rag.rag import evidence, guardrails

    if not guardrails.check_input(it.question, pipe.max_question_chars).ok or guardrails.detect_injection(it.question):
        return {"pre": "rejected"}
    q, hits = pipe.search(it.question)
    tf = evidence.top_tfidf_of(pipe.retriever, q, hits)
    cov = evidence.coverage_of(q, hits, int(pipe.cfg["evidence"].get("coverage_top_n", 3)))
    found = [n for n in q.article_refs if n in pipe.retriever.by_article]
    return {
        "pre": None,
        "tf": tf,
        "cov": cov,
        "refs": q.article_refs,
        "bis": q.bis_refs,
        "found": found,
        "n_hits": len(hits),
        "spec": pipe.specificity(it.question),
    }


def _decide(question: str, f: dict[str, Any], scope_cfg: dict[str, float], ev_cfg: dict[str, float]) -> str:
    """Mirror of the pipeline's pre-generation decision order (kept in sync by a unit test)."""
    from ecc_rag.rag import scope

    if f["pre"]:
        return f["pre"]
    sc = scope.classify(question, f["tf"], f["cov"], bool(f["refs"]) or bool(f["bis"]), scope_cfg, f["spec"])
    if not sc.in_scope:
        return "needs_clarification" if sc.reason == "needs_clarification" else "out_of_scope"
    if f["bis"] or (f["refs"] and not f["found"]):
        return "abstained"
    if f["found"]:
        return "proceed"
    if not f["n_hits"]:
        return "abstained"
    ok = f["tf"] >= ev_cfg["min_top_tfidf"] and f["cov"] >= ev_cfg["min_query_coverage"]
    return "proceed" if ok else "abstained"


def calibrate(pipe: RAGPipeline, items: list[GoldenItem]) -> dict[str, Any]:
    dev = [i for i in items if i.split == "dev"]
    feats = {i.id: _features(pipe, i) for i in dev}
    grid = {
        "min_tfidf_for_in_scope": [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22],
        "min_coverage_for_in_scope": [0.1, 0.2, 0.3, 0.4, 0.5],
        "min_top_tfidf": [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18],
        "min_query_coverage": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    }
    best: tuple[float, float, dict[str, float]] | None = None
    n_trials = 0
    for vals in itertools.product(*grid.values()):
        n_trials += 1
        params = dict(zip(grid.keys(), vals, strict=True))
        sc_cfg = {
            **pipe.cfg["scope"],
            "min_tfidf_for_in_scope": params["min_tfidf_for_in_scope"],
            "min_coverage_for_in_scope": params["min_coverage_for_in_scope"],
        }
        ev_cfg = {**pipe.cfg["evidence"], "min_top_tfidf": params["min_top_tfidf"], "min_query_coverage": params["min_query_coverage"]}
        rows = []
        for it in dev:
            pred = _decide(it.question, feats[it.id], sc_cfg, ev_cfg)
            ok = {OUTCOME_FOR[it.expected_behavior], *(OUTCOME_FOR[b] for b in it.acceptable_behaviors)}
            rows.append({"expected": OUTCOME_FOR[it.expected_behavior], "predicted": pred, "correct": pred in ok})
        s = summarize(rows)
        proceed_ok = s["per_expected_class_accuracy"].get("proceed", 0.0)
        if proceed_ok < 0.90:
            continue
        key = (s["balanced_accuracy"], proceed_ok)
        if best is None or key > (best[0], best[1]):
            best = (key[0], key[1], params)
    if best is None:
        raise RuntimeError("no threshold combination keeps >=90% of answerable dev questions")
    return {
        "objective": "max dev balanced accuracy s.t. dev proceed-rate >= 0.90 (ties: higher proceed-rate, then grid order)",
        "best": best[2],
        "dev_balanced_accuracy": best[0],
        "dev_proceed_rate": best[1],
        "trials": n_trials,
    }


def build_offline_pipeline() -> RAGPipeline:
    from ecc_rag.rag.index import load_index
    from ecc_rag.rag.pipeline import load_articles_map
    from ecc_rag.rag.retriever import Retriever

    cfg = load_rag_config()
    root = PROJECT_ROOT
    li = load_index(root / "data/processed/chunks.jsonl", root / "data/processed/articles.jsonl", root / "data/index", cfg)
    retr = Retriever(li.chunks, li.embedder, li.store, li.manifest["collection"], cfg["retrieval"])
    return RAGPipeline(retr, load_articles_map(root / "data/processed/articles.jsonl"), cfg, llm=None)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the pre-generation decision layer")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/gates_eval.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    version, items = load_golden()
    pipe = build_offline_pipeline()
    cal = None
    if args.calibrate:
        cal = calibrate(pipe, items)
        (PROJECT_ROOT / "reports/threshold_calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
        pipe.cfg["scope"].update({k: cal["best"][k] for k in ("min_tfidf_for_in_scope", "min_coverage_for_in_scope")})
        pipe.cfg["evidence"].update({k: cal["best"][k] for k in ("min_top_tfidf", "min_query_coverage")})
    rows = evaluate(pipe, items)
    out = {
        "dataset_version": version,
        "dataset_hash": dataset_hash(),
        "thresholds": {**pipe.cfg["scope"], **pipe.cfg["evidence"]},
        "calibration": cal,
        "dev": summarize([r for r in rows if r["split"] == "dev"]),
        "test": summarize([r for r in rows if r["split"] == "test"]),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("thresholds", "dev", "test")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
