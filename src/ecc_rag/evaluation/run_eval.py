"""Offline evaluation of the deployed configuration (no LLM needed) + MLflow logging.

Produces:
* ``reports/retrieval_eval.json``  — retrieval metrics of the *production* config on the real index
* ``reports/gates_eval.json``      — decision-layer (scope / evidence / guardrails) accuracy
* ``reports/eval_summary.json``    — compact summary used by the quality gate and Grafana
and logs everything as one MLflow run (params = configs/rag.yaml, tags = git
commit, dataset hash, corpus hash, index collection).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ecc_rag.config import PROJECT_ROOT, config_hash, load_rag_config
from ecc_rag.evaluation import gates_eval
from ecc_rag.evaluation.benchmark import KS, evaluate_config, summarize
from ecc_rag.evaluation.dataset import dataset_hash, load_golden
from ecc_rag.rag.index import git_commit

logger = logging.getLogger(__name__)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
    return out


def run(mlflow_uri: str | None, experiment: str, reports: Path) -> dict[str, Any]:
    cfg = load_rag_config()
    version, items = load_golden()
    pipe = gates_eval.build_offline_pipeline()
    manifest = json.loads((PROJECT_ROOT / "data/index/manifest.json").read_text(encoding="utf-8"))

    rows = evaluate_config(pipe.retriever, items, cfg["retrieval"]["method"], top_k=10)
    retrieval = {s: summarize(rows, s) for s in ("dev", "test")} | {"all": summarize(rows)}
    by_style = {st: summarize(rows, None, "style", st) for st in ("msa", "egyptian", "english")}
    by_type: dict[str, Any] = {}
    for t in sorted({r["type"] for r in rows}):
        by_type[t] = summarize(rows, None, "type", t)
    gate_rows = gates_eval.evaluate(pipe, items)
    gates = {
        "dev": gates_eval.summarize([r for r in gate_rows if r["split"] == "dev"]),
        "test": gates_eval.summarize([r for r in gate_rows if r["split"] == "test"]),
    }

    meta = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset": version,
        "dataset_hash": dataset_hash(),
        "corpus_hash": manifest["corpus_hash"],
        "index_collection": manifest["collection"],
        "embedder": manifest["embedder"],
        "rag_config_hash": config_hash(cfg),
        "python": platform.python_version(),
        "environment": "local" if not Path("/.dockerenv").exists() else "docker",
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "retrieval_eval.json").write_text(
        json.dumps({**meta, "metrics": retrieval, "by_style": by_style, "by_type": by_type, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
        newline="\n",
    )
    (reports / "gates_eval.json").write_text(
        json.dumps({**meta, "thresholds": {**cfg["scope"], **cfg["evidence"]}, **gates, "rows": gate_rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
        newline="\n",
    )
    keep = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@5", "precision@5", "hit@5"]
    summary = {
        **meta,
        "metrics": {f"retrieval_{s}": {k: retrieval[s][k] for k in keep} | {"n": retrieval[s]["n"]} for s in ("dev", "test")}
        | {
            f"gates_{s}": {k: gates[s][k] for k in ("accuracy", "balanced_accuracy", "false_proceed_rate", "false_block_rate", "n")}
            for s in ("dev", "test")
        },
    }
    (reports / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    if mlflow_uri:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=f"eval-{meta['git_commit'][:8]}"):
            mlflow.set_tags(
                {
                    "git_commit": meta["git_commit"],
                    "dataset_version": version,
                    "dataset_hash": meta["dataset_hash"],
                    "corpus_hash": meta["corpus_hash"],
                    "index_collection": meta["index_collection"],
                    "stage": "offline_eval",
                    "llm_generation_evaluated": "false",
                }
            )
            params = _flatten(cfg)
            mlflow.log_params({k[:250]: str(v)[:500] for k, v in params.items()})
            for group, ms in summary["metrics"].items():
                for k, v in ms.items():
                    mlflow.log_metric(f"{group}_{k.replace('@', '_at_')}", float(v))
            for f in ("retrieval_eval.json", "gates_eval.json", "eval_summary.json"):
                mlflow.log_artifact(str(reports / f))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mlflow-uri", default=None, help="e.g. sqlite:///mlflow.db or http://localhost:5000 (omit to skip logging)")
    p.add_argument("--experiment", default="ecc-rag-eval")
    p.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    s = run(args.mlflow_uri, args.experiment, args.reports)
    print(json.dumps(s["metrics"], indent=2))


if __name__ == "__main__":
    main()


__all__ = ["KS", "run"]
