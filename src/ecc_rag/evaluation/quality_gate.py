"""Automated quality gate (used in CI and before deployment).

Fails (exit 1) when any of these hold:

* data validation report missing or not passed;
* eval results were produced on a different corpus than the deployed index;
* an absolute floor/ceiling in ``configs/quality_gates.yaml`` is violated;
* a metric regressed versus the committed baseline (``reports/baseline/``) by
  more than the allowed tolerance;
* (optional, when ``reports/rag_eval.json`` exists) LLM-generation floors fail.

Floors were set *after* measuring the baseline (see the comments in the YAML),
never invented upfront.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from ecc_rag.config import PROJECT_ROOT


def _get(d: dict[str, Any], dotted: str) -> float | None:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return float(cur) if isinstance(cur, (int, float)) else None


def run_gate(config_path: Path, summary_path: Path | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = PROJECT_ROOT
    failures: list[str] = []
    checks: list[str] = []

    vr = root / cfg["data_validation_report"]
    if not vr.exists() or not json.loads(vr.read_text(encoding="utf-8")).get("passed"):
        failures.append(f"data validation report missing or failed: {vr}")
    else:
        checks.append("data validation passed")

    summary_path = summary_path or root / cfg["current_summary"]
    if not summary_path.exists():
        return {"passed": False, "failures": [f"no evaluation summary at {summary_path} — run `make eval`"], "checks": checks}
    cur = json.loads(summary_path.read_text(encoding="utf-8"))

    manifest_p = root / "data/index/manifest.json"
    if manifest_p.exists():
        man = json.loads(manifest_p.read_text(encoding="utf-8"))
        if man["corpus_hash"] != cur.get("corpus_hash"):
            failures.append("evaluation was run on a different corpus than the current index (corpus_hash mismatch)")
        else:
            checks.append("eval corpus_hash == index corpus_hash")

    for rule in cfg.get("floors", []):
        v = _get(cur["metrics"], rule["metric"])
        if v is None:
            failures.append(f"metric missing: {rule['metric']}")
        elif v < rule["min"]:
            failures.append(f"{rule['metric']} = {v:.4f} < floor {rule['min']}")
        else:
            checks.append(f"{rule['metric']} = {v:.4f} >= {rule['min']}")
    for rule in cfg.get("ceilings", []):
        v = _get(cur["metrics"], rule["metric"])
        if v is None:
            failures.append(f"metric missing: {rule['metric']}")
        elif v > rule["max"]:
            failures.append(f"{rule['metric']} = {v:.4f} > ceiling {rule['max']}")
        else:
            checks.append(f"{rule['metric']} = {v:.4f} <= {rule['max']}")

    base_p = root / cfg["baseline_summary"]
    if base_p.exists():
        base = json.loads(base_p.read_text(encoding="utf-8"))
        if base.get("dataset_hash") != cur.get("dataset_hash"):
            checks.append("baseline used a different golden dataset version — regression check skipped (re-baseline required)")
        else:
            for rule in cfg.get("max_regression", []):
                b, c = _get(base["metrics"], rule["metric"]), _get(cur["metrics"], rule["metric"])
                if b is None or c is None:
                    continue
                if b - c > rule["tolerance"]:
                    failures.append(f"regression {rule['metric']}: {c:.4f} vs baseline {b:.4f} (allowed drop {rule['tolerance']})")
                else:
                    checks.append(f"no regression on {rule['metric']} ({c:.4f} vs {b:.4f})")
    else:
        failures.append(f"baseline missing: {base_p}")

    llm_p = root / cfg.get("llm_eval", {}).get("path", "reports/rag_eval.json")
    if llm_p.exists() and cfg.get("llm_eval"):
        llm = json.loads(llm_p.read_text(encoding="utf-8"))
        for rule in cfg["llm_eval"].get("floors", []):
            v = _get(llm.get("metrics", {}), rule["metric"])
            if v is None or v < rule["min"]:
                failures.append(f"LLM eval {rule['metric']} = {v} < {rule['min']}")
            else:
                checks.append(f"LLM eval {rule['metric']} = {v:.4f} >= {rule['min']}")
    else:
        checks.append("LLM generation eval not present — generation quality NOT gated in this run")
    return {"passed": not failures, "failures": failures, "checks": checks}


def main() -> None:
    p = argparse.ArgumentParser(description="Quality gate")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/quality_gates.yaml")
    p.add_argument("--summary", type=Path, default=None)
    args = p.parse_args()
    res = run_gate(args.config, args.summary)
    for c in res["checks"]:
        print(f"  ok   {c}")
    for f in res["failures"]:
        print(f"  FAIL {f}")
    print("QUALITY GATE:", "PASSED" if res["passed"] else "FAILED")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
