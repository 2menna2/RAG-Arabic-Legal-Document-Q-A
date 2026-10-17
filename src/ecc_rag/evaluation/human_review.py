"""Summarise a completed human-review sheet (CSV exported by rag_eval).

Usage: python -m ecc_rag.evaluation.human_review reports/human_review/review_sheet_YYYY-MM-DD.csv
Only rows with a reviewer name are counted; the output states how many were reviewed.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

SCORES = [
    "correctness_0_2",
    "citation_correct_0_1",
    "evidence_complete_0_2",
    "unsupported_claim_0_1",
    "abstention_quality_0_2_or_NA",
    "clarity_0_2",
]


def summarize(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("reviewer") or "").strip()]
    out: dict[str, object] = {"sheet": str(path), "reviewed_rows": len(rows), "reviewers": sorted({r["reviewer"] for r in rows})}
    for col in SCORES:
        vals = [float(r[col]) for r in rows if (r.get(col) or "").strip() not in ("", "NA", "na")]
        out[col] = {"mean": round(statistics.fmean(vals), 3), "n": len(vals)} if vals else {"mean": None, "n": 0}
    return out


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), ensure_ascii=False, indent=2))
