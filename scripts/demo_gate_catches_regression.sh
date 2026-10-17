#!/usr/bin/env bash
# Demonstrates that the quality gate blocks a harmful configuration change.
# A deliberately worse retrieval setting (BM25 only, without question-frame stripping) is evaluated into a
# temporary reports dir and checked against the committed baseline. Expected result: QUALITY GATE: FAILED.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
python - "$TMP" <<'PY'
import sys, yaml, pathlib
cfg = yaml.safe_load(open("configs/rag.yaml", encoding="utf-8"))
cfg["retrieval"]["method"] = "bm25"
cfg["retrieval"]["strip_question_frame"] = False
pathlib.Path(sys.argv[1], "rag_bad.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
PY
RAG_CONFIG_PATH="$TMP/rag_bad.yaml" python -m ecc_rag.evaluation.run_eval --reports "$TMP/reports" > /dev/null
set +e
python -m ecc_rag.evaluation.quality_gate --summary "$TMP/reports/eval_summary.json"
code=$?
set -e
rm -rf "$TMP"
if [ "$code" -eq 0 ]; then echo "UNEXPECTED: gate passed a degraded configuration"; exit 1; fi
echo "OK: gate rejected the degraded configuration (exit $code)"
