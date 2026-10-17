# Evaluation methodology

Results: [`reports/EVALUATION_REPORT.md`](../reports/EVALUATION_REPORT.md) (generated) ·
[`reports/retrieval_benchmark.md`](../reports/retrieval_benchmark.md) · [`reports/mlflow_runs.md`](../reports/mlflow_runs.md).

## 1. Golden dataset

`data/evaluation/golden_v1.yaml` — 241 items: 191 answerable (direct, scenario, exception-sensitive, multi-article,
article lookup, repealed status, false premise, status-unverified) and 50 negative/boundary items (out-of-scope other
laws, non-legal, ambiguous, prompt injection, non-existent / "مكرر" articles, no-evidence topics). Styles: MSA 191,
Egyptian colloquial 37, English 13.

Fields: `question, style, type, difficulty, expected_behavior, acceptable_behaviors, expected_articles,
acceptable_articles, key_points, expected_failure_reason, origin, review_status, split`.

Rules: the split is deterministic (`sha256("golden_v1:"+id) % 100 < 40` → dev). Retrieval configuration, query
processing and gate thresholds were chosen on **dev only**; **test** is reported. The golden set is never used in
prompts. Items are `unreviewed` — legal review has not happened and must not be claimed.

## 2. Retrieval metrics (`src/ecc_rag/evaluation/retrieval_metrics.py`)

Ground truth = `expected_articles` (grade 2) and `acceptable_articles` (grade 1). Rankings are article-level and
de-duplicated.

| metric | definition |
|---|---|
| Recall@K | \|expected ∩ topK\| / \|expected\| |
| Hit@K | 1 if any expected article in topK |
| Precision@K | \|(expected ∪ acceptable) ∩ topK\| / K |
| MRR | 1 / rank of the first expected article |
| nDCG@K | graded gains 2/1, log2 discount, normalised by the ideal ranking |

Selection rule for the benchmark, fixed before running it: highest dev Recall@5, tie-break dev MRR.

## 3. Decision-layer metrics (`gates_eval.py`)

Pre-generation outcome vs expected behaviour (answer→proceed, abstain→abstained, out_of_scope, clarify, reject):
accuracy, balanced accuracy, per-class accuracy, **false-proceed rate** (negatives that would reach the LLM) and
**false-block rate** (answerable questions stopped). Thresholds are grid-searched on dev
(`--calibrate`, objective: balanced accuracy with ≥ 90 % of answerable dev questions proceeding).

## 4. Generation metrics with the LLM (`rag_eval.py`, needs `GEMINI_API_KEY`)

| metric | meaning | nature |
|---|---|---|
| outcome_accuracy | final status equals expected behaviour | deterministic |
| answered_rate / abstention_recall | coverage vs refusal | deterministic |
| expected_article_cited | expected articles among validated citations | deterministic |
| first_pass_validation_rate, citation_failure_rate | how often citations pass without / fail after repair | deterministic |
| heuristic_weak_support_rate | inferred claims with < 20 % lexical overlap with cited text | **heuristic**, not faithfulness |
| judge_key_point_coverage | Gemini-as-judge share of key points conveyed | **LLM-judged**, not human |
| ragas_faithfulness, ragas_answer_relevancy, ragas_llm_context_precision_with_reference, ragas_context_recall | official RAGAS 0.3.9 metrics | LLM-based library metrics |

Rows ending in `error` (provider unavailable / rate-limited / timeout) are excluded from the quality metrics and
reported as `n_errors`, `error_rate`, `error_reasons`. A metric with an empty denominator is `null` (not measured),
never `0`. A run is `complete` only with zero errors and no `--limit`; the quality gate rejects incomplete runs when
LLM floors are configured. Free-tier pacing: `--min-interval 7`; 429 retry-after hints are honoured up to
`--max-retry-wait` (default 90 s) in batch runs and `LLM_MAX_RETRY_WAIT_S` (default 8 s) in the API.

Status: not executed in the build environment (no Gemini access). The project owner ran a 20-question smoke run on
2026-10-17 with `gemini-3.8-flash`; it exposed free-tier 429s and the two metric defects fixed above, and is **not**
a valid evaluation (incomplete, answerable-only items). The RAGAS extra's import compatibility was verified
(`ragas==0.3.9` requires `langchain-community<0.4`).

## 5. Human review protocol

1. `make eval-llm` exports `reports/human_review/review_sheet_<date>.csv`: 40 answers stratified by question type.
2. A reviewer with Egyptian civil-law knowledge fills, per row: `correctness_0_2`, `citation_correct_0_1`,
   `evidence_complete_0_2`, `unsupported_claim_0_1`, `abstention_quality_0_2_or_NA`, `clarity_0_2`, `reviewer`, `notes`.
   Rubric: 2 = fully correct/complete, 1 = partially, 0 = wrong/missing; unsupported claim = 1 if any statement is not
   supported by the cited text.
3. `python -m ecc_rag.evaluation.human_review <sheet.csv>` summarises only rows with a reviewer name.
4. Disagreements between LLM-judge and human scores are investigated, not averaged away. LLM-as-a-judge never replaces
   this step for legal correctness.

## 6. Reproducing

```bash
dvc repro benchmark evaluate            # or: make benchmark eval
python scripts/render_eval_report.py    # regenerates reports/EVALUATION_REPORT.md from the JSON results
make gate
```
