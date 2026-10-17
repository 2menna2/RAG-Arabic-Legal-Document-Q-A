# Reference pipeline (baseline only)

Verbatim copies of `src/rag/ingestion.py` and `src/rag/clean_dataset.py` from the reference
project `egyptian-legal-rag-main` (as provided in the upload). They are kept **only** to rebuild
the baseline corpus used in `reports/retrieval_benchmark.md`, so that the effect of the new data
pipeline can be measured on identical questions. They are not imported by the application.

Known defects (see docs/AUDIT_REPORT.md): lam-alef ligature corruption, reversed numbers, missing
Article 452, hand-typed article texts (1021, 1022, 279, 884, 901, 935), fabricated repeal text.

Rebuild (DVC stage `reference_baseline`):

    python baselines/reference_pipeline/run_reference.py data/raw/Law.pdf data/baselines/reference_civil_code_ready.json
