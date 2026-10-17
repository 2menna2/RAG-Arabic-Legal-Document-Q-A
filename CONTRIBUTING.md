# Contributing

1. `make setup && source .venv/bin/activate && dvc repro index` (the source PDF is in the repository)
2. Branch from `main`; keep changes small and focused.
3. Before pushing: `make lint typecheck test` and, if retrieval/prompt/config/data changed, `make eval gate`.
4. Never tune on the **test** split. Use dev for experiments; report test.
5. Changing the corpus, chunking or embedder requires `dvc repro` (the index manifest will refuse stale indexes) and a
   new evaluation; promote a new baseline only deliberately (`make baseline`) and explain why in the PR.
6. Never commit secrets or derived data (`data/processed`, `data/index`). Never replace `data/raw/Law.pdf`: its
   sha256 is pinned in `configs/source.yaml` and the pipeline fails on a different file.
7. Do not add hand-written legal text to the corpus. Source defects go to `configs/known_source_issues.yaml`.
8. Golden-set changes: new ids only (never reuse), keep `review_status` honest, bump the dataset version if items change.
