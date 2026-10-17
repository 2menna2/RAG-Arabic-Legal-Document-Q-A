# Contributing

1. `make setup && source .venv/bin/activate && dvc pull`
2. Branch from `main`; keep changes small and focused.
3. Before pushing: `make lint typecheck test` and, if retrieval/prompt/config/data changed, `make eval gate`.
4. Never tune on the **test** split. Use dev for experiments; report test.
5. Changing the corpus, chunking or embedder requires `dvc repro` (the index manifest will refuse stale indexes) and a
   new evaluation; promote a new baseline only deliberately (`make baseline`) and explain why in the PR.
6. Never commit secrets, `Law.pdf`, or derived full-text data; DVC handles data.
7. Do not add hand-written legal text to the corpus. Source defects go to `configs/known_source_issues.yaml`.
8. Golden-set changes: new ids only (never reuse), keep `review_status` honest, bump the dataset version if items change.
