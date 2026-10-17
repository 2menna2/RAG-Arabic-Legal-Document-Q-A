# CI/CD

## Workflows

### `.github/workflows/ci.yml` — every PR and push to `main` (also callable by the release)

| job | what it does | fails when |
|---|---|---|
| `lint-typecheck` | `ruff check`, `ruff format --check`, `mypy` | any lint/type error |
| `unit-tests` | `pytest tests/unit` with coverage ≥ 60 % (synthetic fixtures) | any failure / coverage drop |
| `dependency-audit` | `pip-audit` on the locked runtime dependencies | known vulnerability |
| `data-quality-gate` | `dvc pull` the PDF → `dvc repro index` (extract, parse, **validate**, chunk, index) → checks `dvc.lock` unchanged → integration + e2e tests on the real corpus → `run_eval` → **quality gate** | validation error, non-reproducible outputs, test failure, metric below floor or regression vs `reports/baseline/` |
| `docker-build-scan-smoke` | build image tagged with the commit SHA, **Trivy** scan (fixable HIGH/CRITICAL), run read-only container, smoke test, assert non-root and no `.env` | build error, vulnerability, smoke failure |

Without the DVC secrets (e.g. PRs from forks) the data-dependent jobs print a warning and are skipped — CI **never**
generates synthetic data to make the gate pass. Make `data-quality-gate` a required status check on `main`.

### `.github/workflows/release.yml` — tag `v*` or manual

`validate` (re-runs the whole CI) → `publish-image` (GHCR, tags = SHA + version, SBOM + provenance) →
`deploy-production` (GitHub environment `production` = manual approval; Cloud Run revision with no traffic → smoke test
with a real LLM question → traffic shift → automatic rollback on failure).

## Secrets and permissions

| secret | used by | notes |
|---|---|---|
| `DVC_REMOTE_URL` | ci, release | `gdrive://<folder-id>` (free), `s3://…`, `gs://…` or `azure://…` |
| `GDRIVE_CREDENTIALS_DATA` | ci | gdrive only: service-account JSON; share the Drive folder with the service account (Viewer is enough for CI) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | ci | s3 only: read-only credentials |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | ci | gs only: service-account JSON |
| `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA` | release | Workload Identity Federation, no JSON key |
| `GCP_PROJECT`, `GCP_REGION` | release | |
| variable `CLOUD_RUN_MIN_INSTANCES` | release | optional; default `0` (scale to zero) |
| `GITHUB_TOKEN` | release | `packages: write` only in `publish-image` |
| `gemini-api-key` | **GCP Secret Manager**, not GitHub | mounted into Cloud Run |

The DVC remote plugin (`dvc-gdrive`, `dvc-s3`, …) is installed in an isolated `uvx` environment, not in the locked
project environment (`dvc-gdrive` pins an old `cryptography`).

Default workflow permissions are `contents: read`; jobs elevate only what they need (`packages: write`, `id-token: write`).
Actions are referenced by major version; Dependabot (`.github/dependabot.yml`) proposes updates — pin to commit SHAs
when you accept them for stricter supply-chain control.

## Validation status

Both workflows pass `actionlint 1.7.7`. They have **not been executed on GitHub** (no repository/secrets were
available). Each step was exercised locally: lint, mypy, tests, `dvc repro`, eval, gate (including a deliberately
degraded config that the gate rejects: `scripts/demo_gate_catches_regression.sh`), Docker build, read-only run and
smoke test. Trivy and pip-audit were not run locally.
