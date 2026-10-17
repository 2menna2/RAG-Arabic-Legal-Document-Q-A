# Deployment

## 1. Runtime requirements

| Resource | Minimum (measured) | Notes |
|---|---|---|
| CPU | 1 vCPU works; 2 vCPU measured ~80 req/s retrieval | retrieval is CPU-bound; LLM time is I/O wait |
| Memory | 512 MiB idle ≈ 340 MB RSS; **1 GiB limit** held 32 concurrent users (peak RSS 467 MB) | with LSA embeddings; neural local models need far more |
| Disk (image) | ~825 MB uncompressed (217 MB content) | includes index |
| Python | 3.12 (3.11–3.13 supported by `pyproject.toml`) | exact deps in `uv.lock` |
| Docker | Engine 24+ with BuildKit (tested with 29.8.2) | `docker compose` v2 |
| Network egress | `generativelanguage.googleapis.com` (Gemini) | + `QDRANT_URL` host if a server is used |

Files needed at build time: `data/processed/articles.jsonl`, `data/processed/chunks.jsonl` (from `dvc pull` or `dvc repro`).
The image builds and embeds its own index; nothing else from `data/` is required at runtime.

## 2. Environment

Copy `.env.example` → `.env` (never committed). Essential variables:

| variable | purpose |
|---|---|
| `GEMINI_API_KEY` | **required** for `/api/v1/ask` (secret) |
| `LLM_PROVIDER` | `gemini` (default) or `none` = retrieval-only deployment (`/ask` → 503, `/ready` → 200) |
| `GEMINI_MODEL` | default `gemini-3.8-flash` (`gemini-2.5-flash` returned 404 "no longer available to new users" on 2026-10-17); list the models your key can use with `client.models.list()` |
| `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES` | 30 s, 2 retries (only on 429/5xx/timeouts) |
| `QDRANT_URL` / `QDRANT_API_KEY` | use a Qdrant server; unset = embedded index baked into the image |
| `RATE_LIMIT_PER_MINUTE` | per client IP and per replica (default 30) |
| `REQUEST_TIMEOUT_S`, `MAX_QUESTION_CHARS` | 60 s, 1000 |
| `CORS_ORIGINS` | JSON list; empty = CORS disabled |
| `LOG_LEVEL`, `LOG_QUESTIONS` | `hashed` by default (`none` / `redacted` / `full`) |
| `MLFLOW_TRACKING_URI` | evaluation tracking (not needed by the API) |

## 3. From clone to a running API (commands tested in this order)

```bash
git clone <repo> ecc-rag && cd ecc-rag
make setup && source .venv/bin/activate              # uv sync --frozen
dvc remote add -d --local storage <remote-url> && dvc pull   # or place Law.pdf + `dvc repro index`
make test                                            # 127 tests (1 skipped without GEMINI_API_KEY)
make serve                                           # local, embedded index
curl -s localhost:8000/health && curl -s localhost:8000/ready
```

Docker (tested: build, read-only run, health check `healthy`, `/ready`, `/search`, `/ask`):

```bash
make docker-build TAG=$(git rev-parse --short HEAD)
docker run -d --name ecc --read-only --tmpfs /tmp --cpus 2 --memory 1g \
  -p 127.0.0.1:8000:8000 --env-file .env ecc-rag-api:$(git rev-parse --short HEAD)
make smoke URL=http://127.0.0.1:8000                 # add --require-llm via scripts/smoke_test.sh when a key is set
```

Compose profiles: see README. `docker compose --profile qdrant up -d` starts Qdrant, runs the one-shot `indexer` into
the server, then `api-qdrant` (use this layout for several replicas).

## 4. Production target: Google Cloud Run (recommended here)

Why: single stateless container, scale-to-N, managed TLS/ingress, revisions with traffic splitting (= rollback),
Secret Manager integration, same cloud as Gemini. Kubernetes is not needed for one small service; if the organisation
already runs Kubernetes, the same image works with the settings in §6.

One-time setup (manual):

1. GCP project + Artifact Registry or use GHCR (the release workflow pushes to GHCR).
2. Secret Manager secret `gemini-api-key`; grant the Cloud Run runtime service account `secretAccessor`.
3. Workload Identity Federation for GitHub (`GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA` secrets) — no JSON keys in GitHub.
4. GitHub environment `production` with required reviewers.
5. Private service (`--no-allow-unauthenticated`); put an API gateway / IAP / load balancer with Cloud Armor in front
   for public exposure and global rate limiting.

Deployment (automated by `.github/workflows/release.yml` on a `v*` tag, after CI gates + manual approval):

```bash
gcloud run deploy ecc-rag-api --image ghcr.io/<org>/<repo>/ecc-rag-api:<sha> --no-traffic --tag sha-<short> \
  --port 8000 --cpu 2 --memory 1Gi --concurrency 8 --min-instances 0 --max-instances 4 --cpu-boost \
  --set-env-vars LLM_PROVIDER=gemini,LOG_QUESTIONS=hashed,RATE_LIMIT_PER_MINUTE=30 \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest --no-allow-unauthenticated
scripts/smoke_test.sh <tagged-revision-url via proxy> --require-llm
gcloud run services update-traffic ecc-rag-api --to-tags sha-<short>=100
```

Health/readiness: Cloud Run startup probe → `/ready` (HTTP 200 only when index, vector store and LLM config are OK);
liveness → `/health`. `--concurrency 8` matches the measured knee of the latency curve.

**Cost vs. cold start.** `--min-instances 0` (default in `release.yml`) scales to zero and fits the Cloud Run free
tier for low traffic; the first request after an idle period pays a cold start (measured locally: container start →
`/ready` ≈ 7.5 s on 1 vCPU, *plus* image pull time on Cloud Run). Set the repository variable
`CLOUD_RUN_MIN_INSTANCES=1` to keep one instance warm — that instance is billed continuously.
Check current free-tier limits on the Cloud Run pricing page; they are not hard-coded here.

## 5. Rollback

* Automatic: the release job shifts traffic only after the smoke test passes; if any later step fails, it routes 100 %
  back to the previously serving revision.
* Manual: `gcloud run services update-traffic ecc-rag-api --to-revisions <previous-revision>=100`.
* Images are immutable and tagged by git SHA and version; `latest` is never used.
* Data/model rollback: check out the previous Git tag → `dvc pull` → rebuild; the index manifest prevents mixing versions.

## 6. Kubernetes (if used)

`readinessProbe: httpGet /ready`, `livenessProbe: httpGet /health`, `resources: requests {cpu: 500m, memory: 512Mi},
limits {cpu: "2", memory: 1Gi}`, `securityContext: {runAsNonRoot: true, readOnlyRootFilesystem: true}`, an `emptyDir`
for `/tmp`, secret `GEMINI_API_KEY` from a Secret/ExternalSecret, Ingress with TLS (cert-manager), HPA on CPU. For more
than one replica set `QDRANT_URL` (StatefulSet or managed Qdrant) and run the indexer as a Job on each release.

## 7. What was NOT done

No real cloud deployment was performed (no cloud account, registry credentials or Gemini access in the build
environment). The release workflow is validated with `actionlint` but has not run on GitHub.
