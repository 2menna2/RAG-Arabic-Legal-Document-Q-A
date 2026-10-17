# Performance & monitoring validation (measured on 2026-10-17)

Raw results: `reports/loadtest/*.json` (produced by `scripts/loadtest.py`).

## Test setup

| item | value |
|---|---|
| host | 1 VM, 2 vCPU Intel Xeon @ 2.10 GHz, 7.8 GiB RAM (load generator runs **on the same host** and competes for CPU) |
| container | image `ecc-rag-api` (commit `6494003`), `--read-only --tmpfs /tmp --cpus 2 --memory 1g`, 1 uvicorn worker, embedded Qdrant |
| corpus / index | 1,149 articles → 1,149 chunks (`article_ctx`), LSA-256 vectors, hybrid BM25 + char-TF-IDF + dense (RRF) |
| LLM | **disabled** (`LLM_PROVIDER=none`): Gemini was not reachable from this environment, so LLM latency is NOT included |
| traffic | closed loop, each virtual user sends the next request when the previous returns; questions sampled from the 241 golden questions |

## Results — `POST /api/v1/search` (retrieval only), 20 s per level

| concurrency | requests | throughput (req/s) | p50 ms | p95 ms | p99 ms | errors |
|---|---|---|---|---|---|---|
| 1 | 1212 | 60.6 | 15.5 | 20.5 | 35.2 | 0 % |
| 4 | 1633 | 81.5 | 47.4 | 71.2 | 86.4 | 0 % |
| 8 | 1585 | 79.0 | 97.9 | 146.8 | 184.5 | 0 % |
| 16 | 1505 | 75.0 | 204.3 | 311.0 | 403.5 | 0 % |
| 32 | 1423 | 70.6 | 447.9 | 609.6 | 714.4 | 0 % |

Server resident memory (from its own `/metrics`): 341 MB before → 467 MB after the run (no OOM at 32 concurrent).
Throughput saturates at ~80 req/s at 4 concurrent users: retrieval is CPU-bound (BM25 scoring in pure Python + GIL);
beyond that, latency grows linearly with queue length.

## Results — `POST /api/v1/ask` with the LLM disabled, 15 s per level

Measures the full pre-generation path (input checks, injection detection, retrieval, scope, evidence gate). Answerable
questions end in `503 llm_unavailable` (by design: no fake answer), negatives end in `200` with
`rejected/out_of_scope/needs_clarification/abstained` — both are counted as successful responses here.

| concurrency | requests | throughput (req/s) | p50 ms | p95 ms | p99 ms | errors |
|---|---|---|---|---|---|---|
| 1 | 674 | 44.9 | 22.1 | 28.2 | 38.2 | 0 % |
| 4 | 938 | 62.3 | 64.3 | 92.5 | 104.0 | 0 % |
| 8 | 874 | 58.0 | 137.2 | 199.4 | 226.2 | 0 % |

With Gemini enabled, end-to-end latency will be dominated by the LLM call (and a possible repair round after a failed
citation validation); this must be measured with `make loadtest URL=...` against a deployment that has a key.

## Defect found by the load test (and fixed)

The first run OOM-killed the 1 GiB container at 16 concurrent users (`OOMKilled=true`, 99.7 % connection errors).
Profiling showed that `TruncatedSVD.transform` on one sparse query row copies the 256 × 33,987 projection matrix
(≈ 70 MB) on every call. `LsaEmbedder.embed_query` now gathers only the rows of the non-zero features
(≈ 0.2 MB, identical result to 1e-17; retrieval metrics unchanged). After the fix: p50 at 1 user 51.7 → 15.5 ms,
peak throughput 30.7 → 81.5 req/s, no OOM at 32 users.

## Monitoring validation

What was actually exercised:

| check | how | result |
|---|---|---|
| `/metrics` exposition | scraped by a real Prometheus 2.53.2 binary (`promtool` + server) at 5 s interval against both containers | both targets `up` |
| alert rules syntax | `promtool check rules deploy/prometheus/alerts.yml` | 9 rules, SUCCESS |
| alert evaluation | Prometheus rule evaluation during the load test | `EccNotReady` **pending** on the container without `GEMINI_API_KEY` (correct); `EccHigh5xxRate` **pending** while `/ask` returned 503 llm_unavailable (correct) |
| PromQL used by the dashboard | `histogram_quantile(0.95, …ecc_http_request_duration_seconds_bucket{route="/api/v1/search"}…)` queried through the Prometheus API | 0.48 s at the 32-user level (consistent with the load-test p95/p99) |
| RAG outcome counters | `sum by (status)(ecc_rag_outcomes_total)` | error 2003, out_of_scope 275, rejected 95, abstained 70, needs_clarification 43 |
| Grafana | dashboard JSON generated and provisioning files written | **not rendered**: Grafana binaries/images could not be downloaded in this environment; validate with `docker compose --profile monitoring up` |
| Alertmanager / notifications | not configured | routing alerts to Slack/e-mail is a deployment task (see docs/MONITORING.md) |

Quality metrics shown in Grafana (`ecc_eval_metric`) come from the **offline evaluation run baked into the image**
(`reports/eval_summary.json`, labelled with `run_at` and `dataset`), never from live traffic.
