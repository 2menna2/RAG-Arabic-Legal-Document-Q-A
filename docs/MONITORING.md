# Monitoring and observability

## Metrics (`GET /metrics`, Prometheus format)

| metric | type | labels | meaning |
|---|---|---|---|
| `ecc_http_requests_total` | counter | method, route, status_code | requests (route template, not raw path) |
| `ecc_http_request_duration_seconds` | histogram | route | end-to-end latency |
| `ecc_rag_outcomes_total` | counter | status, reason | answered / abstained / out_of_scope / needs_clarification / rejected / error |
| `ecc_abstentions_total` | counter | reason | insufficient_evidence, citation_validation_failed, article_not_in_corpus, … |
| `ecc_guardrail_events_total` | counter | kind | prompt_injection, document_injection, other_law:*, non_legal, needs_clarification |
| `ecc_retrieval_duration_seconds` | histogram | – | retrieval latency |
| `ecc_llm_request_duration_seconds` | histogram | model | Gemini latency |
| `ecc_llm_errors_total` | counter | code | llm_unavailable / llm_timeout / llm_bad_output |
| `ecc_llm_tokens_total` | counter | model, kind | provider-reported prompt/output tokens (cost = tokens × your price; not hard-coded) |
| `ecc_citation_retries_total` | counter | – | generations repaired after failed validation |
| `ecc_vector_store_errors_total`, `ecc_rate_limited_total` | counter | – | |
| `ecc_ready` | gauge | – | 1 when `/ready` would return 200 (set at start-up and on every `/ready`) |
| `ecc_index_info` | gauge (=1) | collection, embedder, corpus_hash, chunking | which index is served |
| `ecc_eval_metric` | gauge | metric, split, run_at, dataset | **offline** evaluation results baked into the image — not live quality |
| `process_*` | – | – | CPU, RSS, open FDs |

Quality is measured offline (`make eval`, `make eval-llm`); per-request "quality" numbers are never invented.

## Dashboards

`deploy/grafana/dashboards/ecc_rag_ops.json` (provisioned automatically): readiness, request rate, 5xx ratio, memory,
latency p50/p95/p99, outcomes by status, abstentions & guardrails by reason, LLM latency/errors/tokens, retrieval latency,
vector-store errors and rate limiting, and a table of the offline evaluation metrics (clearly labelled as offline).

```bash
echo "GRAFANA_ADMIN_PASSWORD=<strong password>" >> .env
docker compose --profile monitoring up -d
# Prometheus http://127.0.0.1:9090  ·  Grafana http://127.0.0.1:3000 (folder "ECC RAG")
```

## Alerts (`deploy/prometheus/alerts.yml`)

| alert | condition | severity |
|---|---|---|
| EccApiDown | target not scraped for 2 m | critical |
| EccNotReady | `ecc_ready == 0` for 5 m | critical |
| EccHigh5xxRate | > 5 % of API requests 5xx for 10 m | warning |
| EccHighLatencyP95 | `/ask` p95 > 15 s for 10 m | warning |
| EccLlmErrors | LLM error rate > 0.05/s for 5 m | critical |
| EccVectorStoreErrors | any vector-store error in 10 m | critical |
| EccAbstentionSpike | > 50 % abstentions over 30 m | warning |
| EccCitationValidationFailures | > 10 citation-validation abstentions in 1 h | warning |
| EccOfflineEvalBelowFloor | latest offline test R@5 < 0.78 | warning |

Deployment/pipeline failures are surfaced by GitHub Actions (failed workflow notifications), not Prometheus.
Routing alerts to people (Alertmanager → Slack/e-mail/PagerDuty) is not configured in this repository.

Validated: rules with `promtool`, scraping and rule evaluation with a real Prometheus binary
(`EccNotReady` and `EccHigh5xxRate` went *pending* in the expected situations) — see
`reports/PERFORMANCE_AND_MONITORING.md`. Grafana rendering was not possible in the build environment.

## Logs

JSON lines on stdout: `ts, level, logger, request_id, msg` + structured fields (route, status, duration). The
`X-Request-ID` header is accepted or generated and returned. User questions are logged according to `LOG_QUESTIONS`
(`hashed` by default; `redacted` masks long digit sequences and e-mails; `full` should only be used with consent).
Secrets that look like API keys/tokens are scrubbed from log messages.

## Tracing

Not enabled. If added (OpenTelemetry → Langfuse/Tempo), export spans for `retrieval`, `scope`, `evidence`, `llm`,
`citations` with timings and article numbers only; question/answer text should stay off by default
(same policy as `LOG_QUESTIONS`). Per-stage timings are already returned in `diagnostics.timings_ms`.
