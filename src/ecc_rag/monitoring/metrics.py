"""Prometheus metrics. Only values that are actually measured are exported.

RAG *quality* is not measurable per request; ``ecc_eval_metric`` exposes the
numbers of the latest offline evaluation run (with its timestamp and dataset
as labels) so dashboards can show them, clearly separated from live signals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, PlatformCollector, ProcessCollector

REGISTRY = CollectorRegistry(auto_describe=True)
ProcessCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)

LAT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60)

HTTP_REQUESTS = Counter("ecc_http_requests_total", "HTTP requests", ["method", "route", "status_code"], registry=REGISTRY)
HTTP_LATENCY = Histogram("ecc_http_request_duration_seconds", "HTTP request latency", ["route"], buckets=LAT_BUCKETS, registry=REGISTRY)
RAG_OUTCOMES = Counter("ecc_rag_outcomes_total", "RAG outcomes by status and reason", ["status", "reason"], registry=REGISTRY)
RETRIEVAL_LATENCY = Histogram("ecc_retrieval_duration_seconds", "Retrieval latency", buckets=LAT_BUCKETS, registry=REGISTRY)
LLM_LATENCY = Histogram("ecc_llm_request_duration_seconds", "LLM call latency", ["model"], buckets=LAT_BUCKETS, registry=REGISTRY)
LLM_ERRORS = Counter("ecc_llm_errors_total", "LLM failures by type", ["code"], registry=REGISTRY)
LLM_TOKENS = Counter("ecc_llm_tokens_total", "Tokens reported by the provider", ["model", "kind"], registry=REGISTRY)
GUARDRAIL_EVENTS = Counter("ecc_guardrail_events_total", "Guardrail decisions", ["kind"], registry=REGISTRY)
ABSTENTIONS = Counter("ecc_abstentions_total", "Abstentions by reason", ["reason"], registry=REGISTRY)
CITATION_RETRIES = Counter("ecc_citation_retries_total", "Generations retried after failed citation validation", registry=REGISTRY)
VECTOR_STORE_ERRORS = Counter("ecc_vector_store_errors_total", "Vector store errors", registry=REGISTRY)
RATE_LIMITED = Counter("ecc_rate_limited_total", "Requests rejected by the rate limiter", registry=REGISTRY)
INDEX_INFO = Gauge(
    "ecc_index_info", "Loaded index identity (value is always 1)", ["collection", "embedder", "corpus_hash", "chunking"], registry=REGISTRY
)
READY = Gauge("ecc_ready", "1 when /ready would return 200", registry=REGISTRY)
EVAL_METRIC = Gauge(
    "ecc_eval_metric", "Latest OFFLINE evaluation metric (not live traffic)", ["metric", "split", "run_at", "dataset"], registry=REGISTRY
)


def observer(event: str, data: dict[str, Any]) -> None:
    """Pipeline observer → metrics (keeps the RAG code free of Prometheus)."""
    if event == "result":
        RAG_OUTCOMES.labels(data.get("status", "unknown"), str(data.get("reason") or "none")).inc()
    elif event == "retrieval":
        RETRIEVAL_LATENCY.observe(float(data["seconds"]))
    elif event == "llm":
        model = data.get("model", "unknown")
        LLM_LATENCY.labels(model).observe(float(data["seconds"]))
        if data.get("prompt_tokens") is not None:
            LLM_TOKENS.labels(model, "prompt").inc(int(data["prompt_tokens"]))
        if data.get("output_tokens") is not None:
            LLM_TOKENS.labels(model, "output").inc(int(data["output_tokens"]))
    elif event == "llm_error":
        LLM_ERRORS.labels(data.get("code", "unknown")).inc()
    elif event == "guardrail":
        GUARDRAIL_EVENTS.labels(str(data.get("kind"))).inc()
    elif event == "abstain":
        ABSTENTIONS.labels(str(data.get("reason"))).inc()
    elif event == "citation_retry":
        CITATION_RETRIES.inc()


def load_eval_metrics(path: Path) -> int:
    """Expose the latest offline evaluation summary (if present). Returns #series set."""
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for split, metrics in data.get("metrics", {}).items():
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                EVAL_METRIC.labels(name, split, data.get("run_at", "unknown"), data.get("dataset", "unknown")).set(float(value))
                n += 1
    return n
