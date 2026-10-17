"""Evaluation tooling: RAG metrics aggregation, review sheets, quality gate, Gemini error mapping."""

import json

import pytest
import yaml

from ecc_rag.evaluation import human_review, quality_gate, rag_eval
from ecc_rag.rag.llm import GeminiClient, LLMTimeoutError, LLMUnavailableError


def _row(**kw):
    base = {
        "id": "q",
        "split": "test",
        "style": "msa",
        "type": "direct",
        "expected_behavior": "answer",
        "status": "answered",
        "reason": None,
        "correct_outcome": True,
        "answer": "x [م 1]",
        "cited_articles": [1],
        "expected_articles": [1, 2],
        "retrieved_articles": [1],
        "first_pass_valid": True,
        "citation_warnings": [],
        "n_claims": 2,
        "latency_s": 1.0,
        "prompt_tokens": 10,
        "output_tokens": 5,
        "key_points": ["k"],
        "question": "q",
        "contexts": ["c"],
    }
    return {**base, **kw}


def test_rag_metrics_aggregation():
    rows = [
        _row(),
        _row(id="n1", expected_behavior="out_of_scope", status="out_of_scope", expected_articles=[], cited_articles=[]),
        _row(id="a2", status="abstained", reason="citation_validation_failed", correct_outcome=False, cited_articles=[]),
        _row(id="e", status="error", reason="llm_timeout", correct_outcome=False),
    ]
    m = rag_eval.compute_metrics(rows, [])
    # the provider error is reported separately and excluded from quality metrics
    assert m["n_errors"] == 1 and m["n_scored"] == 3 and m["error_rate"] == 0.25
    assert m["error_reasons"] == {"llm_timeout": 1}
    assert m["answered_rate"] == 0.5
    assert m["abstention_recall"] == 1.0
    assert m["expected_article_cited"] == 0.5
    assert m["citation_failure_rate"] == 0.5
    assert m["outcome_accuracy"] == pytest.approx(2 / 3, abs=1e-3)


def test_rag_metrics_empty_denominators_are_null_not_zero():
    """20 answerable-only questions must not report abstention_recall = 0.0 (it was not measured)."""
    m = rag_eval.compute_metrics([_row(), _row(id="e", status="error", reason="llm_unavailable")], [])
    assert m["abstention_recall"] is None and m["n_negative_scored"] == 0
    all_err = rag_eval.compute_metrics([_row(status="error", reason="llm_unavailable")], [])
    assert all_err["outcome_accuracy"] is None and all_err["answered_rate"] is None and all_err["latency_p50_s"] is None


class _RateLimited(Exception):
    code = 429
    status = "RESOURCE_EXHAUSTED"
    message = "quota exceeded"

    def __init__(self, delay: str):
        super().__init__(f"429 RESOURCE_EXHAUSTED retry in {delay}")
        self.details = {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}],
            }
        }


def test_provider_error_detail_names_the_exhausted_quota():
    from ecc_rag.rag.llm import provider_error_detail

    exc = _RateLimited("31s")
    exc.details["error"]["details"].append(
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaMetric": "m", "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
        }
    )
    assert "quota=GenerateRequestsPerDayPerProjectPerModel-FreeTier" in provider_error_detail(exc, limit=10)


def test_provider_retry_delay_is_parsed():
    from ecc_rag.rag.llm import provider_retry_delay_s

    assert provider_retry_delay_s(_RateLimited("17s")) == 17.0
    assert provider_retry_delay_s(_RateLimited("0.5s")) == 0.5
    assert provider_retry_delay_s(RuntimeError("503 UNAVAILABLE")) is None


def test_rate_limit_honours_retry_after_then_succeeds(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    c = _client([_RateLimited("3s")])
    c.max_retry_wait_s = 10
    assert c._call("x", None) == "ok"
    assert 3.0 in slept and c._client.models.calls == 2


def test_rate_limit_longer_than_budget_fails_fast(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    c = _client([_RateLimited("45s")] * 3)
    c.max_retry_wait_s = 8
    with pytest.raises(LLMUnavailableError, match="retry after 45s"):
        c._call("x", None)
    assert c._client.models.calls == 1 and 45.0 not in slept


def test_quality_gate_rejects_incomplete_llm_run(tmp_path, monkeypatch):
    monkeypatch.setattr(quality_gate, "PROJECT_ROOT", tmp_path)
    gate_path = _gate_files(tmp_path, {"r": {"recall@5": 0.8}, "g": {"fp": 0.1}}, {"r": {"recall@5": 0.8}})
    cfg = yaml.safe_load(gate_path.read_text())
    cfg["llm_eval"] = {"path": str(tmp_path / "llm.json"), "floors": [{"metric": "outcome_accuracy", "min": 0.8}]}
    gate_path.write_text(yaml.safe_dump(cfg))
    run = {"complete": False, "limit": 20, "metrics": {"n_errors": 8, "outcome_accuracy": 0.95}}
    (tmp_path / "llm.json").write_text(json.dumps(run))
    res = quality_gate.run_gate(gate_path)
    assert not res["passed"] and any("incomplete" in f for f in res["failures"])
    run["complete"] = True
    (tmp_path / "llm.json").write_text(json.dumps(run))
    assert quality_gate.run_gate(gate_path)["passed"]


def test_review_sheet_round_trip(tmp_path):
    rows = [_row(id=f"q{i}", type=t) for i, t in enumerate(["direct", "scenario", "direct", "injection"])]
    p = tmp_path / "sheet.csv"
    rag_eval.export_review_sheet(rows, p, n=3, seed=1)
    text = p.read_text(encoding="utf-8-sig")
    assert text.count("\n") == 4 and "correctness_0_2" in text
    lines = text.splitlines()
    filled = [lines[0]] + [ln.replace(",,,,,,,", ",2,1,2,0,NA,2,rev1,") if i == 1 else ln for i, ln in enumerate(lines[1:], start=1)]
    p.write_text("\n".join(filled), encoding="utf-8-sig")
    s = human_review.summarize(p)
    assert s["reviewed_rows"] == 1 and s["correctness_0_2"]["mean"] == 2.0 and s["abstention_quality_0_2_or_NA"]["n"] == 0


def _gate_files(tmp_path, cur_metrics, base_metrics, validation_passed=True, dataset_hash="h"):
    (tmp_path / "v.json").write_text(json.dumps({"passed": validation_passed}))
    (tmp_path / "cur.json").write_text(json.dumps({"dataset_hash": dataset_hash, "corpus_hash": "c", "metrics": cur_metrics}))
    (tmp_path / "base.json").write_text(json.dumps({"dataset_hash": "h", "corpus_hash": "c", "metrics": base_metrics}))
    cfg = {
        "data_validation_report": str(tmp_path / "v.json"),
        "current_summary": str(tmp_path / "cur.json"),
        "baseline_summary": str(tmp_path / "base.json"),
        "floors": [{"metric": "r.recall@5", "min": 0.7}],
        "ceilings": [{"metric": "g.fp", "max": 0.3}],
        "max_regression": [{"metric": "r.recall@5", "tolerance": 0.02}],
        "llm_eval": {"path": str(tmp_path / "none.json"), "floors": []},
    }
    (tmp_path / "g.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path / "g.yaml"


def test_quality_gate_pass_and_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(quality_gate, "PROJECT_ROOT", tmp_path)
    ok = _gate_files(tmp_path, {"r": {"recall@5": 0.8}, "g": {"fp": 0.1}}, {"r": {"recall@5": 0.81}})
    assert quality_gate.run_gate(ok)["passed"]
    bad = _gate_files(tmp_path, {"r": {"recall@5": 0.75}, "g": {"fp": 0.4}}, {"r": {"recall@5": 0.81}})
    res = quality_gate.run_gate(bad)
    assert not res["passed"] and len(res["failures"]) == 2  # regression + ceiling
    floor = _gate_files(tmp_path, {"r": {"recall@5": 0.6}, "g": {"fp": 0.1}}, {"r": {"recall@5": 0.6}})
    assert any("floor" in f for f in quality_gate.run_gate(floor)["failures"])
    nodata = _gate_files(tmp_path, {"r": {"recall@5": 0.8}, "g": {"fp": 0.1}}, {"r": {"recall@5": 0.8}}, validation_passed=False)
    assert any("validation" in f for f in quality_gate.run_gate(nodata)["failures"])


class _FakeModels:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def generate_content(self, **kw):
        self.calls += 1
        if not self.errors:
            return "ok"
        err = self.errors.pop(0)
        if err:
            raise err
        return "ok"


def _client(errors):
    c = GeminiClient.__new__(GeminiClient)
    c.model, c.max_retries = "m", 2
    c.max_retry_wait_s, c.min_interval_s, c._last_call = 8.0, 0.0, 0.0
    c._client = type("C", (), {"models": _FakeModels(errors)})()
    return c


def test_gemini_error_mapping(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = _client([RuntimeError("503 UNAVAILABLE")] * 3)
    with pytest.raises(LLMUnavailableError):
        c._call("x", None)
    assert c._client.models.calls == 3  # bounded retries
    c = _client([ValueError("400 INVALID_ARGUMENT")])
    with pytest.raises(LLMUnavailableError):
        c._call("x", None)
    assert c._client.models.calls == 1  # non-retryable: no retry
    c = _client([TimeoutError("timeout")] * 3)
    with pytest.raises(LLMTimeoutError):
        c._call("x", None)


def test_gemini_failure_logs_provider_detail_without_secrets(monkeypatch, caplog):
    """Operators must see *why* Gemini failed (code/status/message); the API key must never appear."""
    import logging

    class FakeAPIError(Exception):
        code = 404
        status = "NOT_FOUND"
        message = "models/gemini-x is not found for API version v1beta (key=AIzaSyA1234567890abcdefghijklmnop)"

    monkeypatch.setattr("time.sleep", lambda s: None)
    c = _client([FakeAPIError("404 NOT_FOUND")])
    with caplog.at_level(logging.ERROR, logger="ecc_rag.rag.llm"), pytest.raises(LLMUnavailableError) as ei:
        c._call("x", None)
    logged = " ".join(r.getMessage() for r in caplog.records)
    for text in (logged, str(ei.value)):
        assert "code=404" in text and "NOT_FOUND" in text and "is not found" in text
        assert "AIzaSy" not in text
