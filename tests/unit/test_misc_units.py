"""Rate limiter, logging privacy, golden-set integrity."""

import time
from pathlib import Path

import pytest

from ecc_rag.api.ratelimit import TokenBucketLimiter
from ecc_rag.evaluation.dataset import load_golden, split_of
from ecc_rag.monitoring.logging import question_for_log, scrub


def test_rate_limiter_blocks_after_burst_and_refills():
    lim = TokenBucketLimiter(per_minute=120, burst=2)
    assert lim.allow("a") and lim.allow("a") and not lim.allow("a")
    assert lim.allow("b")
    time.sleep(0.6)
    assert lim.allow("a")


def test_question_logging_policies():
    q = "رقمي القومي 29801011234567 وبريدي a@b.com"
    assert question_for_log(q, "none") is None
    assert question_for_log(q, "hashed").startswith("sha256:")
    red = question_for_log(q, "redacted")
    assert "29801011234567" not in red and "a@b.com" not in red


def test_secrets_are_scrubbed_from_logs():
    assert "AIza" not in scrub("key=AIzaSyA1234567890abcdefghijklmnop")
    assert "hunter2" not in scrub("password=hunter2")


def test_golden_dataset_integrity():
    version, items = load_golden()
    assert version == "golden_v1" and len(items) >= 200
    assert len({i.id for i in items}) == len(items)
    assert {i.split for i in items} == {"dev", "test"}
    assert all(split_of(i.id) == i.split for i in items)  # deterministic
    for i in items:
        if i.expected_behavior == "answer":
            assert i.expected_articles, i.id
        assert i.review_status == "unreviewed"  # never claim legal review that did not happen
    styles = {i.style for i in items}
    assert {"msa", "egyptian", "english"} <= styles


def test_empty_env_values_mean_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`.env.example` / `docker run --env-file` produce `QDRANT_URL=` — that must not become an empty URL/path."""
    from ecc_rag.config import Settings

    monkeypatch.chdir(tmp_path)  # no .env in cwd
    for var in ("QDRANT_URL", "QDRANT_PATH", "QDRANT_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "")
    s = Settings()
    assert s.qdrant_url is None
    assert s.qdrant_path is None
    assert s.qdrant_api_key is None
    assert s.gemini_api_key is None


def test_env_example_has_no_inline_comments() -> None:
    """docker --env-file keeps everything after '=' (including '# ...') as the value."""
    from ecc_rag.config import PROJECT_ROOT

    for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            assert "#" not in line, f"inline comment in .env.example: {line!r}"
