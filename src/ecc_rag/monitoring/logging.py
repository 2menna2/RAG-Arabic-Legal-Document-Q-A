"""Structured JSON logging with request ids and privacy-aware question logging."""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
import sys
import time

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_SECRETISH = re.compile(r"(AIza[0-9A-Za-z\-_]{20,}|sk-[0-9A-Za-z]{20,}|(?i:api[_-]?key|token|password)\s*[:=]\s*\S+)")
_LONG_DIGITS = re.compile(r"[0-9٠-٩]{6,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": scrub(record.getMessage()),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update({k: v for k, v in extra.items()})
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, ensure_ascii=False, default=str)


def scrub(text: str) -> str:
    return _SECRETISH.sub("[REDACTED]", text)


def question_for_log(question: str, policy: str) -> str | None:
    """none → nothing; hashed → sha256 prefix; redacted → ids/emails masked; full → verbatim."""
    if policy == "none":
        return None
    if policy == "hashed":
        return "sha256:" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    if policy == "redacted":
        return _EMAIL.sub("[EMAIL]", _LONG_DIGITS.sub("[NUMBER]", question))[:500]
    return question[:2000]


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
