"""LLM client abstraction and the Gemini implementation.

Failures are *typed* and never converted into a fake answer:

* ``LLMUnavailableError`` — missing key, auth/network/5xx errors after retries
* ``LLMTimeoutError``     — the call exceeded the configured timeout
* ``LLMBadOutputError``   — the model returned something that is not valid JSON
  for the response schema
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


def provider_retry_delay_s(exc: BaseException) -> float | None:
    """Seconds the provider asked us to wait (google.rpc.RetryInfo.retryDelay, e.g. "17s"), if present."""
    details = getattr(exc, "details", None)
    err = details.get("error", details) if isinstance(details, dict) else None
    items = err.get("details", []) if isinstance(err, dict) else []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)s\s*", str(item.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    return None


def provider_error_detail(exc: BaseException, limit: int = 300) -> str:
    """Operator-facing summary of a provider error (HTTP code, status, message) with secrets scrubbed.

    Used for logs and the exception message; never returned to API clients.
    """
    from ecc_rag.monitoring.logging import scrub

    parts = [type(exc).__name__]
    for attr in ("code", "status"):
        val = getattr(exc, attr, None)
        if val not in (None, ""):
            parts.append(f"{attr}={val}")
    # google.rpc.QuotaFailure says WHICH quota was hit (per-minute vs per-day): keep it even if the text is cut
    details = getattr(exc, "details", None)
    err = details.get("error", details) if isinstance(details, dict) else None
    items = err.get("details", []) if isinstance(err, dict) else []
    quotas = [
        str(v.get("quotaId") or v.get("quotaMetric"))
        for item in (items if isinstance(items, list) else [])
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("QuotaFailure")
        for v in item.get("violations", [])
        if isinstance(v, dict) and (v.get("quotaId") or v.get("quotaMetric"))
    ]
    if quotas:
        parts.append(f"quota={','.join(sorted(set(quotas)))}")
    message = getattr(exc, "message", None) or str(exc)
    parts.append(" ".join(str(message).split())[:limit])
    return scrub(" ".join(parts))


class LLMError(RuntimeError):
    code = "llm_error"


class LLMUnavailableError(LLMError):
    code = "llm_unavailable"


class LLMTimeoutError(LLMError):
    code = "llm_timeout"


class LLMBadOutputError(LLMError):
    code = "llm_bad_output"


class Claim(BaseModel):
    statement: str
    articles: list[int] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)
    kind: Literal["stated", "inferred"] = "stated"


class GenerationOutput(BaseModel):
    status: Literal["answered", "insufficient_evidence", "needs_clarification", "out_of_scope"]
    answer: str = ""
    claims: list[Claim] = Field(default_factory=list)
    clarification_question: str | None = None
    missing_information: str | None = None


@dataclass
class LLMResult:
    output: GenerationOutput
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    raw: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    model: str

    def generate(self, system: str, user: str, history: list[dict[str, str]] | None = None) -> LLMResult: ...
    def complete_text(self, prompt: str) -> str: ...
    def ping(self) -> bool: ...


def parse_output(raw: str) -> GenerationOutput:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    try:
        return GenerationOutput.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMBadOutputError(f"model output is not valid JSON for the schema: {type(exc).__name__}") from exc


class GeminiClient:
    """google-genai SDK client with bounded retries and a hard timeout."""

    RETRYABLE = ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE")

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        max_retry_wait_s: float = 8.0,
        min_interval_s: float = 0.0,
    ):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retry_wait_s = max_retry_wait_s
        self.min_interval_s = min_interval_s  # client-side pacing (batch evaluation on rate-limited tiers)
        self._last_call = 0.0
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def _pace(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _call(self, contents: Any, config: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._pace()
                return self._client.models.generate_content(model=self.model, contents=contents, config=config)
            except Exception as exc:
                last = exc
                msg = f"{type(exc).__name__}: {exc}"
                detail = provider_error_detail(exc)
                if "timeout" in msg.lower() or "deadline" in msg.lower():
                    if attempt >= self.max_retries:
                        raise LLMTimeoutError("Gemini call timed out") from exc
                elif not any(code in msg for code in self.RETRYABLE) or attempt >= self.max_retries:
                    logger.error("Gemini call failed: %s", detail, extra={"model": self.model, "attempt": attempt + 1})
                    raise LLMUnavailableError(f"Gemini call failed ({detail})") from exc
                hinted = provider_retry_delay_s(exc)
                if hinted is not None and hinted > self.max_retry_wait_s:
                    # e.g. free-tier quota: waiting would exceed the request budget — fail clearly instead
                    logger.error(
                        "Gemini rate-limited; provider asks to wait %.0fs (> LLM_MAX_RETRY_WAIT_S=%.0fs): %s",
                        hinted,
                        self.max_retry_wait_s,
                        detail,
                    )
                    raise LLMUnavailableError(f"Gemini rate-limited, retry after {hinted:.0f}s ({detail})") from exc
                delay = hinted if hinted is not None else min(4.0, 0.5 * 2**attempt)
                logger.warning("Gemini call failed, retrying in %.1fs: %s", delay, detail, extra={"attempt": attempt + 1})
                time.sleep(delay)
        raise LLMUnavailableError("Gemini call failed") from last

    def generate(self, system: str, user: str, history: list[dict[str, str]] | None = None) -> LLMResult:
        t = self._types
        contents = []
        for turn in history or []:
            contents.append(t.Content(role=turn["role"], parts=[t.Part(text=turn["text"])]))
        contents.append(t.Content(role="user", parts=[t.Part(text=user)]))
        config = t.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_schema=GenerationOutput,
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        t0 = time.perf_counter()
        resp = self._call(contents, config)
        latency = time.perf_counter() - t0
        raw = getattr(resp, "text", None) or ""
        if not raw:
            raise LLMBadOutputError("empty response (possibly blocked by safety filters)")
        usage = getattr(resp, "usage_metadata", None)
        return LLMResult(
            output=parse_output(raw),
            model=self.model,
            latency_s=latency,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            raw=raw,
        )

    def complete_text(self, prompt: str) -> str:
        config = self._types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=128,
            automatic_function_calling=self._types.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self._call(prompt, config)
        return (getattr(resp, "text", "") or "").strip()

    def ping(self) -> bool:
        try:
            self._client.models.get(model=self.model)
            return True
        except Exception:  # noqa: BLE001
            return False


def make_llm_client() -> LLMClient | None:
    from ecc_rag.config import get_settings

    s = get_settings()
    if s.llm_provider == "none":
        return None
    if s.gemini_api_key is None:
        raise LLMUnavailableError("GEMINI_API_KEY is not configured")
    return GeminiClient(
        s.gemini_api_key.get_secret_value(),
        s.gemini_model,
        s.llm_timeout_s,
        s.llm_max_retries,
        s.llm_temperature,
        s.llm_max_output_tokens,
        max_retry_wait_s=s.llm_max_retry_wait_s,
    )
