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
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


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
    ):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def _call(self, contents: Any, config: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.models.generate_content(model=self.model, contents=contents, config=config)
            except Exception as exc:
                last = exc
                msg = f"{type(exc).__name__}: {exc}"
                if "timeout" in msg.lower() or "deadline" in msg.lower():
                    if attempt >= self.max_retries:
                        raise LLMTimeoutError("Gemini call timed out") from exc
                elif not any(code in msg for code in self.RETRYABLE) or attempt >= self.max_retries:
                    raise LLMUnavailableError(f"Gemini call failed ({type(exc).__name__})") from exc
                time.sleep(min(4.0, 0.5 * 2**attempt))
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
        config = self._types.GenerateContentConfig(temperature=0.0, max_output_tokens=128)
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
        s.gemini_api_key.get_secret_value(), s.gemini_model, s.llm_timeout_s, s.llm_max_retries, s.llm_temperature, s.llm_max_output_tokens
    )
