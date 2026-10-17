"""Runtime configuration.

Two layers:

* ``configs/rag.yaml`` — versioned, reviewable RAG parameters (chunking,
  retrieval, thresholds, model names).  Changing it changes results, so it is
  logged to MLflow and hashed into the index manifest.
* Environment variables (``Settings``) — deployment concerns and secrets
  (API keys, URLs, timeouts, rate limits).  Secrets are never read from files
  committed to Git.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root in development; set ECC_HOME (e.g. /app) when the package is installed non-editable.
PROJECT_ROOT = Path(os.environ.get("ECC_HOME") or Path(__file__).resolve().parents[2])


class Settings(BaseSettings):
    # env_ignore_empty: `QDRANT_URL=` (as in .env.example / docker --env-file) means "unset", not "empty string".
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore", env_ignore_empty=True)

    # --- application
    app_env: Literal["dev", "test", "prod"] = Field("dev", alias="APP_ENV")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    rag_config_path: Path = Field(PROJECT_ROOT / "configs" / "rag.yaml", alias="RAG_CONFIG_PATH")
    data_dir: Path = Field(PROJECT_ROOT / "data", alias="DATA_DIR")
    index_dir: Path = Field(PROJECT_ROOT / "data" / "index", alias="INDEX_DIR")

    # --- LLM (Gemini)
    llm_provider: Literal["gemini", "none"] = Field("gemini", alias="LLM_PROVIDER")
    gemini_api_key: SecretStr | None = Field(None, alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-3.8-flash", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field("gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL")
    llm_timeout_s: float = Field(30.0, alias="LLM_TIMEOUT_S")
    llm_max_retries: int = Field(2, alias="LLM_MAX_RETRIES", ge=0, le=5)
    # longest wait honoured for a provider "retry after" hint (429); longer hints fail fast instead of blocking requests
    llm_max_retry_wait_s: float = Field(8.0, alias="LLM_MAX_RETRY_WAIT_S", ge=0)
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    llm_max_output_tokens: int = Field(2048, alias="LLM_MAX_OUTPUT_TOKENS")

    # --- vector store (Qdrant): server URL in compose/prod, embedded path in dev/CI
    qdrant_url: str | None = Field(None, alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(None, alias="QDRANT_API_KEY")
    qdrant_path: Path | None = Field(None, alias="QDRANT_PATH")
    qdrant_timeout_s: float = Field(5.0, alias="QDRANT_TIMEOUT_S")

    # --- API
    cors_origins: list[str] = Field(default_factory=list, alias="CORS_ORIGINS")
    rate_limit_per_minute: int = Field(30, alias="RATE_LIMIT_PER_MINUTE", ge=0)
    max_question_chars: int = Field(1000, alias="MAX_QUESTION_CHARS")
    request_timeout_s: float = Field(60.0, alias="REQUEST_TIMEOUT_S")
    admin_token: SecretStr | None = Field(None, alias="ADMIN_TOKEN")
    log_questions: Literal["none", "hashed", "redacted", "full"] = Field("hashed", alias="LOG_QUESTIONS")

    # --- tracking
    mlflow_tracking_uri: str = Field(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = Field("ecc-rag", alias="MLFLOW_EXPERIMENT_NAME")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_rag_config(path: Path | None = None) -> dict[str, Any]:
    path = path or get_settings().rag_config_path
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def config_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
