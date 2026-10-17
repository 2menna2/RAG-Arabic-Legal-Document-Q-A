"""Golden evaluation dataset: schema, loading and deterministic dev/test split."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ecc_rag.config import PROJECT_ROOT

Behavior = Literal["answer", "abstain", "out_of_scope", "clarify", "reject"]
DEFAULT_PATH = PROJECT_ROOT / "data" / "evaluation" / "golden_v1.yaml"
DEV_FRACTION = 40  # percent


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    style: Literal["msa", "egyptian", "english"]
    type: str
    difficulty: Literal["easy", "medium", "hard"]
    expected_behavior: Behavior
    acceptable_behaviors: list[Behavior] = Field(default_factory=list)  # boundary cases: also-correct outcomes
    expected_articles: list[int] = Field(default_factory=list)
    acceptable_articles: list[int] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    expected_failure_reason: str | None = None
    origin: str = "authored-v1"
    review_status: str = "unreviewed"
    split: Literal["dev", "test"] | None = None

    @property
    def is_retrieval_item(self) -> bool:
        return self.expected_behavior == "answer" and bool(self.expected_articles)


def split_of(item_id: str) -> Literal["dev", "test"]:
    """Deterministic split: stable across runs and machines, independent of order."""
    h = int(hashlib.sha256(f"golden_v1:{item_id}".encode()).hexdigest(), 16) % 100
    return "dev" if h < DEV_FRACTION else "test"


def load_golden(path: Path | None = None) -> tuple[str, list[GoldenItem]]:
    data = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    items = []
    for raw in data["items"]:
        it = GoldenItem.model_validate(raw)
        it.split = it.split or split_of(it.id)
        items.append(it)
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in golden dataset")
    return data["version"], items


def dataset_hash(path: Path | None = None) -> str:
    return hashlib.sha256(Path(path or DEFAULT_PATH).read_bytes()).hexdigest()[:16]
