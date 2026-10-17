"""Pydantic schema for one article record (data contract of ``articles.jsonl``)."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LegalStatus = Literal["repealed_per_source", "unverified"]


class StatusEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_page: int = Field(ge=1)
    note_en: str | None
    note_ar: str | None
    range: list[int]


class ArticleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_id: str = Field(pattern=r"^ecc1948-art-\d{4}$")
    article_number: int = Field(ge=1, le=1149)
    article_text: str
    text_en: str
    language: Literal["ar"]
    translation_language: Literal["en"] | None
    part: str | None
    book: str | None
    chapter: str | None
    section: str | None
    subsection: str | None
    topic: str | None
    headings_en: dict[str, str]
    source_document: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_page: int = Field(ge=1)
    source_pages: list[int]
    source_version: str
    effective_date: str | None
    code_effective_date: str | None
    legal_status: LegalStatus
    status_evidence: StatusEvidence | None
    citation: str
    citation_en: str
    quality_flags: list[str]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _consistency(self) -> ArticleRecord:
        if self.article_id != f"ecc1948-art-{self.article_number:04d}":
            raise ValueError("article_id does not match article_number")
        if self.legal_status == "repealed_per_source":
            if self.status_evidence is None:
                raise ValueError("repealed article without source evidence")
            if self.article_text or self.text_en:
                raise ValueError("repealed record must not carry invented article text")
        if self.source_page != self.source_pages[0]:
            raise ValueError("source_page must equal first of source_pages")
        expected = content_hash(self.article_number, self.article_text, self.text_en, self.legal_status)
        if expected != self.content_hash:
            raise ValueError("content_hash mismatch")
        return self


def content_hash(number: int, text_ar: str, text_en: str, status: str) -> str:
    return hashlib.sha256("␞".join([str(number), text_ar, text_en, status]).encode("utf-8")).hexdigest()
