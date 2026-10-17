"""Request / response contracts of the public API (OpenAPI is generated from these)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Status = Literal["answered", "abstained", "out_of_scope", "needs_clarification", "rejected", "error"]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"question": "ما هو سن الرشد في القانون المدني المصري؟"}]})
    question: str = Field(..., min_length=1, max_length=1000, description="السؤال بالعربية (فصحى أو عامية مصرية) أو بالإنجليزية")
    top_k: int | None = Field(None, ge=1, le=10, description="عدد المواد المسترجعة (افتراضي من الإعدادات)")
    include_diagnostics: bool = Field(False, description="إرجاع معلومات تشخيصية غير حساسة")

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)


class SourceOut(BaseModel):
    article_number: int
    citation: str
    source_page: int | None
    legal_status: Literal["repealed_per_source", "unverified"]
    text: str = Field(description="نص المادة كما ورد في المصدر (مقتطع عند الطول الزائد)")
    rank: int
    status_note: str | None = None


class CitationOut(BaseModel):
    statement: str
    articles: list[int]
    quotes: list[str] = Field(description="اقتباسات تم التحقق آلياً من وجودها حرفياً في نص المادة")
    kind: Literal["stated", "inferred"]


class AskResponse(BaseModel):
    request_id: str
    status: Status
    answer: str | None = None
    reason: str | None = Field(None, description="سبب الامتناع/الرفض/الخطأ بصيغة قابلة للمعالجة آلياً")
    message: str | None = Field(None, description="رسالة موجهة للمستخدم")
    citations: list[CitationOut] = []
    sources: list[SourceOut] = []
    clarification_question: str | None = None
    notices: list[str] = []
    latency_ms: float
    diagnostics: dict[str, Any] | None = None


class SearchHit(BaseModel):
    article_number: int
    citation: str
    source_page: int | None
    legal_status: str
    rank: int
    text: str
    scores: dict[str, float]


class SearchResponse(BaseModel):
    request_id: str
    hits: list[SearchHit]
    latency_ms: float


class ArticleOut(BaseModel):
    article_number: int
    article_id: str
    article_text: str
    text_en: str
    legal_status: str
    status_evidence: dict[str, Any] | None
    book: str | None
    chapter: str | None
    section: str | None
    source_document: str
    source_page: int
    source_pages: list[int]
    source_version: str
    quality_flags: list[str]
    citation: str


class ErrorResponse(BaseModel):
    request_id: str
    status: Literal["error", "rejected"] = "error"
    reason: str
    message: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, Any]
