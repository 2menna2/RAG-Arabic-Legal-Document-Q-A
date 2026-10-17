"""Shared fixtures.

Unit tests use a tiny *synthetic* corpus built in a temp dir — it exercises the
code paths only and is never used to claim retrieval/answer quality.
Integration/e2e tests use the real corpus + index (``make data index``) and are
skipped with an explicit reason when those artefacts are absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from ecc_rag.config import PROJECT_ROOT, config_hash, load_rag_config
from ecc_rag.data_pipeline.schema import content_hash
from ecc_rag.rag.llm import GenerationOutput, LLMBadOutputError, LLMResult, LLMTimeoutError, LLMUnavailableError

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

SYNTHETIC_ARTICLES = [
    (
        44,
        "(١) كل شخص بلغ سن الرشد متمتعا بقواه العقلية ولم يحجر عليه يكون كامل الأهلية لمباشرة حقوقه المدنية.\n(٢) وسن الرشد هي إحدى وعشرون سنة ميلادية كاملة.",
        "Every person who has attained the age of majority is fully capable. The age of majority is twenty-one Gregorian years.",
    ),
    (
        147,
        "(١) العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين، أو للأسباب التي يقررها القانون.",
        "The contract makes the law of the parties.",
    ),
    (163, "كل خطأ سبب ضررا للغير يلزم من ارتكبه بالتعويض.", "Every fault which causes injury to another imposes liability on its author."),
    (
        176,
        "حارس الحيوان، ولو لم يكن مالكا له، مسئول عما يحدثه الحيوان من ضرر، ولو ضل الحيوان أو تسرب.",
        "The keeper of an animal is liable for damage caused by it.",
    ),
    (
        418,
        "البيع عقد يلتزم به البائع أن ينقل للمشتري ملكية شيء أو حقا ماليا آخر في مقابل ثمن نقدي.",
        "Sale is a contract whereby the vendor transfers ownership for a price.",
    ),
    (
        558,
        "الإيجار عقد يلتزم المؤجر بمقتضاه أن يمكن المستأجر من الانتفاع بشيء معين مدة معينة لقاء أجر معلوم.",
        "A lease is a contract for the use of a thing for a fixed period for a rent.",
    ),
    (
        747,
        "التأمين عقد يلتزم المؤمن بمقتضاه أن يؤدي إلى المؤمن له مبلغا من المال في حالة وقوع الحادث المؤمن ضده.",
        "Insurance is a contract.",
    ),
]


def make_article(n: int, ar: str, en: str, status: str = "unverified", evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    if status == "repealed_per_source":
        ar, en = "", ""
    return {
        "article_id": f"ecc1948-art-{n:04d}",
        "article_number": n,
        "article_text": ar,
        "text_en": en,
        "language": "ar",
        "translation_language": "en" if en else None,
        "part": None,
        "book": "الكتاب الأول",
        "chapter": "الباب الأول",
        "section": None,
        "subsection": None,
        "topic": None,
        "headings_en": {},
        "source_document": "data/raw/Law.pdf",
        "source_sha256": "0" * 64,
        "source_page": 1,
        "source_pages": [1],
        "source_version": "synthetic-test",
        "effective_date": None,
        "code_effective_date": "1949-10-15",
        "legal_status": status,
        "status_evidence": evidence,
        "citation": f"القانون المدني المصري، المادة {n}",
        "citation_en": f"Egyptian Civil Code, Article {n}",
        "quality_flags": [],
        "content_hash": content_hash(n, ar, en, status),
    }


def synthetic_articles() -> list[dict[str, Any]]:
    arts = [make_article(n, ar, en) for n, ar, en in SYNTHETIC_ARTICLES]
    arts.append(
        make_article(
            60,
            "",
            "",
            "repealed_per_source",
            {
                "source_page": 7,
                "note_en": "* Articles 54-80 have been repealed by Presidential Decree.",
                "note_ar": "ألغيت المواد من ٥٤ إلى ٨٠",
                "range": [54, 80],
            },
        )
    )
    return sorted(arts, key=lambda a: a["article_number"])


@pytest.fixture
def rag_cfg() -> dict[str, Any]:
    cfg = load_rag_config(PROJECT_ROOT / "configs" / "rag.yaml")
    cfg["embedding"]["lsa"]["n_components"] = 8
    cfg["embedding"]["lsa"]["min_df"] = 1
    return cfg


@pytest.fixture
def synthetic_index(tmp_path: Path, rag_cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    """Build articles → chunks → LSA/Qdrant(embedded) index for the synthetic corpus in tmp_path."""
    from ecc_rag.config import get_settings
    from ecc_rag.data_pipeline.chunking import build_chunks
    from ecc_rag.rag.index import build_index

    data = tmp_path / "data"
    (data / "processed").mkdir(parents=True)
    arts = synthetic_articles()
    (data / "processed" / "articles.jsonl").write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in arts), encoding="utf-8")
    chunks = build_chunks(arts, rag_cfg["chunking"]["strategy"], rag_cfg["chunking"]["params"])
    (data / "processed" / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    monkeypatch.setenv("QDRANT_PATH", str(data / "index" / "qdrant"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    get_settings.cache_clear()
    manifest = build_index(data / "processed" / "chunks.jsonl", data / "processed" / "articles.jsonl", data / "index", rag_cfg)
    yield {"data": data, "index_dir": data / "index", "cfg": rag_cfg, "manifest": manifest, "articles": arts}
    get_settings.cache_clear()


class ScriptedLLM:
    """Test double: returns pre-programmed outputs or raises pre-programmed errors, in order."""

    model = "scripted-test-llm"

    def __init__(self, outputs: list[Any]):
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def generate(self, system: str, user: str, history=None) -> LLMResult:
        self.calls.append({"system": system, "user": user, "history": history})
        nxt = self.outputs.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        out = nxt if isinstance(nxt, GenerationOutput) else GenerationOutput.model_validate(nxt)
        return LLMResult(output=out, model=self.model, latency_s=0.01, prompt_tokens=100, output_tokens=20, raw=out.model_dump_json())

    def complete_text(self, prompt: str) -> str:
        return ""

    def ping(self) -> bool:
        return True


class QuotingTestLLM:
    """Deterministic test double for end-to-end plumbing tests on the real corpus:
    answers by quoting the first sentence of the first retrieved document.
    Exercises prompt → JSON → citation validation → API without a network LLM.
    It is NOT a quality signal and never used outside tests."""

    model = "quoting-test-llm"

    def generate(self, system: str, user: str, history=None) -> LLMResult:
        import re

        m = re.search(r'<document article="(\d+)"[^>]*>\s*(.+?)\s*</document>', user, re.S)
        if not m:
            out = GenerationOutput(status="insufficient_evidence", missing_information="no documents")
        else:
            n, text = int(m.group(1)), m.group(2)
            sentence = re.split(r"[.\n]", re.sub(r"^\([0-9٠-٩]+\)\s*", "", text.strip()))[0].strip()
            words = sentence.split()[:20]
            quote = " ".join(words)
            out = GenerationOutput(
                status="answered",
                answer=f"وفقاً للمادة {n}: {quote} [م {n}]",
                claims=[{"statement": quote, "articles": [n], "quotes": [quote], "kind": "stated"}],
            )
        return LLMResult(output=out, model=self.model, latency_s=0.0, raw=out.model_dump_json())

    def complete_text(self, prompt: str) -> str:
        return ""

    def ping(self) -> bool:
        return True


LLM_ERRORS = {"unavailable": LLMUnavailableError("down"), "timeout": LLMTimeoutError("slow"), "bad": LLMBadOutputError("junk")}

REAL_DATA = PROJECT_ROOT / "data" / "processed" / "articles.jsonl"
REAL_INDEX = PROJECT_ROOT / "data" / "index" / "manifest.json"


def require_real_artifacts() -> None:
    if not (REAL_DATA.exists() and REAL_INDEX.exists()):
        pytest.skip("real corpus/index not present — run `dvc pull` or `make data index` (integration tests need the real Corpus)")


def cfg_hash(cfg: dict[str, Any]) -> str:
    return config_hash(cfg)
