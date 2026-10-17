"""Stage 2 — turn extracted table rows into one structured record per article.

Rules (all deterministic, all logged):

* A row whose English cell starts with ``Article N`` (or whose Arabic cell starts
  with ``مادة``) opens article *N*.  The English header is authoritative for the
  number because it is plain Latin digits; the Arabic header number is parsed
  independently and any disagreement is recorded as a quality flag.
* A bold row that is not an article is a structural heading (part / book /
  chapter / section / sub-heading) and updates the hierarchy metadata.
* Any other row is a continuation of the open article (tables split across
  pages, highlighted paragraphs rendered as separate rows, …).
* Repeal notes that appear in the source ("Articles 54-80 have been repealed
  …", "Articles 389-417 repealed") produce ``legal_status = repealed_per_source``
  records **carrying the verbatim note as evidence** and an empty article text.
  We never invent replacement wording, and articles without such a note get
  ``legal_status = unverified`` (the source does not certify current validity).

Nothing is patched by hand.  Known defects of the *source* document are listed
in ``configs/known_source_issues.yaml`` and surfaced as flags by validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ecc_rag.data_pipeline.arabic import clean_display_text, clean_english_text, to_western_digits
from ecc_rag.data_pipeline.schema import content_hash

logger = logging.getLogger(__name__)

# "rticle 452": the capital A of one header is not extractable from the source PDF
EN_ARTICLE_RE = re.compile(r"^\s*A?rticle\s*(\d{1,4})\b", re.IGNORECASE)
EN_RANGE_REPEAL_RE = re.compile(r"Articles\s+(\d{1,4})\s*-\s*(\d{1,4})\s+(?:have\s+been\s+)?repealed", re.IGNORECASE)
AR_ARTICLE_RE = re.compile(r"^[\s()]*م\s*ا\s*د\s*[ةه][\s()]*([0-9٠-٩][0-9٠-٩ ]{0,5})[\s()]*")
AR_RANGE_RE = re.compile(r"^[\s()]*المواد\s+من")

HEADING_LEVELS = ["part", "book", "chapter", "section", "subsection", "topic"]
AR_LEVEL_PREFIX = {
    "القسم": "part",
    "الكتاب": "book",
    "الباب": "chapter",
    "باب": "chapter",
    "الفصل": "section",
}
EN_LEVEL_PREFIX = {"part": "part", "book": "book", "chapter": "chapter", "section": "section"}


@dataclass
class ArticleBuilder:
    number: int
    pages: list[int]
    en_parts: list[str] = field(default_factory=list)
    ar_parts: list[str] = field(default_factory=list)
    hierarchy_ar: dict[str, str | None] = field(default_factory=dict)
    hierarchy_en: dict[str, str | None] = field(default_factory=dict)
    ar_header_number: int | None = None
    flags: list[str] = field(default_factory=list)
    status_note_en: str | None = None
    status_note_ar: str | None = None
    repeal_range: tuple[int, int] | None = None


def _strip_en_header(text: str, number: int) -> str:
    return re.sub(rf"^\s*A?rticle\s*{number}\b\s*", "", text, count=1, flags=re.IGNORECASE)


def _parse_ar_header(text: str) -> tuple[int | None, str]:
    m = AR_ARTICLE_RE.match(text)
    if not m:
        return None, text
    digits = to_western_digits(m.group(1)).replace(" ", "")
    rest = text[m.end() :]
    try:
        return int(digits), rest
    except ValueError:
        return None, rest


def _heading_level(en: str, ar: str) -> str:
    ar_first = ar.strip().split()[0] if ar.strip() else ""
    for prefix, level in AR_LEVEL_PREFIX.items():
        if ar_first.startswith(prefix):
            return level
    en_low = en.strip().lower()
    for key, level in EN_LEVEL_PREFIX.items():
        if re.match(rf"^(?:first|second|third)?\s*{key}\b", en_low):
            return level
    if re.match(r"^[\s:]*([0-9٠-٩]+|أولا|ثانيا|ثالثا|رابعا|خامسا)", to_western_digits(ar)) or re.match(r"^\s*\d+[.\-]", en):
        return "subsection"
    return "topic"


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_heading(text: str) -> str | None:
    # headings in the source carry a stray trailing "أ" glyph (e.g. "العقود المسماة أ")
    text = re.sub(r"\s+[أإ]$", "", _one_line(text))
    return text or None


def parse_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[ArticleBuilder], list[dict[str, Any]]]:
    hierarchy_ar: dict[str, str | None] = {k: None for k in HEADING_LEVELS}
    hierarchy_en: dict[str, str | None] = {k: None for k in HEADING_LEVELS}
    articles: list[ArticleBuilder] = []
    events: list[dict[str, Any]] = []
    current: ArticleBuilder | None = None

    for row in rows:
        en, ar, page = row["en_text"].strip(), row["ar_text"].strip(), row["page"]
        is_bold = max(row["en_bold_ratio"], row["ar_bold_ratio"]) >= 0.5
        en_m = EN_ARTICLE_RE.match(en)
        range_m = EN_RANGE_REPEAL_RE.search(en)
        ar_num, ar_rest = _parse_ar_header(ar)

        if en_m or ar_num is not None or (range_m and not is_bold) or AR_RANGE_RE.match(ar):
            if en_m:
                number = int(en_m.group(1))
            elif range_m:
                number = int(range_m.group(1))
            elif ar_num is not None:
                number = ar_num
                events.append({"type": "article_from_ar_header_only", "page": page, "number": number})
            else:
                events.append({"type": "unparseable_article_row", "page": page, "en": en[:80], "ar": ar[:80]})
                continue
            current = ArticleBuilder(
                number=number,
                pages=[page],
                hierarchy_ar=dict(hierarchy_ar),
                hierarchy_en=dict(hierarchy_en),
                ar_header_number=ar_num,
            )
            if range_m:
                current.repeal_range = (int(range_m.group(1)), int(range_m.group(2)))
                current.status_note_en = _one_line(_strip_en_header(en, number))
                current.status_note_ar = _one_line(ar)
            else:
                body_en = _strip_en_header(en, number) if en_m else en
                if body_en:
                    current.en_parts.append(body_en)
                if ar:
                    current.ar_parts.append(ar_rest if ar_num is not None else ar)
                if not en_m:
                    current.flags.append("en_header_missing")
                if ar and ar_num is None:
                    current.flags.append("ar_header_unparsed")
                if not ar:
                    current.flags.append("ar_cell_empty_in_header_row")
                if ar_num is not None and en_m and ar_num != number:
                    current.flags.append(f"ar_header_number_mismatch:{ar_num}")
            articles.append(current)
            continue

        if is_bold:
            level = _heading_level(en, ar)
            idx = HEADING_LEVELS.index(level)
            for lvl in HEADING_LEVELS[idx:]:
                hierarchy_ar[lvl] = None
                hierarchy_en[lvl] = None
            hierarchy_ar[level] = _clean_heading(ar)
            hierarchy_en[level] = _clean_heading(en)
            events.append({"type": "heading", "page": page, "level": level, "ar": hierarchy_ar[level], "en": hierarchy_en[level]})
            continue

        # continuation row
        if current is None:
            events.append({"type": "orphan_row_before_first_article", "page": page, "en": en[:80], "ar": ar[:80]})
            continue
        if page not in current.pages:
            current.pages.append(page)
        if en:
            current.en_parts.append(en)
        if ar:
            current.ar_parts.append(ar)
        if not (en and ar):
            current.flags.append(f"one_sided_continuation_row:p{page}")
    return articles, events


def build_records(builders: list[ArticleBuilder], source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen: dict[int, int] = {}
    for b in builders:
        numbers = [b.number]
        if b.repeal_range:
            lo, hi = b.repeal_range
            numbers = list(range(lo, hi + 1))
        for n in numbers:
            if n in seen:
                issues.append({"type": "duplicate_article_number", "article_number": n, "pages": b.pages})
                continue
            seen[n] = len(records)
            if b.repeal_range:
                text_ar, text_en = "", ""
                status = "repealed_per_source"
                evidence = {
                    "source_page": b.pages[0],
                    "note_en": b.status_note_en,
                    "note_ar": b.status_note_ar,
                    "range": list(b.repeal_range),
                }
                flags: list[str] = []
            else:
                text_ar = clean_display_text("\n".join(b.ar_parts))
                text_en = clean_english_text("\n".join(b.en_parts))
                status, evidence = "unverified", None
                flags = sorted(set(b.flags))
                if not text_ar:
                    flags.append("ar_text_missing")
                if not text_en:
                    flags.append("en_text_missing")
            h_ar, h_en = b.hierarchy_ar, b.hierarchy_en
            rec = {
                "article_id": f"ecc1948-art-{n:04d}",
                "article_number": n,
                "article_text": text_ar,
                "text_en": text_en,
                "language": "ar",
                "translation_language": "en" if text_en else None,
                "part": h_ar.get("part") or h_en.get("part"),
                "book": h_ar.get("book") or h_en.get("book"),
                "chapter": h_ar.get("chapter") or h_en.get("chapter"),
                "section": h_ar.get("section") or h_en.get("section"),
                "subsection": h_ar.get("subsection") or h_en.get("subsection"),
                "topic": h_ar.get("topic") or h_en.get("topic"),
                "headings_en": {k: v for k, v in h_en.items() if v},
                "source_document": source["document"],
                "source_sha256": source["sha256"],
                "source_page": b.pages[0],
                "source_pages": b.pages,
                "source_version": source["version"],
                "effective_date": None,
                "code_effective_date": source.get("code_effective_date"),
                "legal_status": status,
                "status_evidence": evidence,
                "citation": f"القانون المدني المصري، المادة {n}",
                "citation_en": f"Egyptian Civil Code, Article {n}",
                "quality_flags": flags,
                "content_hash": content_hash(n, text_ar, text_en, status),
            }
            records.append(rec)
    records.sort(key=lambda r: r["article_number"])
    return records, issues


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse extracted rows into article records")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=Path("configs/source.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    source = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))["source"]
    builders, events = parse_rows(load_rows(args.rows))
    records, issues = build_records(builders, source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    flag_counts: dict[str, int] = {}
    for r in records:
        for fl in r["quality_flags"]:
            key = fl.split(":")[0]
            flag_counts[key] = flag_counts.get(key, 0) + 1
    report = {
        "articles": len(records),
        "headings": sum(1 for e in events if e["type"] == "heading"),
        "parse_events": [e for e in events if e["type"] != "heading"],
        "issues": issues,
        "quality_flag_counts": flag_counts,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %d article records; flags=%s; issues=%d", len(records), flag_counts, len(issues))


if __name__ == "__main__":
    main()
