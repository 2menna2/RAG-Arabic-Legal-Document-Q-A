"""Stage 3 — validate the structured corpus before anything is embedded.

Fails (exit code 1) on any problem that is not an explicitly documented source
issue (``configs/known_source_issues.yaml``).  Writes a machine-readable data
quality report that the evaluation report and CI both consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ecc_rag.data_pipeline.schema import ArticleRecord

logger = logging.getLogger(__name__)

INFO_FLAGS = {"one_sided_continuation_row"}
MAX_ARTICLE_CHARS = 3000  # longest genuine article is ~1.4k chars; a giant record means a failed split
EMBEDDED_HEADER_AR = re.compile(r"(?:^|\n)\s*\(?\s*م\s*ا\s*د\s*ة\s*[()\s]*[0-9٠-٩]")
EMBEDDED_HEADER_EN = re.compile(r"(?:^|\s)Article\s+\d{1,4}\s*\n")
# token artefacts produced by reversed lam-alef ligatures (see pdf_extract.py)
LIGATURE_ARTEFACT = re.compile(r"(?:^|\s)(?:إال|اإل|األ|فال|وال|ال)(?=\s)")
SPLIT_WORD = re.compile(r"(?:^|\s)[ا-ي](?=\s)")


def _allowed_flags(known: dict[str, Any]) -> dict[int, set[str]]:
    allowed: dict[int, set[str]] = {}
    for issue in known.get("issues", []):
        for art, flags in (issue.get("allowed_flags") or {}).items():
            allowed.setdefault(int(art), set()).update(flags)
    return allowed


def validate(
    records: list[dict[str, Any]], known: dict[str, Any], pdf_path: Path | None = None, expected_max: int = 1149
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    allowed = _allowed_flags(known)
    numbers: list[int] = []
    ids: set[str] = set()
    ligature_tokens = 0
    private_use = 0
    split_tokens = 0

    for i, raw in enumerate(records):
        try:
            rec = ArticleRecord.model_validate(raw)
        except ValidationError as exc:
            errors.append(f"record {i}: schema: {exc.errors()[:3]}")
            continue
        n = rec.article_number
        numbers.append(n)
        if rec.article_id in ids:
            errors.append(f"duplicate article_id {rec.article_id}")
        ids.add(rec.article_id)

        flags = {f.split(":")[0] for f in rec.quality_flags}
        blocking = flags - INFO_FLAGS - allowed.get(n, set())
        if blocking:
            errors.append(f"article {n}: unexplained quality flags {sorted(blocking)}")
        if rec.legal_status != "repealed_per_source":
            if not rec.article_text and "ar_text_missing" not in allowed.get(n, set()):
                errors.append(f"article {n}: empty Arabic text")
            if rec.article_text and sum("؀" <= c <= "ۿ" for c in rec.article_text) < 10:
                errors.append(f"article {n}: article_text lacks Arabic characters")
            if rec.text_en and sum(c.isascii() and c.isalpha() for c in rec.text_en) < 10:
                errors.append(f"article {n}: text_en lacks Latin characters")
            if len(rec.article_text) > MAX_ARTICLE_CHARS:
                errors.append(f"article {n}: {len(rec.article_text)} chars — probable merged articles")
            sentences = [x.strip() for x in re.split(r"[.\n]", rec.article_text) if len(x.strip()) >= 30]
            if len(sentences) != len(set(sentences)):
                errors.append(f"article {n}: repeated sentence — probable duplicated extraction")
            if EMBEDDED_HEADER_AR.search(rec.article_text) or EMBEDDED_HEADER_EN.search(rec.text_en + "\n"):
                errors.append(f"article {n}: contains another article header — split failure")
        ligature_tokens += len(LIGATURE_ARTEFACT.findall(rec.article_text))
        private_use += sum(1 for c in rec.article_text + rec.text_en if "\ue000" <= c <= "\uf8ff")
        split_tokens += len(SPLIT_WORD.findall(rec.article_text.replace(" و ", " ")))

    expected = set(range(1, expected_max + 1))
    missing = sorted(expected - set(numbers))
    dupes = sorted({n for n in numbers if numbers.count(n) > 1})
    if missing:
        errors.append(f"missing article numbers (unexplained gaps): {missing[:50]}")
    if dupes:
        errors.append(f"duplicate article numbers: {dupes}")
    if numbers != sorted(numbers):
        errors.append("records are not sorted by article_number")
    if ligature_tokens > 5:
        errors.append(f"{ligature_tokens} lam-alef ligature artefact tokens (expected ~0)")

    source_hash_ok = None
    if pdf_path is not None and pdf_path.exists():
        h = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        source_hash_ok = all(r.get("source_sha256") == h for r in records)
        if not source_hash_ok:
            errors.append("source_sha256 in records does not match the PDF on disk")
    elif pdf_path is not None:
        warnings.append("source PDF not present — source hash not re-verified")

    repealed = [r["article_number"] for r in records if r.get("legal_status") == "repealed_per_source"]
    corpus_hash = hashlib.sha256("".join(r["content_hash"] for r in records).encode()).hexdigest()
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "records": len(records),
            "repealed_per_source": len(repealed),
            "repealed_ranges": _ranges(repealed),
            "with_english": sum(1 for r in records if r.get("text_en")),
            "flagged_records": sum(1 for r in records if r.get("quality_flags")),
            "ligature_artefact_tokens": ligature_tokens,
            "single_letter_tokens_possible_split_words": split_tokens,
            "unmapped_private_use_glyphs": private_use,
            "max_article_chars": max((len(r.get("article_text", "")) for r in records), default=0),
            "source_hash_verified": source_hash_ok,
        },
        "corpus_hash": corpus_hash,
    }


def _ranges(nums: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for n in sorted(nums):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate articles.jsonl")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--known-issues", type=Path, default=Path("configs/known_source_issues.yaml"))
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    known = yaml.safe_load(args.known_issues.read_text(encoding="utf-8"))
    report = validate(records, known, args.pdf)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    for e in report["errors"][:30]:
        logger.error(e)
    logger.info("validation %s — %s", "PASSED" if report["passed"] else "FAILED", report["stats"])
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
