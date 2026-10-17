"""Stage 1 — extract table rows (English cell, Arabic cell) from the bilingual PDF.

The source PDF is a two-column Word export laid out as a table: English on
the left, Arabic on the right, one table row per article or heading.  A naive
``page.get_text()`` dump interleaves the columns and – more dangerously for a
legal corpus – corrupts Arabic in two systematic ways that we measured on the
reference implementation:

1. **Lam-alef ligatures** (ﻻ ﻷ ﻹ ﻵ) are mapped by the PDF's ToUnicode table to
   the two code points in the *wrong* logical order, so ``لا`` comes out as
   ``ال`` and ``إلا`` as ``إال``.  This silently turns negations ("لا يجوز") into
   something that reads like a definite article.  We detect the ligature
   geometrically: the alef half of a ligature is emitted with a *zero-width*
   bounding box immediately before the lam glyph, and we swap the pair back.
2. **Multi-digit numbers inside RTL lines** are emitted right-to-left
   (``١٩٤٩`` → ``٩٤٩١``).  Numbers are always left-to-right, so we re-order every
   run of digits (and date separators between digits) by ascending x.

Every row is written with its page number and bounding box so later stages can
cite ``source_page`` and so extraction problems can be traced back to the PDF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

logger = logging.getLogger(__name__)

ALEF_FAMILY = set("اأإآ")
LAM = "ل"
DIGITS = set("0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")
DIGIT_JOINERS = set("/.,")  # only kept inside a run when surrounded by digits
ZERO_WIDTH_EPS = 0.05


@dataclass
class ExtractedRow:
    page: int
    table_index: int
    row_index: int
    en_text: str
    ar_text: str
    en_bold_ratio: float
    ar_bold_ratio: float
    bbox: list[float]
    cell_count: int
    issues: list[str] = field(default_factory=list)


def _is_arabic_letter(ch: str) -> bool:
    return "؀" <= ch <= "ۿ" and ch not in DIGITS


def _script_counts(text: str) -> tuple[int, int]:
    ar = sum(1 for c in text if _is_arabic_letter(c))
    en = sum(1 for c in text if ("a" <= c.lower() <= "z"))
    return ar, en


def _fix_line(chars: list[dict[str, Any]]) -> str:
    """Return the logical text of one rawdict line with ligature + digit fixes."""
    seq = list(chars)
    # 1) lam-alef ligature repair: [alef(zero width), lam] -> [lam, alef]
    i = 0
    while i < len(seq) - 1:
        c, nxt = seq[i], seq[i + 1]
        width = c["bbox"][2] - c["bbox"][0]
        if c["c"] in ALEF_FAMILY and width < ZERO_WIDTH_EPS and nxt["c"] == LAM:
            seq[i], seq[i + 1] = nxt, c
            i += 2
            continue
        i += 1
    # 2) digit runs are LTR: re-order each run by x position
    out: list[str] = []
    j = 0
    while j < len(seq):
        if seq[j]["c"] in DIGITS:
            k = j
            while k + 1 < len(seq) and (
                seq[k + 1]["c"] in DIGITS or (seq[k + 1]["c"] in DIGIT_JOINERS and k + 2 < len(seq) and seq[k + 2]["c"] in DIGITS)
            ):
                k += 1
            run = seq[j : k + 1]
            if len(run) > 1:
                run = sorted(run, key=lambda ch: ch["origin"][0])
            out.extend(ch["c"] for ch in run)
            j = k + 1
        else:
            out.append(seq[j]["c"])
            j += 1
    return "".join(out)


def extract_cell(page: pymupdf.Page, rect: pymupdf.Rect) -> tuple[str, float]:
    """Extract text of a table cell, returning (text, bold_char_ratio)."""
    d = page.get_text("rawdict", clip=rect)
    lines: list[str] = []
    bold = total = 0
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            chars: list[dict[str, Any]] = []
            for span in line.get("spans", []):
                is_bold = "bold" in span.get("font", "").lower() or bool(span.get("flags", 0) & 16)
                for ch in span.get("chars", []):
                    # rawdict clip keeps chars whose bbox intersects; require the
                    # char origin to be inside the cell to avoid neighbour bleed.
                    x, y = ch["origin"]
                    if not (rect.x0 - 1 <= x <= rect.x1 + 1 and rect.y0 - 1 <= y <= rect.y1 + 1):
                        continue
                    chars.append(ch)
                    if not ch["c"].isspace():
                        total += 1
                        bold += int(is_bold)
            text = _fix_line(chars).strip()
            if text:
                lines.append(text)
    return "\n".join(lines), (bold / total if total else 0.0)


def _contains(big: pymupdf.Rect, small: pymupdf.Rect, tol: float = 2.0) -> bool:
    return small.x0 >= big.x0 - tol and small.y0 >= big.y0 - tol and small.x1 <= big.x1 + tol and small.y1 <= big.y1 + tol and big != small


def _word_set(text: str) -> frozenset[str]:
    import re

    return frozenset(w for w in re.findall(r"[\w\u0600-\u06ff]+", text) if len(w) > 1 and not w.isdigit())


def extract_rows(pdf_path: Path) -> list[ExtractedRow]:
    doc = pymupdf.open(pdf_path)
    rows: list[ExtractedRow] = []
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            tables = page.find_tables().tables
            if not tables:
                logger.warning("page %d: no table detected", pno + 1)
                rows.append(ExtractedRow(pno + 1, -1, -1, "", "", 0, 0, list(page.rect), 0, ["no_table_on_page"]))
                continue
            seen: list[tuple[pymupdf.Rect, frozenset[str]]] = []
            for t_idx, table in enumerate(tables):
                for r_idx, row in enumerate(table.rows):
                    row_rect = pymupdf.Rect(row.bbox)
                    cells = [pymupdf.Rect(c) for c in row.cells if c is not None]
                    en_parts: list[tuple[str, float]] = []
                    ar_parts: list[tuple[str, float]] = []
                    issues: list[str] = []
                    for rect in cells:
                        text, bold = extract_cell(page, rect)
                        if not text.strip():
                            continue
                        ar_n, en_n = _script_counts(text)
                        if ar_n == 0 and en_n == 0:
                            # digits/punctuation only: attach to the side by position
                            (ar_parts if rect.x0 > page.rect.width / 2 else en_parts).append((text, bold))
                        elif ar_n >= en_n:
                            ar_parts.append((text, bold))
                            if en_n > 10:
                                issues.append("mixed_script_cell")
                        else:
                            en_parts.append((text, bold))
                            if ar_n > 10:
                                issues.append("mixed_script_cell")
                    if len(cells) != 2:
                        issues.append(f"cell_count_{len(cells)}")

                    def _join(parts: list[tuple[str, float]]) -> tuple[str, float]:
                        if not parts:
                            return "", 0.0
                        txt = "\n".join(p[0] for p in parts)
                        w = sum(len(p[0]) for p in parts) or 1
                        return txt, sum(p[1] * len(p[0]) for p in parts) / w

                    en_text, en_bold = _join(en_parts)
                    ar_text, ar_bold = _join(ar_parts)
                    if not en_text and not ar_text:
                        continue
                    # Highlighted paragraphs are sometimes detected as a second, overlapping
                    # table. Skip such a nested row only if its text was already captured by
                    # the enclosing row (otherwise it carries text we would lose).
                    words = _word_set(en_text + " " + ar_text)
                    if any(_contains(big, row_rect) and words <= txt for big, txt in seen):
                        logger.info("page %d: skipped duplicate nested row", pno + 1)
                        continue
                    seen.append((row_rect, words))
                    rows.append(
                        ExtractedRow(
                            page=pno + 1,
                            table_index=t_idx,
                            row_index=r_idx,
                            en_text=en_text,
                            ar_text=ar_text,
                            en_bold_ratio=round(en_bold, 3),
                            ar_bold_ratio=round(ar_bold, 3),
                            bbox=[round(v, 2) for v in row.bbox],
                            cell_count=len(cells),
                            issues=issues,
                        )
                    )
    finally:
        doc.close()
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if not args.input.exists():
        raise SystemExit(
            f"Source PDF not found: {args.input}. Run `dvc pull data/raw/Law.pdf.dvc` or place the PDF there (see docs/DATA_SOURCES.md)."
        )
    rows = extract_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    logger.info("extracted %d rows from %s (sha256=%s)", len(rows), args.input, sha256_file(args.input))


if __name__ == "__main__":
    main()
