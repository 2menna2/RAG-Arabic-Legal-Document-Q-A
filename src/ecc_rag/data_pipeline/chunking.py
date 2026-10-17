"""Stage 4 — chunk articles for retrieval.

Every chunk keeps ``article_number`` / ``article_id`` / source page so any
retrieved passage can be traced back to the article and the PDF page.

Strategies (compared in the retrieval benchmark, not assumed):

``article``
    One chunk per article, Arabic text only.
``article_ctx``
    One chunk per article; the *search* text is prefixed with the structural
    headings (book / chapter / section / sub-heading) and followed by the
    English translation, which gives lexical and cross-lingual hooks.  The
    *display* text stays the verbatim article.
``paragraph``
    Articles longer than ``min_chars_to_split`` are split on their numbered
    paragraphs ``(١) (٢) …``.  A paragraph that opens with an exception /
    condition connector (ومع ذلك، على أنه، إلا …) is merged back into the
    previous paragraph so a rule is never separated from its exception.  Each
    sub-chunk carries the article header and headings.

Repealed ranges become a single *status* chunk per article with the verbatim
repeal note from the source, so "is article 60 in force?" is answerable from
evidence rather than from the model's memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from ecc_rag.config import config_hash, load_rag_config

logger = logging.getLogger(__name__)

EXCEPTION_OPENERS = re.compile(
    r"^\s*(?:\([0-9٠-٩]{1,2}\)\s*)?(?:ومع ذلك|ومع هذا|على أن|علي أن|على أنه|إلا|الا|غير أن|غير أنه|ويستثنى|"
    r"وفي هذه الحالة|وفى هذه الحالة|وفي جميع الأحوال|وفى جميع الأحوال|فإذا|وإذا|ولا يجوز|ولكن)"
)
PARA_SPLIT = re.compile(r"\n(?=\([0-9٠-٩]{1,2}\)\s)")


def _headings(a: dict[str, Any]) -> list[str]:
    return [h for h in (a.get("book"), a.get("chapter"), a.get("section"), a.get("subsection"), a.get("topic")) if h]


def _base_meta(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "article_id": a["article_id"],
        "article_number": a["article_number"],
        "citation": a["citation"],
        "source_page": a["source_page"],
        "source_pages": a["source_pages"],
        "legal_status": a["legal_status"],
        "book": a.get("book"),
        "chapter": a.get("chapter"),
        "section": a.get("section"),
        "subsection": a.get("subsection"),
        "topic": a.get("topic"),
        "content_hash": a["content_hash"],
        "quality_flags": a.get("quality_flags", []),
    }


def split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and EXCEPTION_OPENERS.match(p):
            merged[-1] = merged[-1] + "\n" + p
        else:
            merged.append(p)
    return merged


def chunk_article(a: dict[str, Any], strategy: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    n = a["article_number"]
    meta = _base_meta(a)
    heads = " — ".join(_headings(a))
    header = f"المادة {n}"

    if a["legal_status"] == "repealed_per_source":
        ev = a["status_evidence"] or {}
        note = " ".join(x for x in (ev.get("note_ar"), ev.get("note_en")) if x)
        return [
            {
                "chunk_id": f"{a['article_id']}#status",
                "chunk_index": 0,
                "kind": "repeal_status",
                "language": "ar",
                "display_text": note,
                "search_text": f"{header} المادة {n} ملغاة ملغى ألغيت repealed Article {n} {heads} {note}",
                **meta,
            }
        ]

    ar, en = a["article_text"], a.get("text_en", "")
    language = "ar" if ar else "en"
    display = ar or en
    if strategy == "article":
        return [
            {
                "chunk_id": f"{a['article_id']}#0",
                "chunk_index": 0,
                "kind": "article",
                "language": language,
                "display_text": display,
                "search_text": f"{header}\n{display}",
                **meta,
            }
        ]
    if strategy == "article_ctx":
        extra_en = f"\nArticle {n}: {en}" if (params.get("include_english", True) and en and ar) else ""
        return [
            {
                "chunk_id": f"{a['article_id']}#0",
                "chunk_index": 0,
                "kind": "article",
                "language": language,
                "display_text": display,
                "search_text": f"{header} — {heads}\n{display}{extra_en}",
                **meta,
            }
        ]
    if strategy == "paragraph":
        min_chars = int(params.get("min_chars_to_split", 600))
        paras = split_paragraphs(display) if (ar and len(ar) >= min_chars) else [display]
        out = []
        for i, p in enumerate(paras):
            out.append(
                {
                    "chunk_id": f"{a['article_id']}#{i}",
                    "chunk_index": i,
                    "kind": "article" if len(paras) == 1 else "paragraph",
                    "language": language,
                    "display_text": p,
                    "search_text": f"{header} — {heads}\n{p}",
                    **meta,
                }
            )
        return out
    raise ValueError(f"unknown chunking strategy: {strategy}")


def build_chunks(articles: list[dict[str, Any]], strategy: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for a in articles:
        chunks.extend(chunk_article(a, strategy, params))
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk ids")
    return chunks


def load_articles(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk articles.jsonl")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", default=None, help="override configs/rag.yaml chunking.strategy")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = load_rag_config()["chunking"]
    strategy = args.strategy or cfg["strategy"]
    params = cfg.get("params", {})
    chunks = build_chunks(load_articles(args.input), strategy, params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("strategy=%s params_hash=%s chunks=%d", strategy, config_hash(params), len(chunks))


if __name__ == "__main__":
    main()
