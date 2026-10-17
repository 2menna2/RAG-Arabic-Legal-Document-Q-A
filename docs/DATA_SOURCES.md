# Data sources, provenance and usage conditions

## Legal corpus

| Item | Value (only what can be verified) |
|---|---|
| Document | Egyptian Civil Code (Law No. 131 of 1948), bilingual table edition (English left, Arabic right), 170 pages |
| File | `data/raw/Law.pdf` — DVC-tracked, **never committed to Git** |
| Hashes | md5 `5086ef5f16f15d856278db943e58e948` · sha256 `2d6419fb262dab712f7bf5f6e31834bea74fbcf03b78ae081dd35d156915eebf` · 2,599,703 bytes |
| Identity check | identical (md5) to the file referenced by the reference project's `data/raw/Law.pdf.dvc` |
| Origin | provided by the project owner; the ITI × MLOps MENA handbook cites *sadanykhalifa.com — القانون المدني المصري* as the source of this bilingual PDF |
| Retrieval date | unknown (not recorded when the file was obtained) |
| Version | PDF metadata: created 2019-07-22 by Microsoft Word; the date to which the text is consolidated is **unknown** |
| Effective date | original code in force from **15 October 1949** — from the promulgation law printed on page 1 ("ويعمل به ابتداء من ١٥ أكتوبر سنة ١٩٤٩"); per-article effective dates unknown → `effective_date = null` |
| English column | unofficial translation of unknown authorship; used for cross-lingual retrieval only |
| Repeals | only those stated in the document itself: Arts. 54–80 (p. 7, with the decrees cited by the source) and 389–417 (p. 53) → `legal_status = repealed_per_source` with verbatim evidence; all others `unverified` |
| License / rights | unknown. Egyptian statutes are generally not protected as official texts, but the translation and the compilation may be. The PDF and the derived full-text JSONL are therefore kept out of Git and distributed only through a private DVC remote. **Confirm rights before any public deployment.** |

Nothing in the corpus was hand-written: there are no manual text patches. Source defects are documented in
`configs/known_source_issues.yaml` (SRC-001 … SRC-007) and surfaced as `quality_flags` per article.

## Record schema (`data/processed/articles.jsonl`)

`article_id, article_number, article_text (Arabic, verbatim after layout repair), text_en, language, translation_language,
part, book, chapter, section, subsection, topic, headings_en, source_document, source_sha256, source_page, source_pages,
source_version, effective_date (null), code_effective_date, legal_status, status_evidence, citation, citation_en,
quality_flags, content_hash` — enforced by `src/ecc_rag/data_pipeline/schema.py` (Pydantic, `extra=forbid`).

## Getting the data

1. **DVC remote (recommended):** `dvc remote add -d --local storage <url>` then `dvc pull`. A remote is not committed
   because none was available; the mechanism was tested with a local-directory remote: push → fresh `git clone` →
   `dvc pull` restored `Law.pdf` and all outputs with identical hashes.
2. **From the PDF:** place it at `data/raw/Law.pdf`; `configs/source.yaml` holds the expected sha256 and
   `validate.py --pdf` re-verifies it. Then `dvc repro`.

## Evaluation data (`data/evaluation/golden_v1.yaml`)

241 questions authored on 2026-10-17 by the engineering team (AI-assisted) by reading the article texts; expected
articles verified against `articles.jsonl` (an automated check flags key points with low lexical support; the factual
ones were then read against the full article text). 21 items were adapted from the reference project's evaluation questions (its 20-item set and its gate list) and corrected.
`review_status: unreviewed` on every item — **no lawyer has reviewed this set**. Deterministic split by
`sha256(id)`: dev ≈ 40 % (tuning, calibration), test ≈ 60 % (reporting, gates).
