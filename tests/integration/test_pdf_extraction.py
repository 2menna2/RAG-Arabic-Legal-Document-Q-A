"""Extraction on real pages of the source PDF (needs data/raw/Law.pdf from DVC)."""

import pytest

from ecc_rag.config import PROJECT_ROOT

pytestmark = pytest.mark.integration
PDF = PROJECT_ROOT / "data/raw/Law.pdf"


def test_first_page_rows_are_clean():
    if not PDF.exists():
        pytest.skip("data/raw/Law.pdf not present — run `dvc pull`")
    import pymupdf

    from ecc_rag.data_pipeline.pdf_extract import extract_cell

    doc = pymupdf.open(PDF)
    page = doc[0]
    rows = page.find_tables().tables[0].rows
    texts = ["".join(extract_cell(page, pymupdf.Rect(c))[0] for c in r.cells if c) for r in rows]
    joined = "\n".join(texts)
    assert "لا يجوز إلغاء نص تشريعي إلا بتشريع لاحق" in joined  # ligatures repaired
    assert "SECTION I" in joined
    assert extract_cell(page, pymupdf.Rect(rows[0].cells[0]))[1] > 0.9  # heading row is bold
