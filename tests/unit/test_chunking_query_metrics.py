"""Chunking strategies, query analysis and retrieval metric definitions."""

import pytest

from ecc_rag.data_pipeline.chunking import build_chunks, split_paragraphs
from ecc_rag.evaluation.retrieval_metrics import all_metrics, mrr, ndcg_at_k, precision_at_k, recall_at_k
from ecc_rag.rag.query import analyze_query, extract_article_refs
from tests.conftest import synthetic_articles


def test_exception_paragraph_stays_with_rule():
    text = "(١) العقد شريعة المتعاقدين.\n(٢) ومع ذلك إذا طرأت حوادث استثنائية عامة جاز للقاضي.\n(٣) ويقع باطلا كل اتفاق."
    paras = split_paragraphs(text)
    assert len(paras) == 2 and "ومع ذلك" in paras[0]


@pytest.mark.parametrize("strategy", ["article", "article_ctx", "paragraph"])
def test_every_chunk_is_traceable(strategy):
    chunks = build_chunks(synthetic_articles(), strategy, {"include_english": True, "min_chars_to_split": 10})
    for c in chunks:
        assert c["article_number"] and c["article_id"] and c["source_page"] and c["chunk_id"].startswith(c["article_id"])
    repealed = [c for c in chunks if c["article_number"] == 60]
    assert repealed and repealed[0]["kind"] == "repeal_status" and "ألغيت" in repealed[0]["display_text"]


def test_article_ctx_keeps_display_text_verbatim():
    arts = synthetic_articles()
    chunks = build_chunks(arts, "article_ctx", {"include_english": True})
    by = {c["article_number"]: c for c in chunks}
    a147 = next(a for a in arts if a["article_number"] == 147)
    assert by[147]["display_text"] == a147["article_text"]
    assert "Article 147" in by[147]["search_text"]


def test_article_refs():
    assert extract_article_refs("ماذا تنص المادة ١٤٧؟") == [147]
    assert extract_article_refs("What does Article 5 say?") == [5]
    assert extract_article_refs("اذكر نص المادة 3000") == [3000]  # kept so the gate can refuse
    assert extract_article_refs("المواد من 54 إلى 56") == [54, 55, 56]
    assert analyze_query("نص المادة 147 مكرر").bis_refs == [147]


def test_question_frame_words_removed_from_search_text():
    q = analyze_query("ما تعريف عقد التأمين في القانون المدني؟")
    assert "تعريف" not in q.content and "التامين" in q.content and q.is_definition


def test_metric_definitions():
    ranked = [5, 9, 147, 3]
    assert recall_at_k(ranked, [147, 148], 3) == 0.5
    assert mrr(ranked, [147]) == pytest.approx(1 / 3)
    assert precision_at_k(ranked, [147], [5], 2) == 0.5
    assert ndcg_at_k([147], [147], [], 1) == 1.0
    assert ndcg_at_k([5, 147], [147], [5], 2) < 1.0
    m = all_metrics(ranked, [3], [], ks=(1, 5))
    assert m["hit@1"] == 0 and m["hit@5"] == 1 and m["mrr"] == 0.25
    with pytest.raises(ValueError):
        recall_at_k(ranked, [], 3)
