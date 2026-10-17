"""Citation validation is the last gate before an answer is shown."""

from ecc_rag.rag.citations import answer_article_mentions, quote_in_text, validate
from ecc_rag.rag.llm import GenerationOutput, LLMBadOutputError, parse_output

CTX = {
    147: {
        "text": "(١) العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين، أو للأسباب التي يقررها القانون.",
        "legal_status": "unverified",
    },
    60: {"text": "[ملغاة وفقاً لملاحظة المصدر] ألغيت المواد من ٥٤ إلى ٨٠", "legal_status": "repealed_per_source"},
}


def out(answer, claims, status="answered"):
    return GenerationOutput(status=status, answer=answer, claims=claims)


def test_valid_answer_passes():
    o = out(
        "العقد شريعة المتعاقدين [م 147]",
        [{"statement": "لا يجوز نقض العقد إلا باتفاق", "articles": [147], "quotes": ["فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين"]}],
    )
    r = validate(o, CTX)
    assert r.valid and r.quotes_valid == 1 and r.cited_articles == [147]


def test_quote_normalisation_tolerates_hamza_and_ellipsis():
    assert quote_in_text("العقد شريعة المتعاقدين … إلا باتفاق الطرفين", CTX[147]["text"])
    assert quote_in_text("فلا يجوز نقضه ولا تعديله الا باتفاق", CTX[147]["text"])


def test_unknown_article_and_hallucinated_reference_fail():
    o = out("وفقاً للمادة 148 يجب التنفيذ بحسن نية", [{"statement": "x", "articles": [148], "quotes": []}])
    r = validate(o, CTX)
    assert not r.valid
    assert any(i.startswith("unknown_article") for i in r.issues)
    assert any(i.startswith("hallucinated_reference") for i in r.issues)


def test_fabricated_quote_fails():
    o = out("[م 147]", [{"statement": "x", "articles": [147], "quotes": ["يجوز لأحد الطرفين فسخ العقد بإرادته المنفردة"]}])
    assert any(i.startswith("quote_mismatch") for i in validate(o, CTX).issues)


def test_stated_claim_needs_citation_and_answer_needs_claims():
    assert any(i.startswith("missing_citation") for i in validate(out("نص", [{"statement": "x", "articles": []}]), CTX).issues)
    assert not validate(out("إجابة بلا استشهاد", []), CTX).valid


def test_repealed_article_must_be_presented_as_repealed():
    bad = out("المادة 60 تنظم الجمعيات [م 60]", [{"statement": "x", "articles": [60], "quotes": []}])
    assert any(i.startswith("repealed_without_notice") for i in validate(bad, CTX).issues)
    good = out("المادة 60 ملغاة وفقاً للمصدر [م 60]", [{"statement": "ملغاة", "articles": [60], "quotes": ["ألغيت المواد من ٥٤ إلى ٨٠"]}])
    assert validate(good, CTX).valid


def test_non_answer_statuses_are_not_validated_as_answers():
    assert validate(GenerationOutput(status="insufficient_evidence"), CTX).valid


def test_answer_mentions_parsing():
    assert answer_article_mentions("انظر [م 147] والمادة ١٤٨ و Article 5") == [5, 147, 148]


def test_parse_output_rejects_non_json():
    import pytest

    with pytest.raises(LLMBadOutputError):
        parse_output("not json")
    assert parse_output('```json\n{"status": "insufficient_evidence"}\n```').status == "insufficient_evidence"
