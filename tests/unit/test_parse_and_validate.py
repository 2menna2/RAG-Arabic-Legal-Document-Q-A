"""Row parsing into articles + corpus validation rules."""

import copy

from ecc_rag.data_pipeline.parse_articles import build_records, parse_rows
from ecc_rag.data_pipeline.validate import validate

SOURCE = {"document": "data/raw/Law.pdf", "sha256": "a" * 64, "version": "test", "code_effective_date": "1949-10-15"}


def row(en, ar, page=1, bold=0.0):
    return {"en_text": en, "ar_text": ar, "page": page, "en_bold_ratio": bold, "ar_bold_ratio": bold}


def parse(rows):
    builders, events = parse_rows(rows)
    recs, issues = build_records(builders, SOURCE)
    return {r["article_number"]: r for r in recs}, events, issues


def test_articles_headings_and_continuations():
    rows = [
        row("BOOK I\nOBLIGATIONS", "الكتاب الأول\nالالتزامات بوجه عام", bold=1.0),
        row("Article 1\nFirst text.", "مادة ( ١ )\nالنص الأول للمادة"),
        row("continued english", "تكملة النص العربي", page=2),
        row("Article 2\nSecond.", "مادة ( ٢ )\nالنص الثاني"),
    ]
    recs, _, issues = parse(rows)
    assert set(recs) == {1, 2} and not issues
    assert "تكملة النص العربي" in recs[1]["article_text"]
    assert recs[1]["source_pages"] == [1, 2]
    assert recs[1]["book"] == "الكتاب الأول الالتزامات بوجه عام"
    assert recs[1]["legal_status"] == "unverified"


def test_header_without_capital_a_is_still_parsed():
    recs, _, _ = parse([row("rticle 452\nAn action on a warranty", "مادة ٤٥٢\nتسقط بالتقادم دعوى الضمان")])
    assert 452 in recs and recs[452]["text_en"].startswith("An action")


def test_repeal_note_creates_flagged_records_without_invented_text():
    recs, _, _ = parse([row("Articles 389-417 repealed", "المواد من ٣٨٩ إلى ٤١٧ ملغاة", page=53)])
    assert set(recs) == set(range(389, 418))
    r = recs[400]
    assert r["legal_status"] == "repealed_per_source"
    assert r["article_text"] == "" and r["text_en"] == ""
    assert r["status_evidence"]["source_page"] == 53 and "ملغاة" in r["status_evidence"]["note_ar"]


def test_arabic_header_number_mismatch_is_flagged():
    recs, _, _ = parse([row("Article 10\ntext", "مادة ( ١١ )\nنص")])
    assert any(f.startswith("ar_header_number_mismatch") for f in recs[10]["quality_flags"])


def _valid_records(n=3):
    rows = [
        row(f"Article {i}\nEnglish text number {i} here.", f"مادة ( {i} )\nنص عربي طويل بما يكفي للمادة رقم {i} هنا")
        for i in range(1, n + 1)
    ]
    recs, _, _ = parse(rows)
    return [recs[k] for k in sorted(recs)]


def test_validation_passes_on_clean_records():
    rep = validate(_valid_records(), {"issues": []}, expected_max=3)
    assert rep["passed"], rep["errors"]


def test_validation_detects_gap_and_duplicates():
    recs = _valid_records(3)
    rep = validate([recs[0], recs[2]], {"issues": []}, expected_max=3)
    assert not rep["passed"] and any("missing article numbers" in e for e in rep["errors"])
    rep = validate([recs[0], recs[1], recs[1], recs[2]], {"issues": []}, expected_max=3)
    assert any("duplicate" in e for e in rep["errors"])


def test_validation_detects_tampered_text_and_merged_articles():
    recs = _valid_records(2)
    bad = copy.deepcopy(recs)
    bad[0]["article_text"] += " نص مضاف"  # content_hash no longer matches
    assert not validate(bad, {"issues": []}, expected_max=2)["passed"]
    merged = _valid_records(2)
    from ecc_rag.data_pipeline.schema import content_hash

    merged[0]["article_text"] += "\nمادة ( ٢ ) نص مادة أخرى مدمجة"
    merged[0]["content_hash"] = content_hash(1, merged[0]["article_text"], merged[0]["text_en"], "unverified")
    rep = validate(merged, {"issues": []}, expected_max=2)
    assert any("another article header" in e for e in rep["errors"])


def test_validation_detects_repeated_sentence_and_ligature_damage():
    from ecc_rag.data_pipeline.schema import content_hash

    recs = _valid_records(1)
    s = "ولا يجوز للدائن بعد موت المدين أخذ اختصاص على عقار في التركة"
    recs[0]["article_text"] = f"{s}. {s}."
    recs[0]["content_hash"] = content_hash(1, recs[0]["article_text"], recs[0]["text_en"], "unverified")
    assert any("repeated sentence" in e for e in validate(recs, {"issues": []}, expected_max=1)["errors"])
    recs = _valid_records(1)
    recs[0]["article_text"] = "العقد شريعة المتعاقدين فال يجوز نقضه وال تعديله إال باتفاق " * 3
    recs[0]["content_hash"] = content_hash(1, recs[0]["article_text"], recs[0]["text_en"], "unverified")
    assert any("ligature" in e for e in validate(recs, {"issues": []}, expected_max=1)["errors"])


def test_known_source_issue_allows_documented_missing_text():
    recs = _valid_records(1)
    from ecc_rag.data_pipeline.schema import content_hash

    recs[0]["article_text"] = ""
    recs[0]["quality_flags"] = ["ar_text_missing"]
    recs[0]["content_hash"] = content_hash(1, "", recs[0]["text_en"], "unverified")
    assert not validate(recs, {"issues": []}, expected_max=1)["passed"]
    known = {"issues": [{"id": "X", "allowed_flags": {1: ["ar_text_missing"]}}]}
    assert validate(recs, known, expected_max=1)["passed"]
