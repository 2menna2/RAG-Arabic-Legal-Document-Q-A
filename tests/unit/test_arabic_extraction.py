"""Arabic extraction repairs (ligatures, digit order) and normalisation."""

from ecc_rag.data_pipeline.arabic import clean_display_text, light_stem, normalize_for_search, to_western_digits, tokenize_for_search
from ecc_rag.data_pipeline.pdf_extract import _fix_line


def ch(c, x0, width=3.0):
    return {"c": c, "bbox": (x0, 0, x0 + width, 10), "origin": (x0, 9)}


def test_lam_alef_ligature_is_restored():
    # PDF order for the word "لا": zero-width alef then lam (reversed by the ToUnicode map)
    chars = [ch("ا", 100, 0.0), ch("ل", 94, 6.0), ch(" ", 90), ch("ي", 86), ch("ج", 80), ch("و", 75), ch("ز", 70)]
    assert _fix_line(chars) == "لا يجوز"


def test_regular_definite_article_is_untouched():
    chars = [ch("ا", 100, 2.3), ch("ل", 97, 2.3), ch("ع", 92), ch("ق", 88), ch("د", 84)]
    assert _fix_line(chars) == "العقد"


def test_hamza_alef_ligature():
    # "إلا": real alef-hamza, then [zero-width alef, lam] ligature
    chars = [ch("إ", 100, 2.3), ch("ا", 97, 0.0), ch("ل", 91, 6.0)]
    assert _fix_line(chars) == "إلا"


def test_multi_digit_numbers_are_left_to_right():
    # glyphs stored right-to-left: ٦٥٩١ should read ١٩٥٦ (x ascending)
    chars = [ch("ة", 200), ch("ن", 196), ch("س", 192), ch(" ", 188), ch("٦", 180), ch("٥", 176), ch("٩", 172), ch("١", 168)]
    assert _fix_line(chars).endswith("١٩٥٦")


def test_date_with_separators_kept_together():
    chars = [ch("٣", 120), ch("/", 116), ch("١", 112), ch("١", 108), ch("/", 104), ch("٦", 100), ch("٥", 96), ch("٩", 92), ch("١", 88)]
    assert _fix_line(chars) == "١٩٥٦/١١/٣"


def test_clean_display_text_rebuilds_paragraph_markers_and_joins_wraps():
    raw = "( ١ (\nتحسب المواعيد بالتقويم ال\nميلادي ما لم ينص على غير ذلك.\n) ٢ (\nفإذا لم يوجد نص"
    out = clean_display_text(raw)
    assert "بالتقويم الميلادي" in out
    assert "(١)" in out and "(٢)" in out
    assert out.count("\n") == 1


def test_clean_display_text_reattaches_split_definite_article():
    assert clean_display_text("جميع ال دائنين الذين") == "جميع الدائنين الذين"


def test_normalize_for_search():
    assert normalize_for_search("الإيجارُ مسئوليّة ١٤٧") == "الايجار مسيوليه 147"
    assert to_western_digits("المادة ١٤٧") == "المادة 147"


def test_tokenize_removes_stopwords_and_stems():
    toks = tokenize_for_search("ما هي شروط عقد البيع في القانون؟")
    assert "في" not in toks and "ما" not in toks
    assert light_stem("والمستأجر".replace("أ", "ا")) == "مستاجر"
