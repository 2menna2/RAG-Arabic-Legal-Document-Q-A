"""Scope classification: is the question about the Egyptian Civil Code corpus?

Three signals, combined (not a single keyword list):

1. **Other-law detector** — phrases that clearly belong to another Egyptian
   statute that is *not* in the corpus (penal code, personal status, tax,
   administrative procedures, company law, labour-law specifics).  Wording is
   specific on purpose: e.g. "نفقة الأقارب" stays in scope (Civil Code art. 15)
   while "نفقة الزوجة" does not.
2. **Retrieval signal** — the best char-TF-IDF similarity and query-term
   coverage against the corpus.  Thresholds come from ``configs/rag.yaml`` and
   are calibrated on the dev split.
3. **Clarity** — too few content words → ask for clarification.

When an LLM is available, its own ``out_of_scope`` / ``needs_clarification``
judgement is a fourth signal applied after retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ecc_rag.data_pipeline.arabic import light_stem, normalize_for_search, tokenize_for_search

OTHER_LAW: dict[str, list[str]] = {
    # NB: "الحبس" (right of retention, art. 246) and "سرقة" (arts. 220, 728, 977) are civil-law terms too,
    # so only unambiguous criminal-law vocabulary is listed.
    "criminal_law": [
        r"قانون العقوبات",
        r"\bعقوب[هة]\b",
        r"\bالعقوب[هة]\b",
        r"\bجريم[هة]\b",
        r"\bجنح[هة]\b",
        r"\bجناي[هة]\b",
        r"\bالسجن\b",
        r"مخدرات",
        r"شيك بدون رصيد",
        r"penal code",
        r"\bcriminal\b",
        r"\bprison\b",
    ],
    "personal_status": [
        r"نصيب .{0,25}(الميراث|الارث|الورث|التركه)",
        r"انصب[هة] الورث[هة]",
        r"\bالمواريث\b",
        r"\bالفرائض\b",
        r"الاحوال الشخصي[هة]",
        r"\bطلاق\b",
        r"\bالطلاق\b",
        r"\bالخلع\b",
        r"\bالحضان[هة]\b",
        r"نفق[هة] الزوج[هة]",
        r"الزواج العرفي",
        r"اثبات النسب",
        r"\bdivorce\b",
        r"\bcustody\b",
    ],
    # NB: art. 377 (prescription of taxes owed to the State) is in the Civil Code, so only tax-law specifics.
    "tax_customs": [
        r"ضريب[هة] (ال)?دخل",
        r"(نسب[هة]|سعر) (ال)?ضريب[هة]",
        r"قانون (ال)?ضرائب",
        r"القيم[هة] المضاف[هة]",
        r"\bالجمارك\b",
        r"income tax",
        r"\bvat\b",
    ],
    "administrative": [
        r"جواز (ال)?سفر",
        r"رخص[هة] (ال)?قياد[هة]",
        r"\bتاشير[هة]\b",
        r"الرقم القومي",
        r"بطاق[هة] (ال)?رقم",
        r"\bpassport\b",
        r"\bvisa\b",
        r"driving licen[cs]e",
        r"(رسوم|اجراءات) (ال)?شهر العقاري",
        r"توكيل رسمي في الشهر العقاري",
        r"الشهر العقاري.*(رسوم|الرسوم)",
    ],
    "labour_social_insurance": [
        r"قانون العمل",
        r"\bاجاز[هة] سنوي[هة]\b",
        r"\bاجاز[هة]\b.*\bحقي\b",
        r"التامينات الاجتماعي[هة]",
        r"labou?r law",
    ],
    "company_capital_markets": [
        r"شرك[هة] مساهم[هة]",
        r"شركات المساهم[هة]",
        r"قانون الشركات",
        r"\bالبورص[هة]\b",
        r"سجل تجاري",
        r"joint stock",
        r"company law",
    ],
    "constitutional_electoral": [r"\bالدستور\b", r"\bانتخابات\b", r"\bconstitution\b"],
}
_OTHER_LAW_RE = {k: [re.compile(p, re.I) for p in v] for k, v in OTHER_LAW.items()}

LEGAL_CUES = {
    light_stem(normalize_for_search(w))
    for w in """
عقد التزام مادة قانون مدني حق حقه حقي ملكية مالك بيع باع اشتريت شراء مشتري بائع البياع الشاري ايجار إيجار مستأجر مؤجر
أجرة تعويض مسئول مسئولية مسؤولية ضرر دين دائن مدين رهن كفالة كفيل ضامن شفعة حيازة تقادم إرث ميراث ورث ورثنا تركة ورثة
وصية هبة هدية شركة شريك وكالة توكيل وكيل مقاول مقاولة تأمين فوائد بطلان باطل فسخ إبطال أهلية قاصر رشد تدليس إكراه غلط
جار الجيران ارتفاق انتفاع تسجيل عقار منقول أرض شقة عربون قرض سلف ثمن عيب ضمان قضية محكمة القاضي دعوى مشاع شيوع
حادثة صلح وديعة حارس فسخ إخلاء
contract article civil obligation liability lease lessee lessor tenant sale vendor purchaser damages property
mortgage debtor creditor law court claim warranty agency partnership gift heir estate prescription majority capacity
""".split()
}
VAGUE_WORDS = {
    normalize_for_search(w)
    for w in """
مشكله مشكلة سؤال استفسار موضوع حاجه حاجة عندي عندى ذلك هذا هذه ده دي دى كده كدا الامر الأمر الحاله الحالة
المده المدة الشروط شروط يجوز ينفع ممكن
""".split()
}


@dataclass
class ScopeDecision:
    in_scope: bool
    reason: str = "in_scope"  # in_scope | other_law:<topic> | non_legal | needs_clarification
    detail: str = ""


def other_law_topic(question: str) -> str | None:
    norm = normalize_for_search(question)
    for topic, pats in _OTHER_LAW_RE.items():
        if any(p.search(norm) or p.search(question) for p in pats):
            return topic
    return None


def content_tokens(question: str) -> list[str]:
    from ecc_rag.rag.query import QUESTION_FRAME_WORDS

    stop = {light_stem(w) for w in QUESTION_FRAME_WORDS | VAGUE_WORDS} | QUESTION_FRAME_WORDS | VAGUE_WORDS
    return [t for t in tokenize_for_search(question) if t not in stop]


def classify(
    question: str, top_tfidf: float, coverage: float, has_article_ref: bool, cfg: dict[str, float], specificity: float | None = None
) -> ScopeDecision:
    """``specificity`` = max BM25 IDF of the question's content terms (rare legal term → high)."""
    topic = other_law_topic(question)
    # e.g. "عقوبة التبديد" for a debtor who fraudulently causes insolvency is in the Civil Code (art. 260):
    # a very strong corpus match overrides the other-law veto.
    strong_match = coverage >= cfg.get("other_law_override_coverage", 0.8) and top_tfidf >= cfg.get("other_law_override_tfidf", 0.35)
    if topic and not strong_match:
        return ScopeDecision(False, f"other_law:{topic}", "السؤال يتعلق بتشريع آخر غير القانون المدني الموجود في قاعدة البيانات.")
    if has_article_ref:
        return ScopeDecision(True)
    toks = content_tokens(question)
    generic_single = len(toks) == 1 and (specificity is None or specificity < cfg.get("min_idf_single_term", 2.5))
    if not toks or generic_single:
        return ScopeDecision(False, "needs_clarification", "السؤال لا يحدد المسألة القانونية المقصودة.")
    has_cue = bool(set(tokenize_for_search(question)) & LEGAL_CUES)
    english = sum(c.isascii() and c.isalpha() for c in question) > sum("\u0600" <= c <= "\u06ff" for c in question)
    if english and not has_cue:
        # the English column is a general-vocabulary translation: char n-grams match anything
        return ScopeDecision(False, "non_legal", "The question does not appear to concern the Egyptian Civil Code.")
    if not has_cue and (top_tfidf < cfg["min_tfidf_for_in_scope"] or coverage < cfg["min_coverage_for_in_scope"]):
        return ScopeDecision(False, "non_legal", "السؤال لا يبدو متعلقاً بأحكام القانون المدني المصري.")
    return ScopeDecision(True)
