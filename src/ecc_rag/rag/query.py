"""Query analysis: normalisation, dialect expansion and explicit article references."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ecc_rag.config import PROJECT_ROOT
from ecc_rag.data_pipeline.arabic import normalize_for_search, to_western_digits

# "المادة 147", "مادة ١٤٧", "م 147", "Article 147", "art. 147"
ARTICLE_REF_RE = re.compile(
    r"(?:ال)?(?:مادة|ماده|المادتين|المواد)\s*(?:رقم\s*)?\(?\s*([0-9٠-٩]{1,4})\s*\)?"
    r"|\b(?:article|art\.?)\s*([0-9]{1,4})\b",
    re.IGNORECASE,
)
ARTICLE_RANGE_RE = re.compile(r"(?:من\s*)?([0-9٠-٩]{1,4})\s*(?:إلى|الى|حتى|-|–|to)\s*([0-9٠-٩]{1,4})")


# Words that frame a question but carry no legal content; removed from the *search* text only.
QUESTION_FRAME_WORDS = {
    normalize_for_search(w)
    for w in """تعريف عرف حكم احكام أحكام القانون المدني المصري مصر ماذا يعني معنى المقصود اشرح شرح اذكر نص تنص ينص
    قانون مدني لي ليا ايه إيه ماهو ماهي what does say define definition under egyptian civil code law explain""".split()
}
DEFINITION_CUES = {normalize_for_search(w) for w in "تعريف عرف ماهو ماهي المقصود معنى define definition defined".split()}


@dataclass
class AnalyzedQuery:
    original: str
    normalized: str
    expanded: str
    expansions: list[str] = field(default_factory=list)
    article_refs: list[int] = field(default_factory=list)
    is_dialectal: bool = False
    is_definition: bool = False
    content: str = ""
    bis_refs: list[int] = field(default_factory=list)


@lru_cache(maxsize=4)
def load_lexicon(path: str | None = None) -> dict[str, list[str]]:
    p = Path(path) if path else PROJECT_ROOT / "configs" / "dialect_lexicon.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {normalize_for_search(k): v for k, v in data["expansions"].items()}


def extract_article_refs(text: str, max_article: int = 1149) -> list[int]:
    """All article numbers the user refers to — including ones that do not exist
    (e.g. 3000), so that the evidence gate can refuse instead of guessing."""
    refs: list[int] = []
    t = to_western_digits(text)
    for m in ARTICLE_REF_RE.finditer(t):
        num = m.group(1) or m.group(2)
        if num and int(num) >= 0:
            refs.append(int(num))
    if re.search(r"المواد|المادتين|articles", t, re.IGNORECASE):
        for m in ARTICLE_RANGE_RE.finditer(t):
            lo, hi = int(m.group(1)), int(m.group(2))
            if 1 <= lo < hi <= max_article and hi - lo <= 40:
                refs.extend(range(lo, hi + 1))
    return list(dict.fromkeys(refs))  # de-duplicate, keep order


BIS_RE = re.compile(r"(?:مادة|ماده)\s*(?:رقم\s*)?\(?\s*([0-9٠-٩]{1,4})\s*\)?\s*(?:مكرر|مكررا|مكرراً|bis)", re.IGNORECASE)


def analyze_query(question: str, lexicon: dict[str, list[str]] | None = None) -> AnalyzedQuery:
    lexicon = lexicon if lexicon is not None else load_lexicon()
    norm = normalize_for_search(question)
    padded = f" {norm} "
    toks = norm.split()
    content = " ".join(t for t in toks if t not in QUESTION_FRAME_WORDS)
    is_definition = bool(set(toks) & DEFINITION_CUES) or norm.startswith(("ما هو ", "ما هي ", "what is ", "how is "))
    expansions: list[str] = []
    for key, vals in lexicon.items():
        if f" {key} " in padded:
            expansions.extend(vals)
    expanded = content + (" " + " ".join(normalize_for_search(v) for v in expansions) if expansions else "")
    return AnalyzedQuery(
        original=question,
        normalized=norm,
        expanded=expanded,
        expansions=expansions,
        article_refs=extract_article_refs(question),
        is_dialectal=bool(expansions),
        is_definition=is_definition,
        content=content or norm,
        bis_refs=[int(to_western_digits(m.group(1))) for m in BIS_RE.finditer(question)],
    )
