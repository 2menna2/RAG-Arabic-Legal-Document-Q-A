"""Arabic text utilities shared by the data pipeline and retrieval.

Two very different operations live here and must not be confused:

* ``clean_display_text`` — conservative repairs of *extraction artefacts* in
  the stored legal text (mirrored brackets, line-wrap joins).  It never changes
  words, spelling, hamza or diacritics: the stored ``article_text`` must stay a
  faithful copy of the source.
* ``normalize_for_search`` — aggressive, lossy normalisation used only to build
  search keys (BM25 tokens, TF-IDF features, citation matching).  It is never
  shown to users.
"""

from __future__ import annotations

import re
import unicodedata

ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
TASHKEEL_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"

# paragraph markers rendered with mirrored brackets in the RTL column, e.g.
# ")١ (", "(١ (", "١ (", ") أ)".  We rebuild them as "(١)" / "(أ)".
_PARA_MARK_RE = re.compile(r"[()]\s*([0-9٠-٩]{1,2}|[أبجدهوزحطي])\s*[()]")
_LETTER_MARK_LINE_RE = re.compile(r"^\(?\s*\n?([أبجدهوزحطي])\)\s*", re.MULTILINE)


def to_western_digits(text: str) -> str:
    return text.translate(ARABIC_INDIC_DIGITS)


def clean_display_text(text: str) -> str:
    """Repair layout artefacts without altering legal wording."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).replace(" ", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and ln not in {"(", ")", "()", ")(", "( )", ") ("}]
    # A lone bracket line followed by a letter-marker line: ")\nأ) ..."
    merged: list[str] = []
    for ln in lines:
        # re-attach the definite article split by a visual line wrap ("ال" + "\n" + "ميلادي")
        if merged and merged[-1].endswith(" ال"):
            merged[-1] = merged[-1] + ln
            continue
        if merged and merged[-1] == "ال":
            merged[-1] = "ال" + ln
            continue
        merged.append(ln)
    text = " ".join(merged)
    # a free-standing definite article is never a word: it is a glyph-spacing
    # artefact of the source ("ال دائنين" -> "الدائنين"); re-attach it.
    text = re.sub(r"(^|\s)(ال|وال|بال|فال|لل) (?=[\u0621-\u064A])", r"\1\2", text)
    # final-alef that slipped after the full stop: "فحواه.ا" -> "فحواها."
    text = re.sub(r"\.ا(\s|$)", r"ا.\1", text)
    # leading paragraph marker split from the header: "١ ( نص" / "١ )نص" / "١.) نص"
    text = re.sub(r"^[()]?\s*([0-9٠-٩]{1,2})\s*\.?\s*[()]\s*", r"(\1) ", text)
    text = _PARA_MARK_RE.sub(lambda m: f"({m.group(1)})", text)
    # lettered sub-paragraph with a lost opening bracket: " ب) " -> " (ب) "
    text = re.sub(r"(^|\s)([أبجده])\)\s", r"\1(\2) ", text)
    # put each numbered/lettered paragraph on its own line for readability
    text = re.sub(r"\s*(\([0-9٠-٩]{1,2}\))\s*", r"\n\1 ", text)
    text = re.sub(r"\s+([.،,؛:])", r"\1", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def clean_english_text(text: str) -> str:
    if not text:
        return ""
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return " ".join(ln for ln in lines if ln).strip()


def strip_tashkeel(text: str) -> str:
    return TASHKEEL_RE.sub("", text).replace(TATWEEL, "")


def normalize_for_search(text: str) -> str:
    """Lossy normalisation for matching (never for display)."""
    text = unicodedata.normalize("NFKC", text)
    text = strip_tashkeel(text)
    text = to_western_digits(text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --- light stemming (prefix/suffix stripping in the spirit of Larkey's light10)
_PREFIXES = ("وبال", "وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ل")
_SUFFIXES = ("هما", "تين", "ات", "ان", "ين", "ون", "ها", "ية", "ه", "ي")

ARABIC_STOPWORDS = {
    normalize_for_search(w)
    for w in """
    في من على علي الى إلى عن مع او أو ان أن إن لا ما لم لن هل هو هي هذا هذه ذلك تلك التي الذي الذين
    كان كانت يكون تكون اذا إذا كل بعض غير قد ثم بين حتى عند وقد فيها فيه منه منها به بها له لها
    كما ايضا أيضا اي أي التى الذى ده دي دى ايه إيه اللي إللي عشان علشان مين فين ازاي إزاي امتى إمتى
    هو وهو وهي يعني بس لو ولا مش هل يا انا أنا احنا إحنا انت إنت عايز عاوز
    """.split()
}
ENGLISH_STOPWORDS = set(
    "the a an of to in on for by with is are be or and as at from that this which what who how when "
    "can may shall will it its his her their there if not no any all such".split()
)


def light_stem(token: str) -> str:
    if not token or not re.search("[؀-ۿ]", token):
        return token
    for p in _PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            token = token[len(p) :]
            break
    for s in _SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[: -len(s)]
            break
    return token


def tokenize_for_search(text: str, stem: bool = True) -> list[str]:
    toks = normalize_for_search(text).split()
    out = []
    for t in toks:
        if t in ARABIC_STOPWORDS or t in ENGLISH_STOPWORDS:
            continue
        if len(t) < 2 and not t.isdigit():
            continue
        out.append(light_stem(t) if stem else t)
    return out
