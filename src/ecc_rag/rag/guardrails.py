"""Input validation and prompt-injection defences.

Layered, because no pattern list catches every attack:

1. ``check_input``       — length / content sanity (cheap, deterministic).
2. ``detect_injection``  — multilingual patterns + structural signals (role
   tags, fake system delimiters, chat-template tokens).  A hit rejects the
   request before retrieval.
3. ``scan_documents``    — the same detector over *retrieved* text; a hit
   removes that document from the context (documents are untrusted data).
4. Prompt isolation      — documents are wrapped in ``<document>`` tags with
   ``<``/``>`` escaped, and the system prompt states that document content is
   never an instruction.
5. Output checks         — citation validation (``citations.py``) rejects
   answers that cite anything outside the retrieved context.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override_en",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instructions?|rules?|prompt|guidelines|above|previous|prior|system)\b",
            re.I | re.S,
        ),
    ),
    (
        "reveal_prompt_en",
        re.compile(
            r"\b(reveal|show|print|repeat|leak|tell me)\b.{0,30}\b(system prompt|instructions|hidden prompt|your prompt|rules)\b",
            re.I | re.S,
        ),
    ),
    (
        "role_switch_en",
        re.compile(
            r"\b(you are now|act as|pretend (to be|you are)|from now on you|developer mode|jailbreak|DAN\b|no restrictions|without (any )?restrictions)",
            re.I,
        ),
    ),
    (
        "override_ar",
        re.compile(r"(تجاهل|تجاهلي|انس|انسى|اترك|تخط|تخطى|تجاوز|ألغ|الغ)\s*(كل|جميع)?\s*(ال)?(تعليمات|أوامر|اوامر|قواعد|القيود|السابق)"),
    ),
    ("reveal_ar", re.compile(r"(اكشف|أظهر|اظهر|اعرض|اطبع|كرر)\s*(لي)?\s*(ال)?(تعليمات|موجه|برومبت|البرومبت|قواعدك|أوامرك|الأوامر)")),
    ("role_switch_ar", re.compile(r"(أنت|انت)\s*(الآن|الان|دلوقتي)\s|تصرف\s*(كأنك|ك)|بلا\s*قيود|بدون\s*قيود|من\s*غير\s*قيود|وضع\s*المطور")),
    (
        "fake_delimiter",
        re.compile(
            r"(###\s*(system|assistant|instruction)|<\s*/?\s*(system|assistant|instructions?)\s*>|\[/?INST\]|<\|im_(start|end)\|>|^\s*(system|assistant)\s*:)",
            re.I | re.M,
        ),
    ),
    (
        "forced_answer",
        re.compile(
            r"\b(you must|you have to|always) (answer|say|respond|reply)\b|(يجب|لازم)\s*(أن|ان)?\s*(تجيب|تقول|ترد)\s*(بأن|ان|أن)", re.I
        ),
    ),
]

CONTROL_CHARS = re.compile("[\\u0000-\\u0008\\u000b\\u000c\\u000e-\\u001f\\u007f\\u202a-\\u202e\\u2066-\\u2069]")


@dataclass
class InputCheck:
    ok: bool
    code: str = "ok"
    message: str = ""


def check_input(question: str, max_chars: int = 1000) -> InputCheck:
    if question is None:
        return InputCheck(False, "empty", "السؤال فارغ.")
    q = unicodedata.normalize("NFC", question).strip()
    if not q:
        return InputCheck(False, "empty", "السؤال فارغ.")
    if len(q) > max_chars:
        return InputCheck(False, "too_long", f"السؤال أطول من الحد المسموح ({max_chars} حرف).")
    letters = sum(1 for c in q if c.isalpha())
    if letters < 3:
        return InputCheck(False, "too_short", "السؤال قصير جداً أو لا يحتوي على نص مفهوم.")
    if CONTROL_CHARS.search(q):
        return InputCheck(False, "control_characters", "السؤال يحتوي على رموز تحكم غير مسموح بها.")
    return InputCheck(True)


def detect_injection(text: str) -> list[str]:
    return [name for name, pat in INJECTION_PATTERNS if pat.search(text or "")]


def escape_document(text: str) -> str:
    """Neutralise tag-like content so a document cannot close its own <document> wrapper."""
    return text.replace("<", "‹").replace(">", "›")


def scan_documents(texts: list[str]) -> list[list[str]]:
    return [detect_injection(t) for t in texts]
