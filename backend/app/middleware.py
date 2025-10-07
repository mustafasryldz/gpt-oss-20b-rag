import re
import unicodedata
from langdetect import detect, DetectorFactory
from typing import Tuple

DetectorFactory.seed = 0  # deterministic

EMAIL_RE   = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE   = re.compile(r"\+?\d[\d\s\-]{8,}\d")
APIKEY_RE  = re.compile(r"\b(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36,})\b", re.I)
INJECTION_RE = re.compile(r"(ignore\s+previous|disregard\s+above|you\s+are\s+now|system\s+prompt)", re.I)
BADWORDS_TR = [r"\bkahpe\b", r"\boglum\b", r"\bomurga?s[ıi]z\b"]
BADWORDS_RE = re.compile("|".join(BADWORDS_TR), re.I)

MASK = "█"

def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("\u00A0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        return "tr" if lang.startswith("tr") else ("en" if lang.startswith("en") else lang)
    except Exception:
        return "unknown"

def mask_pii(text: str) -> Tuple[str, dict]:
    flags = {"email": False, "phone": False, "apikey": False, "badwords": False, "injection": False}

    def _mask(pattern, key, s: str) -> str:
        if pattern.search(s):
            flags[key] = True
            s = pattern.sub(MASK, s)
        return s

    text = _mask(EMAIL_RE, "email", text)
    text = _mask(PHONE_RE, "phone", text)
    text = _mask(APIKEY_RE, "apikey", text)

    if BADWORDS_RE.search(text):
        flags["badwords"] = True
        text = BADWORDS_RE.sub(MASK, text)

    if INJECTION_RE.search(text):
        flags["injection"] = True

    return text, flags
