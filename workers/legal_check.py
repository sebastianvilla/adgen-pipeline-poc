import unicodedata

def _normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold().strip()

# Simple check of prohibited words list by locale
PROHIBITED_BY_LOCALE = {
    "en-US": ["sexy", "addictive", "politics"],
    "es-ES": ["sexy", "adictivo", "política"],
    "fr-FR": ["sexy", "addictif", "politique"],
    "ja-JP": ["セクシー", "中毒性", "政治"],
}

def check(message: str, locale: str = "en-US") -> bool:
    msg = _normalize(message)
    words = PROHIBITED_BY_LOCALE.get(locale, PROHIBITED_BY_LOCALE["en-US"])
    for w in words:
        if _normalize(w) in msg:
            return False
    return True