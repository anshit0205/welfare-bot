"""
language.py — Fast, accurate language detection.

Stage 1: Unicode codepoint detection for non-ASCII input (instant, 99% accurate)
Stage 2: Piggybacked onto 8B intent classification call for ASCII/romanised input
         (zero extra latency — runs in same LLM call as intent detection)

Supports: English, Hindi, Marathi, Bengali, Tamil
"""

# Marathi-specific vocabulary — distinguishes Marathi from Hindi in Devanagari
_MARATHI_MARKERS = {
    "आहे", "नाही", "आणि", "मला", "तुम्ही", "करा", "होय", "नको",
    "सांगा", "माझा", "माझी", "शेतकरी", "मजूर", "मुलगी", "योजना",
    "कागदपत्रे", "अर्ज", "कुठे", "कसे", "काय", "जमीन", "माझे",
    "आम्ही", "तुमची", "असेल", "नसेल", "करायचे", "मिळेल", "पाहिजे",
}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "mr": "Marathi (मराठी)",
    "bn": "Bengali (বাংলা)",
    "ta": "Tamil (தமிழ்)",
}

SUPPORTED_LANGUAGES = list(LANGUAGE_NAMES.keys())


def detect_language_from_script(text: str) -> str | None:
    """
    Stage 1: Detect language purely from Unicode script ranges.
    Returns language code if confident, None if text is ASCII (needs Stage 2).
    """
    if not text or not text.strip():
        return None

    # Count characters per script block
    tamil    = sum(1 for c in text if 0x0B80 <= ord(c) <= 0x0BFF)
    bengali  = sum(1 for c in text if 0x0980 <= ord(c) <= 0x09FF)
    devanagari = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    telugu   = sum(1 for c in text if 0x0C00 <= ord(c) <= 0x0C7F)
    kannada  = sum(1 for c in text if 0x0C80 <= ord(c) <= 0x0CFF)
    latin    = sum(1 for c in text if c.isascii() and c.isalpha())

    total_non_latin = tamil + bengali + devanagari + telugu + kannada

    # If mostly ASCII → needs Stage 2 (LLM detection)
    if total_non_latin == 0:
        return None

    # If mixed but mostly Latin → likely English with some special chars → Stage 2
    if total_non_latin < 3 and latin > total_non_latin * 3:
        return None

    # Clear winner
    scores = {
        "ta": tamil,
        "bn": bengali,
        "hi": devanagari,  # default Devanagari → Hindi, check Marathi below
    }

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "en"  # fallback

    # Devanagari: distinguish Marathi vs Hindi by vocabulary
    if best == "hi":
        words = set(text.split())
        marathi_hits = words & _MARATHI_MARKERS
        if marathi_hits:
            return "mr"
        return "hi"

    return best


def get_language_name(code: str) -> str:
    """Get full language name for a code."""
    return LANGUAGE_NAMES.get(code, "Hindi (हिंदी)")


def parse_detected_language(raw: str) -> str:
    """
    Parse language code returned by LLM in classifier response.
    Handles slight variations and maps to our 5 supported codes.
    """
    if not raw:
        return "hi"

    raw = raw.strip().lower()

    # Direct code match
    if raw in ("en", "hi", "mr", "bn", "ta"):
        return raw

    # Name mapping
    name_map = {
        "english":  "en",
        "hindi":    "hi",
        "marathi":  "mr",
        "bengali":  "bn",
        "tamil":    "ta",
        "hinglish": "hi",
    }
    return name_map.get(raw, "hi")