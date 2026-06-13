"""
intent_classifier.py — Intent + language + context resolution.

Uses 70B model (not 8B) because context resolution requires real reasoning.
Getting "madhya pradesh ka batao" → "NREGA wages in Madhya Pradesh" right
is the difference between a useful answer and a completely wrong one.
"""

from src.utils.llm_client      import call_strong, parse_json_response,call_fast
from src.utils.language        import detect_language_from_script, parse_detected_language
from src.utils.prompts         import CLASSIFIER_SYSTEM, CLASSIFIER_PROMPT
from src.memory.session_memory import get_history_text, session_to_profile_text

VALID_INTENTS = {
    "eligibility_check", "scheme_query", "document_query",
    "how_to_apply", "general_info", "slot_fill_response", "off_topic",
}


def classify(message: str, session: dict, session_id: str) -> dict:
    """
    Single 70B call: language detection + intent + context resolution + slot extraction.

    Returns:
    {
        "language":         "hi",
        "language_source":  "script_detected" | "auto_detected",
        "intent":           "scheme_query",
        "new_facts":        {"has_land": true},
        "scheme_mentioned": "nrega",
        "resolved_query":   "NREGA daily wage in West Bengal",
    }
    """
    # ── Stage 1: Unicode script detection (instant, no LLM) ──────────────────
    script_lang = detect_language_from_script(message)
    known_lang = session.get("language")

    # ── Build context for prompt ──────────────────────────────────────────────
    history_text = get_history_text(session, turns=4)
    known_facts  = session_to_profile_text(session)
    entity_mem   = session.get("entity_memory", {})
    last_scheme  = entity_mem.get("last_scheme_id") or "none"
    summary      = session.get("conversation_summary", "")

    # Inject summary for richer context on long conversations
    full_history = f"Summary: {summary}\n\n{history_text}" if summary else history_text

    prompt = CLASSIFIER_PROMPT.format(
    message=message,
    history=full_history,
    known_facts=known_facts,
    last_scheme=last_scheme,
    )

    if known_lang:
        prompt += f"""

    KNOWN USER LANGUAGE:
    {known_lang}

    IMPORTANT:
    Do NOT perform language detection.
    The user's language is already known.
    Use "{known_lang}" as the language.
    """

    raw    = call_fast(
        system=CLASSIFIER_SYSTEM,
        user=prompt,
        max_tokens=350,
        json_mode=True,
        session_id=session_id,
        call_type="intent_classifier",
    )
    result = parse_json_response(raw)

    # ── Apply language detection result ───────────────────────────────────────
    if script_lang:
        # Script detection is definitive — override LLM
        result["language"]        = script_lang
        result["language_source"] = "script_detected"
    elif known_lang:
        result["language"] = known_lang
        result["language_source"] = "session"

    else:
        result["language"] = parse_detected_language(
            result.get("language", "hi")
        )
        result["language_source"] = "auto_detected"

    # ── Sanitise ──────────────────────────────────────────────────────────────
    if result.get("intent") not in VALID_INTENTS:
        result["intent"] = _keyword_fallback(message, session)

    result.setdefault("new_facts",        {})
    result.setdefault("scheme_mentioned", None)
    result.setdefault("resolved_query",   message)

    # Clean nulls from new_facts
    result["new_facts"] = {
        k: v for k, v in (result.get("new_facts") or {}).items()
        if v is not None and v != ""
    }

    return result


def _keyword_fallback(message: str, session: dict) -> str:
    """Last-resort keyword fallback if 70B returns invalid intent."""
    msg = message.lower().strip()

    yes_w = {"yes", "हाँ", "haan", "ha", "ஆம்", "হ্যাঁ", "होय", "ji"}
    no_w  = {"no", "नहीं", "nahi", "இல்லை", "না", "नाही", "nope", "nhi"}

    if session.get("in_eligibility_flow") and not session.get("eligibility_done"):
        if any(w in msg for w in yes_w) or any(w in msg for w in no_w):
            return "slot_fill_response"

    if any(w in msg for w in ["document", "kागज", "दस्तावेज", "aadhaar", "papers"]):
        return "document_query"
    if any(w in msg for w in ["apply", "आवेदन", "application", "how to", "steps"]):
        return "how_to_apply"
    if any(w in msg for w in ["eligible", "पात्र", "qualify", "which scheme", "कौन सी"]):
        return "eligibility_check"

    return "scheme_query"