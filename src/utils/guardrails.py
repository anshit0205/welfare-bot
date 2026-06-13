"""
guardrails.py — Fast pre-LLM input guardrail layer.

Runs BEFORE the classifier (intent_classifier.py) on every user message.
Two-stage design:

  Stage 1 — Regex/keyword fast-path (instant, no LLM call)
    Catches obvious off-topic, abusive, prompt-injection, or unsafe
    requests without spending a single token.

  Stage 2 — Lightweight 8B classifier (only if Stage 1 is ambiguous)
    For messages that don't match Stage 1 patterns but also don't look
    like a welfare-scheme question, ask the fast model a single yes/no
    "is this on-topic" question. Cheap (~20 tokens).

On-topic = anything about: government welfare schemes, eligibility,
documents, benefits, applying, helplines, related general govt-scheme
chat, greetings, or slot-fill answers (yes/no/occupation/income/state
during eligibility flow — these are NOT off-topic even though they look
like one-word replies).

Returns a GuardrailResult.
  • result.allowed=False  → return result.message directly, skip everything else.
  • result.stage          → 1 (regex) or 2 (LLM).  None if allowed.
  • result.matched        → the specific pattern/keyword that fired.  None if allowed.
  • result.reason         → 'offtopic' | 'injection' | 'abuse'.  None if allowed.

Always call log_guardrail() after check_message(), whether allowed or not —
blocked messages never reach the classifier and would be invisible otherwise.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.utils.llm_client import call_fast
from src.memory.usage_logger import log_llm_call


# ══════════════════════════════════════════════════════════════════════════════
# RESULT TYPE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GuardrailResult:
    allowed:  bool
    reason:   Optional[str] = None   # 'offtopic' | 'injection' | 'abuse' | None
    message:  Optional[str] = None   # user-facing refusal, in their language
    stage:    Optional[int] = None   # 1=regex, 2=LLM; None if allowed
    matched:  Optional[str] = None   # pattern/keyword that fired; None if allowed


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — FAST REGEX / KEYWORD PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

# Obvious prompt-injection / jailbreak attempts
_INJECTION_PATTERNS = [
    r"ignore (all|previous|above) instructions",
    r"you are now",
    r"system prompt",
    r"act as (a|an)\s",
    r"pretend (you|to be)",
    r"jailbreak",
    r"dan mode",
    r"\bsudo\b",
    r"reveal your (prompt|instructions|system)",
]

# Clearly off-topic subjects.
# ⚠️  IMPORTANT: keep these SPECIFIC enough not to hit welfare-related queries.
#     e.g. "translate this scheme letter to Hindi" must NOT be blocked —
#     so the translate pattern anchors on "translate this/the following ... to a
#     foreign language" rather than any sentence containing "translate".
_OFFTOPIC_PATTERNS = [
    r"write (a |me )?(code|program|script|poem|essay|song|story)\b",
    # Code/debug help — only fire when a language token AND a code task token co-occur
    r"\b(python|javascript|java|c\+\+|html|css|sql)\b.{0,60}\b(code|function|program|error|bug)\b",
    # General trivia / news — NOT welfare queries
    r"who (won|is the (prime minister|president|ceo))\b",
    r"\b(movie|film|cricket|football|match score|song lyrics|recipe)\b",
    r"\bmeaning of life\b",
    # Translation of arbitrary content — but NOT "translate this scheme letter"
    # Anchored: "translate this to English/Hindi" with no scheme-context word nearby
    r"translate (this|the following)\b(?!.{0,40}(scheme|yojana|form|letter|document))"
    r".{0,60}\b(to|into)\s+(english|hindi|french|spanish|bengali|tamil|telugu|marathi)\b",
    r"what('?s| is) the weather\b",
    r"\bstock (price|market)\b",
    r"\b(cryptocurrency|bitcoin|ethereum)\b",
]

# Abusive / harmful content.
# Use {0,80} to bound the lookahead distance — prevents a false positive where
# an innocuous clause crosses a 200-word message into a harmful one.
_ABUSE_PATTERNS = [
    r"\b(kill|suicide|bomb|weapon|hack|exploit)\b.{0,80}\b(make|build|create|how to)\b",
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)
_OFFTOPIC_RE  = re.compile("|".join(_OFFTOPIC_PATTERNS),  re.IGNORECASE | re.DOTALL)
_ABUSE_RE     = re.compile("|".join(_ABUSE_PATTERNS),      re.IGNORECASE | re.DOTALL)

# Things that LOOK short/ambiguous but are valid slot-fill / on-topic replies
# during the eligibility flow — never block these.
_SAFE_SHORT_REPLIES = {
    "yes", "no", "ha", "haan", "हाँ", "नहीं", "nahi", "होय", "नाही",
    "ஆம்", "இல்லை", "হ্যাঁ", "না",
    "farmer", "labourer", "student", "woman hoh",
    "किसान", "मजदूर", "छात्र", "महिला",
    "🌾 farmer", "👷 labourer", "👩 woman hoh", "📚 student",
}

# Welfare-domain keywords that instantly allow a message, skipping Stage 2.
_WELFARE_KEYWORDS = {
    "scheme", "yojana", "eligib", "पात्र", "योजना", "document", "दस्तावेज़",
    "apply", "आवेदन", "benefit", "लाभ", "helpline", "wage", "मजदूरी",
    "pension", "पेंशन", "subsidy", "सब्सिडी", "card", "कार्ड", "income",
    "आय", "land", "ज़मीन", "bpl", "aadhaar", "आधार", "nrega", "mgnrega",
    "pmay", "pm kisan", "ration", "राशन",
}


# ══════════════════════════════════════════════════════════════════════════════
# REFUSAL MESSAGES — multilingual
# ══════════════════════════════════════════════════════════════════════════════

REFUSAL_MESSAGES = {
    "offtopic": {
        "en": "I'm a welfare scheme assistant, so I can't help with that. Ask me about government schemes, eligibility, documents, or how to apply — I'm happy to help with those!",
        "hi": "मैं एक कल्याण योजना सहायक हूँ, इसलिए इसमें मदद नहीं कर सकता। मुझसे सरकारी योजनाओं, पात्रता, दस्तावेज़ों या आवेदन के बारे में पूछें — मैं उसमें खुशी से मदद करूँगा।",
        "mr": "मी कल्याण योजना सहाय्यक आहे, त्यामुळे यात मदत करू शकत नाही. सरकारी योजना, पात्रता, कागदपत्रे किंवा अर्जाबद्दल विचारा.",
        "bn": "আমি একটি কল্যাণ প্রকল্প সহকারী, তাই এতে সাহায্য করতে পারি না। সরকারি প্রকল্প, যোগ্যতা, কাগজপত্র বা আবেদন সম্পর্কে জিজ্ঞাসা করুন।",
        "ta": "நான் ஒரு நலன் திட்ட உதவியாளர், எனவே இதில் உதவ முடியாது. அரசு திட்டங்கள், தகுதி, ஆவணங்கள் அல்லது விண்ணப்பிப்பது பற்றி கேளுங்கள்.",
    },
    "injection": {
        "en": "I can't follow instructions like that. I'm here only to help with government welfare scheme questions — feel free to ask about eligibility, documents, or benefits.",
        "hi": "मैं ऐसे निर्देशों का पालन नहीं कर सकता। मैं केवल सरकारी कल्याण योजना संबंधी सवालों में मदद के लिए हूँ — पात्रता, दस्तावेज़ या लाभ के बारे में पूछें।",
        "mr": "मी असे निर्देश पाळू शकत नाही. मी फक्त सरकारी कल्याण योजनेच्या प्रश्नांसाठी आहे.",
        "bn": "আমি এই ধরনের নির্দেশ অনুসরণ করতে পারি না। আমি শুধুমাত্র সরকারি কল্যাণ প্রকল্প সম্পর্কিত প্রশ্নে সাহায্য করি।",
        "ta": "நான் அத்தகைய வழிமுறைகளைப் பின்பற்ற முடியாது. நான் அரசு நலன் திட்டக் கேள்விகளுக்கு மட்டுமே உதவ முடியும்.",
    },
    "abuse": {
        "en": "I can't help with that request. If you have questions about welfare schemes, eligibility, or benefits, I'm here for that.",
        "hi": "मैं इस अनुरोध में मदद नहीं कर सकता। यदि आपके पास कल्याण योजनाओं के बारे में सवाल हैं, तो मैं यहाँ मदद के लिए हूँ।",
        "mr": "मी या विनंतीमध्ये मदत करू शकत नाही. कल्याण योजनांबद्दल प्रश्न असल्यास मी मदत करू शकतो.",
        "bn": "আমি এই অনুরোধে সাহায্য করতে পারি না। কল্যাণ প্রকল্প সম্পর্কে প্রশ্ন থাকলে আমি সাহায্য করতে পারি।",
        "ta": "இந்த கோரிக்கையில் எனக்கு உதவ முடியாது. நலன் திட்டங்கள் பற்றிய கேள்விகளுக்கு நான் இங்கே இருக்கிறேன்.",
    },
}


def get_refusal_message(reason: str, lang: str) -> str:
    bucket = REFUSAL_MESSAGES.get(reason, REFUSAL_MESSAGES["offtopic"])
    return bucket.get(lang, bucket["en"])


def _first_match(pattern: re.Pattern, text: str) -> Optional[str]:
    """Return the matched substring (truncated) for logging, or None."""
    m = pattern.search(text)
    return m.group(0)[:120] if m else None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — LIGHTWEIGHT LLM CHECK (only for ambiguous cases)
# ══════════════════════════════════════════════════════════════════════════════

_STAGE2_SYSTEM = (
    "You classify whether a user message is related to Indian government "
    "welfare schemes (eligibility, benefits, documents, how to apply, helplines, "
    "general greetings, or short answers like yes/no/occupation/income/state "
    "given during an eligibility questionnaire). "
    "Return ONLY valid JSON: {\"on_topic\": true} or {\"on_topic\": false}. No prose."
)

_STAGE2_PROMPT = """\
User message: "{message}"

Is this on-topic for a welfare scheme assistant (as defined)?
Return ONLY: {{"on_topic": true}} or {{"on_topic": false}}
"""


def _stage2_check(message: str, session_id: str) -> bool:
    """
    Returns True if on-topic, False if off-topic.
    Defaults to True on failure (fail-open — never block a real user due to
    our own classification error).
    Also logs the LLM token usage so Stage 2 costs are visible in the dashboard.
    """
    from src.utils.llm_client import parse_json_response

    raw = call_fast(
        system=_STAGE2_SYSTEM,
        user=_STAGE2_PROMPT.format(message=message),
        max_tokens=20,
        json_mode=True,
        session_id=session_id,
        call_type="guardrail_check",
    )

    # Log Stage 2 token usage — previously this import existed but was never called.
    # call_fast should return a dict with usage; adapt to your actual return shape.
    if isinstance(raw, dict) and "usage" in raw:
        u = raw["usage"]
        log_llm_call(
            session_id=session_id,
            call_type="guardrail_check",
            model=raw.get("model", "fast"),
            prompt_tokens=u.get("input_tokens"),
            completion_tokens=u.get("output_tokens"),
            total_tokens=u.get("input_tokens", 0) + u.get("output_tokens", 0),
            query_text=message,
        )
        parsed = parse_json_response(raw.get("content", raw))
    else:
        parsed = parse_json_response(raw)

    return parsed.get("on_topic", True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def check_message(
    message: str,
    session: dict,
    session_id: str = "unknown",
) -> GuardrailResult:
    """
    Run guardrail checks on a user message.

    Call this FIRST, before intent_classifier.classify().
    After calling this, ALWAYS call log_guardrail() with the result —
    blocked messages are otherwise invisible in usage_events.

    If result.allowed is False → return result.message to the user immediately.
    If result.allowed is True  → proceed to classifier as normal.
    """
    lang      = session.get("language", "hi")
    msg       = message.strip()
    msg_lower = msg.lower()

    # ── Always allow special internal triggers ────────────────────────────────
    if msg == "__init__":
        return GuardrailResult(allowed=True)

    # ── Always allow known short slot-fill replies (yes/no/occupation etc.) ───
    if msg_lower in _SAFE_SHORT_REPLIES:
        return GuardrailResult(allowed=True)

    # ── During active eligibility flow: permissive for short free-text replies ─
    # Numbers, state names, "2 lakh" etc. are slot answers, not open chat.
    # Still run injection/abuse regex (zero cost) but skip off-topic checks.
    in_flow = session.get("in_eligibility_flow", False) and not session.get("eligibility_done", False)
    if in_flow and len(msg.split()) <= 6:
        matched = _first_match(_INJECTION_RE, msg_lower)
        if matched:
            return GuardrailResult(
                allowed=False, reason="injection", stage=1, matched=matched,
                message=get_refusal_message("injection", lang),
            )
        return GuardrailResult(allowed=True)

    # ── Stage 1a: prompt injection / jailbreak ────────────────────────────────
    matched = _first_match(_INJECTION_RE, msg_lower)
    if matched:
        return GuardrailResult(
            allowed=False, reason="injection", stage=1, matched=matched,
            message=get_refusal_message("injection", lang),
        )

    # ── Stage 1b: abuse / harmful intent ──────────────────────────────────────
    matched = _first_match(_ABUSE_RE, msg_lower)
    if matched:
        return GuardrailResult(
            allowed=False, reason="abuse", stage=1, matched=matched,
            message=get_refusal_message("abuse", lang),
        )

    # ── Stage 1c: clearly off-topic keyword match ─────────────────────────────
    matched = _first_match(_OFFTOPIC_RE, msg_lower)
    if matched:
        return GuardrailResult(
            allowed=False, reason="offtopic", stage=1, matched=matched,
            message=get_refusal_message("offtopic", lang),
        )

    # ── Stage 1d: welfare keyword fast-allow → skip Stage 2 ──────────────────
    if any(kw in msg_lower for kw in _WELFARE_KEYWORDS):
        return GuardrailResult(allowed=True)

    # ── Stage 1e: very short messages (greetings etc.) → allow ────────────────
    if len(msg.split()) <= 3:
        return GuardrailResult(allowed=True)

    # ── Stage 2: ambiguous longer message — one cheap 8B call ─────────────────
    try:
        on_topic = _stage2_check(msg, session_id)
    except Exception as e:
        print(f"[guardrail Stage2 ERROR] {e}")
        on_topic = True  # fail-open: never penalise user for our infra error

    if not on_topic:
        return GuardrailResult(
            allowed=False, reason="offtopic", stage=2, matched=None,
            message=get_refusal_message("offtopic", lang),
        )

    return GuardrailResult(allowed=True)