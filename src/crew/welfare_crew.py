"""
welfare_crew.py — Main orchestrator. v2 — loop bug fixed.

Root cause of eligibility loop:
  Old flow: user answers → classifier tries to extract slot → sometimes fails
  → slot not stored → same question asked again → infinite loop

New flow:
  user answers → dedicated slot extractor (8B, single job) → store → next slot
  The slot extractor ONLY extracts one value. Cannot fail silently.

Root cause of context loss:
  Old: resolved_query used raw message if classifier uncertain
  New: classifier is 70B and always produces a resolved_query.
       Entity memory (last_scheme_id) is injected into every classifier call.
"""

from typing import Callable, Optional
from src.memory.session_memory import (
    get_session, save_session, append_turn, merge_facts,
    update_language, update_entity_memory, get_next_pending_slot,
    is_eligibility_complete, needs_binary_buttons, needs_occupation_buttons,
    ELIGIBILITY_SLOTS,
)
from src.agents.intent_classifier import classify
from src.agents.answer_agent      import (
    answer_scheme_query, answer_general_query,
    ask_next_slot_question, extract_slot_value,
    update_conversation_summary,
)
from src.agents.eligibility_agent import run_eligibility_check
from src.utils.prompts            import get_status
from src.utils.language           import get_language_name
from src.data_pipeline.embedder import BM25
import sys
from src.utils.guardrails     import check_message
from src.memory.usage_logger  import log_guardrail
sys.modules['__main__'].BM25 = BM25
import time
# ── Fast gate — pure Python, no LLM ──────────────────────────────────────────
_YES_WORDS = {
    "yes", "हाँ", "हां", "ha", "haan", "ஆம்", "হ্যাঁ", "होय",
    "✅ yes", "✅ हाँ", "✅ ஆம்", "✅ হ্যাঁ", "✅ होय",
    "ji", "haa", "bilkul",
}
_NO_WORDS = {
    "no", "नहीं", "nahi", "nein", "இல்லை", "না", "नाही",
    "❌ no", "❌ नहीं", "❌ இல்லை", "❌ না", "❌ नाही",
    "nhi", "nope", "na",
}

_OCC_BUTTON_MAP = {
    "🌾 farmer": "farmer",       "🌾 किसान": "farmer",
    "🌾 விவசாயி": "farmer",     "🌾 কৃষক": "farmer",
    "🌾 शेतकरी": "farmer",
    "👷 labourer": "labourer",   "👷 मजदूर": "labourer",
    "👷 தொழிலாளர்": "labourer","👷 শ্রমিক": "labourer",
    "👷 मजूर": "labourer",
    "👩 woman hoh": "woman_hoh", "👩 महिला": "woman_hoh",
    "👩 பெண் தலைவி": "woman_hoh","👩 মহিলা": "woman_hoh",
    "👩 महिला प्रमुख": "woman_hoh",
    "📚 student": "student",     "📚 छात्र": "student",
    "📚 மாணவர்": "student",     "📚 ছাত্র": "student",
    "📚 विद्यार्थी": "student",
}

# Binary slots — answered with yes/no buttons
_BINARY_SLOTS = {"has_land", "is_bpl", "has_girl_child", "is_pregnant"}


def _fast_gate(message: str, session: dict) -> Optional[dict]:
    """
    No-LLM fast path for unambiguous inputs.
    Returns classification dict if matched, None if LLM needed.
    """
    msg_lower = message.lower().strip()
    in_flow   = session.get("in_eligibility_flow", False)
    done      = session.get("eligibility_done", False)
    lang      = session.get("language", "hi")

    # ── Occupation button ─────────────────────────────────────────────────────
    for label, occ in _OCC_BUTTON_MAP.items():
        if msg_lower == label.lower() or message.strip() == label.strip():
            # woman_hoh implies female — infer gender so we skip that question
            inferred_facts = {"occupation": occ}
            if occ == "woman_hoh":
                inferred_facts["gender"] = "female"
            return {
                "intent":          "slot_fill_response",
                "new_facts":       inferred_facts,
                "language":        lang,
                "language_source": "session",
                "scheme_mentioned": None,
                "resolved_query":  f"My occupation is {occ}",
                "_slot_filled":    "occupation",
                "_slot_value":     occ,
            }

    # ── Yes/No for binary slots ───────────────────────────────────────────────
    if in_flow and not done:
        next_slot = get_next_pending_slot(session)
        if next_slot in _BINARY_SLOTS:
            if msg_lower in _YES_WORDS:
                return {
                    "intent":          "slot_fill_response",
                    "new_facts":       {next_slot: True},
                    "language":        lang,
                    "language_source": "session",
                    "scheme_mentioned": None,
                    "resolved_query":  "yes",
                    "_slot_filled":    next_slot,
                    "_slot_value":     True,
                }
            if msg_lower in _NO_WORDS:
                return {
                    "intent":          "slot_fill_response",
                    "new_facts":       {next_slot: False},
                    "language":        lang,
                    "language_source": "session",
                    "scheme_mentioned": None,
                    "resolved_query":  "no",
                    "_slot_filled":    next_slot,
                    "_slot_value":     False,
                }

    return None


def run(
    sid:       str,
    message:   str,
    status_cb: Callable[[str], None] = lambda x: None,
) -> tuple[str, list[str]]:

    # ── Memory hydration ──────────────────────────────────────────────────────
    # Must happen FIRST — check_message needs session for language + flow state
    status_cb(get_status("loading_memory", "en"))
    session = get_session(sid)
    lang    = session.get("language", "hi")

    # ── Guardrail check ───────────────────────────────────────────────────────
      # add at top of file if not already there

    t0 = time.monotonic()
    result = check_message(message, session, session_id=sid)
    guardrail_ms = (time.monotonic() - t0) * 1000
    log_guardrail(
        session_id=sid,
        query_text=message,
        allowed=result.allowed,
        stage=result.stage,
        reason=result.reason,
        matched=result.matched,
        latency_ms=guardrail_ms,
    )
    if not result.allowed:
        return result.message, []

    # ── Init eligibility flow ─────────────────────────────────────────────────
    if message == "__init__":
        session["in_eligibility_flow"] = True
        session["confirmed_facts"]     = {}
        session["pending_slots"]       = list(ELIGIBILITY_SLOTS)
        session["eligibility_done"]    = False
        # Store what question we just asked, so slot extractor has context
        first_q = ask_next_slot_question("occupation", session,session_id=sid)
        session["last_slot_question"]  = first_q
        session["last_slot_asked"]     = "occupation"
        append_turn(session, "__start__", first_q)
        save_session(sid, session)
        return first_q, []

    eligible_ids = []
    reply        = ""
    in_flow      = session.get("in_eligibility_flow", False)
    done         = session.get("eligibility_done", False)

    # ══════════════════════════════════════════════════════════════
    # ELIGIBILITY FLOW — slot filling
    # Separate path from general Q&A to prevent cross-contamination
    # ══════════════════════════════════════════════════════════════
    if in_flow and not done:

        # ── Fast gate first (yes/no buttons, occupation buttons) ──────────────
        fast = _fast_gate(message, session)

        if fast:
            # Fast gate directly gives us the slot + value
            merge_facts(session, fast.get("new_facts", {}))
            update_language(session, fast["language"], fast["language_source"])

        else:
            # ── Dedicated slot extractor for free-text answers ────────────────
            # e.g. user types "2 lakhs" for income, "maharashtra" for state
            next_slot     = get_next_pending_slot(session)
            last_question = session.get("last_slot_question", "")

            if next_slot:
                status_cb(get_status("extracting_slot", lang))
                extracted = extract_slot_value(
                    slot_name=next_slot,
                    user_reply=message,
                    question_asked=last_question,
                    session=session,
                    status_cb=status_cb,
                    session_id=sid
                )

                if extracted is not None:
                    merge_facts(session, {next_slot: extracted})
                else:
                    # Extraction failed — user may have asked a question instead
                    # Check if it's a general question, not a slot answer
                    if _looks_like_question(message):
                        # Answer their question briefly, then re-ask the slot
                        brief = answer_general_query(message, session, status_cb,session_id=sid)
                        re_q  = ask_next_slot_question(next_slot, session, session_id=sid)
                        reply = f"{brief}\n\n{re_q}"
                        append_turn(session, message, reply)
                        session["last_slot_question"] = re_q
                        session["last_slot_asked"]    = next_slot
                        save_session(sid, session)
                        return reply, []
                    # Otherwise re-ask the same slot — once only
                    re_q  = ask_next_slot_question(next_slot, session, session_id=sid)
                    session["last_slot_question"] = re_q
                    session["last_slot_asked"]    = next_slot
                    append_turn(session, message, re_q)
                    save_session(sid, session)
                    return re_q, []

            # Detect language from message even during slot fill
            from src.utils.language import detect_language_from_script
            script_lang = detect_language_from_script(message)
            if script_lang:
                update_language(session, script_lang, "script_detected")
            lang = session.get("language", "hi")

        # ── Check if flow is complete ─────────────────────────────────────────
        next_slot = get_next_pending_slot(session)

        if next_slot:
            # Ask next question
            q = ask_next_slot_question(next_slot, session, session_id=sid)
            session["last_slot_question"] = q
            session["last_slot_asked"]    = next_slot
            reply = q

        else:
            # All slots filled — run eligibility check
            if is_eligibility_complete(session):
                status_cb(get_status("checking_eligibility", lang))
                status_cb(get_status("compiling_eligibility", lang))
                reply, eligible_ids = run_eligibility_check(session)
                session["eligibility_done"]    = True
                session["in_eligibility_flow"] = False
                if eligible_ids:
                    session["entity_memory"]["last_schemes_discussed"] = eligible_ids[:5]
            else:
                # Fallback — still need occupation at minimum
                q = ask_next_slot_question("occupation", session, session_id=sid)
                session["last_slot_question"] = q
                session["last_slot_asked"]    = "occupation"
                reply = q

        append_turn(session, message, reply)
        if session.get("turn", 0) % 4 == 0 and session.get("turn", 0) > 0:
            session["conversation_summary"] = update_conversation_summary(session, session_id=sid)
        save_session(sid, session)
        return reply, eligible_ids

    # ══════════════════════════════════════════════════════════════
    # GENERAL Q&A FLOW — outside eligibility
    # ══════════════════════════════════════════════════════════════

    # Fast gate (for yes/no outside flow — treated as general)
    fast = _fast_gate(message, session)

    if fast:
        classification = fast
    else:
        status_cb(get_status("understanding_question", lang))
        classification = classify(message, session, session_id=sid)

    # Update language
    update_language(
        session,
        classification.get("language", lang),
        classification.get("language_source", "auto_detected"),
    )
    lang = session.get("language", "hi")

    # Merge any facts extracted (e.g. user mentions their state in a question)
    if classification.get("new_facts"):
        merge_facts(session, classification["new_facts"])

    update_entity_memory(
        session,
        scheme_id=classification.get("scheme_mentioned"),
        intent=classification.get("intent"),
    )

    intent         = classification.get("intent", "scheme_query")
    resolved_query = classification.get("resolved_query", message)

    # Route
    if intent == "eligibility_check":
        # Start eligibility flow
        session["in_eligibility_flow"] = True
        session["confirmed_facts"]     = {}
        session["pending_slots"]       = list(ELIGIBILITY_SLOTS)
        session["eligibility_done"]    = False
        first_q = ask_next_slot_question("occupation", session, session_id=sid)
        session["last_slot_question"]  = first_q
        session["last_slot_asked"]     = "occupation"
        reply = first_q

    elif intent in ("scheme_query", "document_query", "how_to_apply"):
        reply = answer_scheme_query(
            resolved_query=resolved_query,
            intent=intent,
            session=session,
            status_cb=status_cb,
            session_id=sid
        )

    else:
        reply = answer_general_query(
            resolved_query=resolved_query,
            session=session,
            status_cb=status_cb,
            session_id=sid
        )

    append_turn(session, message, reply)
    if session.get("turn", 0) % 4 == 0 and session.get("turn", 0) > 0:
        session["conversation_summary"] = update_conversation_summary(session,session_id=sid)
    save_session(sid, session)
    return reply, eligible_ids


def _looks_like_question(message: str) -> bool:
    """Heuristic: did the user ask a question instead of answering one?"""
    msg = message.lower().strip()
    question_signals = [
        "?", "kya", "what", "how", "why", "when", "where", "which",
        "क्या", "कैसे", "कब", "कहाँ", "कौन", "என்ன", "எப்படி", "কি", "কেন",
        "batao", "bataiye", "tell me", "explain",
    ]
    return any(s in msg for s in question_signals)