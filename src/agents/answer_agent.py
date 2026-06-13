"""
answer_agent.py — Answer generation + dedicated slot extractor.

Key fix: slot extraction is now a SEPARATE dedicated LLM call.
This eliminates the loop bug where the LLM would re-ask confirmed questions
because it conflated "classify intent" with "extract slot value".

Slot extractor: 8B, fast, single job — parse one value from user reply.
Answer generator: 70B, full context, generates final response.

CACHING (Phase 1 — exact match, see src/utils/cache.py):
  - ask_next_slot_question : cached on (slot_name, lang, known_fact_keys)
  - answer_scheme_query    : cached on (resolved_query, lang, intent)
                              — only cached when NOT served by Tavily live
                              search (or with a short TTL if it was)
  - answer_general_query   : cached on (resolved_query, lang)

All cache hits are logged via log_cache_hit() so the usage dashboard can
show hit-rate and estimated tokens/latency saved.
"""

from typing import Callable, Optional
from src.tools.scheme_search_tool  import search, format_results_for_prompt, CONF_HIGH, CONF_MEDIUM
from src.utils.tavily_search       import search_welfare_schemes, is_available
from src.utils.llm_client          import call_strong, call_fast, parse_json_response
from src.utils.prompts             import (
    SCHEME_ANSWER_SYSTEM, SCHEME_ANSWER_PROMPT,
    ELIGIBILITY_ANSWER_SYSTEM, ELIGIBILITY_ANSWER_PROMPT,
    GENERAL_ANSWER_SYSTEM, GENERAL_ANSWER_PROMPT,
    SLOT_QUESTION_PROMPT, SLOT_EXTRACTOR_SYSTEM, SLOT_EXTRACTOR_PROMPT,
    SUMMARY_PROMPT, TAVILY_BLOCK_TEMPLATE, get_status, LANGUAGE_NAMES,
)
from src.utils.language            import get_language_name
from src.memory.session_memory     import get_history_text, session_to_profile_text
import time
from src.memory.usage_logger import log_retrieval, log_tavily, log_cache_hit
from src.utils.cache import get_cached, set_cached, make_key


# ── Slot extractor — dedicated single-purpose call ───────────────────────────

def extract_slot_value(
    slot_name:       str,
    user_reply:      str,
    question_asked:  str,
    session:         dict,
    status_cb:       Callable[[str], None] = lambda x: None,
    session_id:      str = "unknown",
) -> Optional[object]:
    """
    Extract a single slot value from the user's reply.
    Dedicated 8B call — does ONE thing only.
    Returns the extracted value, or None if extraction failed.

    NOT cached: depends on free-text user_reply, which is effectively unique
    per user and not worth caching.
    """
    status_cb(get_status("extracting_slot", session.get("language", "hi")))

    raw = call_fast(
        system=SLOT_EXTRACTOR_SYSTEM,
        user=SLOT_EXTRACTOR_PROMPT.format(
            question_asked=question_asked,
            user_reply=user_reply,
            slot_name=slot_name,
        ),
        max_tokens=60,
        json_mode=True,
        session_id=session_id,
        call_type="slot_extractor",
    )

    parsed = parse_json_response(raw)
    value  = parsed.get(slot_name)

    # Convert string booleans if needed
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "1"):
            value = True
        elif value.lower() in ("false", "no", "0"):
            value = False

    return value  # None means extraction failed


# ── Slot question — ask user for next missing slot ───────────────────────────

def ask_next_slot_question(slot_name: str, session: dict, session_id: str = "unknown") -> str:
    """
    Generate a warm, conversational question for the next eligibility slot.

    CACHED: the question text depends only on (slot_name, lang, which fact
    keys are already known) — not on their values. Two users at the same
    point in the eligibility flow, in the same language, get the same
    question. Long TTL (7 days) since wording rarely needs to change.
    """
    lang      = session.get("language", "hi")
    lang_name = get_language_name(lang)
    profile   = session_to_profile_text(session)
    known_keys = sorted(session.get("confirmed_facts", {}).keys())

    cache_key = make_key("slot_question", slot_name, lang, known_keys)

    t0 = time.monotonic()
    cached = get_cached(cache_key)
    if cached is not None:
        log_cache_hit(
            session_id=session_id,
            call_type="slot_question",
            query_text=f"{slot_name}|{lang}|{known_keys}",
            latency_ms=(time.monotonic() - t0) * 1000,
        )
        return cached

    response = call_fast(
        system=f"You are a warm welfare scheme assistant. Respond in {lang_name} only.",
        user=SLOT_QUESTION_PROMPT.format(
            language=lang,
            language_name=lang_name,
            known_facts=profile,
            slot_name=slot_name,
        ),
        max_tokens=100,
        session_id=session_id,
        call_type="slot_question",
    )

    set_cached(cache_key, response, call_type="slot_question", lang=lang)
    return response


# ── Main scheme/doc/apply answer ─────────────────────────────────────────────

def answer_scheme_query(
    resolved_query: str,
    intent:         str,
    session:        dict,
    status_cb:      Callable[[str], None] = lambda x: None,
    session_id:     str = "unknown",
) -> str:
    """
    CACHED: keyed on (resolved_query, lang, intent). Since resolved_query is
    already a self-contained English query (produced by the classifier),
    identical/near-identical questions across different users and sessions
    hit the same cache entry.

    Cache TTL differs based on whether the answer leaned on live Tavily
    search results (shorter TTL — "live search" implies freshness) vs. pure
    KB (longer TTL).
    """
    lang      = session.get("language", "hi")
    lang_name = get_language_name(lang)
    profile   = session_to_profile_text(session)
    history   = get_history_text(session, turns=4)

    cache_key = make_key("scheme_answer", resolved_query, lang, intent)

    t0 = time.monotonic()
    cached = get_cached(cache_key)
    if cached is not None:
        log_cache_hit(
            session_id=session_id,
            call_type="scheme_answer",
            query_text=resolved_query,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
        return cached

    # ── Hybrid RAG search ─────────────────────────────────────────────────────
    status_cb(get_status("searching_kb", lang))
    t0 = time.monotonic()
    results, confidence = search(resolved_query, top_k=4, lang=lang)
    retrieval_ms = (time.monotonic() - t0) * 1000
    kb_text = format_results_for_prompt(results)

    log_retrieval(
        session_id=session_id,
        query_text=resolved_query,
        confidence=confidence,
        latency_ms=retrieval_ms,
    )

    # ── Tavily routing ────────────────────────────────────────────────────────
    tavily_block = ""
    tavily_used  = False

    if confidence >= CONF_HIGH:
        log_tavily(session_id=session_id, query_text=resolved_query,
                   fired=False, confidence=confidence)

    elif confidence >= CONF_MEDIUM:
        if is_available():
            status_cb(get_status("enhancing_search", lang))
            t0 = time.monotonic()
            tavily_text = search_welfare_schemes(resolved_query)
            tavily_ms = (time.monotonic() - t0) * 1000
            if tavily_text:
                tavily_block = TAVILY_BLOCK_TEMPLATE.format(tavily_results=tavily_text)
                tavily_used  = True
            log_tavily(session_id=session_id, query_text=resolved_query,
                       fired=True, confidence=confidence, latency_ms=tavily_ms)
        else:
            log_tavily(session_id=session_id, query_text=resolved_query,
                       fired=False, confidence=confidence)

    else:
        if is_available():
            status_cb(get_status("live_search", lang))
            t0 = time.monotonic()
            tavily_text = search_welfare_schemes(resolved_query)
            tavily_ms = (time.monotonic() - t0) * 1000
            if tavily_text:
                tavily_block = TAVILY_BLOCK_TEMPLATE.format(tavily_results=tavily_text)
                tavily_used  = True
            log_tavily(session_id=session_id, query_text=resolved_query,
                       fired=True, confidence=confidence, latency_ms=tavily_ms)
        else:
            log_tavily(session_id=session_id, query_text=resolved_query,
                       fired=False, confidence=confidence)

    # ── Generate answer ───────────────────────────────────────────────────────
    status_cb(get_status("generating_answer", lang))

    response = call_strong(
        system=SCHEME_ANSWER_SYSTEM.format(language_name=lang_name),
        user=SCHEME_ANSWER_PROMPT.format(
            language_name=lang_name,
            user_profile=profile,
            intent=intent,
            resolved_query=resolved_query,
            confidence=confidence,
            kb_results=kb_text,
            tavily_block=tavily_block,
            history=history,
        ),
        max_tokens=600,
        session_id=session_id,
        call_type="scheme_answer",
    )

    # Cache: shorter TTL if Tavily live-search contributed to the answer
    cache_call_type = "scheme_answer_live" if tavily_used else "scheme_answer"
    set_cached(cache_key, response, call_type=cache_call_type, lang=lang)

    return response


def answer_general_query(
    resolved_query: str,
    session:        dict,
    status_cb:      Callable[[str], None] = lambda x: None,
    session_id:     str = "unknown",
) -> str:
    """
    Answer general welfare questions. Fast 8B model, no RAG.

    CACHED: keyed on (resolved_query, lang). No profile-dependence in the
    prompt's factual content (it only echoes profile back), so identical
    general questions across users share a cache entry.
    """
    lang      = session.get("language", "hi")
    lang_name = get_language_name(lang)
    profile   = session_to_profile_text(session)
    history   = get_history_text(session, turns=3)

    cache_key = make_key("general_answer", resolved_query, lang)

    t0 = time.monotonic()
    cached = get_cached(cache_key)
    if cached is not None:
        log_cache_hit(
            session_id=session_id,
            call_type="general_answer",
            query_text=resolved_query,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
        return cached

    status_cb(get_status("generating_answer", lang))

    response = call_fast(
        system=GENERAL_ANSWER_SYSTEM.format(language_name=lang_name),
        user=GENERAL_ANSWER_PROMPT.format(
            language_name=lang_name,
            user_profile=profile,
            resolved_query=resolved_query,
            history=history,
        ),
        max_tokens=350,
        session_id=session_id,
        call_type="general_answer",
    )

    set_cached(cache_key, response, call_type="general_answer", lang=lang)
    return response


def update_conversation_summary(session: dict, session_id: str = "unknown") -> str:
    """
    Regenerate summary every 4 turns. 8B, background.

    NOT cached: history is unique per session, so a cache hit would be
    statistically near-impossible and not worth the lookup cost.
    """
    history = get_history_text(session, turns=8)
    if not history or history == "No previous conversation.":
        return ""
    return call_fast(
        system="You summarise conversations in 2 sentences. Output only the summary.",
        user=SUMMARY_PROMPT.format(history=history),
        max_tokens=120,
        session_id=session_id,
        call_type="summary"
    ).strip()