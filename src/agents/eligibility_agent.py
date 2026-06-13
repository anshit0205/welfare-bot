"""
eligibility_agent.py — LLM-based eligibility reasoning.

Replaces brittle rule-based boolean matching with 70B LLM reasoning
over actual scheme JSON eligibility_rules.

The LLM reads the real rules and reasons naturally:
- "farmer who also does daily labour" → qualifies for both
- "woman head of household" → qualifies for women-only schemes
- Income edge cases handled gracefully
"""

import glob
import json
from src.utils.llm_client    import call_strong, parse_json_response
from src.utils.prompts       import ELIGIBILITY_REASONING_PROMPT, ELIGIBILITY_ANSWER_PROMPT
from src.utils.language      import get_language_name
from src.memory.session_memory import get_history_text, session_to_profile_text


def run_eligibility_check(session: dict) -> tuple[str, list[str]]:
    """
    Run full eligibility check for the user.

    Returns:
        answer:       formatted answer string in user's language
        eligible_ids: list of scheme IDs user qualifies for
    """
    profile      = session_to_profile_text(session)
    facts        = session.get("confirmed_facts", {})
    lang         = session.get("language", "hi")
    lang_name    = get_language_name(lang)
    history_text = get_history_text(session, turns=3)

    # ── Load all scheme JSON files ────────────────────────────────────────────
    schemes = []
    for jf in sorted(glob.glob("data/schemes/*.json")):
        try:
            s = json.load(open(jf, encoding="utf-8"))
            # Only pass eligibility-relevant fields to keep prompt lean
            schemes.append({
                "id":                 s.get("id"),
                "name_en":            s.get("name_en"),
                "name_hi":            s.get("name_hi"),
                "benefit_en":         s.get("benefit_en"),
                "eligibility_rules":  s.get("eligibility_rules", {}),
                "exclusion_criteria": s.get("exclusion_criteria", []),
                "documents_en":       s.get("documents_en", [])[:5],
                "application_url":    s.get("application_url"),
                "helpline":           s.get("helpline"),
            })
        except Exception:
            continue

    if not schemes:
        return _no_schemes_response(lang_name, lang), []

    # ── 70B eligibility reasoning ─────────────────────────────────────────────
    reasoning_raw = call_strong(
        system=(
            "You are an expert in Indian government welfare scheme eligibility. "
            "Return ONLY valid JSON. No markdown. No explanation. Start with { end with }."
        ),
        user=ELIGIBILITY_REASONING_PROMPT.format(
            user_profile=profile,
            schemes_json=json.dumps(schemes, ensure_ascii=False, indent=1),
        ),
        max_tokens=3000,
        json_mode=True,
    )

    reasoning = parse_json_response(reasoning_raw)
    eligible   = reasoning.get("eligible",   [])
    ineligible = reasoning.get("ineligible", [])

    eligible_ids = [s["id"] for s in eligible if s.get("id")]

    # ── 70B answer generation in user's language ──────────────────────────────
    eligible_text   = _format_eligible(eligible)
    ineligible_text = _format_ineligible(ineligible[:5])  # top 5 only

    answer = call_strong(
        system=(
            f"You are a helpful Indian welfare scheme advisor. "
            f"Respond entirely in {lang_name}. "
            f"Use simple language. Be warm and encouraging."
        ),
        user=ELIGIBILITY_ANSWER_PROMPT.format(
            user_profile=profile,
            eligible_schemes=eligible_text,
            ineligible_schemes=ineligible_text,
            language=lang,
            language_name=lang_name,
            history=history_text,
        ),
        max_tokens=1500,
    )

    return answer, eligible_ids


def _format_eligible(eligible: list) -> str:
    if not eligible:
        return "None"
    blocks = []
    for s in eligible:
        docs = ", ".join(s.get("documents", [])[:4])
        blocks.append(
            f"SCHEME: {s.get('name_en','')}\n"
            f"BENEFIT: {s.get('benefit_en','')}\n"
            f"WHY ELIGIBLE: {s.get('reason','')}\n"
            f"KEY DOCUMENTS: {docs}\n"
            f"URL: {s.get('url','')}\n"
            f"HELPLINE: {s.get('helpline','')}"
        )
    return "\n\n".join(blocks)


def _format_ineligible(ineligible: list) -> str:
    if not ineligible:
        return "None"
    return "\n".join(
        f"- {s.get('name_en','')}: {s.get('reason','')}"
        for s in ineligible
    )


def _no_schemes_response(lang_name: str, lang: str) -> str:
    messages = {
        "hi": "माफ़ करें, अभी योजना डेटाबेस उपलब्ध नहीं है। कृपया बाद में प्रयास करें।",
        "en": "Sorry, the scheme database is currently unavailable. Please try again later.",
        "mr": "माफ करा, योजना डेटाबेस उपलब्ध नाही. कृपया नंतर प्रयत्न करा.",
        "bn": "দুঃখিত, স্কিম ডেটাবেস এখন পাওয়া যাচ্ছে না। পরে আবার চেষ্টা করুন।",
        "ta": "மன்னிக்கவும், திட்ட தரவுத்தளம் கிடைக்கவில்லை. பிறகு முயற்சிக்கவும்.",
    }
    return messages.get(lang, messages["en"])