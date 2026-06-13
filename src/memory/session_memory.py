"""
session_memory.py — 3-layer memory system.

Layer 1 — Working Memory: confirmed user facts (occupation, income, etc.)
Layer 2 — Conversation Buffer: last 8 turns of raw dialogue
Layer 3 — Entity Memory: schemes discussed, references resolved

Resets on session end (browser refresh generates new sid).
All data stored locally in SQLite at data/sessions.db
"""

import sqlite3
import json
import os
from typing import Optional
from datetime import datetime

DB_PATH = "data/sessions.db"

# Slots we collect during eligibility flow, in order
ELIGIBILITY_SLOTS = [
    "occupation",
    "gender",
    "has_land",
    "is_bpl",
    "has_girl_child",
    "is_pregnant",
    "annual_income",
    "state",
    "age"
    
]

# Slots that need yes/no button UI
BINARY_SLOTS = {"has_land", "is_bpl", "has_girl_child", "is_pregnant"}

# Slots that need occupation button UI
OCCUPATION_SLOTS = {"occupation"}


def init_db():
    """Create tables if they don't exist."""
    os.makedirs("data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id                  TEXT PRIMARY KEY,
            language            TEXT DEFAULT 'hi',
            language_source     TEXT DEFAULT 'default',
            turn                INTEGER DEFAULT 0,
            confirmed_facts     TEXT DEFAULT '{}',
            pending_slots       TEXT DEFAULT '[]',
            conversation_buffer TEXT DEFAULT '[]',
            entity_memory       TEXT DEFAULT '{}',
            eligibility_done    INTEGER DEFAULT 0,
            in_eligibility_flow INTEGER DEFAULT 0,
            conversation_summary TEXT DEFAULT '',
            last_slot_question  TEXT DEFAULT '',
            last_slot_asked     TEXT DEFAULT '',
            updated_at          TEXT DEFAULT (datetime('now'))
        )
    """)
    con.commit()
    con.close()


def _row_to_session(row) -> dict:
    """Convert SQLite row tuple to session dict."""
    if not row:
        return _empty_session()
    return {
        "language":             row[1],
        "language_source":      row[2],
        "turn":                 row[3],
        "confirmed_facts":      json.loads(row[4]),
        "pending_slots":        json.loads(row[5]),
        "conversation_buffer":  json.loads(row[6]),
        "entity_memory":        json.loads(row[7]),
        "eligibility_done":     bool(row[8]),
        "in_eligibility_flow":  bool(row[9]),
        "conversation_summary": row[10],
        "last_slot_question":   row[11] if len(row) > 11 else "",
        "last_slot_asked":      row[12] if len(row) > 12 else "",
    }


def _empty_session() -> dict:
    return {
        "language":             "hi",
        "language_source":      "default",
        "turn":                 0,
        "confirmed_facts":      {},
        "pending_slots":        list(ELIGIBILITY_SLOTS),
        "conversation_buffer":  [],
        "entity_memory":        {
            "last_schemes_discussed": [],
            "last_scheme_id":         None,
            "last_intent":            None,
            "resolved_references":    {},
        },
        "eligibility_done":     False,
        "in_eligibility_flow":  False,
        "conversation_summary": "",
        "last_slot_question":   "",   # what we last asked during eligibility flow
        "last_slot_asked":      "",   # which slot we last asked about
    }


def get_session(sid: str) -> dict:
    """Load session from DB. Returns empty session if not found."""
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    con.close()
    return _row_to_session(row)


def save_session(sid: str, session: dict):
    """Persist full session state to DB."""
    # Keep buffer to last 8 turns
    buf = session.get("conversation_buffer", [])[-8:]
    session["conversation_buffer"] = buf

    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT OR REPLACE INTO sessions
        (id, language, language_source, turn, confirmed_facts, pending_slots,
         conversation_buffer, entity_memory, eligibility_done,
         in_eligibility_flow, conversation_summary,
         last_slot_question, last_slot_asked, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        sid,
        session.get("language", "hi"),
        session.get("language_source", "default"),
        session.get("turn", 0),
        json.dumps(session.get("confirmed_facts", {}),      ensure_ascii=False),
        json.dumps(session.get("pending_slots",   []),      ensure_ascii=False),
        json.dumps(buf,                                      ensure_ascii=False),
        json.dumps(session.get("entity_memory",   {}),      ensure_ascii=False),
        int(session.get("eligibility_done",   False)),
        int(session.get("in_eligibility_flow", False)),
        session.get("conversation_summary", ""),
        session.get("last_slot_question", ""),
        session.get("last_slot_asked", ""),
    ))
    con.commit()
    con.close()


def reset_session(sid: str):
    """Delete session — called on browser reset button."""
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM sessions WHERE id=?", (sid,))
    con.commit()
    con.close()


def append_turn(session: dict, user_msg: str, bot_reply: str):
    """Add a turn to the conversation buffer."""
    session["conversation_buffer"].append({
        "role_user": user_msg,
        "role_bot":  bot_reply,
        "turn":      session.get("turn", 0),
    })
    session["conversation_buffer"] = session["conversation_buffer"][-8:]
    session["turn"] = session.get("turn", 0) + 1


def merge_facts(session: dict, new_facts: dict):
    """
    Merge newly extracted facts into confirmed_facts.
    Also recalculates pending_slots.
    """
    if not new_facts:
        return

    facts = session.setdefault("confirmed_facts", {})
    for key, val in new_facts.items():
        if val is not None and val != "":
            facts[key] = val

    # Recalculate pending slots — only slots we still need
    if session.get("in_eligibility_flow"):
        session["pending_slots"] = [
            s for s in ELIGIBILITY_SLOTS
            if s not in facts
        ]


def update_language(session: dict, lang: str, source: str = "auto_detected"):
    """
    Update language in session.
    user_selected always wins over auto_detected.
    """
    current_source = session.get("language_source", "default")
    if current_source == "user_selected" and source != "user_selected":
        return  # user explicitly chose, don't override
    session["language"] = lang
    session["language_source"] = source


def update_entity_memory(session: dict, scheme_id: Optional[str], intent: Optional[str]):
    """Update entity memory with latest scheme and intent."""
    em = session.setdefault("entity_memory", {})
    if scheme_id:
        em["last_scheme_id"] = scheme_id
        discussed = em.setdefault("last_schemes_discussed", [])
        if scheme_id not in discussed:
            discussed.insert(0, scheme_id)
        em["last_schemes_discussed"] = discussed[:5]  # keep last 5
    if intent:
        em["last_intent"] = intent


def get_history_text(session: dict, turns: int = 4) -> str:
    """Format last N turns as readable text for prompt injection."""
    buf = session.get("conversation_buffer", [])[-turns:]
    if not buf:
        return "No previous conversation."
    lines = []
    for turn in buf:
        lines.append(f"User: {turn['role_user']}")
        lines.append(f"Bot: {turn['role_bot']}")
    return "\n".join(lines)


def get_next_pending_slot(session: dict) -> Optional[str]:

    facts = session.get("confirmed_facts", {})

    for slot in ELIGIBILITY_SLOTS:

        if slot in facts:
            continue

        # Skip gender if occupation already implies female
        if slot == "gender":
            occupation = str(facts.get("occupation", "")).lower()

            if occupation in {
                
                "women hoh",
                "woman hoh",
                "woman_hoh",
                "female farmer",
                "female labourer",
                "female laborer",
                }:
                facts["gender"] = "female"
                continue

        return slot

    return None


def is_eligibility_complete(session: dict) -> bool:
    return get_next_pending_slot(session) is None


def needs_binary_buttons(session: dict) -> bool:
    """Should UI show Yes/No buttons?"""
    next_slot = get_next_pending_slot(session)
    return (
        session.get("in_eligibility_flow", False)
        and not session.get("eligibility_done", False)
        and next_slot in BINARY_SLOTS
    )


def needs_occupation_buttons(session: dict) -> bool:
    """Should UI show occupation buttons?"""
    next_slot = get_next_pending_slot(session)
    return (
        session.get("in_eligibility_flow", False)
        and not session.get("eligibility_done", False)
        and next_slot in OCCUPATION_SLOTS
    )


def session_to_profile_text(session: dict) -> str:
    """Format confirmed facts as readable profile for prompt injection."""
    facts = session.get("confirmed_facts", {})
    if not facts:
        return "No profile information collected yet."

    lines = []
    label_map = {
        "occupation":    "Occupation",
        "gender":        "Gender",
        "has_land":      "Owns agricultural land",
        "is_bpl":        "Has BPL ration card",
        "has_girl_child":"Has girl child under 10",
        "is_pregnant":   "Pregnant/recently delivered woman in household",
        "annual_income": "Annual household income",
        "state":         "State",
        "age":           "Age",
    }
    for key, label in label_map.items():
        if key in facts:
            val = facts[key]
            if isinstance(val, bool):
                val = "Yes" if val else "No"
            lines.append(f"{label}: {val}")

    return "\n".join(lines) if lines else "No profile information collected yet."