"""
cache.py — Exact-match response cache (Phase 1 of intelligent caching).

Caches deterministic-ish LLM outputs that repeat heavily across sessions:
  - slot questions          (slot_name, lang, known_fact_keys)
  - scheme/document/howto answers (resolved_query, lang, intent)
  - general answers         (resolved_query, lang)

NOT cached:
  - eligibility reasoning (depends on full per-user profile)
  - classifier output with non-empty history (context-dependent)
  - Tavily-fired scheme answers get a short TTL (live-search implies freshness)

Storage: SQLite at data/cache.db, single table `response_cache`.

Usage:
    from src.utils.cache import get_cached, set_cached, make_key

    key = make_key("scheme_answer", resolved_query, lang, intent)
    hit = get_cached(key)
    if hit is not None:
        return hit, True   # (response, was_cache_hit)

    response = <expensive call>
    set_cached(key, response, call_type="scheme_answer", lang=lang, ttl_seconds=...)
    return response, False
"""

import sqlite3
import os
import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = "data/cache.db"

# Default TTLs per call_type, in seconds
DEFAULT_TTL = {
    "slot_question":  7 * 24 * 3600,   # 7 days — wording rarely changes
    "scheme_answer":  24 * 3600,       # 24h — KB-backed answers
    "scheme_answer_live": 6 * 3600,    # 6h  — Tavily-sourced answers (freshness matters)
    "general_answer": 24 * 3600,
    "classifier_opener": 24 * 3600,    # first-turn classifier results
}


def init_cache_db():
    os.makedirs("data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS response_cache (
            cache_key   TEXT PRIMARY KEY,
            cache_type  TEXT NOT NULL,   -- 'exact' (phase 1) | 'semantic' (phase 2)
            call_type   TEXT NOT NULL,   -- slot_question / scheme_answer / general_answer / ...
            response    TEXT NOT NULL,
            lang        TEXT,
            created_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            hit_count   INTEGER DEFAULT 0
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_expires ON response_cache(expires_at)")
    con.commit()
    con.close()


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise for stable hashing."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF]", "", text)
    return text


def make_key(call_type: str, *parts) -> str:
    """
    Build a stable cache key from call_type + arbitrary parts.
    Parts are normalized (if str) and joined before hashing.
    """
    norm_parts = []
    for p in parts:
        if isinstance(p, str):
            norm_parts.append(_normalize(p))
        elif isinstance(p, (list, tuple, set)):
            norm_parts.append(",".join(sorted(str(x) for x in p)))
        else:
            norm_parts.append(str(p))
    raw = call_type + "|" + "|".join(norm_parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{call_type}:{digest[:24]}"


def get_cached(cache_key: str) -> Optional[str]:
    """
    Return cached response if present and not expired, else None.
    Increments hit_count on hit (fire-and-forget, best effort).
    """
    init_cache_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT response, expires_at FROM response_cache WHERE cache_key=?",
        (cache_key,)
    ).fetchone()

    if not row:
        con.close()
        return None

    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        # expired — clean up lazily
        con.execute("DELETE FROM response_cache WHERE cache_key=?", (cache_key,))
        con.commit()
        con.close()
        return None

    con.execute(
        "UPDATE response_cache SET hit_count = hit_count + 1 WHERE cache_key=?",
        (cache_key,)
    )
    con.commit()
    con.close()
    return row["response"]


def set_cached(
    cache_key: str,
    response: str,
    call_type: str,
    lang: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
    cache_type: str = "exact",
):
    """Store a response in the cache with TTL (defaults based on call_type)."""
    if not response or not response.strip():
        return  # never cache empty/failed responses

    init_cache_db()
    ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_TTL.get(call_type, 3600)
    now = datetime.utcnow()
    expires = now + timedelta(seconds=ttl)

    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT OR REPLACE INTO response_cache
        (cache_key, cache_type, call_type, response, lang, created_at, expires_at, hit_count)
        VALUES (?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT hit_count FROM response_cache WHERE cache_key=?), 0))
    """, (
        cache_key, cache_type, call_type, response, lang,
        now.isoformat(), expires.isoformat(), cache_key,
    ))
    con.commit()
    con.close()


def get_cache_stats() -> dict:
    """Global cache stats — size, hit counts, breakdown by call_type."""
    init_cache_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    total = con.execute("SELECT COUNT(*) AS n FROM response_cache").fetchone()["n"]
    total_hits = con.execute("SELECT COALESCE(SUM(hit_count),0) AS n FROM response_cache").fetchone()["n"]

    by_type = con.execute("""
        SELECT call_type,
               COUNT(*) AS entries,
               COALESCE(SUM(hit_count),0) AS hits
        FROM response_cache
        GROUP BY call_type
        ORDER BY hits DESC
    """).fetchall()

    con.close()
    return {
        "total_entries": total,
        "total_hits": total_hits,
        "by_call_type": [dict(r) for r in by_type],
    }


def purge_expired():
    """Remove expired entries — call periodically (e.g. on app start)."""
    init_cache_db()
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM response_cache WHERE expires_at < ?", (datetime.utcnow().isoformat(),))
    con.commit()
    con.close()