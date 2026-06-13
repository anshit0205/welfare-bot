"""
usage_logger.py — Persistent traceability log.

Tracks, per session (sid), every LLM call, Tavily call, guardrail event, and
cache hit/miss:
  - timestamp
  - session_id
  - call_type (classifier / slot_extractor / scheme_answer / eligibility / general / tavily)
  - model used
  - prompt_tokens, completion_tokens, total_tokens
  - latency_ms (wall-clock duration of the call in milliseconds)
  - confidence score (for retrieval calls)
  - tavily_fired (true/false), tavily_error (if any)
  - guardrail_stage (1=regex, 2=LLM, 0=allowed), guardrail_reason, guardrail_matched
  - cache_hit (0/1), cache_type ('exact'/'semantic'/NULL)
  - query text (truncated)

Stored in SQLite at data/usage_logs.db — survives restarts, never resets.
Each row is one event. Query by sid to get full session history.

── Measuring latency (caller's responsibility) ──────────────────────────────

    import time
    t0 = time.monotonic()
    response = call_llm(...)
    latency_ms = (time.monotonic() - t0) * 1000

    log_llm_call(..., latency_ms=latency_ms)

time.monotonic() is preferred over time.time() because it is unaffected by
system-clock adjustments and never goes backwards.

── Cache events ───────────────────────────────────────────────────────────

On a CACHE HIT for what would otherwise be an LLM call, log it via
log_llm_call(..., model="cache", prompt_tokens=0, completion_tokens=0,
total_tokens=0, cache_hit=True, cache_type="exact", latency_ms=<lookup_ms>),
or use the log_cache_hit() convenience wrapper below.
This keeps token/latency dashboards meaningful (cache hits show as ~0 cost)
while still being filterable/countable via cache_hit=1.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import Optional

DB_PATH = "data/usage_logs.db"


def init_usage_db():
    os.makedirs("data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT NOT NULL,
            timestamp           TEXT NOT NULL,
            event_type          TEXT NOT NULL,   -- 'llm_call' | 'retrieval' | 'tavily' | 'guardrail'
            call_type           TEXT,            -- classifier/slot_extractor/scheme_answer/eligibility/general/summary
            model               TEXT,
            prompt_tokens       INTEGER,
            completion_tokens   INTEGER,
            total_tokens        INTEGER,
            latency_ms          REAL,            -- wall-clock duration of the call (milliseconds)
            confidence          REAL,
            tavily_fired        INTEGER,         -- 0/1
            tavily_error        TEXT,
            guardrail_allowed   INTEGER,         -- 0/1 (NULL for non-guardrail events)
            guardrail_stage     INTEGER,         -- 1=regex fast-path, 2=LLM check (NULL if allowed)
            guardrail_reason    TEXT,            -- 'offtopic' | 'injection' | 'abuse' | NULL
            guardrail_matched   TEXT,            -- the pattern or token that triggered the block
            cache_hit           INTEGER,         -- 0/1 (NULL for non-cacheable events)
            cache_type          TEXT,            -- 'exact' | 'semantic' | NULL
            query_text          TEXT,
            extra_json          TEXT
        )
    """)
    # Indices
    con.execute("CREATE INDEX IF NOT EXISTS idx_session   ON usage_events(session_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON usage_events(timestamp)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_guardrail ON usage_events(event_type, guardrail_allowed)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cache     ON usage_events(event_type, cache_hit)")
    con.commit()
    con.close()


# ── Migrate existing DBs that pre-date new columns ───────────────────────────

def _migrate_columns():
    """
    Safe, idempotent migration: adds any columns that are missing from an
    existing usage_events table.  No-ops if a column already exists
    (SQLite raises OperationalError on duplicate ADD COLUMN).
    """
    con = sqlite3.connect(DB_PATH)
    for col_def in [
        # guardrail columns (original migration)
        "ADD COLUMN guardrail_allowed  INTEGER",
        "ADD COLUMN guardrail_stage    INTEGER",
        "ADD COLUMN guardrail_reason   TEXT",
        "ADD COLUMN guardrail_matched  TEXT",
        # latency column
        "ADD COLUMN latency_ms         REAL",
        # cache columns
        "ADD COLUMN cache_hit          INTEGER",
        "ADD COLUMN cache_type         TEXT",
    ]:
        try:
            con.execute(f"ALTER TABLE usage_events {col_def}")
        except sqlite3.OperationalError:
            pass  # column already exists
    con.commit()
    con.close()


# keep the old name as an alias so any external callers don't break
_migrate_guardrail_columns = _migrate_columns


# ── Public logging helpers ────────────────────────────────────────────────────

def log_llm_call(
    session_id: str,
    call_type: str,
    model: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    latency_ms: Optional[float] = None,
    query_text: str = "",
    extra: Optional[dict] = None,
    cache_hit: Optional[bool] = None,
    cache_type: Optional[str] = None,
):
    """
    Log a single LLM call with token usage and wall-clock latency.

    Parameters
    ----------
    latency_ms : elapsed time from sending the request to receiving the full
                 response, in milliseconds.  Measure with time.monotonic():

                     t0 = time.monotonic()
                     response = call_llm(...)
                     latency_ms = (time.monotonic() - t0) * 1000

    cache_hit  : True if this "call" was actually served from cache (no LLM
                 invoked). Pass model="cache", token counts=0 in that case.
    cache_type : 'exact' | 'semantic' | None. Only meaningful when cache_hit=True.
    """
    _insert(
        session_id=session_id,
        event_type="llm_call",
        call_type=call_type,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        cache_hit=int(cache_hit) if cache_hit is not None else None,
        cache_type=cache_type,
        query_text=query_text[:300],
        extra=extra,
    )


def log_cache_hit(
    session_id: str,
    call_type: str,
    query_text: str,
    latency_ms: Optional[float] = None,
    cache_type: str = "exact",
    extra: Optional[dict] = None,
):
    """
    Convenience wrapper for the common case: a cache hit replacing what would
    have been an LLM call. Logs model="cache", all token counts=0,
    cache_hit=1, cache_type=cache_type.

    Usage
    -----
    t0 = time.monotonic()
    cached = get_cached(key)
    if cached is not None:
        log_cache_hit(session_id=sid, call_type="scheme_answer",
                       query_text=resolved_query,
                       latency_ms=(time.monotonic()-t0)*1000)
        return cached
    """
    log_llm_call(
        session_id=session_id,
        call_type=call_type,
        model="cache",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        latency_ms=latency_ms,
        query_text=query_text,
        extra=extra,
        cache_hit=True,
        cache_type=cache_type,
    )


def log_retrieval(
    session_id: str,
    query_text: str,
    confidence: float,
    latency_ms: Optional[float] = None,
    extra: Optional[dict] = None,
):
    """
    Log a KB retrieval event with its confidence score and latency.

    Parameters
    ----------
    latency_ms : time from issuing the retrieval query to receiving ranked
                 results, in milliseconds.
    """
    _insert(
        session_id=session_id,
        event_type="retrieval",
        confidence=confidence,
        latency_ms=latency_ms,
        query_text=query_text[:300],
        extra=extra,
    )


def log_tavily(
    session_id: str,
    query_text: str,
    fired: bool,
    error: Optional[str] = None,
    confidence: Optional[float] = None,
    latency_ms: Optional[float] = None,
    extra: Optional[dict] = None,
):
    """
    Log whether Tavily was fired for this query.

    fired=True, error=None    → Tavily called successfully
    fired=True, error="<msg>" → Tavily was attempted but failed
    fired=False, error=None   → Tavily not needed (confidence high enough)

    Parameters
    ----------
    latency_ms : round-trip time for the Tavily HTTP request, in milliseconds.
                 Only meaningful when fired=True; pass None when fired=False.
    """
    _insert(
        session_id=session_id,
        event_type="tavily",
        confidence=confidence,
        tavily_fired=int(fired),
        tavily_error=error,
        latency_ms=latency_ms,
        query_text=query_text[:300],
        extra=extra,
    )


def log_guardrail(
    session_id: str,
    query_text: str,
    allowed: bool,
    stage: Optional[int] = None,
    reason: Optional[str] = None,
    matched: Optional[str] = None,
    latency_ms: Optional[float] = None,
    extra: Optional[dict] = None,
):
    """
    Log a guardrail check result.

    Parameters
    ----------
    session_id  : the session this message belongs to
    query_text  : the raw user message (truncated to 300 chars)
    allowed     : True if the message passed all checks
    stage       : which stage blocked it — 1 (regex) or 2 (LLM). None if allowed.
    reason      : 'offtopic' | 'injection' | 'abuse'. None if allowed.
    matched     : the specific pattern/keyword that triggered the block, for
                  tuning false-positives later. None if allowed or Stage 2 block.
    latency_ms  : total time spent in check_message(), in milliseconds.
                  Useful for flagging slow LLM-stage checks (Stage 2).

    Usage
    -----
    Call this ONCE per incoming message, right after check_message() returns,
    regardless of whether the message was allowed or blocked.  Blocked messages
    that never reach the classifier are otherwise invisible in the logs.

    Example
    -------
    import time
    t0 = time.monotonic()
    result = check_message(user_message, session, session_id=sid)
    latency_ms = (time.monotonic() - t0) * 1000

    log_guardrail(
        session_id=sid,
        query_text=user_message,
        allowed=result.allowed,
        stage=result.stage,
        reason=result.reason if not result.allowed else None,
        matched=result.matched,
        latency_ms=latency_ms,
    )
    if not result.allowed:
        return result.message, []
    """
    _insert(
        session_id=session_id,
        event_type="guardrail",
        guardrail_allowed=int(allowed),
        guardrail_stage=stage,
        guardrail_reason=reason,
        guardrail_matched=matched,
        latency_ms=latency_ms,
        query_text=query_text[:300],
        extra=extra,
    )


# ── Internal insert ───────────────────────────────────────────────────────────

def _insert(
    session_id: str,
    event_type: str,
    call_type: Optional[str] = None,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    latency_ms: Optional[float] = None,
    confidence: Optional[float] = None,
    tavily_fired: Optional[int] = None,
    tavily_error: Optional[str] = None,
    guardrail_allowed: Optional[int] = None,
    guardrail_stage: Optional[int] = None,
    guardrail_reason: Optional[str] = None,
    guardrail_matched: Optional[str] = None,
    cache_hit: Optional[int] = None,
    cache_type: Optional[str] = None,
    query_text: str = "",
    extra: Optional[dict] = None,
):
    init_usage_db()
    _migrate_columns()  # safe no-op for existing DBs
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO usage_events
        (session_id, timestamp, event_type, call_type, model,
         prompt_tokens, completion_tokens, total_tokens,
         latency_ms,
         confidence, tavily_fired, tavily_error,
         guardrail_allowed, guardrail_stage, guardrail_reason, guardrail_matched,
         cache_hit, cache_type,
         query_text, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        datetime.utcnow().isoformat(),
        event_type,
        call_type,
        model,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        latency_ms,
        confidence,
        tavily_fired,
        tavily_error,
        guardrail_allowed,
        guardrail_stage,
        guardrail_reason,
        guardrail_matched,
        cache_hit,
        cache_type,
        query_text,
        json.dumps(extra, ensure_ascii=False) if extra else None,
    ))
    con.commit()
    con.close()


# ── Reporting helpers ─────────────────────────────────────────────────────────

def get_session_log(session_id: str) -> list[dict]:
    """Return all events for a session, oldest first."""
    init_usage_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM usage_events WHERE session_id=? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_session_token_summary(session_id: str) -> dict:
    """
    Total tokens and latency used in this session, broken down by call_type.

    Each breakdown entry includes:
      - calls              : number of LLM calls of this type
      - cache_hits         : number of those calls served from cache
      - prompt_tokens      : total prompt tokens
      - completion_tokens  : total completion tokens
      - total_tokens       : total tokens
      - avg_latency_ms     : average latency across calls that reported it
      - p95_latency_ms     : 95th-percentile latency (None if < 20 samples)
    """
    init_usage_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    # Aggregate totals per call_type
    rows = con.execute("""
        SELECT call_type,
               COUNT(*)                                        AS calls,
               COALESCE(SUM(cache_hit), 0)                    AS cache_hits,
               SUM(prompt_tokens)                             AS prompt_tokens,
               SUM(completion_tokens)                         AS completion_tokens,
               SUM(total_tokens)                              AS total_tokens,
               AVG(CASE WHEN latency_ms IS NOT NULL
                        THEN latency_ms END)                  AS avg_latency_ms,
               COUNT(CASE WHEN latency_ms IS NOT NULL
                          THEN 1 END)                         AS latency_samples
        FROM usage_events
        WHERE session_id=? AND event_type='llm_call'
        GROUP BY call_type
    """, (session_id,)).fetchall()

    # p95 requires the raw latency values — fetch once per call_type that has
    # enough samples to make a percentile meaningful (≥ 20 rows).
    breakdown = []
    for r in rows:
        entry = dict(r)
        entry["avg_latency_ms"] = (
            round(entry["avg_latency_ms"], 2)
            if entry["avg_latency_ms"] is not None else None
        )
        if entry["latency_samples"] >= 20:
            latencies = [
                row[0] for row in con.execute("""
                    SELECT latency_ms
                    FROM usage_events
                    WHERE session_id=? AND event_type='llm_call'
                      AND call_type=? AND latency_ms IS NOT NULL
                    ORDER BY latency_ms ASC
                """, (session_id, entry["call_type"])).fetchall()
            ]
            idx = int(len(latencies) * 0.95)
            entry["p95_latency_ms"] = round(latencies[min(idx, len(latencies) - 1)], 2)
        else:
            entry["p95_latency_ms"] = None
        breakdown.append(entry)

    con.close()

    grand_total = sum(r["total_tokens"] or 0 for r in breakdown)
    return {"breakdown": breakdown, "grand_total_tokens": grand_total}


def get_tavily_summary(session_id: str) -> list[dict]:
    """All Tavily events for this session — fired/not fired/errors/latency."""
    init_usage_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT timestamp, query_text, confidence,
               tavily_fired, tavily_error, latency_ms
        FROM usage_events
        WHERE session_id=? AND event_type='tavily'
        ORDER BY id ASC
    """, (session_id,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_guardrail_summary(session_id: Optional[str] = None) -> dict:
    """
    Guardrail block statistics.

    If session_id is given → stats for that session only.
    If session_id is None  → global stats across all sessions (useful for
                             tuning: which patterns fire most often?).

    Returns
    -------
    {
        "total_checks": int,
        "blocked": int,
        "allowed": int,
        "block_rate_pct": float,
        "by_reason": {"offtopic": int, "injection": int, "abuse": int},
        "by_stage":  {1: int, 2: int},
        "top_matched_patterns": [{"pattern": str, "count": int}, ...],  # top 10
        "latency": {
            "avg_ms":      float | None,   # average check duration
            "avg_stage1_ms": float | None, # avg when regex fast-path fired
            "avg_stage2_ms": float | None, # avg when LLM check was needed
        }
    }
    """
    init_usage_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    where = "WHERE event_type='guardrail'"
    params: tuple = ()
    if session_id:
        where += " AND session_id=?"
        params = (session_id,)

    total = con.execute(
        f"SELECT COUNT(*) as n FROM usage_events {where}", params
    ).fetchone()["n"]

    blocked = con.execute(
        f"SELECT COUNT(*) as n FROM usage_events {where} AND guardrail_allowed=0",
        params
    ).fetchone()["n"]

    by_reason_rows = con.execute(
        f"""SELECT guardrail_reason, COUNT(*) as n
            FROM usage_events {where} AND guardrail_allowed=0
            GROUP BY guardrail_reason""",
        params
    ).fetchall()

    by_stage_rows = con.execute(
        f"""SELECT guardrail_stage, COUNT(*) as n
            FROM usage_events {where} AND guardrail_allowed=0
            GROUP BY guardrail_stage""",
        params
    ).fetchall()

    top_patterns_rows = con.execute(
        f"""SELECT guardrail_matched, COUNT(*) as n
            FROM usage_events {where} AND guardrail_allowed=0
                AND guardrail_matched IS NOT NULL
            GROUP BY guardrail_matched
            ORDER BY n DESC
            LIMIT 10""",
        params
    ).fetchall()

    # Latency breakdown — overall and split by stage
    latency_row = con.execute(
        f"""SELECT
                AVG(latency_ms)                                            AS avg_ms,
                AVG(CASE WHEN guardrail_stage=1 THEN latency_ms END)      AS avg_stage1_ms,
                AVG(CASE WHEN guardrail_stage=2 THEN latency_ms END)      AS avg_stage2_ms
            FROM usage_events {where} AND latency_ms IS NOT NULL""",
        params
    ).fetchone()

    con.close()

    def _round(v):
        return round(v, 2) if v is not None else None

    allowed = total - blocked
    return {
        "total_checks": total,
        "blocked": blocked,
        "allowed": allowed,
        "block_rate_pct": round(blocked / total * 100, 1) if total else 0.0,
        "by_reason": {r["guardrail_reason"]: r["n"] for r in by_reason_rows},
        "by_stage":  {r["guardrail_stage"]:  r["n"] for r in by_stage_rows},
        "top_matched_patterns": [
            {"pattern": r["guardrail_matched"], "count": r["n"]}
            for r in top_patterns_rows
        ],
        "latency": {
            "avg_ms":        _round(latency_row["avg_ms"]),
            "avg_stage1_ms": _round(latency_row["avg_stage1_ms"]),
            "avg_stage2_ms": _round(latency_row["avg_stage2_ms"]),
        },
    }


def get_cache_summary(session_id: Optional[str] = None) -> dict:
    """
    Cache hit-rate stats for llm_call events.

    If session_id is given → stats for that session only.
    If session_id is None  → global stats across all sessions.

    Returns
    -------
    {
        "total_llm_calls": int,
        "cache_hits": int,
        "hit_rate_pct": float,
        "by_call_type": [
            {"call_type": str, "calls": int, "cache_hits": int, "hit_rate_pct": float,
             "tokens_saved": int, "avg_latency_cached_ms": float|None,
             "avg_latency_uncached_ms": float|None},
            ...
        ],
        "estimated_tokens_saved": int
    }
    """
    init_usage_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    where = "WHERE event_type='llm_call'"
    params: tuple = ()
    if session_id:
        where += " AND session_id=?"
        params = (session_id,)

    total = con.execute(f"SELECT COUNT(*) as n FROM usage_events {where}", params).fetchone()["n"]
    hits = con.execute(
        f"SELECT COUNT(*) as n FROM usage_events {where} AND cache_hit=1", params
    ).fetchone()["n"]

    rows = con.execute(f"""
        SELECT call_type,
               COUNT(*) AS calls,
               COALESCE(SUM(cache_hit), 0) AS cache_hits,
               AVG(CASE WHEN cache_hit=1 THEN latency_ms END) AS avg_latency_cached_ms,
               AVG(CASE WHEN (cache_hit IS NULL OR cache_hit=0) THEN latency_ms END) AS avg_latency_uncached_ms,
               AVG(CASE WHEN (cache_hit IS NULL OR cache_hit=0) THEN total_tokens END) AS avg_tokens_uncached
        FROM usage_events {where}
        GROUP BY call_type
    """, params).fetchall()

    by_call_type = []
    estimated_tokens_saved = 0
    for r in rows:
        entry = dict(r)
        avg_tokens = entry.pop("avg_tokens_uncached") or 0
        tokens_saved = int(round(avg_tokens * entry["cache_hits"]))
        estimated_tokens_saved += tokens_saved
        entry["tokens_saved"] = tokens_saved
        entry["hit_rate_pct"] = round(entry["cache_hits"] / entry["calls"] * 100, 1) if entry["calls"] else 0.0
        for k in ("avg_latency_cached_ms", "avg_latency_uncached_ms"):
            entry[k] = round(entry[k], 2) if entry[k] is not None else None
        by_call_type.append(entry)

    con.close()

    return {
        "total_llm_calls": total,
        "cache_hits": hits,
        "hit_rate_pct": round(hits / total * 100, 1) if total else 0.0,
        "by_call_type": by_call_type,
        "estimated_tokens_saved": estimated_tokens_saved,
    }


def get_all_sessions(limit: int = 50) -> list[dict]:
    """List recent distinct sessions with first/last activity timestamps."""
    init_usage_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT session_id,
               MIN(timestamp) as first_seen,
               MAX(timestamp) as last_seen,
               COUNT(*) as events
        FROM usage_events
        GROUP BY session_id
        ORDER BY last_seen DESC
        LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]