"""llm_client.py — Single NVIDIA NIM client for all models.

One API key, all models. Right model for right task:
- llama-3.1-8b  → fast classification, slot extraction, general answers
- llama-3.3-70b → complex reasoning, eligibility, scheme answers
- qwen2.5-72b   → structured data extraction fallback

All calls include timeout + retry logic.

CHANGES (traceability):
- Every _call() now logs prompt/completion/total tokens to usage_logger,
  tagged with session_id and call_type.
- call_fast/call_strong/call_struct now accept session_id + call_type
  so logging is automatic — callers just pass these two extra kwargs.
"""

import os
import time
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

from src.memory.usage_logger import log_llm_call

load_dotenv()

_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

# Model aliases
MODEL_FAST   = "meta/llama-3.1-8b-instruct"    # intent, slots, general
MODEL_STRONG = "meta/llama-3.3-70b-instruct"   # eligibility, scheme answers
MODEL_STRUCT = "qwen/qwen2.5-72b-instruct"     # structured extraction fallback

def _call(
    model, system, user,
    max_tokens=1024, temperature=0.1, json_mode=False,
    retries=2, session_id="unknown", call_type="unspecified",
) -> str:
    """
    Core LLM call with retry logic.
    json_mode=True requests JSON object output (supported by Llama 3.x on NIM).

    Logs token usage (prompt/completion/total) to usage_logger for every
    successful call, tagged with session_id and call_type.
    """
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries + 1):
        try:
            t0 = time.monotonic()                          # ← start timer
            resp = _client.chat.completions.create(**kwargs)
            latency_ms = (time.monotonic() - t0) * 1000   # ← stop timer

            usage = getattr(resp, "usage", None)
            try:
                log_llm_call(
                    session_id=session_id,
                    call_type=call_type,
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                    completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                    total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                    latency_ms=latency_ms, 
                    cache_hit=False,                      # ← always False for actual calls
                    cache_type=None,
                    query_text=user[:1000],  # log a snippet of the query for debugging (truncate if too long)  
                )
            except Exception as log_exc:
                print(f"[usage_logger ERROR] {log_exc}")

            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == retries:
                raise
            time.sleep(3 * (attempt + 1))
    return ""

def call_fast(
    system: str,
    user:   str,
    max_tokens: int = 512,
    json_mode:  bool = False,
    session_id: str = "unknown",
    call_type:  str = "unspecified",
) -> str:
    """Fast 8B model — classification, slot extraction, general chat."""
    return _call(
        MODEL_FAST, system, user, max_tokens=max_tokens, json_mode=json_mode,
        session_id=session_id, call_type=call_type,
    )


def call_strong(
    system: str,
    user:   str,
    max_tokens: int = 1570,
    json_mode:  bool = False,
    session_id: str = "unknown",
    call_type:  str = "unspecified",
) -> str:
    """Strong 70B model — eligibility reasoning, scheme answers."""
    return _call(
        MODEL_STRONG, system, user, max_tokens=max_tokens, json_mode=json_mode,
        session_id=session_id, call_type=call_type,
    )


def call_struct(
    system: str,
    user:   str,
    max_tokens: int = 2048,
    session_id: str = "unknown",
    call_type:  str = "unspecified",
) -> str:
    """Qwen 72B — structured JSON extraction fallback."""
    return _call(
        MODEL_STRUCT, system, user, max_tokens=max_tokens, json_mode=True,
        session_id=session_id, call_type=call_type,
    )


def parse_json_response(raw: str) -> dict:
    """
    Robustly parse JSON from LLM output.
    Handles: markdown fences, leading prose, trailing commas, truncation.
    """
    if not raw:
        return {}

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())

    # Find first { ... last }
    s = raw.find("{")
    e = raw.rfind("}")
    if s != -1 and e != -1:
        raw = raw[s: e + 1]

    # Strategy 1: direct
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: fix trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: try json_repair if available
    try:
        import json_repair  # type: ignore
        result = json_repair.repair_json(raw, return_objects=True)
        if isinstance(result, dict) and result:
            return result
    except (ImportError, Exception):
        pass

    return {}