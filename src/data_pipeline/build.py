"""build_data.py — v11
==================
Changes from v10:
  FIX 10 — Translation split into two sequential calls (A + B) to stay well
            under token ceiling and eliminate mid-stream ServerDisconnected errors.
            Trans-A (name/description/benefit): max_tokens=2200
            Trans-B (documents/apply_steps):    max_tokens=2800
  FIX 11 — FAQ split into two sequential calls (A + B):
            FAQ-A  (Q&A with hi+ta translations): max_tokens=2200
            FAQ-B  (mistakes/tips/state_wise):    max_tokens=1800
  FIX 12 — Logging: Phase-2 and Phase-3 log lines now print actual max_tokens
            values per sub-call instead of the old stale hardcoded string.

Architecture:
  SCRAPING   : Playwright (async, headless Chromium) — single browser, parallel contexts
  SEARCH     : Tavily advanced deep search — max_results=8 per query
  LLM        : NVIDIA AsyncOpenAI (NIM) — single shared client, round-robin llama-3.3-70b pool
  DECOMPOSED : 5 sequential sub-tasks per scheme:
                 Core → Trans-A → Trans-B → FAQ-A → FAQ-B
  CONCURRENCY: 1 scheme at a time (safe for 40 req/min budget)

Sub-task call graph per scheme:
  Phase 1: [Core-English]   — full English extraction,           max_tokens=6500
  Phase 2a:[Trans-A]        — name/description/benefit ×4 langs, max_tokens=2200
  Phase 2b:[Trans-B]        — documents/apply_steps   ×4 langs,  max_tokens=2800
  Phase 3a:[FAQ-A]          — Q&A (8 items) with hi+ta,          max_tokens=2200
  Phase 3b:[FAQ-B]          — mistakes/tips/state_wise,           max_tokens=1800

Rate-limit math (40 req/min budget):
  1 scheme × 5 sequential calls = max 1 simultaneous LLM call at any time.
  Well within limits with room for retries.

Install:
  pip install playwright tavily-python openai httpx python-dotenv tqdm json-repair
  playwright install chromium

.env:
  NVIDIA_API_KEY=...
  TAVILY_API_KEY=...
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
from playwright.async_api import async_playwright, Browser, BrowserContext
from tavily import TavilyClient
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE GLOBAL CLIENT  ← FIX 1 (v10)
# ══════════════════════════════════════════════════════════════════════════════

def _make_client() -> AsyncOpenAI:
    timeout = httpx.Timeout(
        connect=45.0,
        read=1600.0,
        write=60.0,
        pool=62.0,
    )
    return AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        timeout=timeout,
        max_retries=0,
    )

CLIENT: AsyncOpenAI = _make_client()


# ══════════════════════════════════════════════════════════════════════════════
# ROUND-ROBIN LLM POOL
# ══════════════════════════════════════════════════════════════════════════════

_POOL_SIZE = 2

PRIMARY_MODEL   = "meta/llama-3.3-70b-instruct"
FALLBACK_MODELS = [
    "meta/llama-3.1-70b-instruct",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "microsoft/phi-4-mini-instruct",
]

_slot_sem:      list[asyncio.Semaphore] = []
_slot_failures: list[int]              = [0] * _POOL_SIZE
_slot_cooldown: list[float]            = [0.0] * _POOL_SIZE

_fallback_failures: dict[str, int]   = defaultdict(int)
_fallback_cooldown: dict[str, float] = {}

_rr_counter = 0


def _next_slot() -> int:
    global _rr_counter
    now = time.time()
    for _ in range(_POOL_SIZE):
        idx = _rr_counter % _POOL_SIZE
        _rr_counter += 1
        if _slot_cooldown[idx] <= now:
            return idx
    return min(range(_POOL_SIZE), key=lambda i: _slot_cooldown[i])


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

Path("data").mkdir(exist_ok=True)
_fmt    = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
_file_h = logging.FileHandler("data/scrape.log", encoding="utf-8")
_file_h.setFormatter(_fmt)
_tqdm_h = TqdmLoggingHandler()
_tqdm_h.setFormatter(_fmt)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(_file_h)
log.addHandler(_tqdm_h)
log.propagate = False


# ══════════════════════════════════════════════════════════════════════════════
# PLAYWRIGHT SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

_SCRAPE_CONCURRENCY = 4
_scrape_sem: asyncio.Semaphore | None = None

PLAYWRIGHT_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-blink-features=AutomationControlled",
    "--disable-software-rasterizer",
]

STEALTH_JS = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'plugins',   { get: () => [1,2,3,4,5] });
    window.chrome = { runtime: {} };
"""

_CONTENT_SELECTORS = ["main", "article", "#content", ".content", ".scheme-detail", "body"]
CONTENT_SELECTORS  = ", ".join(_CONTENT_SELECTORS)


def _html_to_text(html: str) -> str:
    html = re.sub(
        r"<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>",
        " ", html, flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s{2,}", " ", text).strip()


# ── Per-domain timeout overrides ──────────────────────────────────────────────
# Sites known to be slow, JS-heavy, or running legacy IIS/NIC stacks get extra
# time on the first Playwright attempt.  All other sites use DEFAULT_TIMEOUT_MS.
DEFAULT_TIMEOUT_MS = 25_000
_DOMAIN_TIMEOUTS: dict[str, int] = {
    "nabard.org":       45_000,   # ASPX/IIS — slow server + heavy JS
    "nsdl.co.in":       40_000,   # NSDL financial portal — bot-resistant
    "indiapost.gov.in": 40_000,   # India Post SPA — deferred JS rendering
    "wcd.nic.in":       35_000,   # NIC legacy CMS — variable response time
    "nrega.nic.in":     35_000,   # NIC legacy — intermittently slow
    "nsap.nic.in":      35_000,   # NIC legacy — intermittently slow
    "yet.nta.ac.in":    35_000,   # NTA exam portal — aggressive bot checks
    "mohua.gov.in":     35_000,   # Ministry portal — variable uptime
    "mnre.gov.in":      35_000,   # Ministry portal — variable uptime
}

def _timeout_for(url: str) -> int:
    """Return the appropriate Playwright timeout for a given URL."""
    from urllib.parse import urlparse
    netloc = urlparse(url).netloc.lower()
    # Strip exactly the 'www.' prefix if present — lstrip() is wrong here
    # because it strips any combination of characters, not the literal prefix.
    host = netloc[4:] if netloc.startswith("www.") else netloc
    for domain, ms in _DOMAIN_TIMEOUTS.items():
        if host == domain or host.endswith("." + domain):
            return ms
    return DEFAULT_TIMEOUT_MS


async def _httpx_fallback(url: str, label: str) -> str:
    """
    Plain HTTP GET with httpx — no JavaScript execution.
    Used as last resort when Playwright times out twice.
    Works well for static/server-rendered pages (nabard.org ASPX, NIC pages).
    Tavily deep search already covers JS-rendered content, so data loss is minimal.
    """
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=5.0),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
            },
            follow_redirects=True,
            verify=False,
        ) as client:
            resp = await client.get(url)
            if resp.status_code >= 400:
                log.warning(f"  {label}: httpx fallback HTTP {resp.status_code}")
                return ""
            text = _html_to_text(resp.text)[:20_000]
            log.info(f"  {label}: httpx fallback OK — {len(text):,} chars")
            return text
    except Exception as exc:
        log.warning(f"  {label}: httpx fallback failed — {type(exc).__name__}: {exc}")
        return ""


async def _playwright_get(
    browser: Browser,
    url: str,
    label: str,
    timeout_ms: int,
    wait_until: str = "domcontentloaded",
) -> str:
    """
    Single Playwright page load attempt.  Raises on timeout so the caller can
    decide whether to retry with a different strategy.
    Returns extracted text on success, "" on HTTP error.
    """
    ctx: BrowserContext | None = None
    try:
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        page = await ctx.new_page()
        await page.add_init_script(STEALTH_JS)
        await page.route(
            "**/*",
            lambda route: (
                route.abort()
                if route.request.resource_type in {"image", "font", "media", "stylesheet"}
                else route.continue_()
            ),
        )
        response = await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
        if response and response.status >= 400:
            log.warning(f"  {label}: HTTP {response.status}")
            return ""
        if wait_until != "commit":
            # Only wait for networkidle on full-load strategies; skip for commit
            try:
                await page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass
        body_html = ""
        for selector in _CONTENT_SELECTORS:
            try:
                el = page.locator(selector).first
                if await el.count() > 0:
                    html = await el.inner_html(timeout=3_000)
                    if len(html) > 500:
                        body_html = html
                        break
            except Exception:
                continue
        if not body_html:
            body_html = await page.content()
        return _html_to_text(body_html)[:20_000]
    finally:
        if ctx:
            try:
                await ctx.close()
            except Exception:
                pass


async def scrape_url_playwright(
    browser: Browser,
    url: str,
    label: str,
    timeout_ms: int | None = None,   # None → auto from _DOMAIN_TIMEOUTS
) -> str:
    """
    Three-tier scraping strategy — never hard-fails on a single slow site:

    Tier 1 — Playwright, wait_until=domcontentloaded, domain-aware timeout.
              Handles the majority of sites cleanly.

    Tier 2 — Playwright, wait_until=commit (first byte only), +10s timeout.
              Catches sites where domcontentloaded fires late or not at all
              (e.g. nabard.org ASPX, NIC CMS pages).  Extracts whatever HTML
              the server has sent by the time the connection is established.

    Tier 3 — httpx plain GET, no JavaScript.
              Last resort for completely Playwright-resistant servers.
              Works fine for server-rendered pages; Tavily covers JS content.
    """
    global _scrape_sem
    async with _scrape_sem:

        effective_timeout = timeout_ms if timeout_ms is not None else _timeout_for(url)

        # ── Tier 1: normal domcontentloaded ───────────────────────────────
        log.info(f"  {label}: loading {url[:70]}... (timeout={effective_timeout//1000}s)")
        try:
            text = await _playwright_get(browser, url, label, effective_timeout, "domcontentloaded")
            if text:
                log.info(f"  {label}: {len(text):,} chars")
                return text
            # Empty but no exception means HTTP error was already logged
            return ""
        except Exception as exc1:
            log.warning(
                f"  {label}: Tier-1 failed ({type(exc1).__name__}) "
                f"— retrying with wait_until=commit"
            )

        # ── Tier 2: commit (first-byte) — bypasses slow domcontentloaded ──
        commit_timeout = effective_timeout + 10_000
        try:
            text = await _playwright_get(browser, url, label, commit_timeout, "commit")
            if text:
                log.info(f"  {label}: {len(text):,} chars (commit-mode)")
                return text
            return ""
        except Exception as exc2:
            log.warning(
                f"  {label}: Tier-2 failed ({type(exc2).__name__}) "
                f"— falling back to httpx"
            )

        # ── Tier 3: httpx plain GET — no JS, server-rendered HTML only ────
        return await _httpx_fallback(url, label)


# ══════════════════════════════════════════════════════════════════════════════
# TAVILY DEEP SEARCH
# ══════════════════════════════════════════════════════════════════════════════

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

TAVILY_QUERIES: dict[str, list[str]] = {
    "pm_kisan": [
        "PM Kisan Samman Nidhi eligibility criteria documents required 2024 2025",
        "PM Kisan Rs 6000 three installments dates schedule helpline 155261",
        "PM Kisan excluded ineligible farmers income tax payer government employee",
        "PM Kisan eKYC Aadhaar linking process pmkisan.gov.in 2025",
        "PM Kisan application rejected reasons appeal process corrections",
        "PM Kisan new farmer registration state portals documents list",
    ],
    "pm_fasal_bima": [
        "PMFBY crop insurance eligibility premium 2024-25 farmers",
        "PMFBY premium rate state-wise kharif rabi 2024-25",
        "PMFBY claim process documents flood drought natural disaster",
        "PMFBY enrollment deadline CSC banks last date 2024",
    ],
    "kisan_credit_card": [
        "Kisan Credit Card KCC eligibility documents 2024 2025",
        "KCC loan limit interest rate 4% subvention subsidy 3 lakh",
        "KCC apply online PM Kisan beneficiaries fishermen SHG process",
        "Kisan Credit Card ATM withdrawal insurance crop cattle documents",
    ],
    "pm_kusum": [
        "PM KUSUM solar pump subsidy eligibility farmers 2024 components",
        "PM KUSUM component A B C solar plant income 60% subsidy",
        "PM KUSUM application state portal documents required 2024",
    ],
    "nrega": [
        "MGNREGA NREGA eligibility job card documents 2024 2025 gram panchayat",
        "MGNREGA state-wise wage rate 2024-25 all major states",
        "MGNREGA 100 days employment guarantee unemployment allowance rights",
        "MGNREGA payment DBT Aadhaar grievance complaint ombudsman",
    ],
    "ayushman_bharat": [
        "Ayushman Bharat PMJAY eligibility SECC 2024 2025 beneficiary",
        "Ayushman Bharat Rs 5 lakh coverage hospitals procedures list",
        "Ayushman card apply CSC documents Aadhaar ration card process",
        "PMJAY excluded diseases helpline 14555 grievance cashless",
    ],
    "pm_jan_arogya": [
        "AB PM-JAY SEHAT Jammu Kashmir eligibility 2024 2025",
        "SEHAT scheme Rs 5 lakh hospital list documents J&K beneficiary",
    ],
    "pmay_gramin": [
        "PMAY Gramin eligibility SECC BPL homeless 2024 2025",
        "PMAY Gramin Rs 1.20 lakh plain Rs 1.30 lakh hilly installments",
        "PMAY Gramin AwaasSoft portal documents gram sabha selection",
    ],
    "pmay_urban": [
        "PMAY Urban eligibility EWS LIG MIG income 2024 2025",
        "PMAY Urban CLSS credit linked subsidy components BLC AHP",
        "PMAY Urban apply pmaymis.gov.in documents process",
    ],
    "pm_matru_vandana": [
        "PMMVY Rs 5000 eligibility pregnant women installments 2024 2025",
        "PMMVY documents Aadhaar MCP card apply Anganwadi CDPO",
        "PMMVY second child girl Rs 6000 helpline 7998799804",
    ],
    "sukanya_samriddhi": [
        "Sukanya Samriddhi interest rate 8.2% 2024-25 details",
        "SSY account open post office bank documents 80C tax benefit",
        "SSY maturity 21 years withdrawal 18 marriage education closure",
    ],
    "beti_bachao": [
        "Beti Bachao Beti Padhao eligibility benefits 2024 2025",
        "BBBP Sukanya Samriddhi Anganwadi school district implementation",
    ],
    "ujjwala": [
        "PM Ujjwala 2.0 eligibility BPL ration card documents 2024 2025",
        "Ujjwala free LPG connection Rs 1600 waiver apply CSC",
        "Ujjwala 2.0 migrants self-declaration refill subsidy DBT PAHAL",
    ],
    "atal_pension": [
        "Atal Pension Yojana eligibility 18-40 contribution chart 2024",
        "APY Rs 1000 Rs 5000 pension co-contribution bank enrollment",
        "APY nominee corpus exit rules income tax excluded documents",
    ],
    "pm_jeevan_jyoti": [
        "PMJJBY Rs 330 premium Rs 2 lakh death coverage 2024 2025",
        "PMJJBY eligibility 18-50 claim process death certificate nominee",
    ],
    "pm_suraksha_bima": [
        "PMSBY Rs 20 premium Rs 2 lakh accidental death 2024 2025",
        "PMSBY eligibility 18-70 claim FIR hospital report documents",
    ],
    "nsap_old_age": [
        "IGNOAPS old age pension 60 years BPL Rs 200 Rs 500 2024 2025",
        "NSAP pension state top-up Maharashtra UP Tamil Nadu apply",
    ],
    "nsap_widow": [
        "IGNWPS widow pension BPL Rs 300 documents apply 2024 2025",
        "Widow pension state top-up death certificate Aadhaar bank",
    ],
    "nsap_disability": [
        "IGNDPS disability pension 80% BPL Rs 300 documents 2024 2025",
        "Disability pension state top-up certificate apply process",
    ],
    "nsp_scholarship": [
        "NSP scholarship eligibility income Rs 2.5 lakh 2024-25 apply",
        "NSP amount pre-matric post-matric SC ST OBC minority documents",
        "NSP scholarships.gov.in deadline Aadhaar DBT bank renewal",
    ],
    "pm_yasasvi": [
        "PM YASASVI OBC EBC DNT eligibility Rs 75000 Rs 125000 2024-25",
        "YASASVI NTA exam date yet.nta.ac.in documents process",
    ],
    "pm_jan_dhan": [
        "PM Jan Dhan zero balance account RuPay Rs 2 lakh insurance 2024",
        "PMJDY Rs 10000 overdraft DBT documents apply reactivation",
    ],
    "pm_mudra": [
        "PM MUDRA Shishu Kishore Tarun Rs 50000 Rs 5L Rs 10L 2024 2025",
        "MUDRA loan eligibility documents apply bank NBFC collateral free",
    ],
    "pm_svanidhi": [
        "PM SVANidhi street vendor Rs 10000 Rs 20000 Rs 50000 2024 2025",
        "SVANidhi eligibility vendor certificate apply digital payment cashback",
    ],
    "e_shram": [
        "e-Shram registration eligibility unorganised workers Rs 2 lakh 2024",
        "e-Shram eshram.gov.in Aadhaar mobile OTP documents BOCW benefits",
    ],
}


def _tavily_one_query(query: str) -> tuple[str, str]:
    for attempt in range(3):
        try:
            res = tavily.search(
                query=query,
                max_results=8,
                search_depth="advanced",
            )
            combined = "\n".join(
                f"[{item.get('title', '')}] {item.get('content', '')[:1200]}"
                for item in res.get("results", [])
            )
            return query, combined
        except Exception as exc:
            if attempt == 2:
                log.warning(f"  Tavily '{query[:40]}' failed: {exc}")
            time.sleep(3 * (attempt + 1))
    return query, ""


async def search_tavily_deep(sid: str) -> str:
    queries = TAVILY_QUERIES.get(sid, [
        f"{sid.replace('_', ' ')} India government scheme eligibility documents 2025",
        f"{sid.replace('_', ' ')} benefit amount apply process helpline official",
    ])
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, _tavily_one_query, q) for q in queries]
    pairs = await asyncio.gather(*tasks)
    supp = "\n\n".join(
        f"=== Query: {q} ===\n{text}"
        for q, text in pairs if text
    )
    log.info(f"  [{sid}] Tavily: {len(supp):,} chars from {len(queries)} queries")
    return supp


# ══════════════════════════════════════════════════════════════════════════════
# NVIDIA ASYNC LLM — ROUND-ROBIN POOL
# ══════════════════════════════════════════════════════════════════════════════

async def _call_slot(
    slot_idx: int,
    prompt_text: str,
    max_tokens: int,
    task_label: str,
) -> str:
    for attempt in range(5):
        try:
            resp = await CLIENT.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system",  "content": SYSTEM_PROMPT},
                    {"role": "user",    "content": prompt_text},
                ],
                temperature=0.05,
                top_p=0.92,
                stream=False,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
            result = resp.choices[0].message.content.strip()
            _slot_failures[slot_idx] = 0
            log.info(f"  slot[{slot_idx}] {task_label}: OK {len(result):,} chars")
            return result

        except RateLimitError:
            wait = 25 + (attempt * 10)
            log.warning(f"  slot[{slot_idx}] {task_label}: rate-limited, wait {wait}s")
            await asyncio.sleep(wait)

        except APITimeoutError as e:
            wait = min(10 * (attempt + 1), 90)
            log.warning(
                f"  slot[{slot_idx}] {task_label}: timeout attempt {attempt+1}, "
                f"retry in {wait}s"
            )
            log.warning(f"  FULL ERROR: {repr(e)}")
            if e.__cause__:
                log.warning(f"  CAUSE: {repr(e.__cause__)}")
            await asyncio.sleep(wait)
            continue

        except APIConnectionError as e:
            wait = min(10 * (attempt + 1), 90)
            log.warning(
                f"  slot[{slot_idx}] {task_label}: connection error attempt {attempt+1}, "
                f"retry in {wait}s"
            )
            log.warning(f"  FULL ERROR: {repr(e)}")
            if e.__cause__:
                log.warning(f"  CAUSE: {repr(e.__cause__)}")
            await asyncio.sleep(wait)
            continue

        except InternalServerError:
            wait = 20
            log.warning(f"  slot[{slot_idx}] {task_label}: server error, wait {wait}s")
            await asyncio.sleep(wait)

        except Exception as e:
            log.warning(f"  slot[{slot_idx}] {task_label}: {type(e).__name__}: {e}")
            await asyncio.sleep(8)

        _slot_failures[slot_idx] += 1

    if _slot_failures[slot_idx] >= 3:
        _slot_cooldown[slot_idx] = time.time() + 600
        log.warning(f"  slot[{slot_idx}] disabled for 10 min after 3 model failures")
    raise RuntimeError(f"slot[{slot_idx}] exhausted 5 attempts for {task_label}")


async def call_llm(
    prompt_text: str,
    max_tokens: int,
    task_label: str = "llm",
) -> str:
    now = time.time()
    all_cooling = all(_slot_cooldown[i] > now for i in range(_POOL_SIZE))

    if not all_cooling:
        slot_idx = _next_slot()
        async with _slot_sem[slot_idx]:
            try:
                return await _call_slot(slot_idx, prompt_text, max_tokens, task_label)
            except RuntimeError:
                pass

    log.warning(f"  All llama-3.3 slots unavailable, trying fallback models")
    for model in FALLBACK_MODELS:
        if _fallback_cooldown.get(model, 0) > time.time():
            continue
        for attempt in range(4):
            try:
                resp = await CLIENT.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt_text},
                    ],
                    temperature=0.05,
                    top_p=0.92,
                    stream=False,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                )
                result = resp.choices[0].message.content.strip()
                _fallback_failures[model] = 0
                log.info(f"  fallback [{model}] {task_label}: OK {len(result):,} chars")
                return result
            except (RateLimitError, APIConnectionError, InternalServerError) as e:
                wait = min(10 * (attempt + 1), 60)
                log.warning(
                    f"  fallback [{model}] attempt {attempt+1}: "
                    f"{type(e).__name__} — retry in {wait}s"
                )
                await asyncio.sleep(wait)
            except Exception as e:
                log.warning(f"  fallback [{model}]: {type(e).__name__}: {e}")
                await asyncio.sleep(5)
        _fallback_failures[model] += 1
        if _fallback_failures[model] >= 3:
            _fallback_cooldown[model] = time.time() + 600

    raise RuntimeError(f"All models unavailable for task: {task_label}")


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a senior policy analyst and multilingual data engineer specializing in
Indian government welfare schemes. Your task is to extract COMPLETE, ACCURATE,
DETAILED structured data from raw web content.

RULES:
1. Output ONLY valid raw JSON — no markdown fences, no prose, no explanation.
2. Start with { and end with }. Every field must be present.
3. Use null only when data is genuinely unavailable after exhausting all sources.
4. Never invent amounts, dates, or eligibility rules — only extract what is stated.
5. Amounts must include exact figures: Rs 6000, Rs 1.20 lakh, Rs 5 lakh, etc.
6. Lists must have MINIMUM required items — do not truncate.
7. Complete every array fully — never leave a list half-finished."""


# ─── PROMPT 1: Core English extraction ───────────────────────────────────────
# max_tokens: 6500

CORE_ENGLISH_PROMPT = """\
TASK: Extract a COMPLETE, production-quality English knowledge base entry for
Indian government scheme id="{sid}".

Think step by step before writing JSON:
  Step 1 — Identify scheme purpose, ministry, launch year, and scale.
  Step 2 — Extract EVERY eligibility rule including all exclusions (min 6 exclusions).
  Step 3 — List EVERY benefit with exact rupee amounts and payment schedule.
  Step 4 — List EVERY required document with its official name (minimum 10).
  Step 5 — Write detailed step-by-step application process (minimum 10 steps).
  Step 6 — Extract payment mode, frequency, schedule, processing time.
  Step 7 — Extract all URLs, helplines, renewal details, related schemes.
  Step 8 — Write 25+ diverse search keywords covering Hindi name variants too.

Return this EXACT JSON (all English fields only — translations come in next pass):
{{
  "id": "{sid}",

  "name_en": "Full official English name",

  "description_en": "6-8 sentences: objective, target beneficiaries, implementing ministry, launch year, geographic coverage, scale (crore beneficiaries), funding pattern, and key impact metric.",

  "benefit_en": "Exhaustive paragraph covering: (1) primary financial benefit with exact rupee amount and payment schedule, (2) secondary/non-monetary benefits, (3) linked scheme benefits and convergence, (4) top-up benefits where applicable, (5) insurance coverage if any, (6) subsidy details.",

  "eligibility_rules": {{
    "occupation": ["farmer", "agricultural_labourer", "rural_household", "or other applicable categories — list ALL"],
    "gender": "any OR male OR female",
    "bpl_required": false,
    "land_ownership_required": false,
    "girl_child_required": false,
    "pregnant_woman_only": false,
    "age_min": null,
    "age_max": null,
    "income_max_annual": null,
    "caste_preference": ["any OR sc/st/obc/minority/dntebc — list ALL applicable"],
    "bank_account_required": true,
    "aadhaar_required": true,
    "other_conditions": [
      "Every additional condition explicitly stated — land holding limits, number of children, residence requirements, employment status, institutional landholding exclusions, constitutional post exclusions, etc. Minimum 5 conditions."
    ]
  }},

  "exclusion_criteria": [
    "Income tax payees (previous assessment year)",
    "Constitutional post holders (President, VP, Ministers, MPs, MLAs, etc.)",
    "Serving/retired Central & State government employees (except Group D/MTS)",
    "Government pensioners drawing Rs 10,000+/month",
    "All professionals registered with professional bodies (doctors, lawyers, CAs, architects, engineers)",
    "Any other explicitly stated exclusion from the scheme guidelines — list minimum 6"
  ],

  "documents_en": [
    "01. Aadhaar Card — mandatory for eKYC and DBT",
    "02. Land ownership documents / Khasra-Khatauni / Patta",
    "03. Bank passbook or cancelled cheque — for DBT linkage",
    "04. [Continue listing ALL required documents with their official names and purpose — minimum 10 documents]"
  ],

  "apply_steps_en": [
    "Step 01: [Portal/office name] — [detailed action with URL if applicable]",
    "Step 02: [Document preparation details]",
    "Step 03: [Registration/form filling details]",
    "Step 04: [Verification process — who verifies, timeline]",
    "Step 05: [Field officer visit or physical verification if applicable]",
    "Step 06: [Approval process — committee/authority that approves]",
    "Step 07: [DBT linkage / bank seeding step]",
    "Step 08: [First disbursement — trigger and timeline]",
    "Step 09: [Status tracking — portal URL, SMS, IVRS]",
    "Step 10: [Grievance redressal — portal, helpline, escalation path]",
    "[Add further steps specific to this scheme if applicable]"
  ],

  "application_url":    "https://official-portal",
  "helpline":           "Helpline number(s) with name",
  "ministry":           "Full official ministry name",
  "nodal_department":   "Implementing department if different",
  "scheme_type":        "central",
  "launch_year":        2000,
  "last_updated":       "2024",
  "processing_time":    "Estimated days from application to first benefit",
  "renewal_required":   false,
  "renewal_frequency":  null,
  "renewal_process":    "Details if renewal applies",

  "payment_mode":       "DBT / direct bank transfer / cheque / in-kind",
  "payment_frequency":  "annual / quarterly / monthly / one-time",
  "payment_schedule":   "Specific dates, triggers, or installment breakdown",

  "related_schemes":    ["List all convergence schemes, linked schemes, complementary programmes"],
  "tags":               ["farmer", "women", "pension", "insurance", "etc — minimum 6"],
  "source_url":         "https://primary-source-url",
  "keywords": [
    "25+ search terms covering: English scheme name, Hindi scheme name transliterated,",
    "common abbreviations, eligibility terms, benefit amounts, target beneficiary types,",
    "common questions like 'how to apply', 'documents needed', 'amount received'"
  ]
}}

=== SOURCE 1: myscheme.gov.in ===
{myscheme_text}

=== SOURCE 2: Ministry / Official Website ===
{ministry_text}

=== SOURCE 3: Tavily Deep Search Results ===
{tavily_text}

CRITICAL: Be exhaustive. Every list must be complete. Every amount must be exact.
Incomplete data directly harms beneficiaries who rely on this information."""


# ─── PROMPT 2a: Translation A — name / description / benefit ─────────────────
# FIX 10: Split translations into two calls.
# Trans-A handles only the prose fields (short-to-medium output).
# Estimated output: 4 langs × (name ~10 + description ~200 + benefit ~150) = ~1440 tokens.
# max_tokens=2200 gives ~50% headroom — no truncation risk.

TRANSLATION_A_PROMPT = """\
TASK: Translate ONLY the name, description, and benefit of this Indian government
scheme into Hindi, Tamil, Bengali, and Marathi.

Translate precisely — do not add, remove, or change any facts.
Use formal register appropriate for government communications.
Transliterate scheme names where standard (e.g. PM Kisan → पीएम किसान).
Keep all rupee amounts, portal names, and abbreviations intact.

Return this EXACT JSON (prose fields only):
{{
  "name_hi": "पूर्ण आधिकारिक हिंदी नाम",
  "name_ta": "முழு அதிகாரப்பூர்வ தமிழ் பெயர்",
  "name_bn": "সম্পূর্ণ সরকারি বাংলা নাম",
  "name_mr": "पूर्ण अधिकृत मराठी नाव",

  "description_hi": "मूल अंग्रेजी विवरण का पूरा हिंदी अनुवाद — सभी वाक्य",
  "description_ta": "மூல ஆங்கில விளக்கத்தின் முழுமையான தமிழ் மொழிபெயர்ப்பு",
  "description_bn": "মূল ইংরেজি বিবরণের সম্পূর্ণ বাংলা অনুবাদ",
  "description_mr": "मूळ इंग्रजी वर्णनाचे संपूर्ण मराठी भाषांतर",

  "benefit_hi": "लाभ का पूरा हिंदी अनुवाद — सभी राशियाँ सहित",
  "benefit_ta": "நன்மைகளின் முழுமையான தமிழ் மொழிபெயர்ப்பு",
  "benefit_bn": "সুবিধার সম্পূর্ণ বাংলা অনুবাদ",
  "benefit_mr": "लाभांचे संपूर्ण मराठी भाषांतर"
}}

=== ENGLISH SOURCE ===
Scheme name: {name_en}

Description:
{description_en}

Benefit:
{benefit_en}"""


# ─── PROMPT 2b: Translation B — documents / apply_steps ──────────────────────
# FIX 10: List fields are the heaviest part (10+ items × 4 langs each).
# Estimated output: 4 langs × (10 docs ~120 + 10 steps ~200) = ~1280 tokens.
# max_tokens=2800 gives ~55% headroom — safe even for 15-item lists.

TRANSLATION_B_PROMPT = """\
TASK: Translate ONLY the documents list and application steps of this Indian
government scheme into Hindi, Tamil, Bengali, and Marathi.

Translate precisely — do not add, remove, or change any facts.
Translate EVERY item in both lists — never truncate arrays.
Keep all portal names, official document names, and rupee amounts intact.
Use formal register appropriate for government communications.

Return this EXACT JSON (list fields only):
{{
  "documents_hi": ["प्रत्येक दस्तावेज़ का हिंदी अनुवाद — सभी आइटम"],
  "documents_ta": ["ஒவ்வொரு ஆவணத்தின் தமிழ் மொழிபெயர்ப்பு"],
  "documents_bn": ["প্রতিটি নথির বাংলা অনুবাদ"],
  "documents_mr": ["प्रत्येक कागदपत्राचे मराठी भाषांतर"],

  "apply_steps_hi": ["प्रत्येक चरण का हिंदी अनुवाद — सभी चरण"],
  "apply_steps_ta": ["ஒவ்வொரு படியின் தமிழ் மொழிபெயர்ப்பு"],
  "apply_steps_bn": ["প্রতিটি পদক্ষেপের বাংলা অনুবাদ"],
  "apply_steps_mr": ["प्रत्येक पायरीचे मराठी भाषांतर"]
}}

=== ENGLISH SOURCE ===
Scheme name: {name_en}

Documents list ({doc_count} items):
{documents_en_str}

Application steps ({step_count} items):
{apply_steps_en_str}"""


# ─── PROMPT 3a: FAQ-A — Q&A with hi + ta translations ────────────────────────
# FIX 11: Split FAQ into two calls.
# FAQ-A handles only the 8 Q&A items with Hindi+Tamil translations.
# Estimated output: 8 items × (en Q+A ~80 + hi Q+A ~90 + ta Q+A ~90) = ~2080 tokens.
# max_tokens=2200 gives adequate headroom without the state_wise overhead.

FAQ_A_PROMPT = """\
TASK: Generate 8 FAQ items for "{scheme_name}" (id: {sid}) covering the
most common real-world questions from beneficiaries.

Think step by step:
  Step 1 — Identify 8 common questions covering: eligibility edge cases,
            document pitfalls, payment delays, portal/login issues,
            eKYC failures, rejection reasons, appeal process, benefit amounts.
  Step 2 — Write precise 2-3 sentence answers with exact amounts/dates/portal names.
  Step 3 — Translate each question AND answer into Hindi and Tamil accurately.

Return this EXACT JSON:
{{
  "faq": [
    {{
      "question_en": "Specific question a beneficiary would search for",
      "answer_en":   "Precise 2-3 sentence answer with exact amounts/dates/portal names",
      "question_hi": "हिंदी में प्रश्न",
      "answer_hi":   "हिंदी में उत्तर — सटीक राशि और तारीख सहित",
      "question_ta": "தமிழில் கேள்வி",
      "answer_ta":   "தமிழில் பதில் — சரியான தொகை மற்றும் தேதி உடன்"
    }}
  ]
}}

The faq array MUST contain exactly 8 items. No more, no less.

=== Tavily Deep Search Results (eligibility, payment, portal details) ===
{tavily_text}

=== Core Scheme Data (for context) ===
Name: {scheme_name}
Benefits: {benefit_en}
Eligibility summary: {eligibility_summary}"""


# ─── PROMPT 3b: FAQ-B — mistakes / tips / state_wise ─────────────────────────
# FIX 11: FAQ-B has no translations — only English structured data.
# Estimated output: mistakes(6) + tips(4) + state_wise(18 states) ≈ 900 tokens.
# max_tokens=1800 is very comfortable; keeps this call fast and reliable.

FAQ_B_PROMPT = """\
TASK: Extract common mistakes, pro tips, and state-wise variations for
"{scheme_name}" (id: {sid}).

Think step by step:
  Step 1 — Identify 6 common mistakes that cause rejection or payment failure.
  Step 2 — Write 4 actionable pro tips to maximise benefit / speed up application.
  Step 3 — Identify state-specific benefit top-ups or implementation differences
            for the 18 major states listed below.

Return this EXACT JSON:
{{
  "common_mistakes": [
    "Mistake 1: [What goes wrong] — [How to avoid it specifically]",
    "Mistake 2: ...",
    "Mistake 3: ...",
    "Mistake 4: ...",
    "Mistake 5: ...",
    "Mistake 6: ..."
  ],

  "pro_tips": [
    "Tip 1: Specific actionable advice to speed up application or maximise benefit",
    "Tip 2: ...",
    "Tip 3: ...",
    "Tip 4: ..."
  ],

  "state_wise_amounts": {{
    "note": "What specifically varies by state",
    "varies_by": "amount / eligibility / wage_rate / none",
    "states": {{
      "andhra_pradesh": "state top-up or additional benefit or 'same as central'",
      "assam":          "...",
      "bihar":          "...",
      "gujarat":        "...",
      "haryana":        "...",
      "jharkhand":      "...",
      "karnataka":      "...",
      "kerala":         "...",
      "madhya_pradesh": "...",
      "maharashtra":    "...",
      "odisha":         "...",
      "punjab":         "...",
      "rajasthan":      "...",
      "tamil_nadu":     "...",
      "telangana":      "...",
      "uttar_pradesh":  "...",
      "uttarakhand":    "...",
      "west_bengal":    "..."
    }}
  }},

  "state_specific_portals": {{
    "note": "State-level application portals if different from central portal",
    "portals": {{}}
  }}
}}

=== Tavily Deep Search Results ===
{tavily_text}

=== Core Scheme Data (for context) ===
Name: {scheme_name}
Benefits: {benefit_en}
Eligibility summary: {eligibility_summary}"""


# ══════════════════════════════════════════════════════════════════════════════
# JSON REPAIR
# ══════════════════════════════════════════════════════════════════════════════

def _close_open_json(s: str) -> str:
    stack, in_string, escape = [], False, False
    for ch in s:
        if escape:                   escape = False; continue
        if ch == "\\" and in_string: escape = True;  continue
        if ch == '"':                in_string = not in_string; continue
        if not in_string:
            if ch in "{[":    stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack and stack[-1] == ch: stack.pop()
    if in_string: s += '"'
    return s + "".join(reversed(stack))


def repair_and_parse_json(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    s = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())
    a, b = s.find("{"), s.rfind("}")
    s = s[a:b+1].strip() if a != -1 and b != -1 else s.strip()

    for fn in [
        lambda x: json.loads(x),
        lambda x: json.loads(re.sub(r",\s*([}\]])", r"\1", x)),
        lambda x: __import__("json_repair").repair_json(x, return_objects=True),
        lambda x: json.loads(_close_open_json(x)),
        lambda x: json.loads(re.sub(r",\s*([}\]])", r"\1", _close_open_json(x))),
    ]:
        try:
            r = fn(s)
            if isinstance(r, dict) and r:
                return r
        except Exception:
            pass
    raise ValueError("JSON repair exhausted all 5 strategies")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEME CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════

SCHEMES: dict[str, tuple[str, str]] = {
    "pm_kisan":         ("https://myscheme.gov.in/schemes/pmkisan",        "https://pmkisan.gov.in"),
    "pm_fasal_bima":    ("https://myscheme.gov.in/schemes/pmfby",          "https://pmfby.gov.in"),
    "kisan_credit_card":("https://myscheme.gov.in/schemes/kcc",            "https://www.nabard.org/content1.aspx?id=572"),
    "pm_kusum":         ("https://myscheme.gov.in/schemes/pmkusum",        "https://mnre.gov.in/solar/schemes"),
    "nrega":            ("https://myscheme.gov.in/schemes/mgnregs",        "https://nrega.nic.in"),
    "ayushman_bharat":  ("https://myscheme.gov.in/schemes/pmjay",          "https://pmjay.gov.in"),
    "pm_jan_arogya":    ("https://myscheme.gov.in/schemes/ab-pmjay-sehat", "https://pmjay.gov.in/sehat"),
    "pmay_gramin":      ("https://myscheme.gov.in/schemes/pmayg",          "https://pmayg.nic.in"),
    "pmay_urban":       ("https://myscheme.gov.in/schemes/pmay-u",         "https://pmaymis.gov.in"),
    "pm_matru_vandana": ("https://myscheme.gov.in/schemes/pmmvy",          "https://wcd.nic.in/schemes/pradhan-mantri-matru-vandana-yojana"),
    "sukanya_samriddhi":("https://myscheme.gov.in/schemes/ssy",            "https://www.indiapost.gov.in/Financial/Pages/Content/Sukanya-Samriddhi-Account.aspx"),
    "beti_bachao":      ("https://myscheme.gov.in/schemes/bbbp",           "https://wcd.nic.in/bbbp-schemes"),
    "ujjwala":          ("https://myscheme.gov.in/schemes/pmuy",           "https://www.pmuy.gov.in"),
    "atal_pension":     ("https://myscheme.gov.in/schemes/apyvp",          "https://www.npscra.nsdl.co.in/scheme-details.php"),
    "pm_jeevan_jyoti":  ("https://myscheme.gov.in/schemes/pmjjby",         "https://jansuraksha.gov.in"),
    "pm_suraksha_bima": ("https://myscheme.gov.in/schemes/pmsby",          "https://jansuraksha.gov.in"),
    "nsap_old_age":     ("https://myscheme.gov.in/schemes/ignoaps",        "https://nsap.nic.in"),
    "nsap_widow":       ("https://myscheme.gov.in/schemes/ignwps",         "https://nsap.nic.in"),
    "nsap_disability":  ("https://myscheme.gov.in/schemes/igndps",         "https://nsap.nic.in"),
    "nsp_scholarship":  ("https://myscheme.gov.in/schemes/nsp",            "https://scholarships.gov.in"),
    "pm_yasasvi":       ("https://myscheme.gov.in/schemes/pmyasasvi",      "https://yet.nta.ac.in"),
    "pm_jan_dhan":      ("https://myscheme.gov.in/schemes/pmjdy",          "https://pmjdy.gov.in"),
    "pm_mudra":         ("https://myscheme.gov.in/schemes/mudra",          "https://www.mudra.org.in"),
    "pm_svanidhi":      ("https://myscheme.gov.in/schemes/pmsvanidhi",     "https://pmsvanidhi.mohua.gov.in"),
    "e_shram":          ("https://myscheme.gov.in/schemes/eshram",         "https://eshram.gov.in"),
}


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_scheme(data: dict) -> list[str]:
    issues = []
    if len(data.get("documents_en",   [])) < 8:   issues.append("documents_en < 8")
    if len(data.get("apply_steps_en", [])) < 8:   issues.append("apply_steps_en < 8")
    if not data.get("helpline"):                   issues.append("helpline missing")
    if len(data.get("benefit_en",     "")) < 100:  issues.append("benefit_en too short")
    if not data.get("name_hi"):                    issues.append("name_hi missing")
    if not data.get("name_ta"):                    issues.append("name_ta missing")
    if len(data.get("description_en", "")) < 150:  issues.append("description_en too short")
    if not data.get("exclusion_criteria"):         issues.append("exclusion_criteria missing")
    if len(data.get("keywords",       [])) < 15:   issues.append("keywords < 15")
    if not data.get("eligibility_rules"):          issues.append("eligibility_rules missing")
    if not data.get("faq"):                        issues.append("faq missing")
    if len(data.get("faq",            [])) < 5:    issues.append("faq < 5 items")
    if not data.get("documents_hi"):               issues.append("documents_hi missing")
    if not data.get("apply_steps_hi"):             issues.append("apply_steps_hi missing")
    return issues


# ══════════════════════════════════════════════════════════════════════════════
# PER-SCHEME ASYNC WORKER — 5-phase sequential pipeline
# ══════════════════════════════════════════════════════════════════════════════

SCHEME_TIMEOUT = 2400   # 40 min ceiling


async def _process_one_scheme(
    browser: Browser,
    sid: str,
    myscheme_url: str,
    ministry_url: str,
) -> dict:

    log.info(f"\n{'─'*60}\n  Processing: {sid}")

    # ── PHASE 0: I/O (scraping + Tavily) — fully parallel ─────────────────
    myscheme_text, ministry_text, supp = await asyncio.gather(
        scrape_url_playwright(browser, myscheme_url, "myscheme.gov.in"),
        scrape_url_playwright(browser, ministry_url,  "Ministry site"),
        search_tavily_deep(sid),
    )

    Path(f"data/raw/{sid}_myscheme.txt").write_text(myscheme_text or "EMPTY", encoding="utf-8")
    Path(f"data/raw/{sid}_ministry.txt").write_text(ministry_text or "EMPTY", encoding="utf-8")
    Path(f"data/raw/{sid}_supp.txt").write_text(supp or "EMPTY", encoding="utf-8")

    if not myscheme_text and not ministry_text and not supp:
        raise RuntimeError(f"[{sid}] Zero data from all three sources")

    text_src = (myscheme_text or "")[:7_000]
    min_src  = (ministry_text or "")[:6_000]
    supp_src = supp[:6_500]

    # ── PHASE 1: Core English extraction ──────────────────────────────────
    # FIX 12: log actual max_tokens value
    _core_tokens = 6500
    log.info(f"  [{sid}] Phase 1: Core English extraction (max_tokens={_core_tokens})...")
    raw_core = await asyncio.wait_for(
        call_llm(
            CORE_ENGLISH_PROMPT.format(
                sid=sid,
                myscheme_text=text_src or "Not available",
                ministry_text=min_src  or "Not available",
                tavily_text=supp_src   or "Not available",
            ),
            max_tokens=_core_tokens,
            task_label=f"{sid}/core",
        ),
        timeout=900,
    )
    Path(f"data/raw/{sid}_llm_core.txt").write_text(raw_core, encoding="utf-8")
    data = repair_and_parse_json(raw_core)
    data["id"] = sid
    log.info(f"  [{sid}] Core parsed: {len(data.get('documents_en',[]))} docs, "
             f"{len(data.get('apply_steps_en',[]))} steps")

    # FIX 4 (v10): Only check English-only fields for core retry
    core_issues = [
        w for w in validate_scheme(data)
        if (
            ("documents_en" in w)
            or ("apply_steps_en" in w)
            or ("benefit_en" in w)
            or ("eligibility" in w)
        )
    ]
    if core_issues:
        log.warning(f"  [{sid}] Core thin: {core_issues} — retrying core once")
        try:
            raw_core2 = await asyncio.wait_for(
                call_llm(
                    CORE_ENGLISH_PROMPT.format(
                        sid=sid,
                        myscheme_text=text_src or "Not available",
                        ministry_text=min_src  or "Not available",
                        tavily_text=supp_src   or "Not available",
                    ) + "\n\nIMPORTANT: Previous extraction was incomplete. "
                        "documents_en must have 10+ items. apply_steps_en must have 10+ steps. "
                        "Expand every list fully. Do not truncate any array.",
                    max_tokens=_core_tokens,
                    task_label=f"{sid}/core-retry",
                ),
                timeout=900,
            )
            data2 = repair_and_parse_json(raw_core2)
            data2["id"] = sid
            for field in ("documents_en", "apply_steps_en", "exclusion_criteria",
                          "keywords", "tags", "related_schemes"):
                if len(data2.get(field, [])) > len(data.get(field, [])):
                    data[field] = data2[field]
            for field in ("benefit_en", "description_en"):
                if len(str(data2.get(field, ""))) > len(str(data.get(field, ""))):
                    data[field] = data2[field]
            if data2.get("eligibility_rules"):
                data["eligibility_rules"] = data2["eligibility_rules"]
            log.info(f"  [{sid}] Core retry merged successfully")
        except Exception as exc:
            log.warning(f"  [{sid}] Core retry failed: {exc} — continuing with original")

    # ── PHASE 2a: Translation A — name / description / benefit ────────────
    # FIX 10: Prose fields only; well within max_tokens=2200.
    _trans_a_tokens = 2200
    log.info(f"  [{sid}] Phase 2a: Trans-A name/desc/benefit (max_tokens={_trans_a_tokens})...")
    try:
        raw_trans_a = await asyncio.wait_for(
            call_llm(
                TRANSLATION_A_PROMPT.format(
                    name_en=data.get("name_en", sid),
                    description_en=data.get("description_en", ""),
                    benefit_en=data.get("benefit_en", ""),
                ),
                max_tokens=_trans_a_tokens,
                task_label=f"{sid}/trans-a",
            ),
            timeout=480,
        )
        Path(f"data/raw/{sid}_llm_trans_a.txt").write_text(raw_trans_a, encoding="utf-8")
        trans_a = repair_and_parse_json(raw_trans_a)
        lang_a_keys = [k for k in trans_a if any(
            k.endswith(f"_{lang}") for lang in ("hi", "ta", "bn", "mr")
        )]
        log.info(f"  [{sid}] Trans-A: {len(lang_a_keys)} fields "
                 f"(name×4, description×4, benefit×4)")
        data.update(trans_a)
    except asyncio.TimeoutError:
        log.warning(f"  [{sid}] Trans-A timed out — skipping")
    except Exception as exc:
        log.warning(f"  [{sid}] Trans-A failed: {exc}")

    # ── PHASE 2b: Translation B — documents / apply_steps ─────────────────
    # FIX 10: List fields only; max_tokens=2800 handles even 15-item lists.
    _trans_b_tokens = 2800
    docs_str  = "\n".join(data.get("documents_en",   []))
    steps_str = "\n".join(data.get("apply_steps_en", []))
    doc_count  = len(data.get("documents_en",   []))
    step_count = len(data.get("apply_steps_en", []))

    log.info(f"  [{sid}] Phase 2b: Trans-B docs({doc_count})/steps({step_count}) "
             f"(max_tokens={_trans_b_tokens})...")
    try:
        raw_trans_b = await asyncio.wait_for(
            call_llm(
                TRANSLATION_B_PROMPT.format(
                    name_en=data.get("name_en", sid),
                    documents_en_str=docs_str,
                    apply_steps_en_str=steps_str,
                    doc_count=doc_count,
                    step_count=step_count,
                ),
                max_tokens=_trans_b_tokens,
                task_label=f"{sid}/trans-b",
            ),
            timeout=600,
        )
        Path(f"data/raw/{sid}_llm_trans_b.txt").write_text(raw_trans_b, encoding="utf-8")
        trans_b = repair_and_parse_json(raw_trans_b)
        lang_b_keys = [k for k in trans_b if any(
            k.endswith(f"_{lang}") for lang in ("hi", "ta", "bn", "mr")
        )]
        log.info(f"  [{sid}] Trans-B: {len(lang_b_keys)} fields "
                 f"(documents×4, apply_steps×4)")
        data.update(trans_b)
    except asyncio.TimeoutError:
        log.warning(f"  [{sid}] Trans-B timed out — skipping")
    except Exception as exc:
        log.warning(f"  [{sid}] Trans-B failed: {exc}")

    # ── PHASE 3a: FAQ-A — Q&A with hi + ta translations ───────────────────
    # FIX 11: Q&A translation is heavy; isolated to its own call.
    _faq_a_tokens = 2200
    eligibility_summary = json.dumps(
        data.get("eligibility_rules", {}), ensure_ascii=False
    )[:800]

    log.info(f"  [{sid}] Phase 3a: FAQ-A Q&A×8 with hi+ta (max_tokens={_faq_a_tokens})...")
    faq_data: list = []
    try:
        raw_faq_a = await asyncio.wait_for(
            call_llm(
                FAQ_A_PROMPT.format(
                    scheme_name=data.get("name_en", sid),
                    sid=sid,
                    tavily_text=supp_src[:4_000],
                    benefit_en=data.get("benefit_en", "")[:600],
                    eligibility_summary=eligibility_summary,
                ),
                max_tokens=_faq_a_tokens,
                task_label=f"{sid}/faq-a",
            ),
            timeout=540,
        )
        Path(f"data/raw/{sid}_llm_faq_a.txt").write_text(raw_faq_a, encoding="utf-8")
        parsed_faq_a = repair_and_parse_json(raw_faq_a)
        faq_data = parsed_faq_a.get("faq", [])
        log.info(f"  [{sid}] FAQ-A: {len(faq_data)} Q&A items with hi+ta")
    except asyncio.TimeoutError:
        log.warning(f"  [{sid}] FAQ-A timed out — skipping")
    except Exception as exc:
        log.warning(f"  [{sid}] FAQ-A failed: {exc}")

    if faq_data:
        data["faq"] = faq_data

    # ── PHASE 3b: FAQ-B — mistakes / tips / state_wise ────────────────────
    # FIX 11: No translations here — fast and reliable.
    _faq_b_tokens = 1800
    log.info(f"  [{sid}] Phase 3b: FAQ-B mistakes/tips/state (max_tokens={_faq_b_tokens})...")
    try:
        raw_faq_b = await asyncio.wait_for(
            call_llm(
                FAQ_B_PROMPT.format(
                    scheme_name=data.get("name_en", sid),
                    sid=sid,
                    tavily_text=supp_src[:4_000],
                    benefit_en=data.get("benefit_en", "")[:600],
                    eligibility_summary=eligibility_summary,
                ),
                max_tokens=_faq_b_tokens,
                task_label=f"{sid}/faq-b",
            ),
            timeout=420,
        )
        Path(f"data/raw/{sid}_llm_faq_b.txt").write_text(raw_faq_b, encoding="utf-8")
        parsed_faq_b = repair_and_parse_json(raw_faq_b)
        merged_b = {k: parsed_faq_b[k] for k in (
            "common_mistakes", "pro_tips",
            "state_wise_amounts", "state_specific_portals"
        ) if parsed_faq_b.get(k)}
        log.info(f"  [{sid}] FAQ-B: mistakes={len(parsed_faq_b.get('common_mistakes', []))} "
                 f"tips={len(parsed_faq_b.get('pro_tips', []))} "
                 f"states={len(parsed_faq_b.get('state_wise_amounts', {}).get('states', {}))}")
        data.update(merged_b)
    except asyncio.TimeoutError:
        log.warning(f"  [{sid}] FAQ-B timed out — skipping")
    except Exception as exc:
        log.warning(f"  [{sid}] FAQ-B failed: {exc}")

    # ── Final validation report ────────────────────────────────────────────
    final_issues = validate_scheme(data)
    if final_issues:
        for issue in final_issues:
            log.warning(f"  [{sid}] FINAL QUALITY: {issue}")
    else:
        log.info(f"  [{sid}] ✓ All quality checks passed")

    stats = (
        f"docs={len(data.get('documents_en', []))} "
        f"steps={len(data.get('apply_steps_en', []))} "
        f"faq={len(data.get('faq', []))} "
        f"keywords={len(data.get('keywords', []))} "
        f"langs={'hi' if data.get('name_hi') else '-'}"
        f"{'ta' if data.get('name_ta') else '-'}"
        f"{'bn' if data.get('name_bn') else '-'}"
        f"{'mr' if data.get('name_mr') else '-'}"
    )
    log.info(f"  [{sid}] COMPLETE — {stats}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ASYNC PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

_SCHEME_CONCURRENCY = 1
_scheme_sem: asyncio.Semaphore | None = None


async def _process_with_sem(
    browser: Browser,
    sid: str,
    myscheme_url: str,
    ministry_url: str,
    pbar: tqdm,
    results: dict,
):
    global _scheme_sem
    async with _scheme_sem:
        out_path = Path(f"data/schemes2/{sid}.json")
        try:
            data = await asyncio.wait_for(
                _process_one_scheme(browser, sid, myscheme_url, ministry_url),
                timeout=SCHEME_TIMEOUT,
            )
            out_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            results["success"].append(sid)
        except asyncio.TimeoutError:
            log.error(f"  [{sid}] timed out after {SCHEME_TIMEOUT}s")
            results["failed"].append(sid)
        except Exception as exc:
            log.error(f"  [{sid}] FAILED: {exc}")
            Path(f"data/raw/{sid}_FAILED.txt").write_text(str(exc), encoding="utf-8")
            results["failed"].append(sid)
        finally:
            pbar.update(1)
            pbar.set_postfix_str(
                f"✓{len(results['success'])} ✗{len(results['failed'])}"
            )


async def _run_async(force: bool = False, schemes_filter: Optional[list] = None):
    global _scrape_sem, _scheme_sem, _slot_sem

    _scrape_sem = asyncio.Semaphore(_SCRAPE_CONCURRENCY)
    _scheme_sem = asyncio.Semaphore(_SCHEME_CONCURRENCY)
    _slot_sem   = [asyncio.Semaphore(1) for _ in range(_POOL_SIZE)]

    for i in range(_POOL_SIZE):
        _slot_failures[i] = 0
        _slot_cooldown[i] = 0.0

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/schemes2").mkdir(parents=True, exist_ok=True)

    target = {k: v for k, v in SCHEMES.items()
              if schemes_filter is None or k in schemes_filter}

    results: dict[str, list] = {"success": [], "failed": [], "skipped": []}

    if not force:
        remaining, skipped = {}, []
        for sid, urls in target.items():
            if Path(f"data/schemes2/{sid}.json").exists():
                skipped.append(sid)
            else:
                remaining[sid] = urls
        results["skipped"] = skipped
        if skipped:
            log.info(f"Skipping {len(skipped)} existing: {', '.join(skipped)}")
        target = remaining

    if not target:
        log.info("Nothing to do.")
        return results

    pbar = tqdm(
        total=len(target), desc="Schemes", unit="scheme",
        ncols=100, colour="cyan", dynamic_ncols=True,
    )

    log.info(
        f"Starting pipeline: {len(target)} schemes, "
        f"{_SCHEME_CONCURRENCY} concurrent (sequential), "
        f"{_POOL_SIZE} llama-3.3 slots, "
        f"5 LLM calls/scheme (core + trans-a + trans-b + faq-a + faq-b)"
    )

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=PLAYWRIGHT_LAUNCH_ARGS,
        )
        log.info(f"Browser: {browser.browser_type.name} {browser.version}")

        tasks = [
            _process_with_sem(browser, sid, my_url, min_url, pbar, results)
            for sid, (my_url, min_url) in target.items()
        ]
        await asyncio.gather(*tasks)
        await browser.close()

    pbar.close()

    tqdm.write(f"""
{'='*60}
BUILD COMPLETE
  Success  ({len(results['success'])}): {', '.join(results['success']) or 'none'}
  Failed   ({len(results['failed'])}): {', '.join(results['failed'])  or 'none'}
  Skipped  ({len(results['skipped'])}): {', '.join(results['skipped']) or 'none'}
{'='*60}""")
    return results


def scrape_and_structure(force: bool = False, schemes_filter: Optional[list] = None):
    asyncio.run(_run_async(force=force, schemes_filter=schemes_filter))


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Welfare scheme KB builder v11")
    parser.add_argument("--force",       action="store_true",
                        help="Re-scrape all (ignore existing JSONs)")
    parser.add_argument("--schemes",     nargs="*",
                        help="Run specific schemes e.g. --schemes pm_kisan nrega")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallel schemes (default 1; increase only on paid NVIDIA tier)")
    parser.add_argument("--pool-size",   type=int, default=2,
                        help="Number of llama-3.3 slots in round-robin pool (default 2)")
    args = parser.parse_args()

    if args.concurrency is not None:
        _SCHEME_CONCURRENCY = args.concurrency
    if args.pool_size is not None:
        _POOL_SIZE = args.pool_size
        _slot_failures[:] = [0] * _POOL_SIZE
        _slot_cooldown[:] = [0.0] * _POOL_SIZE

    scrape_and_structure(force=args.force, schemes_filter=args.schemes)