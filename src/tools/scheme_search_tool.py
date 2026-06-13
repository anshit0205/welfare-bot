"""
scheme_search_tool.py — Hybrid FAISS + BM25 retrieval (v4, Priority 1)

Changes matching build_vectorstore v4:
- Loads new metadata structure: {meta, scheme_lookup, chunk_weights}
- Chunk-type weighted RRF — faq/eligibility chunks score higher than metadata chunks
- Confidence = top FAISS cosine similarity (unchanged, still correct)
- scheme_lookup used to hydrate full scheme data after dedup
- Precomputed BM25 TF maps — score() is now much faster
"""

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional
from collections import defaultdict
from src.data_pipeline.embedder import BM25
import sys
sys.modules['__main__'].BM25 = BM25
# ── Lazy singletons — loaded once per process ────────────────────────────────
_model:         Optional[SentenceTransformer] = None
_index:         Optional[faiss.Index]         = None
_chunks:        Optional[list]                = None
_meta:          Optional[list]                = None   # list of {id, chunk_type}
_scheme_lookup: Optional[dict]               = None   # sid → full scheme JSON
_bm25:          Optional[object]             = None
_chunk_weights: Optional[dict]              = None

# Must match build_vectorstore.py exactly
EMBED_MODEL = "intfloat/multilingual-e5-large"
# CRITICAL: queries must use "query: " prefix to match "passage: " at build time

# Confidence thresholds for Tavily routing
CONF_HIGH   = 0.65
CONF_MEDIUM = 0.50

# Default chunk weights (fallback if not stored in metadata.pkl)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "faq":          1.3,
    "eligibility":  1.2,
    "benefit":      1.15,
    "documents":    1.1,
    "apply":        1.1,
    "exclusion":    1.1,
    "state_wise":   1.05,
    "mistakes":     1.0,
    "overview":     1.0,
    "hindi":        1.0,
    "hindi_apply":  1.0,
    "hindi_docs":   1.0,
    "tamil":        0.95,  # separate per language — cleaner embeddings
    "bengali":      0.95,
    "marathi":      0.95,
    "renewal":      0.9,
    "keywords":     0.85,
    "related":      0.8,
    "metadata":     0.75,
}


def _load_resources():
    global _model, _index, _chunks, _meta, _scheme_lookup, _bm25, _chunk_weights

    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)

    if _index is None:
        _index = faiss.read_index("data/vectorstore/schemes.index")
        store  = pickle.load(open("data/vectorstore/metadata.pkl", "rb"))

        _chunks        = store["chunks"]
        _meta          = store["meta"]           # lightweight: [{id, chunk_type}, ...]
        _scheme_lookup = store["scheme_lookup"]  # full JSON per scheme, keyed by id
        _chunk_weights = store.get("chunk_weights", _DEFAULT_WEIGHTS)

    if _bm25 is None:
        try:
            _bm25 = pickle.load(open("data/vectorstore/bm25.pkl", "rb"))
        except FileNotFoundError:
            _bm25 = None


def _weighted_rrf(
    dense:   list[tuple[int, float]],
    sparse:  list[tuple[int, float]],
    meta:    list[dict],
    weights: dict[str, float],
    k:       int   = 60,
    dw:      float = 0.65,
    sw:      float = 0.35,
) -> list[tuple[int, float]]:
    """
    Chunk-type weighted Reciprocal Rank Fusion.

    Each chunk's RRF score is multiplied by its chunk_type weight.
    A FAQ chunk answering the exact question scores higher than a metadata
    chunk that just happens to mention the scheme name.
    """
    scores: dict[int, float] = defaultdict(float)

    for rank, (idx, _) in enumerate(dense):
        w = weights.get(meta[idx]["chunk_type"], 1.0) if idx < len(meta) else 1.0
        scores[idx] += w * dw / (k + rank + 1)

    for rank, (idx, _) in enumerate(sparse):
        w = weights.get(meta[idx]["chunk_type"], 1.0) if idx < len(meta) else 1.0
        scores[idx] += w * sw / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: -x[1])


def search(
    query: str,
    top_k: int = 4,
    lang:  str = "en",
) -> tuple[list[dict], float]:
    """
    Hybrid FAISS + BM25 search with chunk-type weighted RRF.

    Returns:
        results:    list of full scheme dicts (deduplicated, hydrated from scheme_lookup)
        confidence: float 0-1, top FAISS cosine similarity score
    """
    _load_resources()

    # ── Dense (FAISS) ─────────────────────────────────────────────────────────
    # "query: " prefix required — matches "passage: " prefix used at build time
    q_vec = _model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = _index.search(q_vec, top_k * 5)

    confidence = float(scores[0][0]) if len(scores[0]) > 0 else 0.0

    dense_ranking = [
        (int(idx), float(sc))
        for sc, idx in zip(scores[0], indices[0])
        if idx >= 0
    ]

    # ── Sparse (BM25 — fast, uses precomputed TF maps) ────────────────────────
    sparse_ranking = _bm25.score(query, top_k=top_k * 5) if _bm25 else []

    # ── Weighted RRF merge ───────────────────────────────────────────────────
    if sparse_ranking:
        merged = _weighted_rrf(dense_ranking, sparse_ranking, _meta, _chunk_weights)
    else:
        # FAISS only — still apply chunk weights
        merged = []
        for rank, (idx, sc) in enumerate(dense_ranking):
            w = _chunk_weights.get(_meta[idx]["chunk_type"], 1.0) if idx < len(_meta) else 1.0
            merged.append((idx, w * sc))
        merged.sort(key=lambda x: -x[1])

    # ── Deduplicate by scheme ID, hydrate from scheme_lookup ─────────────────
    seen:    set  = set()
    results: list = []

    doc_key = {
        "hi": "documents_hi",
        "mr": "documents_mr",
        "bn": "documents_bn",
        "ta": "documents_ta",
    }.get(lang, "documents_en")

    for idx, _ in merged:
        if idx >= len(_meta):
            continue

        sid = _meta[idx]["id"]
        if sid in seen:
            continue
        seen.add(sid)

        # Hydrate full scheme data from lookup (single dict, no duplication)
        s = _scheme_lookup.get(sid)
        if not s:
            continue

        results.append({
            "id":               sid,
            "name_en":          s.get("name_en", ""),
            "name_hi":          s.get("name_hi", ""),
            "benefit_en":       s.get("benefit_en", ""),
            "eligibility_rules": s.get("eligibility_rules", {}),
            "exclusion_criteria": s.get("exclusion_criteria", []),
            "documents":        s.get(doc_key) or s.get("documents_en", []),
            "documents_en":     s.get("documents_en", []),
            "apply_steps_en":   s.get("apply_steps_en", []),
            "application_url":  s.get("application_url", ""),
            "helpline":         s.get("helpline", ""),
            "ministry":         s.get("ministry", ""),
            "faq":              s.get("faq", []),
            "state_wise_amounts": s.get("state_wise_amounts", {}),
            # Surface the matched chunk type — useful for answer agent context
            "matched_chunk_type": _meta[idx]["chunk_type"],
        })

        if len(results) >= top_k:
            break

    return results, confidence


def format_results_for_prompt(results: list[dict]) -> str:
    """Format search results as clean text for LLM prompt injection."""
    if not results:
        return "No matching schemes found in knowledge base."

    blocks = []
    for r in results:
        docs  = "\n  - ".join(r.get("documents_en", [])[:6])
        steps = "\n  ".join(
            f"{i+1}. {s}" for i, s in enumerate(r.get("apply_steps_en", [])[:5])
        )

        # Include state-wise amounts if present — useful for follow-up queries
        sw = r.get("state_wise_amounts", {})
        sw_text = ""
        if sw and sw.get("states"):
            state_parts = [f"{st}: {amt}" for st, amt in sw["states"].items() if amt]
            if state_parts:
                sw_text = f"\nSTATE-WISE AMOUNTS: {'; '.join(state_parts)}"

        blocks.append(
            f"── SCHEME: {r['name_en']} ({r.get('name_hi','')}) ──\n"
            f"BENEFIT: {r.get('benefit_en','')}{sw_text}\n"
            f"ELIGIBILITY: {r.get('eligibility_rules',{})}\n"
            f"EXCLUSIONS: {'; '.join(r.get('exclusion_criteria',[])[:4])}\n"
            f"DOCUMENTS:\n  - {docs}\n"
            f"HOW TO APPLY:\n  {steps}\n"
            f"URL: {r.get('application_url','')}\n"
            f"HELPLINE: {r.get('helpline','')}\n"
            f"[matched via: {r.get('matched_chunk_type','')}]"
        )
    return "\n\n".join(blocks)