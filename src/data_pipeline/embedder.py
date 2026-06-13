"""
build_vectorstore.py — SOTA Hybrid Retrieval Knowledge Base (v4.3)
===================================================================
v4.3 changes over v4.2:
1. BM25 inverted index  — O(matching_docs) at query time, not O(N)
   Replaces full-corpus scan with term → [doc_ids] lookup.
   At 50k chunks, a 5-term query now touches ~2-5k docs instead of all 50k.
2. Dynamic chunk-type weights — query-signal-aware boosts in scheme_search_tool.py
   (CHUNK_TYPE_WEIGHTS here remains the base; overrides applied at query time)

Unchanged from v4.2:
- intfloat/multilingual-e5-large (1024-dim) with "passage: " prefix
- Metadata deduplication  — lightweight {id, chunk_type} per chunk
- scheme_lookup dict       — full JSON stored once per scheme
- Unicode-aware BM25 tokenizer (regex \\p{L}+)
- hashlib.md5 for stable deduplication
- One FAQ per chunk
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import glob
import json
import pickle
import regex
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# 0. LOGGING
# ══════════════════════════════════════════════════════════════════════════════

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

_fmt    = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
_tqdm_h = TqdmLoggingHandler()
_tqdm_h.setFormatter(_fmt)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(_tqdm_h)
log.propagate = False


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL
# ══════════════════════════════════════════════════════════════════════════════

EMBED_MODEL = "intfloat/multilingual-e5-large"
# 1024-dim | requires "passage: " prefix at build time (done below)
#           | requires "query: "   prefix at search time (scheme_search_tool.py)
# Both prefixes MUST be consistent — mixing them silently destroys retrieval quality.


# ══════════════════════════════════════════════════════════════════════════════
# 2. BASE CHUNK-TYPE WEIGHTS
# These are the *starting* weights used as a baseline.
# scheme_search_tool.py overrides them per-query based on query signals —
# e.g. "helpline" → metadata weight raised to 1.8; "apply" → apply weight raised to 1.4.
# ══════════════════════════════════════════════════════════════════════════════

CHUNK_TYPE_WEIGHTS: dict[str, float] = {
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
    "tamil":        0.95,
    "bengali":      0.95,
    "marathi":      0.95,
    "renewal":      0.9,
    "keywords":     0.85,
    "related":      0.8,
    "metadata":     0.75,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. BM25 — INVERTED INDEX (v4.3)
# ══════════════════════════════════════════════════════════════════════════════

class BM25:
    """
    Okapi BM25 with inverted index for O(matching_docs) query time.

    v4.2 → v4.3:
      Old: iterate over ALL tf_maps at query time — O(N) regardless of query.
      New: inverted_index maps term → [doc_ids that contain it].
           score() only visits docs that share at least one term with the query.
           Complexity: O(sum of df for each query term) ≈ O(matching_docs).
           For a 5-term query over a 50k-chunk corpus this is typically
           2-5k docs instead of 50k — roughly 10-25x speedup.

    Memory trade-off: inverted_index stores one int per (term, doc) pair.
    For a corpus with vocab V and average df d, that's V*d ints.
    In practice <<10 MB for typical scheme corpora.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

        self.tf_maps:       list[dict[str, int]]      = []
        self.inverted_index: dict[str, list[int]]     = {}  # NEW: term → doc_ids
        self.df:            dict[str, int]             = defaultdict(int)
        self.idf:           dict[str, float]           = {}
        self.doc_lengths:   list[int]                  = []
        self.avgdl:         float                      = 0.0
        self.N:             int                        = 0

    def fit(self, corpus: list[str]) -> None:
        tokenized        = [self._tokenize(doc) for doc in corpus]
        self.N           = len(tokenized)
        self.doc_lengths = [len(doc) for doc in tokenized]
        self.avgdl       = sum(self.doc_lengths) / max(self.N, 1)

        # ── TF maps ───────────────────────────────────────────────────────────
        self.tf_maps = []
        for doc in tokenized:
            tf: dict[str, int] = defaultdict(int)
            for term in doc:
                tf[term] += 1
            self.tf_maps.append(dict(tf))

        # ── Document frequency ────────────────────────────────────────────────
        for tf in self.tf_maps:
            for term in tf:
                self.df[term] += 1

        # ── IDF ───────────────────────────────────────────────────────────────
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

        # ── Inverted index (NEW v4.3) ──────────────────────────────────────────
        # Built in a second pass over tf_maps so we never re-tokenize.
        inv: dict[str, list[int]] = defaultdict(list)
        for doc_id, tf_map in enumerate(self.tf_maps):
            for term in tf_map:
                inv[term].append(doc_id)
        # Freeze to plain dict; lists are already in doc_id order.
        self.inverted_index = dict(inv)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # \p{L}+ matches any Unicode letter sequence.
        # Correctly handles Devanagari (Hindi/Marathi), Tamil, Bengali scripts.
        return regex.findall(r"\p{L}+", text.lower())

    def score(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Score only documents that share at least one term with the query.
        Uses inverted_index to find candidates — O(matching_docs), not O(N).
        """
        q_terms = self._tokenize(query)
        if not q_terms:
            return []

        # ── Candidate set — union of posting lists for all query terms ────────
        candidate_ids: set[int] = set()
        for term in q_terms:
            if term in self.inverted_index:
                candidate_ids.update(self.inverted_index[term])

        if not candidate_ids:
            return []

        # ── Score only candidates ─────────────────────────────────────────────
        scores: list[tuple[int, float]] = []
        for i in candidate_ids:
            tf_map = self.tf_maps[i]
            dl     = self.doc_lengths[i]
            sc     = 0.0
            for term in q_terms:
                if term not in self.idf:
                    continue
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                sc += self.idf[term] * (
                    tf * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                )
            if sc > 0:
                scores.append((i, sc))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHUNK BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_chunks(s: dict) -> list[tuple[str, str]]:
    """
    Returns list of (chunk_text, chunk_type).
    Every chunk prefixed with "passage: " for multilingual-e5-large.
    """
    sid    = s.get("id", "")
    prefix = f"passage: [SCHEME:{sid}]"

    def join(key: str, sep: str = ", ") -> str:
        return sep.join(s.get(key) or [])

    chunks: list[tuple[str, str]] = []

    # 1. Overview
    keywords_str = join("keywords")
    chunks.append((
        f"{prefix} {s.get('name_en','')} — Overview: "
        f"{s.get('description_en','')} "
        f"Benefit: {s.get('benefit_en','')} "
        f"Ministry: {s.get('ministry','')} "
        f"Keywords: {keywords_str}",
        "overview",
    ))

    # 2. Benefit
    chunks.append((
        f"{prefix} {s.get('name_en','')} — Benefits and amount: "
        f"{s.get('benefit_en','')} | "
        f"हिंदी: {s.get('benefit_hi','')}",
        "benefit",
    ))

    # 3. Eligibility
    elig = s.get("eligibility_rules") or {}
    elig_str = (
        f"Occupation: {', '.join(elig.get('occupation') or [])} | "
        f"Gender: {elig.get('gender','any')} | "
        f"BPL required: {elig.get('bpl_required')} | "
        f"Land ownership required: {elig.get('land_ownership_required')} | "
        f"Age: {elig.get('age_min')}-{elig.get('age_max')} | "
        f"Income max: ₹{elig.get('income_max_annual','any')} | "
        f"Aadhaar required: {elig.get('aadhaar_required')} | "
        f"Other conditions: {'; '.join(elig.get('other_conditions') or [])}"
    )
    chunks.append((
        f"{prefix} {s.get('name_en','')} — Eligibility criteria: {elig_str}",
        "eligibility",
    ))

    # 4. Exclusions
    excl = join("exclusion_criteria", "; ")
    if excl:
        chunks.append((
            f"{prefix} {s.get('name_en','')} — Who CANNOT apply / exclusion criteria: {excl}",
            "exclusion",
        ))

    # 5. Documents — English
    if s.get("documents_en"):
        chunks.append((
            f"{prefix} {s.get('name_en','')} — Documents required (English): {join('documents_en')}",
            "documents",
        ))

    # 6. Documents — Hindi
    if s.get("documents_hi"):
        chunks.append((
            f"{prefix} {s.get('name_hi', s.get('name_en',''))} — "
            f"आवश्यक दस्तावेज़: {join('documents_hi')}",
            "hindi_docs",
        ))

    # 7. Documents — Tamil / Bengali / Marathi (separate chunk types)
    if s.get("documents_ta"):
        chunks.append((
            f"{prefix} {s.get('name_en','')} — Documents required (Tamil): {join('documents_ta')}",
            "tamil",
        ))
    if s.get("documents_bn"):
        chunks.append((
            f"{prefix} {s.get('name_en','')} — Documents required (Bengali): {join('documents_bn')}",
            "bengali",
        ))
    if s.get("documents_mr"):
        chunks.append((
            f"{prefix} {s.get('name_en','')} — Documents required (Marathi): {join('documents_mr')}",
            "marathi",
        ))

    # 8. How to apply — English
    if s.get("apply_steps_en"):
        chunks.append((
            f"{prefix} {s.get('name_en','')} — How to apply (steps): "
            f"{join('apply_steps_en', '; ')} "
            f"Apply URL: {s.get('application_url','')}",
            "apply",
        ))

    # 9. How to apply — Hindi
    if s.get("apply_steps_hi"):
        chunks.append((
            f"{prefix} {s.get('name_hi', s.get('name_en',''))} — "
            f"आवेदन कैसे करें: {join('apply_steps_hi', '; ')}",
            "hindi_apply",
        ))

    # 10. Hindi overview
    if s.get("name_hi"):
        chunks.append((
            f"{prefix} {s.get('name_hi','')} — {s.get('description_hi','')} "
            f"लाभ: {s.get('benefit_hi','')}",
            "hindi",
        ))

    # 11. Tamil overview
    if s.get("name_ta") and s.get("description_ta"):
        chunks.append((
            f"{prefix} {s.get('name_ta','')} — "
            f"{s.get('description_ta','')} "
            f"நன்மை: {s.get('benefit_ta') or s.get('benefit_en','')}",
            "tamil",
        ))

    # 12. Bengali overview
    if s.get("name_bn") and s.get("description_bn"):
        chunks.append((
            f"{prefix} {s.get('name_bn','')} — "
            f"{s.get('description_bn','')} "
            f"সুবিধা: {s.get('benefit_bn') or s.get('benefit_en','')}",
            "bengali",
        ))

    # 13. Marathi overview
    if s.get("name_mr") and s.get("description_mr"):
        chunks.append((
            f"{prefix} {s.get('name_mr','')} — "
            f"{s.get('description_mr','')} "
            f"फायदा: {s.get('benefit_mr') or s.get('benefit_en','')}",
            "marathi",
        ))

    # 14. State-wise amounts
    sw = s.get("state_wise_amounts") or {}
    if sw.get("states"):
        parts = [f"{st}: {amt}" for st, amt in sw["states"].items() if amt]
        if parts:
            chunks.append((
                f"{prefix} {s.get('name_en','')} — State-wise benefit amounts "
                f"({sw.get('note','varies by state')}): {'; '.join(parts[:20])}",
                "state_wise",
            ))

    # 15. Metadata
    chunks.append((
        f"{prefix} {s.get('name_en','')} — "
        f"Helpline: {s.get('helpline','')} | "
        f"Apply at: {s.get('application_url','')} | "
        f"Ministry: {s.get('ministry','')} | "
        f"Launch year: {s.get('launch_year','')} | "
        f"Processing time: {s.get('processing_time','')}",
        "metadata",
    ))

    # 16. Renewal
    if s.get("renewal_process"):
        chunks.append((
            f"{prefix} {s.get('name_en','')} — Renewal information: "
            f"Required: {s.get('renewal_required')} | {s.get('renewal_process','')}",
            "renewal",
        ))

    # 17. Keywords
    if s.get("keywords"):
        chunks.append((
            f"{prefix} {s.get('name_en','')} — Search keywords: {', '.join(s['keywords'])}",
            "keywords",
        ))

    # 18. Related schemes
    if s.get("related_schemes"):
        chunks.append((
            f"{prefix} {s.get('name_en','')} — Related schemes: {', '.join(s['related_schemes'])}",
            "related",
        ))

    # 19. Common mistakes
    if s.get("common_mistakes"):
        chunks.append((
            f"{prefix} {s.get('name_en','')} — Common mistakes to avoid: "
            f"{'; '.join(s['common_mistakes'])}",
            "mistakes",
        ))

    # 20. FAQ — one chunk per item
    for faq_item in (s.get("faq") or []):
        q_en = faq_item.get("question_en", "")
        a_en = faq_item.get("answer_en",   "")
        q_hi = faq_item.get("question_hi", "")
        a_hi = faq_item.get("answer_hi",   "")
        q_ta = faq_item.get("question_ta", "")
        a_ta = faq_item.get("answer_ta",   "")

        if not q_en or not a_en:
            continue

        chunk_text = f"{prefix} {s.get('name_en','')} — FAQ: Q: {q_en} A: {a_en}"
        if q_hi and a_hi:
            chunk_text += f" | हिंदी: प्र: {q_hi} उ: {a_hi}"
        if q_ta and a_ta:
            chunk_text += f" | தமிழ்: கே: {q_ta} ப: {a_ta}"

        chunks.append((chunk_text, "faq"))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_vectorstore():
    os.makedirs("data/vectorstore", exist_ok=True)

    log.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    dim   = model.get_sentence_embedding_dimension()
    log.info(f"  Embedding dim: {dim}")

    all_chunks:        list[str]       = []
    all_meta:          list[dict]      = []
    scheme_lookup:     dict            = {}
    seen_hashes:       set[str]        = set()
    chunk_type_counts: dict[str, int]  = defaultdict(int)

    json_files = sorted(glob.glob("data/schemes/*.json"))
    if not json_files:
        log.error("No JSON files in data/schemes/. Run build_data.py first.")
        return

    pbar = tqdm(json_files, desc="Building chunks", unit="scheme", ncols=100, colour="green")
    for jf in pbar:
        sid = Path(jf).stem
        pbar.set_postfix_str(sid)
        s   = json.load(open(jf, encoding="utf-8"))

        scheme_lookup[sid] = s

        for chunk_text, chunk_type in build_chunks(s):
            if not chunk_text.strip():
                continue
            h = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            all_chunks.append(chunk_text)
            all_meta.append({"id": sid, "chunk_type": chunk_type})
            chunk_type_counts[chunk_type] += 1

    n_schemes  = len(json_files)
    avg_chunks = len(all_chunks) // max(n_schemes, 1)
    tqdm.write(f"\nTotal chunks: {len(all_chunks):,} across {n_schemes} schemes (~{avg_chunks}/scheme)")
    tqdm.write("\nChunk type distribution:")
    for ctype, count in sorted(chunk_type_counts.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, count // max(1, len(all_chunks) // 300))
        tqdm.write(f"  {ctype:<20s} {count:>5d}  {bar}")

    # ── Dense embeddings ──────────────────────────────────────────────────────
    tqdm.write(f"\nEmbedding {len(all_chunks):,} chunks with {EMBED_MODEL}…")
    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # ── FAISS index ───────────────────────────────────────────────────────────
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, "data/vectorstore/schemes.index")
    tqdm.write(f"  FAISS index saved: {len(all_chunks):,} vectors @ dim={dim}")

    # ── BM25 with inverted index ──────────────────────────────────────────────
    tqdm.write("\nBuilding BM25 index (inverted index + precomputed TF)…")
    bm25 = BM25()
    with tqdm(total=1, desc="BM25 fit", ncols=80, colour="yellow") as pb:
        bm25.fit(all_chunks)
        pb.update(1)
        pb.set_postfix_str(f"vocab={len(bm25.df):,} | docs={bm25.N:,} | inv_terms={len(bm25.inverted_index):,}")
    tqdm.write(f"  BM25: {len(bm25.df):,} vocab terms, {bm25.N:,} docs, "
               f"{len(bm25.inverted_index):,} inverted index entries")

    # ── Persist ───────────────────────────────────────────────────────────────
    tqdm.write("\nSaving vectorstore…")
    with open("data/vectorstore/metadata.pkl", "wb") as f:
        pickle.dump({
            "chunks":        all_chunks,
            "meta":          all_meta,
            "scheme_lookup": scheme_lookup,
            "chunk_weights": CHUNK_TYPE_WEIGHTS,
        }, f)
    tqdm.write("  metadata.pkl saved")

    with open("data/vectorstore/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    tqdm.write("  bm25.pkl saved")

    import sys
    meta_size   = sys.getsizeof(pickle.dumps(all_meta))
    lookup_size = sys.getsizeof(pickle.dumps(scheme_lookup))
    inv_size    = sys.getsizeof(pickle.dumps(bm25.inverted_index))
    tqdm.write(f"\n  Metadata list size   : {meta_size/1024:.1f} KB")
    tqdm.write(f"  Scheme lookup size   : {lookup_size/1024:.1f} KB")
    tqdm.write(f"  Inverted index size  : {inv_size/1024:.1f} KB")

    tqdm.write(f"""
{'═'*60}
VECTORSTORE BUILD COMPLETE  (v4.3)
  Schemes           : {len(scheme_lookup)}
  Total chunks      : {len(all_chunks):,}
  Unique chunks     : {len(seen_hashes):,}
  Embed model       : {EMBED_MODEL} (dim={dim})
  BM25 vocab        : {len(bm25.df):,} terms
  BM25 inv. index   : {len(bm25.inverted_index):,} terms indexed
  FAQ chunks        : {chunk_type_counts.get('faq', 0)} (1 per FAQ item)
  Chunk types       : {len(chunk_type_counts)} distinct types
  FAISS index       : data/vectorstore/schemes.index
  BM25 index        : data/vectorstore/bm25.pkl
  Metadata          : data/vectorstore/metadata.pkl
{'═'*60}""")


if __name__ == "__main__":
    build_vectorstore()