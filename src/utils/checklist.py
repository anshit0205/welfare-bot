"""
checklist.py — PDF document checklist generator.
Generates a downloadable PDF the user can bring to CSC/Jan Seva Kendra.
"""

import json
import glob
from fpdf import FPDF

LANG_TITLES = {
    "hi": "आवश्यक दस्तावेज़ सूची",
    "ta": "தேவையான ஆவணங்கள்",
    "bn": "প্রয়োজনীয় নথির তালিকা",
    "mr": "आवश्यक कागदपत्रे यादी",
    "en": "Required Documents Checklist",
}

_DOC_KEY = {
    "hi": "documents_hi",
    "ta": "documents_ta",
    "bn": "documents_bn",
    "mr": "documents_mr",
    "en": "documents_en",
}


def _safe_cell(pdf: FPDF, text: str, **kwargs):
    """Write cell, falling back to ASCII-safe version if encoding fails."""
    try:
        pdf.cell(0, **kwargs, txt=text)
    except Exception:
        safe = text.encode("latin-1", "replace").decode("latin-1")
        pdf.cell(0, **kwargs, txt=safe)


def generate_checklist_pdf(eligible_ids: list, lang: str = "en") -> bytes:
    """Generate a PDF checklist for all eligible scheme IDs."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    _safe_cell(pdf, LANG_TITLES.get(lang, LANG_TITLES["en"]), h=10, ln=True, align="C")

    pdf.set_font("Helvetica", "", 9)
    _safe_cell(
        pdf,
        "Present at nearest Common Service Centre (CSC) / Jan Seva Kendra",
        h=6, ln=True, align="C",
    )
    pdf.ln(6)

    doc_key = _DOC_KEY.get(lang, "documents_en")

    # ── One section per eligible scheme ──────────────────────────────────────
    for jf in sorted(glob.glob("data/schemes/*.json")):
        s = json.load(open(jf, encoding="utf-8"))
        if s.get("id") not in eligible_ids:
            continue

        # Scheme header
        pdf.set_font("Helvetica", "B", 13)
        _safe_cell(pdf, s.get("name_en", s["id"]), h=9, ln=True)

        # Benefit line
        pdf.set_font("Helvetica", "I", 10)
        benefit = s.get("benefit_en", "")
        if benefit:
            _safe_cell(pdf, f"Benefit: {benefit[:120]}", h=6, ln=True)

        # Documents checklist
        pdf.set_font("Helvetica", "", 11)
        docs = s.get(doc_key) or s.get("documents_en", [])
        for doc in docs:
            _safe_cell(pdf, f"  [ ]  {doc}", h=7, ln=True)

        # Helpline
        helpline = s.get("helpline", "")
        if helpline:
            pdf.set_font("Helvetica", "I", 9)
            _safe_cell(pdf, f"  Helpline: {helpline}", h=6, ln=True)

        # How to apply (first 3 steps)
        steps = s.get("apply_steps_en", [])
        if steps:
            pdf.set_font("Helvetica", "B", 10)
            _safe_cell(pdf, "  How to apply:", h=7, ln=True)
            pdf.set_font("Helvetica", "", 10)
            for i, step in enumerate(steps[:3], 1):
                _safe_cell(pdf, f"  {i}. {step[:100]}", h=6, ln=True)

        # URL
        url = s.get("application_url", "")
        if url:
            pdf.set_font("Helvetica", "U", 9)
            pdf.set_text_color(0, 0, 200)
            _safe_cell(pdf, f"  Apply online: {url}", h=6, ln=True)
            pdf.set_text_color(0, 0, 0)

        pdf.ln(4)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(4)

    return bytes(pdf.output())


def generate_checklist_sms(eligible_ids: list) -> str:
    """Short SMS version (under 320 chars per scheme)."""
    lines = ["YOUR ELIGIBLE SCHEMES:"]
    for jf in glob.glob("data/schemes/*.json"):
        s = json.load(open(jf, encoding="utf-8"))
        if s.get("id") not in eligible_ids:
            continue
        docs = s.get("documents_en", [])[:3]
        lines.append(f"\n{s.get('name_en','')}: {s.get('benefit_en','')[:80]}")
        lines.append(f"Docs: {', '.join(docs[:3])}")
        if s.get("helpline"):
            lines.append(f"Call: {s['helpline']}")
    return "\n".join(lines)