"""
app.py — Welfare Scheme Assistant
Streamlit UI — fully integrated with new session memory + welfare_crew.

Supports: English · हिंदी · मराठी · বাংলা · தமிழ்
"""

import sys
print("Python:", sys.version, file=sys.stderr)

import streamlit as st
import re
import uuid
import os
from dotenv import load_dotenv
from src.data_pipeline.embedder import BM25
from src.memory.session_memory import (
    init_db, get_session, save_session, reset_session,
    update_language, needs_binary_buttons, needs_occupation_buttons,
    get_next_pending_slot,
)
from src.utils.checklist import generate_checklist_pdf
from src.crew.welfare_crew    import run
from src.data_pipeline.embedder import BM25
load_dotenv()
init_db()
import sys
sys.modules['__main__'].BM25 = BM25
# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Welfare Scheme Assistant · कल्याण योजना",
    page_icon="🇮🇳",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
body, .stApp { background: #ECE5DD; }

.wa-header {
    background: linear-gradient(135deg, #075E54, #128C7E);
    color: white; padding: 14px 20px;
    border-radius: 14px; margin-bottom: 12px;
    display: flex; align-items: center; gap: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
}
.wa-avatar {
    width: 48px; height: 48px; border-radius: 50%;
    background: rgba(255,255,255,0.2);
    display: flex; align-items: center;
    justify-content: center; font-size: 24px; flex-shrink: 0;
}
.wa-name  { font-weight: 700; font-size: 17px; }
.wa-sub   { font-size: 11px; opacity: .85; margin-top: 2px; }
.wa-langs { font-size: 11px; opacity: .70; margin-top: 2px; }

.msg-user {
    background: #DCF8C6; color: #111;
    padding: 10px 14px;
    border-radius: 18px 18px 4px 18px;
    margin: 5px 0 5px 80px;
    font-size: 15px; line-height: 1.6;
    word-wrap: break-word;
    box-shadow: 0 1px 2px rgba(0,0,0,.15);
}
.msg-bot {
    background: #FFFFFF; color: #111;
    padding: 10px 14px;
    border-radius: 18px 18px 18px 4px;
    margin: 5px 80px 5px 0;
    font-size: 15px; line-height: 1.6;
    border: 1px solid #E8E8E8;
    word-wrap: break-word;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
.msg-bot em {
    font-size: 11px; color: #999; font-style: normal;
    display: block; margin-top: 6px;
}
.section-label {
    font-size: 11px; font-weight: 700; color: #666;
    text-transform: uppercase; letter-spacing: .06em;
    margin: 14px 0 6px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("sid",           None),
    ("messages",      []),
    ("eligible_ids",  []),
    ("started",       False),
    ("pending_input", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.sid is None:
    st.session_state.sid = str(uuid.uuid4())

sid = st.session_state.sid

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="wa-header">
  <div class="wa-avatar">🤖</div>
  <div>
    <div class="wa-name">Welfare Scheme Assistant &nbsp;🇮🇳</div>
    <div class="wa-sub">कल्याण योजना सहायक · நலன் திட்ட உதவியாளர் · কল্যাণ প্রকল্প · कल्याण योजना</div>
    <div class="wa-langs">Supports: English &nbsp;·&nbsp; हिंदी &nbsp;·&nbsp; मराठी &nbsp;·&nbsp; বাংলা &nbsp;·&nbsp; தமிழ்</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE SELECTOR + RESET
# ══════════════════════════════════════════════════════════════════════════════

LANG_OPTIONS = {
    "हिंदी (Hindi)":   "hi",
    "English":          "en",
    "मराठी (Marathi)": "mr",
    "বাংলা (Bengali)": "bn",
    "தமிழ் (Tamil)":   "ta",
}

col_lang, col_reset = st.columns([4, 1])

with col_lang:
    session      = get_session(sid)
    current_lang = session.get("language", "hi")
    lang_keys    = list(LANG_OPTIONS.keys())
    lang_vals    = list(LANG_OPTIONS.values())
    default_idx  = lang_vals.index(current_lang) if current_lang in lang_vals else 0

    lang_choice = st.selectbox(
        "Language", lang_keys,
        index=default_idx,
        label_visibility="collapsed",
        key="lang_select",
    )
    chosen_code = LANG_OPTIONS[lang_choice]

    # User explicitly changed language — user_selected always wins
    if chosen_code != current_lang:
        update_language(session, chosen_code, "user_selected")
        save_session(sid, session)
        st.rerun()

with col_reset:
    if st.button("🔄 Reset", use_container_width=True):
        reset_session(sid)
        st.session_state.messages      = []
        st.session_state.eligible_ids  = []
        st.session_state.started       = False
        st.session_state.pending_input = None
        st.session_state.sid           = str(uuid.uuid4())
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# RENDER CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def _render_message(msg: dict):
    cls     = "msg-user" if msg["role"] == "user" else "msg-bot"
    content = msg["content"].replace("\n", "<br>")
    content = re.sub(r"_([^_]+)_", r"<em>\1</em>", content)
    st.markdown(f'<div class="{cls}">{content}</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    _render_message(msg)

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME MESSAGE — first load only
# ══════════════════════════════════════════════════════════════════════════════

WELCOME_MSGS = {
    "en": (
        "🙏 Welcome to Welfare Scheme Assistant.\n\n"
        "I can help you with:\n"
        "• Which government schemes you qualify for\n"
        "• Documents needed for any scheme\n"
        "• How to apply — step by step\n"
        "• Benefits, amounts, and helpline numbers\n\n"
        "Ask me anything, or tap **Start Eligibility Check** below.\n\n"
        "_I remember our conversation — no need to repeat yourself!_"
    ),
    "hi": (
        "🙏 कल्याण योजना सहायक में आपका स्वागत है।\n\n"
        "मैं आपकी मदद कर सकता हूँ:\n"
        "• कौन सी सरकारी योजनाएँ आपके लिए हैं\n"
        "• किसी भी योजना के लिए जरूरी दस्तावेज़\n"
        "• आवेदन कैसे करें — एक-एक कदम\n"
        "• लाभ, राशि और हेल्पलाइन नंबर\n\n"
        "कुछ भी पूछें, या नीचे **Start Eligibility Check** दबाएँ।\n\n"
        "_मुझे हमारी बातचीत याद रहती है — बार-बार दोहराने की जरूरत नहीं!_"
    ),
    "mr": (
        "🙏 कल्याण योजना सहाय्यकामध्ये आपले स्वागत आहे.\n\n"
        "मी मदत करू शकतो:\n"
        "• कोणत्या सरकारी योजना तुमच्यासाठी आहेत\n"
        "• कागदपत्रे कोणती लागतात\n"
        "• अर्ज कसा करायचा\n\n"
        "काहीही विचारा, किंवा **Start Eligibility Check** दाबा.\n\n"
        "_मला आपली संभाषण आठवते!_"
    ),
    "bn": (
        "🙏 কল্যাণ প্রকল্প সহকারীতে আপনাকে স্বাগত জানাই।\n\n"
        "আমি সাহায্য করতে পারি:\n"
        "• কোন সরকারি প্রকল্পে যোগ্য\n"
        "• প্রয়োজনীয় কাগজপত্র\n"
        "• কীভাবে আবেদন করবেন\n\n"
        "যেকোনো প্রশ্ন করুন, বা **Start Eligibility Check** চাপুন।\n\n"
        "_আমি আমাদের কথোপকথন মনে রাখি!_"
    ),
    "ta": (
        "🙏 நலன் திட்ட உதவியாளரில் உங்களை வரவேற்கிறோம்.\n\n"
        "நான் உதவ முடியும்:\n"
        "• எந்த அரசு திட்டங்களுக்கு தகுதி உள்ளது\n"
        "• தேவையான ஆவணங்கள்\n"
        "• விண்ணப்பிக்கும் முறை\n\n"
        "எதையும் கேளுங்கள், அல்லது **Start Eligibility Check** அழுத்தவும்.\n\n"
        "_உங்கள் உரையாடல் எனக்கு நினைவிருக்கும்!_"
    ),
}

if not st.session_state.started:
    st.session_state.started = True
    session      = get_session(sid)
    lang_code    = session.get("language", "hi")
    welcome_text = WELCOME_MSGS.get(lang_code, WELCOME_MSGS["hi"])
    st.session_state.messages.append({"role": "assistant", "content": welcome_text})
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — process pending input
# Separated from input handling to prevent double-render on rerun
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.pending_input is not None:
    pending = st.session_state.pending_input
    st.session_state.pending_input = None

    session = get_session(sid)
    lang    = session.get("language", "hi")

    # Live transparency — each status update appears immediately as it happens
    status_placeholder = st.empty()
    status_lines: list[str] = []

    def _on_status(msg: str):
        status_lines.append(msg)
        # Show all accumulated status lines, newest at bottom
        status_placeholder.markdown(
            "\n\n".join(f"<span style='color:#666;font-size:13px'>{line}</span>"
                        for line in status_lines),
            unsafe_allow_html=True,
        )

    reply, ids = run(sid, pending, status_cb=_on_status)

    # Clear status after answer is ready
    status_placeholder.empty()

    if ids:
        st.session_state.eligible_ids = ids

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ELIGIBILITY CHECK CTA
# Shown only when not in flow and no occupation collected yet
# ══════════════════════════════════════════════════════════════════════════════

session  = get_session(sid)
in_flow  = session.get("in_eligibility_flow", False)
done     = session.get("eligibility_done", False)
facts    = session.get("confirmed_facts", {})
lang     = session.get("language", "hi")

CTA_LABELS = {
    "en": "🔍 Start Eligibility Check",
    "hi": "🔍 पात्रता जाँच शुरू करें",
    "mr": "🔍 पात्रता तपासणी सुरू करा",
    "bn": "🔍 যোগ্যতা পরীক্ষা শুরু করুন",
    "ta": "🔍 தகுதி சரிபார்ப்பை தொடங்குங்கள்",
}

if not in_flow and not done and not facts.get("occupation"):
    if st.button(CTA_LABELS.get(lang, CTA_LABELS["en"]), use_container_width=True, type="primary"):
        cta_placeholder = st.empty()
        cta_placeholder.markdown(
            "<span style='color:#666;font-size:13px'>⏳ Starting eligibility check...</span>",
            unsafe_allow_html=True,
        )
        reply, _ = run(sid, "__init__", status_cb=lambda x: None)
        cta_placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# YES / NO QUICK-REPLY BUTTONS
# Shown only when session memory says next slot is a binary (has_land etc.)
# ══════════════════════════════════════════════════════════════════════════════

session = get_session(sid)
if needs_binary_buttons(session):
    lang = session.get("language", "hi")
    YES_LABELS = {"en": "✅ Yes", "hi": "✅ हाँ", "mr": "✅ होय", "bn": "✅ হ্যাঁ", "ta": "✅ ஆம்"}
    NO_LABELS  = {"en": "❌ No",  "hi": "❌ नहीं", "mr": "❌ नाही", "bn": "❌ না",  "ta": "❌ இல்லை"}

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button(YES_LABELS.get(lang, "✅ Yes"), key="btn_yes", use_container_width=True):
            lbl = YES_LABELS.get(lang, "✅ Yes")
            st.session_state.messages.append({"role": "user", "content": lbl})
            st.session_state.pending_input = lbl
            st.rerun()
    with col_no:
        if st.button(NO_LABELS.get(lang, "❌ No"), key="btn_no", use_container_width=True):
            lbl = NO_LABELS.get(lang, "❌ No")
            st.session_state.messages.append({"role": "user", "content": lbl})
            st.session_state.pending_input = lbl
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# OCCUPATION QUICK-REPLY BUTTONS
# Shown only when next pending slot is "occupation"
# ══════════════════════════════════════════════════════════════════════════════

session = get_session(sid)
if needs_occupation_buttons(session):
    lang = session.get("language", "hi")
    OCC_LABELS = {
        "en": ["🌾 Farmer", "👷 Labourer", "👩 Woman HoH", "📚 Student"],
        "hi": ["🌾 किसान",  "👷 मजदूर",   "👩 महिला",      "📚 छात्र"],
        "mr": ["🌾 शेतकरी", "👷 मजूर",    "👩 महिला प्रमुख","📚 विद्यार्थी"],
        "bn": ["🌾 কৃষক",   "👷 শ্রমিক",  "👩 মহিলা",       "📚 ছাত্র"],
        "ta": ["🌾 விவசாயி","👷 தொழிலாளர்","👩 பெண் தலைவி", "📚 மாணவர்"],
    }
    labels = OCC_LABELS.get(lang, OCC_LABELS["en"])
    cols   = st.columns(len(labels))
    for i, lbl in enumerate(labels):
        with cols[i]:
            if st.button(lbl, key=f"occ_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": lbl})
                st.session_state.pending_input = lbl
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TEXT INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════

session = get_session(sid)
lang    = session.get("language", "hi")

PLACEHOLDERS = {
    "en": "Type your question...",
    "hi": "अपना सवाल लिखें...",
    "mr": "तुमचा प्रश्न लिहा...",
    "bn": "আপনার প্রশ্ন লিখুন...",
    "ta": "உங்கள் கேள்வியை தட்டச்சு செய்யுங்கள்...",
}
SEND_LABELS = {
    "en": "Send ➤", "hi": "भेजें ➤", "mr": "पाठवा ➤", "bn": "পাঠান ➤", "ta": "அனுப்பு ➤",
}

with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    with c1:
        user_input = st.text_input(
            "msg",
            label_visibility="collapsed",
            placeholder=PLACEHOLDERS.get(lang, PLACEHOLDERS["en"]),
        )
    with c2:
        send = st.form_submit_button(
            SEND_LABELS.get(lang, "Send ➤"),
            use_container_width=True,
        )

if send and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending_input = user_input
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT CHECKLIST
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    '<div class="section-label">📋 Your Document Checklist</div>',
    unsafe_allow_html=True,
)

session = get_session(sid)
lang    = session.get("language", "en")

if st.session_state.eligible_ids:
    c1, c2 = st.columns([2, 3])
    with c1:
        pdf_bytes = generate_checklist_pdf(st.session_state.eligible_ids, lang)
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name="welfare_checklist.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with c2:
        n = len(st.session_state.eligible_ids)
        st.success(f"✓ {n} eligible scheme{'s' if n != 1 else ''} found")
        st.caption("Bring this PDF to your nearest Common Service Centre (CSC) / Jan Seva Kendra")
        st.caption("Schemes: " + ", ".join(st.session_state.eligible_ids))
else:
    CHECKLIST_INFO = {
        "en": "Complete the eligibility check above to unlock your personalised PDF checklist.",
        "hi": "अपनी व्यक्तिगत PDF चेकलिस्ट पाने के लिए ऊपर पात्रता जाँच पूरी करें।",
        "mr": "वैयक्तिक PDF चेकलिस्ट मिळवण्यासाठी वरील पात्रता तपासणी पूर्ण करा.",
        "bn": "ব্যক্তিগত PDF চেকলিস্ট পেতে উপরে যোগ্যতা পরীক্ষা সম্পন্ন করুন।",
        "ta": "தனிப்பயன் PDF பட்டியலைப் பெற மேலே தகுதி சரிபார்ப்பை முடிக்கவும்.",
    }
    st.info(CHECKLIST_INFO.get(lang, CHECKLIST_INFO["en"]))

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "Powered by NVIDIA NIM · Llama 3.1 8B + Llama 3.3 70B · "
    "FAISS + BM25 Hybrid RAG · MyScheme.gov.in · Tavily Live Search"
)