"""
webhook.py — FastAPI webhook for Twilio WhatsApp + SMS.

Changes from original:
1. ALL run() calls are async (including __init__) — Twilio always gets an
   immediate ACK, never waits for LLM processing.
2. Occupation + binary slot options are texted to the user since WhatsApp
   has no buttons — they type a number to pick.
3. Session ID normalised: strips "whatsapp:" prefix so same phone number
   on WA and SMS shares one session.
4. Error messages are sent in the user's detected language, not always English.
5. cache.purge_expired() called on startup to clean stale cache entries.
6. Consistent async path for all messages — no more split sync/async logic.
"""

import os
import threading

from fastapi import FastAPI, Response, Form
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

from src.memory.session_memory import (
    init_db, get_session, save_session, reset_session,
    update_language, needs_binary_buttons, needs_occupation_buttons,
    get_next_pending_slot,
)
from src.utils.checklist import generate_checklist_sms
from src.crew.welfare_crew import run
from src.utils.cache import purge_expired
from dotenv import load_dotenv

load_dotenv()

twilio_client = Client(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN"),
)
TWILIO_WA_NUMBER = os.getenv("TWILIO_WA_NUMBER", "whatsapp:+14155238886")

init_db()
purge_expired()   # clean stale cache entries on startup

app = FastAPI(title="Welfare Bot — Twilio Webhook")


# ══════════════════════════════════════════════════════════════════════════════
# STATIC COPY
# ══════════════════════════════════════════════════════════════════════════════

RESET_WORDS = {
    "reset", "start", "restart", "new", "begin",
    "शुरू", "नया", "start over", "शुरुवात",
    "புதிய", "নতুন", "नव्याने", "नए सिरे",
}

LANG_TRIGGER = {
    "hindi": "hi", "हिंदी": "hi", "1": "hi",
    "tamil": "ta", "தமிழ்": "ta", "2": "ta",
    "bengali": "bn", "বাংলা": "bn", "3": "bn",
    "marathi": "mr", "मराठी": "mr", "4": "mr",
    "english": "en",               "5": "en",
}

ELIGIBILITY_TRIGGERS = {
    "check", "eligibility", "पात्रता", "योजना जाँच",
    "எனக்கு தகுதி", "আমার যোগ্যতা", "पात्रता तपासणी",
    "which schemes", "kaunsi yojana",
}

WELCOME_MSG = (
    "🙏 Welfare Scheme Assistant\n\n"
    "Type your language / अपनी भाषा चुनें:\n"
    "1. Hindi (हिंदी)\n"
    "2. Tamil (தமிழ்)\n"
    "3. Bengali (বাংলা)\n"
    "4. Marathi (मराठी)\n"
    "5. English\n\n"
    "Or just ask your question in any language!"
)

LANG_ACK = {
    "hi": "ठीक है! मैं हिंदी में जवाब दूँगा। 🙏\nकौन सी योजना के बारे में जानना है? या पात्रता जाँचने के लिए 'check' लिखें।",
    "ta": "சரி! நான் தமிழில் பதில் சொல்கிறேன். 🙏\nஎந்த திட்டம் பற்றி தெரிந்துகொள்ள விரும்புகிறீர்கள்?",
    "bn": "ঠিক আছে! আমি বাংলায় উত্তর দেব। 🙏\nকোন প্রকল্প সম্পর্কে জানতে চান?",
    "mr": "ठीक आहे! मी मराठीत उत्तर देईन. 🙏\nकोणत्या योजनेबद्दल जाणून घ्यायचे आहे?",
    "en": "Got it! I'll respond in English. 🙏\nWhich scheme would you like to know about, or type 'check' to start your eligibility check.",
}

PROCESSING_MSG = {
    "hi": "🔍 जानकारी खोजी जा रही है। कृपया प्रतीक्षा करें...",
    "en": "🔍 Looking up the information. Please wait...",
    "bn": "🔍 তথ্য খোঁজা হচ্ছে। অনুগ্রহ করে অপেক্ষা করুন...",
    "mr": "🔍 माहिती शोधत आहोत. कृपया प्रतीक्षा करा...",
    "ta": "🔍 தகவல் தேடப்படுகிறது. தயவுசெய்து காத்திருக்கவும்...",
}

ERROR_MSG = {
    "hi": "❌ कुछ गड़बड़ हो गई। कृपया दोबारा कोशिश करें।",
    "en": "❌ Something went wrong. Please try again.",
    "bn": "❌ কিছু একটা সমস্যা হয়েছে। আবার চেষ্টা করুন।",
    "mr": "❌ काहीतरी चूक झाली. पुन्हा प्रयत्न करा.",
    "ta": "❌ ஏதோ தவறு நடந்தது. மீண்டும் முயற்சிக்கவும்.",
}

# ── WhatsApp text menus replacing Streamlit buttons ──────────────────────────
# Sent after the slot question so user knows what to type.

OCC_MENU = {
    "en": "Reply with a number:\n1️⃣ Farmer\n2️⃣ Labourer\n3️⃣ Woman Head of Household\n4️⃣ Student",
    "hi": "नीचे से नंबर लिखें:\n1️⃣ किसान\n2️⃣ मजदूर\n3️⃣ महिला मुखिया\n4️⃣ छात्र",
    "mr": "खालीलपैकी नंबर लिहा:\n1️⃣ शेतकरी\n2️⃣ मजूर\n3️⃣ महिला प्रमुख\n4️⃣ विद्यार्थी",
    "bn": "নম্বর লিখুন:\n1️⃣ কৃষক\n2️⃣ শ্রমিক\n3️⃣ মহিলা কর্তা\n4️⃣ ছাত্র",
    "ta": "எண் தட்டச்சு செய்யுங்கள்:\n1️⃣ விவசாயி\n2️⃣ தொழிலாளர்\n3️⃣ பெண் தலைவி\n4️⃣ மாணவர்",
}

# Maps typed number → occupation value the bot understands
OCC_NUMBER_MAP = {
    "1": "🌾 farmer", "2": "👷 labourer",
    "3": "👩 woman hoh", "4": "📚 student",
}

YESNO_MENU = {
    "en": "Reply *1* for Yes or *2* for No.",
    "hi": "*1* हाँ के लिए, *2* नहीं के लिए।",
    "mr": "*1* होय साठी, *2* नाही साठी.",
    "bn": "*1* হ্যাঁ বলুন, *2* না বলুন।",
    "ta": "*1* ஆம் என்றால், *2* இல்லை என்றால்.",
}

# Maps typed number → yes/no token the bot understands
YESNO_NUMBER_MAP = {"1": "✅ yes", "2": "❌ no"}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_sender(from_: str) -> str:
    """
    Strip the 'whatsapp:' prefix so the same phone number on WA and SMS
    shares one session. Returns just the E.164 number, e.g. +919876543210.
    """
    return from_.replace("whatsapp:", "").strip()


def _lang(sid: str) -> str:
    return get_session(sid).get("language", "hi")


def _send(to: str, body: str):
    """Send a WhatsApp message via Twilio. Truncates to 1597 chars."""
    twilio_client.messages.create(
        body=body[:1597],
        from_=TWILIO_WA_NUMBER,
        to=to if to.startswith("whatsapp:") else f"whatsapp:{to}",
    )


def _twiml(msg: str) -> Response:
    """Return a minimal TwiML response with a single message."""
    tr = MessagingResponse()
    tr.message(msg[:1597])
    return Response(content=str(tr), media_type="application/xml")


def _maybe_append_menu(sid: str, reply: str) -> str:
    """
    After a slot question, append the relevant numbered menu so the user
    knows what to type — replacing the Streamlit buttons.
    """
    session = get_session(sid)
    lang    = session.get("language", "hi")

    if needs_occupation_buttons(session):
        menu = OCC_MENU.get(lang, OCC_MENU["en"])
        return f"{reply}\n\n{menu}"

    if needs_binary_buttons(session):
        menu = YESNO_MENU.get(lang, YESNO_MENU["en"])
        return f"{reply}\n\n{menu}"

    return reply


def _translate_numbered_input(text: str, sid: str) -> str:
    """
    During eligibility flow, translate a bare "1"/"2" reply into the
    occupation label or yes/no token the bot's fast-gate understands.
    Only fires when the session is in the eligibility flow.
    """
    session = get_session(sid)
    if not session.get("in_eligibility_flow") or session.get("eligibility_done"):
        return text

    t = text.strip()

    if needs_occupation_buttons(session) and t in OCC_NUMBER_MAP:
        return OCC_NUMBER_MAP[t]

    if needs_binary_buttons(session) and t in YESNO_NUMBER_MAP:
        return YESNO_NUMBER_MAP[t]

    return text


# ══════════════════════════════════════════════════════════════════════════════
# ASYNC PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _process_async(sender_raw: str, sid: str, message: str):
    """
    Run the LLM pipeline in a background thread and send the reply back
    via Twilio. Used for ALL messages so the webhook always ACKs instantly.

    sender_raw : original From value including 'whatsapp:' prefix (needed
                 by Twilio send API).
    sid        : normalised session ID (no prefix).
    message    : the (possibly translated) user message.
    """
    try:
        reply, eligible_ids = run(
            sid=sid,
            message=message,
            status_cb=lambda x: None,   # can't stream status over Twilio
        )

        # Append numbered menu if we're mid-eligibility-flow
        reply = _maybe_append_menu(sid, reply)

        _send(sender_raw, reply)

        # Send document checklist as a follow-up message if schemes found
        if eligible_ids:
            checklist = generate_checklist_sms(eligible_ids)
            if checklist:
                _send(sender_raw, checklist)

    except Exception as exc:
        print(f"[ASYNC ERROR] {exc}")
        lang = _lang(sid)
        _send(sender_raw, ERROR_MSG.get(lang, ERROR_MSG["en"]))


# ══════════════════════════════════════════════════════════════════════════════
# WEBHOOK ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "service": "welfare-bot-v1"}


@app.post("/webhook")
async def webhook(
    Body: str = Form(default=""),
    From: str = Form(default=""),
    To:   str = Form(default=""),
):
    """
    Main Twilio webhook. Must return within ~15s or Twilio retries.
    Strategy: ACK immediately with a "processing" message, do all heavy
    work in a background thread, send the real reply via Twilio REST API.
    """
    text       = Body.strip()
    sender_raw = From                        # keep original for Twilio send
    sid        = _normalise_sender(From)     # normalised for session lookup
    lang       = _lang(sid)

    print(f"[webhook] {sid}: {text[:80]!r}")

    # ── 1. RESET ──────────────────────────────────────────────────────────────
    if text.lower() in RESET_WORDS:
        reset_session(sid)
        return _twiml(WELCOME_MSG)

    # ── 2. LANGUAGE SELECTION ─────────────────────────────────────────────────
    lang_choice = LANG_TRIGGER.get(text.lower()) or LANG_TRIGGER.get(text)
    if lang_choice:
        session = get_session(sid)
        update_language(session, lang_choice, "user_selected")
        save_session(sid, session)
        return _twiml(LANG_ACK.get(lang_choice, LANG_ACK["hi"]))

    # ── 3. TRANSLATE NUMBERED REPLIES (1/2/3/4) ───────────────────────────────
    # Must happen before eligibility trigger check so "1" during occ question
    # doesn't accidentally hit the eligibility trigger.
    message = _translate_numbered_input(text, sid)

    # ── 4. ELIGIBILITY FLOW INIT ──────────────────────────────────────────────
    if text.lower() in ELIGIBILITY_TRIGGERS:
        message = "__init__"

    # ── 5. FIRE ASYNC + ACK IMMEDIATELY ──────────────────────────────────────
    threading.Thread(
        target=_process_async,
        args=(sender_raw, sid, message),
        daemon=True,
    ).start()

    ack = PROCESSING_MSG.get(lang, PROCESSING_MSG["hi"])
    return _twiml(ack)