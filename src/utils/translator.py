"""
Translation using AI4Bharat IndicTrans2 via HuggingFace Inference API.
Used specifically to translate:
  - Document names from English → user's language
  - Scheme descriptions from English → user's language
  - Application steps from English → user's language

Why IndicTrans2 over Llama for translation:
  - Purpose-built for 22 Indian languages ↔ English
  - Much higher accuracy for Tamil/Bengali document terminology
  - Deterministic (no hallucination risk on proper nouns like scheme names)
  - Free on HuggingFace Inference API
"""
import os, httpx, json
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")

# IndicTrans2 models — 1B param versions, faster on HF inference
_IT2_EN_TO_INDIC = (
    "https://api-inference.huggingface.co/models/"
    "ai4bharat/indictrans2-en-indic-1B"
)
_IT2_INDIC_TO_EN = (
    "https://api-inference.huggingface.co/models/"
    "ai4bharat/indictrans2-indic-en-1B"
)

# IndicTrans2 language codes
_LANG_TO_IT2 = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "te": "tel_Telu",
    "kn": "kan_Knda",
}

_HEADERS = {"Content-Type": "application/json"}
if HF_TOKEN:
    _HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"


def _call_indictrans2(url: str, payload: dict, timeout: float = 8.0) -> str | None:
    try:
        resp = httpx.post(url, headers=_HEADERS, json=payload, timeout=timeout)
        if resp.status_code == 200:
            result = resp.json()
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "").strip()
            if isinstance(result, dict):
                return result.get("generated_text", "").strip()
    except Exception as e:
        print(f"[IndicTrans2] API error: {e}")
    return None


@lru_cache(maxsize=512)
def translate_en_to_lang(text: str, target_lang: str) -> str:
    """
    Translate English text to target_lang using IndicTrans2.
    Results are cached — scheme document names asked repeatedly won't re-call API.
    Falls back to original English text if translation fails.
    """
    if target_lang == "en" or not text.strip():
        return text

    it2_code = _LANG_TO_IT2.get(target_lang)
    if not it2_code:
        return text

    result = _call_indictrans2(
        _IT2_EN_TO_INDIC,
        {
            "inputs": text,
            "parameters": {
                "src_lang": "eng_Latn",
                "tgt_lang": it2_code,
                "max_length": 512,
            }
        }
    )
    return result if result else text


def translate_docs_to_lang(doc_list: list[str], target_lang: str) -> list[str]:
    """
    Translate a list of document names.
    Joins them, translates as a batch (fewer API calls), splits back.
    """
    if target_lang == "en":
        return doc_list

    # Batch translate with separator
    separator = " | "
    joined = separator.join(doc_list)
    translated = translate_en_to_lang(joined, target_lang)

    # Split back — if translation failed, parts will still be in English
    parts = [p.strip() for p in translated.split("|")]
    if len(parts) == len(doc_list):
        return parts
    # If split went wrong, fall back to original
    return doc_list


def translate_steps_to_lang(steps: list[str], target_lang: str) -> list[str]:
    """Translate application steps individually for accuracy."""
    if target_lang == "en":
        return steps
    return [translate_en_to_lang(step, target_lang) for step in steps[:5]]