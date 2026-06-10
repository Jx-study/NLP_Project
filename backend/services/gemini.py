import os
import google.generativeai as genai

_client = None
DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _clean_spoiler(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    return text.removeprefix("SPOILER:").strip()


def _load():
    global _client
    if _client is not None:
        return
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    _client = genai.GenerativeModel(model_name)


def analyze(title: str, content: str, lang: str = "auto", verdict_needed: bool = False) -> dict:
    """
    Call Gemini for spoiler (and optionally re-verdict).

    verdict_needed=True  → ask both "is clickbait?" + "spoiler" in one call (fuzzy region)
    verdict_needed=False → ask for spoiler only (high-confidence clickbait confirmed by model)

    Returns {"is_clickbait": bool, "spoiler": str | None}
    """
    _load()

    if lang == "zh":
        lang_hint = "繁體中文"
        if verdict_needed:
            prompt = f"""你是一個標題黨偵測助理。請根據以下標題和內文回答。

標題：{title}
內文：{content[:600]}

請以以下格式回答（不要加任何其他說明）：
VERDICT: yes
SPOILER: 一句可作為非標題黨新聞標題的中性繁體中文句子，直接說明文章真正重點

若不是標題黨：
VERDICT: no
SPOILER:"""
        else:
            prompt = f"""你是一個標題黨破梗助理。請用一句{lang_hint}改寫成中性、具體、可作為非標題黨新聞標題的句子。

要求：
- 直接說明文章真正重點
- 不要使用「驚人」、「你不會相信」、「竟然」、「這個」、「那個」等吊胃口寫法
- 不要用問句、感嘆句或模糊代稱
- 不要評論也不要誇大

標題：{title}
內文：{content[:600]}

SPOILER:"""
    else:
        if verdict_needed:
            prompt = f"""You are a clickbait detection assistant.

Title: {title}
Content: {content[:600]}

Reply in exactly this format (no extra text):
VERDICT: yes
SPOILER: one neutral, specific, non-clickbait news headline that reveals the real point

If NOT clickbait:
VERDICT: no
SPOILER:"""
        else:
            prompt = f"""You are a clickbait spoiler assistant. Rewrite the clickbait into one neutral, specific, non-clickbait news headline in English.

Requirements:
- Directly reveal the article's real point
- Do not use curiosity-gap words like "this", "that", "these", "shocking", or "you won't believe"
- Do not use a question, exclamation, vague pronoun, commentary, or exaggeration

Title: {title}
Content: {content[:600]}

SPOILER:"""

    response = _client.generate_content(prompt)
    text = response.text.strip()

    is_clickbait = True  # default: model already confirmed clickbait
    spoiler = None

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            verdict_val = line.split(":", 1)[1].strip().lower()
            is_clickbait = verdict_val in ("yes", "true", "1")
        elif line.upper().startswith("SPOILER:"):
            val = _clean_spoiler(line.split(":", 1)[1])
            spoiler = val if val else None

    # If only spoiler was requested, is_clickbait stays True
    if not verdict_needed:
        is_clickbait = True
        if spoiler is None and text:
            spoiler = _clean_spoiler(text)

    return {"is_clickbait": is_clickbait, "spoiler": spoiler}
