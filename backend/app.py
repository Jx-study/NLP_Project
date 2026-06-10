import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from services import classifier, gemini

load_dotenv()

app = Flask(__name__)
CORS(app)

CONFIDENCE_THRESHOLD = 0.80


def detect_lang(title: str, content: str) -> str:
    text = title + content
    zh_count = sum(1 for c in text if "一" <= c <= "鿿")
    return "zh" if zh_count / max(len(text), 1) > 0.2 else "en"


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    content = (data.get("content") or "").strip()
    cascade_enabled = bool(data.get("cascade", True))

    if not title:
        return jsonify({"error": "title is required"}), 400

    lang = detect_lang(title, content)

    # Run both models
    baseline = classifier.predict_baseline(title, content)
    transformer = classifier.predict_transformer(title, content)

    # ── Cascade logic ──────────────────────────────────────────
    from services.scale_lexicon import scale_lexicon_score

    gemini_used = False
    spoiler = None
    confident = transformer["confidence"] >= CONFIDENCE_THRESHOLD
    scale_score = scale_lexicon_score(title, content)

    needs_gemini = cascade_enabled and (
        transformer["is_clickbait"]                       # 判 clickbait → 一律複核
        or not confident                                  # 模糊區 → 複核
        or scale_score >= 3                               # 標題誇大 + 內文兜不上 → 複核
        or (scale_score == 2 and transformer["confidence"] < 0.9)  # 標題有大詞 + 信心不高 → 複核
    )

    if needs_gemini:
        try:
            gem = gemini.analyze(title, content, lang=lang, verdict_needed=True)
            transformer["is_clickbait"] = gem["is_clickbait"]
            transformer["label"] = 1 if gem["is_clickbait"] else 0
            spoiler = gem["spoiler"] if gem["is_clickbait"] else None
            gemini_used = True
        except Exception as e:
            spoiler = f"[Analysis unavailable: {e}]"
            gemini_used = True

    return jsonify({
        "baseline": {
            "label": baseline["label"],
            "is_clickbait": baseline["is_clickbait"],
            "confidence": baseline["confidence"],
        },
        "transformer": {
            "label": transformer["label"],
            "is_clickbait": transformer["is_clickbait"],
            "confidence": transformer["confidence"],
            "gemini_used": gemini_used,
            "spoiler": spoiler,
        },
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
