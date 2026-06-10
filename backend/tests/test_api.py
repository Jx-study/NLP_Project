import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch


BASELINE_SAFE = {"label": 0, "is_clickbait": False, "confidence": 0.92}
BASELINE_CLICK = {"label": 1, "is_clickbait": True, "confidence": 0.88}
TRANSFORMER_SAFE = {"label": 0, "is_clickbait": False, "confidence": 0.95}
TRANSFORMER_CLICK = {"label": 1, "is_clickbait": True, "confidence": 0.91}


@pytest.fixture
def client():
    with patch("services.classifier.predict_baseline") as mock_baseline, \
         patch("services.classifier.predict_transformer") as mock_transformer, \
         patch("services.gemini.analyze") as mock_gemini:

        mock_baseline.return_value = BASELINE_SAFE
        mock_transformer.return_value = TRANSFORMER_SAFE
        mock_gemini.return_value = {"is_clickbait": False, "spoiler": None}

        from app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c, mock_baseline, mock_transformer, mock_gemini


def test_health(client):
    c, *_ = client
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_missing_title(client):
    c, *_ = client
    r = c.post("/api/analyze", json={"content": "some content"})
    assert r.status_code == 400


def test_missing_content_ok(client):
    """content is optional — title alone should succeed."""
    c, *_ = client
    r = c.post("/api/analyze", json={"title": "Some headline"})
    assert r.status_code == 200


def test_response_shape(client):
    """Response must have baseline and transformer keys with correct fields."""
    c, *_ = client
    r = c.post("/api/analyze", json={"title": "Normal news", "content": "Normal content"})
    data = r.get_json()
    assert r.status_code == 200

    for key in ("baseline", "transformer"):
        assert key in data, f"Missing top-level key: {key}"

    for field in ("label", "is_clickbait", "confidence"):
        assert field in data["baseline"], f"baseline missing field: {field}"
        assert field in data["transformer"], f"transformer missing field: {field}"

    for field in ("gemini_used", "spoiler"):
        assert field in data["transformer"], f"transformer missing field: {field}"


def test_non_clickbait_no_gemini_call(client):
    c, mock_baseline, mock_transformer, mock_gemini = client
    mock_transformer.return_value = {**TRANSFORMER_SAFE, "confidence": 0.95}

    r = c.post("/api/analyze", json={"title": "Normal news", "content": "Normal content"})
    data = r.get_json()

    assert data["transformer"]["is_clickbait"] is False
    assert data["transformer"]["spoiler"] is None
    assert data["transformer"]["gemini_used"] is False
    mock_gemini.assert_not_called()


def test_high_confidence_clickbait_reverdict(client):
    """High confidence clickbait now ALSO gets a Gemini re-verdict (not spoiler-only)."""
    c, mock_baseline, mock_transformer, mock_gemini = client
    mock_transformer.return_value = {**TRANSFORMER_CLICK, "confidence": 0.91}
    mock_gemini.return_value = {"is_clickbait": True, "spoiler": "The answer is 42."}

    r = c.post("/api/analyze", json={"title": "You won't believe this!", "content": "..."})
    data = r.get_json()

    mock_gemini.assert_called_once()
    # verdict_needed must be True so Gemini can overturn a tone false-positive
    assert mock_gemini.call_args.kwargs.get("verdict_needed") is True
    assert data["transformer"]["is_clickbait"] is True
    assert data["transformer"]["spoiler"] == "The answer is 42."
    assert data["transformer"]["gemini_used"] is True


def test_high_confidence_clickbait_overturned(client):
    """Tone false-positive: model says clickbait, Gemini overturns to non-clickbait."""
    c, mock_baseline, mock_transformer, mock_gemini = client
    mock_transformer.return_value = {**TRANSFORMER_CLICK, "confidence": 0.93}
    mock_gemini.return_value = {"is_clickbait": False, "spoiler": None}

    r = c.post("/api/analyze", json={"title": "Plain factual headline", "content": "..."})
    data = r.get_json()

    assert data["transformer"]["is_clickbait"] is False
    assert data["transformer"]["label"] == 0
    assert data["transformer"]["spoiler"] is None
    assert data["transformer"]["gemini_used"] is True


def test_low_confidence_gemini_verdict(client):
    """Low confidence: Gemini provides both verdict and spoiler."""
    c, mock_baseline, mock_transformer, mock_gemini = client
    mock_transformer.return_value = {**TRANSFORMER_CLICK, "confidence": 0.65}
    mock_gemini.return_value = {"is_clickbait": True, "spoiler": "It was a cat video."}

    r = c.post("/api/analyze", json={"title": "Something happened...", "content": "..."})
    data = r.get_json()

    assert data["transformer"]["gemini_used"] is True
    assert data["transformer"]["spoiler"] == "It was a cat video."


def test_high_confidence_non_clickbait_scale_word_triggers_gemini(client):
    """Scale word in title forces Gemini re-check even on high-confidence non-clickbait."""
    c, mock_baseline, mock_transformer, mock_gemini = client
    mock_transformer.return_value = {**TRANSFORMER_SAFE, "confidence": 0.96}
    mock_gemini.return_value = {"is_clickbait": True, "spoiler": "He won a school race."}

    r = c.post("/api/analyze", json={"title": "我是世界冠軍", "content": "學校運動會第一名"})
    data = r.get_json()

    mock_gemini.assert_called_once()
    assert data["transformer"]["gemini_used"] is True
    assert data["transformer"]["is_clickbait"] is True
    assert data["transformer"]["label"] == 1
    assert data["transformer"]["spoiler"] == "He won a school race."


def test_high_confidence_non_clickbait_no_scale_word_skips_gemini(client):
    """Plain non-clickbait with no scale word must NOT call Gemini (cost control)."""
    c, mock_baseline, mock_transformer, mock_gemini = client
    mock_transformer.return_value = {**TRANSFORMER_SAFE, "confidence": 0.96}

    r = c.post("/api/analyze", json={"title": "市議會通過預算案", "content": "詳細內容"})
    data = r.get_json()

    mock_gemini.assert_not_called()
    assert data["transformer"]["gemini_used"] is False
    assert data["transformer"]["is_clickbait"] is False
