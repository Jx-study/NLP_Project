import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TRANSFORMER_MODEL_PATH = "../models/xlm-roberta-clickbait-g7"
BASELINE_VECTORIZER_PATH = "../models/baseline/tfidf_vectorizer.joblib"
BASELINE_MODEL_PATH = "../models/baseline/logistic_regression.joblib"
MAX_LENGTH = 256

_tokenizer = None
_model = None
_device = None
_vectorizer = None
_lr_model = None


def _load_transformer():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_PATH, clean_up_tokenization_spaces=True)
    _model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_PATH)
    _model.to(_device)
    _model.eval()


def _load_baseline():
    global _vectorizer, _lr_model
    if _lr_model is not None:
        return
    _vectorizer = joblib.load(BASELINE_VECTORIZER_PATH)
    _lr_model = joblib.load(BASELINE_MODEL_PATH)


def predict_baseline(title: str, content: str) -> dict:
    _load_baseline()
    text = f"{title} {content}"
    vec = _vectorizer.transform([text])
    label = int(_lr_model.predict(vec)[0])
    proba = _lr_model.predict_proba(vec)[0]
    return {
        "label": label,
        "is_clickbait": label == 1,
        "confidence": round(float(proba[label]), 4),
    }


def predict_transformer(title: str, content: str) -> dict:
    _load_transformer()
    text = f"{title} </s> {content[:400]}"
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label = int(torch.argmax(probs).item())
    return {
        "label": label,
        "is_clickbait": label == 1,
        "confidence": round(float(probs[label].item()), 4),
    }


# Keep old single-model entry point for CLI usage
def predict(title: str, content: str, model: str = "transformer") -> dict:
    if model == "baseline":
        result = predict_baseline(title, content)
        result["model"] = "baseline"
    else:
        result = predict_transformer(title, content)
        result["model"] = "transformer"
    return result
