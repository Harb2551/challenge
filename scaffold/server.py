"""Sensitive Content Detection API — Starter Scaffold.

Loads the detector from models/detector_v1 if present, else downloads from Hugging Face Hub.

Run:
    uvicorn scaffold.server:app --host 0.0.0.0 --port 8000
"""

import os
import torch
from pydantic import BaseModel
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Sensitive Content Detector")

MODEL_PATH = "models/detector_v1"
HF_MODEL_ID = "harshit2551/challenge-detector-v1"  # public fallback when local model missing
MAX_LENGTH = 512

# Load model and tokenizer once at startup
_tokenizer = None
_model = None
_device = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.isdir(MODEL_PATH):
        load_path = MODEL_PATH
        print(f"Loading model from {load_path}...")
    else:
        load_path = HF_MODEL_ID
        print(f"Local model not found at {MODEL_PATH}, downloading from Hugging Face ({HF_MODEL_ID})...")
    _tokenizer = AutoTokenizer.from_pretrained(load_path)
    _model = AutoModelForSequenceClassification.from_pretrained(
        load_path, num_labels=1, use_safetensors=True
    ).to(_device)
    _model.eval()
    _model = _model.float()
    if load_path == HF_MODEL_ID:
        os.makedirs(MODEL_PATH, exist_ok=True)
        _tokenizer.save_pretrained(MODEL_PATH)
        _model.save_pretrained(MODEL_PATH, safe_serialization=True)
        print(f"Cached model to {MODEL_PATH} for next run.")


@app.on_event("startup")
def startup():
    _load_model()


# ── Request / Response Models ──────────────────────────────────────

class DetectRequest(BaseModel):
    text: str

class DetectResponse(BaseModel):
    has_sensitive_content: bool
    confidence: float

class BatchDetectRequest(BaseModel):
    texts: list[str]

class BatchDetectResponse(BaseModel):
    results: list[DetectResponse]


# ── Detection Logic ────────────────────────────────────────────────

def detect_sensitive_content(text: str) -> DetectResponse:
    """Run detector model and return has_sensitive_content and confidence in [0, 1]."""
    _load_model()
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    raw = logits.squeeze().item()
    confidence = max(0.0, min(1.0, raw))
    has_sensitive_content = confidence >= 0.5
    return DetectResponse(
        has_sensitive_content=has_sensitive_content,
        confidence=round(confidence, 4),
    )


# ── Endpoints ─────────────────────────────────────────────────────

@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    return detect_sensitive_content(req.text)


@app.post("/detect/batch", response_model=BatchDetectResponse)
def detect_batch(req: BatchDetectRequest):
    results = [detect_sensitive_content(t) for t in req.texts]
    return BatchDetectResponse(results=results)


@app.get("/health")
def health():
    return {"status": "ok"}
