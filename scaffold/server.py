"""Sensitive Content Detection API — Starter Scaffold.

Loads the detector from models/detector_v1 if present, else downloads from Hugging Face Hub.
After downloading from HF, exports to ONNX and uses ONNX Runtime for <10ms latency.

Run:
    uvicorn scaffold.server:app --host 0.0.0.0 --port 8000
"""

import os
import torch
from pydantic import BaseModel
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .onnx_inference import ONNXDetector, ONNXExporter

app = FastAPI(title="Sensitive Content Detector")

MODEL_PATH = "models/detector_v1"
ONNX_PATH = os.path.join(MODEL_PATH, "model.onnx")
HF_MODEL_ID = "harshit2551/challenge-detector-v1"  # public fallback when local model missing
MAX_LENGTH = 512

_tokenizer = None
_model = None
_onnx_detector: ONNXDetector | None = None
_device = None


def _load_model():
    global _tokenizer, _model, _onnx_detector, _device
    if _tokenizer is not None:
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
        print("Exporting to ONNX...")
        ONNXExporter().export(_model, _tokenizer, ONNX_PATH, MODEL_PATH, MAX_LENGTH)
        load_path = MODEL_PATH

    _onnx_detector = ONNXDetector.load(ONNX_PATH, use_cuda=(_device == "cuda"))
    if _onnx_detector is not None:
        print(f"Using ONNX Runtime for inference ({ONNX_PATH}).")
    else:
        print("Using PyTorch for inference.")


@app.on_event("startup")
def startup():
    _load_model()


# ── Request / Response ─────────────────────────────────────────────

class DetectRequest(BaseModel):
    text: str

class DetectResponse(BaseModel):
    has_sensitive_content: bool
    confidence: float

class BatchDetectRequest(BaseModel):
    texts: list[str]

class BatchDetectResponse(BaseModel):
    results: list[DetectResponse]


def _logits_to_response(logit_val: float) -> DetectResponse:
    confidence = max(0.0, min(1.0, float(logit_val)))
    return DetectResponse(
        has_sensitive_content=confidence >= 0.5,
        confidence=round(confidence, 4),
    )


def _run_pytorch_single(text: str) -> DetectResponse:
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
    return _logits_to_response(logits.squeeze().item())


def _run_pytorch_batch(texts: list[str]) -> list[DetectResponse]:
    """One forward pass for the whole batch instead of N separate runs."""
    if not texts:
        return []
    inputs = _tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    return [_logits_to_response(logits[i].squeeze().item()) for i in range(len(texts))]


def detect_sensitive_content(text: str) -> DetectResponse:
    _load_model()
    if _onnx_detector is not None:
        [logit] = _onnx_detector.run(_tokenizer, [text], MAX_LENGTH)
        return _logits_to_response(logit)
    return _run_pytorch_single(text)


def detect_sensitive_content_batch(texts: list[str]) -> list[DetectResponse]:
    _load_model()
    if _onnx_detector is not None:
        logits = _onnx_detector.run(_tokenizer, texts, MAX_LENGTH)
        return [_logits_to_response(logit) for logit in logits]
    return _run_pytorch_batch(texts)


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    return detect_sensitive_content(req.text)


@app.post("/detect/batch", response_model=BatchDetectResponse)
def detect_batch(req: BatchDetectRequest):
    return BatchDetectResponse(results=detect_sensitive_content_batch(req.texts))


@app.get("/health")
def health():
    return {"status": "ok"}
