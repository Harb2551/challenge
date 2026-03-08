"""Sensitive Content Detection API — Starter Scaffold.

Loads the detector from models/detector_v1 if present, else downloads from Hugging Face Hub.
After downloading from HF, exports to ONNX and uses ONNX Runtime for <10ms latency.

Run:
    uvicorn scaffold.server:app --host 0.0.0.0 --port 8000
"""

import os
from typing import Optional

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
# Smaller batches often faster on GPU; chunk large requests to avoid one huge forward
INFERENCE_CHUNK_SIZE = 32
# Optional: truncate batch requests earlier to speed up (most snippets are short)
BATCH_MAX_LENGTH = 256

_tokenizer = None
_model = None
_onnx_detector: Optional[ONNXDetector] = None
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

    # Set FORCE_PYTORCH=1 to skip ONNX and compare latency (e.g. for benchmarking)
    force_pytorch = os.environ.get("FORCE_PYTORCH", "").strip().lower() in ("1", "true", "yes")
    if not force_pytorch:
        _onnx_detector = ONNXDetector.load(ONNX_PATH, use_cuda=(_device == "cuda"))
    else:
        _onnx_detector = None
        print("FORCE_PYTORCH=1: Using PyTorch for inference (ONNX disabled).")
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
    with torch.inference_mode():
        logits = _model(**inputs).logits
    return _logits_to_response(logits.squeeze().item())


def _run_pytorch_batch(texts: list[str]) -> list[DetectResponse]:
    """Chunked forward passes for better GPU utilization and lower latency."""
    if not texts:
        return []
    out: list[DetectResponse] = []
    for i in range(0, len(texts), INFERENCE_CHUNK_SIZE):
        chunk = texts[i : i + INFERENCE_CHUNK_SIZE]
        inputs = _tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=BATCH_MAX_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.inference_mode():
            logits = _model(**inputs).logits
        for j in range(len(chunk)):
            out.append(_logits_to_response(logits[j].squeeze().item()))
    return out


def detect_sensitive_content(text: str) -> DetectResponse:
    _load_model()
    if _onnx_detector is not None:
        [logit] = _onnx_detector.run(_tokenizer, [text], MAX_LENGTH)
        return _logits_to_response(logit)
    return _run_pytorch_single(text)


def detect_sensitive_content_batch(texts: list[str]) -> list[DetectResponse]:
    _load_model()
    if _onnx_detector is not None:
        # Chunk for ONNX too (often faster than one huge batch)
        out: list[DetectResponse] = []
        for i in range(0, len(texts), INFERENCE_CHUNK_SIZE):
            chunk = texts[i : i + INFERENCE_CHUNK_SIZE]
            logits = _onnx_detector.run(_tokenizer, chunk, BATCH_MAX_LENGTH)
            out.extend([_logits_to_response(logit) for logit in logits])
        return out
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
