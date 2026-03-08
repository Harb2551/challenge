"""Sensitive Content Detection API — Starter Scaffold.

Replace the detect_sensitive_content() function with your detection logic.

Run:
    uvicorn scaffold.server:app --host 0.0.0.0 --port 8000
"""

import time
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI(title="Sensitive Content Detector")


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


# ── Detection Logic (REPLACE THIS) ────────────────────────────────

def detect_sensitive_content(text: str) -> DetectResponse:
    """Detect whether text contains sensitive content.

    This is a stub that always returns False.
    Replace this with your actual detection logic.
    """
    return DetectResponse(
        has_sensitive_content=False,
        confidence=0.5,
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
