# Sensitive Content Detection Challenge

## Problem

Build an API that detects whether a given text contains **sensitive content** — information that should not be sent to external services.

Sensitive content includes:
- **Credentials**: passwords, API keys, tokens, private keys
- **Personal identifiers**: SSN, credit card numbers, bank accounts
- **Contact information**: phone numbers, email addresses, physical addresses
- **Other PII**: full names + date of birth, medical/financial account numbers

Your detector must distinguish real sensitive content from benign text that merely _discusses_ these topics (e.g., "the password policy requires 8 characters" is **not** sensitive).

## What You Get

- A cloud VM with GPU (NVIDIA 24GB RAM or equivalent)
- Claude API Key however u want to use 
- This scaffold repo with a FastAPI stub and sample data
- 48 hours

## What You Build

A REST API running on the VM with the following endpoint:

### `POST /detect`

**Request:**
```json
{
  "text": "my password is fluffy2024"
}
```

**Response:**
```json
{
  "has_sensitive_content": true,
  "confidence": 0.99
}
```

| Field | Type | Description |
|-------|------|-------------|
| `has_sensitive_content` | `bool` | Whether the text contains sensitive content |
| `confidence` (optional) | `float` | Model confidence, 0.0 to 1.0 |

### `POST /detect/batch` (bonus)

**Request:**
```json
{
  "texts": [
    "my password is fluffy2024",
    "what's the weather today"
  ]
}
```

**Response:**
```json
{
  "results": [
    {"has_sensitive_content": true, "confidence": 0.99},
    {"has_sensitive_content": false, "confidence": 0.98}
  ]
}
```

## Constraints

- API should response asap, latency is one of the metrics we care
- Must run on the provided VM (do not call external APIs for inference, we will check your codebase implementation as well)
- You may use any language, framework, or approach

## Evaluation Criteria

Your solution will be evaluated by calling your API with a blind test set (>500 labeled samples). We will measure:

| Metric | Weight | Description |
|--------|--------|-------------|
| **F1 Score** | Primary | Harmonic mean of precision and recall |
| **Recall** | High | Percentage of sensitive content correctly caught |
| **Precision** | High | Percentage of flagged content that is actually sensitive |
| **FP Rate** | Medium | How often safe text is incorrectly flagged |
| **Latency** | Medium | Average and p95 response time |

## Sample Data

See `scaffold/sample_data.json` for 20 labeled examples covering the expected input types. This is a small representative sample — the blind test set is larger and more diverse.

## Getting Started

### 1. Clone and install

```bash
git clone <this-repo-url>
cd challenge
pip install -r scaffold/requirements.txt
```

### 2. Run the server

From the **repo root**:

```bash
uvicorn scaffold.server:app --host 0.0.0.0 --port 8000
```

On first run, if the model is not already at `models/detector_v1`, the server will download it and cache it locally.

### 3. Call the API

If you don’t have `models/detector_v1`, the server will download the public model from Hugging Face on first run, cache it, **then export to ONNX** and use ONNX Runtime for &lt;10ms latency.

```bash
pip install -r scaffold/requirements.txt
uvicorn scaffold.server:app --host 0.0.0.0 --port 8000
```

Model ID: **harshit2551/challenge-detector-v1** (set in `scaffold/server.py` as `HF_MODEL_ID`).

## Submission

Ensure your API is running on port **8000** on the provided VM. We will call your endpoint during evaluation. No other submission is required.

## Tips

- Start with something simple that works, then iterate
- Consider what makes this problem hard: "my password is fluffy2024" vs "the password policy requires 8 characters"
- Think about both structured secrets (API keys, tokens) and conversational PII ("my SSN is ...")
- The test set includes edge cases — placeholder values, code snippets, technical discussions
