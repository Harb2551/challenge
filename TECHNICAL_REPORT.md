# Technical Report: Sensitive Content Detection

## 1. Problem Statement

The task is to build an API that detects whether a given text contains **sensitive content** — information that should not be sent to external services. Sensitive content includes:

- **Credentials**: passwords, API keys, tokens, private keys  
- **Personal identifiers**: SSN, credit card numbers, bank accounts  
- **Contact information**: phone numbers, email addresses, physical addresses  
- **Other PII**: full names with date of birth, medical/financial account numbers

The detector must distinguish real sensitive content from benign text that merely *discusses* these topics (e.g., “the password policy requires 8 characters” is **not** sensitive).

**Constraints:** The API must respond with low latency, run on the provided VM (no external inference APIs), and be evaluated on F1, recall, precision, FP rate, and latency over a blind test set of 500+ samples.

---

## 2. Approach Overview

The solution uses a **knowledge-distillation-style** pipeline:

1. **Synthetic data generation** — Use a strong LLM (Claude) as a “teacher” to generate diverse, soft-labeled examples across sensitivity categories.
2. **Training** — Fine-tune a small student model (DeBERTa-v3-small) on this data with regression (continuous sensitivity score in [0, 1]).
3. **Inference** — Serve the model via a FastAPI server with ONNX Runtime when available for low-latency single and batch detection.

The rest of this report details each stage, design choices, and issues encountered.

---

## 3. Synthetic Data Generation

### 3.1 Purpose

The generated data is used to fine-tune a smaller student model for sensitive content detection. We need sufficient volume, category coverage, and **soft labels** (continuous scores) so the student learns the decision boundary (e.g., around 0.5) and nuance, not just binary 0/1.

### 3.2 Evolution of the Strategy

The data-generation strategy evolved through several iterations to fix quality and scale issues.

**1. Initial monolithic approach**  

- A single script (`generate_data.py`) used one broad prompt to generate “sensitive and non-sensitive” examples in one large API call.  
- **Problems:** Skewed data, generic examples, mostly binary labels (0.0 or 1.0), and output token limits when generating ~1,250 examples at once.

**2. Category-specific prompts**  

- Generation was split by category to improve diversity and specificity.  
- A `prompts/` directory was added with five text files: `credentials.txt`, `identifiers.txt`, `contact_info.txt`, `other_pii.txt`, `safe_cases.txt`.  
- Each prompt describes what that category looks like, edge cases, and how to score (e.g., test credentials ≈ 0.88, real credentials = 1.0).

**3. Decision threshold and soft labels**  

- Prompts were updated to center explicitly around a **0.5** decision threshold.  
- Claude was instructed to produce varying shades of sensitivity (e.g., 0.4 for masked credit card, 0.7 for real address) so the student learns the boundary.

**4. Static category-wise few-shot examples**  

- To ground Claude’s behavior, few-shot examples were introduced.  
- A `few_shot_examples/` directory was added with **10 curated examples per category** (positive and negative), stored as JSON and injected into each category prompt.

**5. SOLID refactoring**  

- The monolithic script was refactored into a modular `src/` layout:  
  - `config.py` — environment and paths  
  - `prompt_manager.py` — loads prompts and few-shots  
  - `generator.py` — AsyncAnthropic API calls  
  - `data_saver.py` — writes chunked outputs to `data/`  
  - `pipeline.py` — orchestrates chunking, context injection, and balance tracking
- This improved error handling, testability, and separation of concerns.

**6. Context-aware chunking and balanced generation**  

- **Chunking:** Generating 250 items at once exceeded Claude’s max output tokens and caused JSON truncation. The pipeline batches generation in **chunks of 50**.  
- **Context-aware deduplication:** Previously generated texts are fed back into the system prompt for the next chunk with instructions not to repeat them.  
- **Dynamic balance:** A 50/50 mix of positive (score ≥ 0.5) and negative (< 0.5) is enforced per category. After each chunk, the running totals are computed and passed to the model so it can correct the balance in the next chunk.

### 3.3 Directory Structure and Usage

- `**generate_data.py`** — Main CLI entrypoint.  
- `**prompts/**` — Category-specific instruction files.  
- `**few_shot_examples/**` — 10-shot JSON per category.  
- `**src/**` — `config.py`, `prompt_manager.py`, `generator.py`, `data_saver.py`, `pipeline.py`.  
- **Output** — Chunked arrays written under the repo’s `data/` directory (or configured path).

**Usage:**

- Generate all five categories (1,250 examples total):  
`python3 synthetic_data_generation/generate_data.py`  
- Generate a single category (e.g. for iteration):  
`python3 synthetic_data_generation/generate_data.py --category contact_info`

The resulting data is then split and formatted into `datasets/train.json`, `datasets/val.json`, and `datasets/test.json` for training (splits and formatting can be done by separate scripts or manually).

---

## 4. Training

### 4.1 Overview

- **Base model:** `microsoft/deberta-v3-small`  
- **Task:** Regression — predict a continuous **sensitivity score** in [0, 1]. The API later thresholds at 0.5 for `has_sensitive_content` and exposes the score as `confidence`.  
- **Data:** JSON under `datasets/` with `text` and either `sensitivity_score` or `label` (float in [0, 1]).  
- **Output:** `models/detector_v1` (PyTorch model + tokenizer, safetensors).

### 4.2 How Training Is Performed

1. **Data loading**
  `PiiDataLoader` reads `datasets/train.json` and `datasets/val.json`. Each item must have `text` and a numeric label. The loader maps `sensitivity_score` or `label` to the key `label` and casts to float so the Hugging Face Trainer treats it as regression.
2. **Tokenization**
  Texts are tokenized with the DeBERTa tokenizer, with truncation to `max_length=512` and no padding at map time; padding is applied per batch by the collator.
3. **Model**
  `AutoModelForSequenceClassification.from_pretrained(..., num_labels=1)` gives a regression head (single output). The model is kept in **float32** (`model.float()`) for the full run.
4. **Training loop**
  - **TrainingArguments:** 5 epochs, batch size 16, learning rate 2e-5, weight decay 0.01.  
  - Eval and save every epoch; best model selected by **MSE** (lower is better).  
  - **No mixed precision:** `bf16=False`, `fp16=False` to avoid NaNs (see below).  
  - `DataCollatorWithPadding` pads each batch to the longest sequence in the batch.
5. **Saving**
  Model and tokenizer are saved to `models/detector_v1` with `save_pretrained(..., safe_serialization=True)` (safetensors).

### 4.3 Nuances and Issues Encountered

- **Regression vs classification**  
We use `num_labels=1` (regression), not a two-class classifier. The head is a single linear layer with no sigmoid. At inference we clip the raw logit to [0, 1] and use 0.5 as the decision threshold. All dataset and metric code uses float labels and treats the output as a continuous score.
- **Float32 only**  
Training with `bf16` or `fp16` produced NaNs in the forward pass with this head/DeBERTa setup. Both were disabled; the model is kept in float32. Training is slower and uses more GPU memory but is stable.
- **Dataset label key**  
Hugging Face regression expects a float `label`. Our JSON uses `sensitivity_score` or `label`; the loader normalizes to the `label` key and `float(...)` so the Trainer and `compute_metrics` see consistent regression labels.
- **Metrics**  
We report MSE and R² for regression, and F1, precision, recall, accuracy, and FP-rate using a 0.5 threshold on predictions and labels. The best checkpoint is chosen by **MSE** only.
- **Saving / loading**  
We save with `safe_serialization=True`. When loading the model (e.g. in the server or export), we always pass `num_labels=1` and `use_safetensors=True` so the regression head and weights load correctly.
- **Collapsed model**  
If every test input gets the same score after training, the model may have collapsed. The test script in `test_files/` warns in that case; remedies include more data, different hyperparameters, or checking label balance.

### 4.4 Training Pipeline Structure

- `**TrainingConfig`** — Dataclass for paths, batch size, epochs, LR, max length, device.  
- `**PiiDataLoader**` — Loads and tokenizes datasets.  
- `**DetectorModelManager**` — Loads model, defines `compute_metrics`, runs the Trainer, saves to `models/detector_v1`.  
- `**DistillationPipeline**` — Orchestrates load → tokenize → train → save.

**Run (from repo root):**

```bash
pip install -r scaffold/requirements.txt
python training/train.py
```

---

## 5. Inference and Serving

### 5.1 API

- **Single:** `POST /detect` — body `{"text": "..."}`; response `{"has_sensitive_content": bool, "confidence": float}`.  
- **Batch:** `POST /detect/batch` — body `{"texts": ["...", ...]}`; response `{"results": [{...}, ...]}`.  
- Confidence is the model score clipped to [0, 1]; `has_sensitive_content` is `confidence >= 0.5`.

### 5.2 Model Loading and ONNX

- The server loads from `models/detector_v1` if present; otherwise it downloads a public model from Hugging Face, caches it under `models/detector_v1`, and then exports to ONNX.  
- If `models/detector_v1/model.onnx` exists, the server uses **ONNX Runtime** (with CUDA when available) for inference; otherwise it falls back to PyTorch.  
- ONNX export uses dynamic batch and sequence axes; DeBERTa uses only `input_ids` and `attention_mask` (no `token_type_ids`).  
- Batch requests are processed in **chunks of 256** for better GPU utilization and latency; batch inputs are truncated to `BATCH_MAX_LENGTH=256` tokens.

### 5.3 Implementation Notes

- `**scaffold/server.py`** — FastAPI app, model/ONNX loading, single and chunked batch handlers.  
- `**scaffold/onnx_inference.py**` — Class-based design: `ONNXExporter` (export only), `ONNXSessionLoader`, `ONNXDetector` (inference), and a minimal `TokenizerProtocol` for dependency injection.  
- Inference is in float32; logits are squeezed and clipped to [0, 1] for the API response.

---

## 6. End-to-End Pipeline Summary


| Stage               | Input                               | Output                                         | Main components                                                                                                                           |
| ------------------- | ----------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Synthetic data**  | Prompts, few-shots, category list   | JSON with `text` + `sensitivity_score`         | `synthetic_data_generation/`: prompts, few_shot_examples, src (config, prompt_manager, generator, data_saver, pipeline), generate_data.py |
| **Train/val split** | Raw generated + any seed data       | `datasets/train.json`, `val.json`, `test.json` | Manual or scripted split; keys `text` and `sensitivity_score` (or `label`)                                                                |
| **Training**        | train.json, val.json                | `models/detector_v1`                           | `training/train.py`: PiiDataLoader, DetectorModelManager, DistillationPipeline                                                            |
| **Serving**         | models/detector_v1 (or HF download) | REST API                                       | scaffold/server.py, onnx_inference.py; ONNX optional for low latency                                                                      |


---

## 7. How This Addresses the Evaluation Criteria

- **F1 / Recall / Precision:** Regression with soft labels and a 0.5 threshold lets the model learn the boundary; category-specific synthetic data and balance (50/50 positive/negative per category) improve discrimination.  
- **FP rate:** Balanced data and explicit “safe” prompts (e.g. `safe_cases.txt`) reduce over-flagging of benign text.  
- **Latency:** ONNX Runtime with GPU, chunked batch inference (256), and batch max length 256 keep single and batch latency low.

---

## 8. Repo Layout (Relevant Parts)

- `**synthetic_data_generation/`** — Data generation pipeline (prompts, few-shots, src, generate_data.py).  
- `**training/**` — Training pipeline (train.py, README).  
- `**scaffold/**` — Server (server.py), ONNX module (onnx_inference.py), requirements.  
- `**datasets/**` — train.json, val.json, test.json.  
- `**models/detector_v1/**` — Trained model and tokenizer; optional `model.onnx`.  
- `**test_files/**` — Scripts for latency tests, model checks, pretrained vs distilled comparison.

This report summarizes the full approach: synthetic data generation with Claude, training of DeBERTa-v3-small with regression and the above nuances, and serving via FastAPI with optional ONNX for low-latency detection.