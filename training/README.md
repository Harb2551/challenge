# Training the sensitive-content detector

This folder contains the pipeline to train the detector model.

## Overview

- **Base model:** `microsoft/deberta-v3-small`
- **Task:** Regression — predict a continuous **sensitivity score** in `[0, 1]`. Downstream (e.g. the API) thresholds at 0.5 for `has_sensitive_content` and can expose the score as confidence.
- **Data:** JSON files under `datasets/` with `text` and either `sensitivity_score` or `label` (float in [0,1]).
- **Output:** `models/detector_v1` (PyTorch + tokenizer, safetensors).

## How to run

From the **repo root**:

```bash
# 1. Install dependencies (includes transformers, datasets, torch, etc.)
pip install -r scaffold/requirements.txt

# 2. Train (reads datasets/train.json, datasets/val.json; writes models/detector_v1)
python training/train.py
```

## How training is performed

1. **Data loading**  
   `PiiDataLoader` reads `datasets/train.json` and `datasets/val.json`. Each item must have `text` and a numeric label. We map `sensitivity_score` or `label` to the key `label` and cast to float so Hugging Face treats it as regression.

2. **Tokenization**  
   Texts are tokenized with the DeBERTa tokenizer, truncation to `max_length=512`, no padding at map time (padding is done in the collator per batch).

3. **Model**  
   `AutoModelForSequenceClassification.from_pretrained(..., num_labels=1)` gives a **regression** head (single output). The model is kept in **float32** (`model.float()`) for the whole run.

4. **Training loop**  
   - `TrainingArguments`: 5 epochs, batch size 16, LR 2e-5, weight decay 0.01.  
   - Eval and save every epoch; best model selected by **MSE** (lower is better).  
   - **No mixed precision:** `bf16=False`, `fp16=False` to avoid NaNs (see nuances below).  
   - `DataCollatorWithPadding` pads batches to the longest sequence in the batch.

5. **Saving**  
   Model and tokenizer are saved to `models/detector_v1` with `save_pretrained(..., safe_serialization=True)` so weights are safetensors (no legacy `torch.load`).

## Nuances and issues we hit

- **Regression vs classification**  
  We use `num_labels=1` (regression), not a 2-class classifier. The head is a single linear layer; there is no sigmoid. At inference we clip the raw logit to `[0, 1]` and use 0.5 as the decision threshold. Any dataset or metric code must use float labels and treat the output as a continuous score.

- **Float32 only**  
  Training with `bf16` or `fp16` led to NaNs in the forward pass (especially with this head/DeBERTa combo). We disabled both and keep the model in float32. Training is a bit slower and uses more GPU memory but is stable.

- **Dataset label key**  
  Hugging Face regression expects a float `label`. Our JSON uses `sensitivity_score` (or `label`). The loader normalizes to the `label` key and `float(...)` so the Trainer and `compute_metrics` see consistent regression labels.

- **Metrics**  
  We report MSE and R² for regression, and F1/precision/recall/accuracy/FP-rate using a 0.5 threshold on predictions and labels. Best checkpoint is chosen by **MSE** only.

- **Saving / loading**  
  We save with `safe_serialization=True`. When loading the model elsewhere, always pass `num_labels=1` and `use_safetensors=True` so the regression head and weights load correctly.

- **Collapsed model**  
  If after training every test input gets the same score, the model may have collapsed. The test script in `test_files/` warns in that case; consider more data, different hyperparameters, or checking for label imbalance.

## Files in this folder

| File | Purpose |
|------|--------|
| `train.py` | Main training pipeline: config, data loader, model manager, Trainer, save to `models/detector_v1`. |
