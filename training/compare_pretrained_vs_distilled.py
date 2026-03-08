"""
Compare pretrained (microsoft/deberta-v3-small) vs distilled model on the test set.
Run: python3 training/compare_pretrained_vs_distilled.py
"""
import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score, accuracy_score

TEST_FILE = "datasets/test.json"
PRETRAINED_NAME = "microsoft/deberta-v3-small"
DISTILLED_PATH = "models/detector_v1"
MAX_LENGTH = 512
BATCH_SIZE = 32
FIRST_N = 50  # show first N test samples with predictions
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "..", "pretrained_vs_distilled_results.json")


def load_test_data(path: str):
    with open(path) as f:
        data = json.load(f)
    texts = [x["text"] for x in data]
    labels = np.array([float(x.get("sensitivity_score", x.get("label", 0))) for x in data])
    return texts, labels


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    predictions = np.asarray(predictions).squeeze()
    labels = np.asarray(labels)
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    bin_labels = (labels >= 0.5).astype(int)
    bin_preds = (predictions >= 0.5).astype(int)
    fps = np.sum((bin_preds == 1) & (bin_labels == 0))
    tns = np.sum((bin_preds == 0) & (bin_labels == 0))
    fp_rate = fps / (fps + tns) if (fps + tns) > 0 else 0
    return {
        "mse": mse,
        "r2": r2,
        "f1": f1_score(bin_labels, bin_preds, zero_division=0),
        "precision": precision_score(bin_labels, bin_preds, zero_division=0),
        "recall": recall_score(bin_labels, bin_preds, zero_division=0),
        "accuracy": accuracy_score(bin_labels, bin_preds),
        "fp_rate": fp_rate,
    }


def run_eval(model, tokenizer, texts, labels, device, name: str) -> dict:
    model.eval()
    model = model.float()
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    all_preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        enc = tokenizer(batch_texts, truncation=True, max_length=MAX_LENGTH, padding=False)
        batch = collator(enc)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        preds = out.logits.squeeze(-1).cpu().numpy()
        if preds.ndim == 0:
            preds = np.array([preds])
        preds = np.clip(preds, 0.0, 1.0)  # head is linear (no sigmoid); clip for [0,1] score
        all_preds.append(preds)
    predictions = np.concatenate(all_preds)
    return compute_metrics(labels, predictions), predictions


def print_metrics(metrics: dict, title: str):
    print(f"\n--- {title} ---")
    print(f"  MSE:       {metrics['mse']:.4f}")
    print(f"  R2:        {metrics['r2']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  FP rate:   {metrics['fp_rate']:.4f}")


def main():
    if not os.path.exists(TEST_FILE):
        print(f"Test file not found: {TEST_FILE}")
        return
    texts, labels = load_test_data(TEST_FILE)
    print(f"Loaded {len(texts)} test samples from {TEST_FILE}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Pretrained (untrained head — baseline)
    print(f"\nLoading pretrained: {PRETRAINED_NAME} ...")
    tokenizer_p = AutoTokenizer.from_pretrained(PRETRAINED_NAME)
    model_p = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_NAME, num_labels=1).to(device)
    metrics_pretrained, preds_pretrained = run_eval(model_p, tokenizer_p, texts, labels, device, "Pretrained")
    print_metrics(metrics_pretrained, "Pretrained (no fine-tuning)")

    # 2) Distilled (fine-tuned)
    if not os.path.isdir(DISTILLED_PATH):
        print(f"\nDistilled model not found: {DISTILLED_PATH} (train first)")
        return
    print(f"\nLoading distilled: {DISTILLED_PATH} ...")
    tokenizer_d = AutoTokenizer.from_pretrained(DISTILLED_PATH)
    model_d = AutoModelForSequenceClassification.from_pretrained(DISTILLED_PATH, num_labels=1).to(device)
    metrics_distilled, preds_distilled = run_eval(model_d, tokenizer_d, texts, labels, device, "Distilled")
    print_metrics(metrics_distilled, "Distilled (fine-tuned)")

    # 3) First N samples: text, label, pretrained pred, distilled pred
    n_show = min(FIRST_N, len(texts))
    print(f"\n--- First {n_show} samples (label | pretrained | distilled) ---")
    for i in range(n_show):
        text_preview = (texts[i][:70] + "…") if len(texts[i]) > 70 else texts[i]
        label = float(labels[i])
        p_p = float(preds_pretrained[i])
        p_d = float(preds_distilled[i])
        ok_p = "✓" if (p_p >= 0.5) == (label >= 0.5) else "✗"
        ok_d = "✓" if (p_d >= 0.5) == (label >= 0.5) else "✗"
        print(f"  [{i+1}] {text_preview}")
        print(f"       label={label:.2f} | pretrained={p_p:.3f} {ok_p} | distilled={p_d:.3f} {ok_d}")

    # 4) Summary
    print("\n--- Summary (distillation effect) ---")
    print(f"  MSE:       {metrics_pretrained['mse']:.4f} -> {metrics_distilled['mse']:.4f} (lower is better)")
    print(f"  R2:        {metrics_pretrained['r2']:.4f} -> {metrics_distilled['r2']:.4f} (higher is better)")
    print(f"  F1:        {metrics_pretrained['f1']:.4f} -> {metrics_distilled['f1']:.4f} (higher is better)")
    print(f"  Accuracy:  {metrics_pretrained['accuracy']:.4f} -> {metrics_distilled['accuracy']:.4f} (higher is better)")

    # 5) Save full results to JSON
    results = [
        {
            "query": text,
            "ground_truth_score": round(float(labels[i]), 4),
            "pretrained_score": round(float(preds_pretrained[i]), 4),
            "distilled_score": round(float(preds_distilled[i]), 4),
        }
        for i, text in enumerate(texts)
    ]
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON} ({len(results)} rows)")


if __name__ == "__main__":
    main()
