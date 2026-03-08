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
DISTILLED_PATH = "/workspace/models/detector_v1"
MAX_LENGTH = 512
BATCH_SIZE = 32


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
        all_preds.append(preds)
    predictions = np.concatenate(all_preds)
    return compute_metrics(labels, predictions)


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
    metrics_pretrained = run_eval(model_p, tokenizer_p, texts, labels, device, "Pretrained")
    print_metrics(metrics_pretrained, "Pretrained (no fine-tuning)")

    # 2) Distilled (fine-tuned)
    if not os.path.isdir(DISTILLED_PATH):
        print(f"\nDistilled model not found: {DISTILLED_PATH} (train first)")
        return
    print(f"\nLoading distilled: {DISTILLED_PATH} ...")
    tokenizer_d = AutoTokenizer.from_pretrained(DISTILLED_PATH)
    model_d = AutoModelForSequenceClassification.from_pretrained(DISTILLED_PATH, num_labels=1).to(device)
    metrics_distilled = run_eval(model_d, tokenizer_d, texts, labels, device, "Distilled")
    print_metrics(metrics_distilled, "Distilled (fine-tuned)")

    # 3) Summary
    print("\n--- Summary (distillation effect) ---")
    print(f"  MSE:       {metrics_pretrained['mse']:.4f} -> {metrics_distilled['mse']:.4f} (lower is better)")
    print(f"  R2:        {metrics_pretrained['r2']:.4f} -> {metrics_distilled['r2']:.4f} (higher is better)")
    print(f"  F1:        {metrics_pretrained['f1']:.4f} -> {metrics_distilled['f1']:.4f} (higher is better)")
    print(f"  Accuracy:  {metrics_pretrained['accuracy']:.4f} -> {metrics_distilled['accuracy']:.4f} (higher is better)")


if __name__ == "__main__":
    main()
