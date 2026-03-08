"""
Export the trained detector (models/detector_v1) to ONNX for fast inference.

Run after training, or use the server (it auto-exports after downloading from HF).

Usage:
  python training/export_onnx.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/detector_v1"
ONNX_PATH = os.path.join(MODEL_PATH, "model.onnx")
MAX_LENGTH = 512


def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train or run the server once to download the model.")
        return 1

    print(f"Loading from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=1, use_safetensors=True
    )
    model.eval()

    dummy = tokenizer(
        "dummy",
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    if "token_type_ids" in dummy:
        token_type_ids = dummy["token_type_ids"]
        args = (input_ids, attention_mask, token_type_ids)
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        }
    else:
        args = (input_ids, attention_mask)
        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        }

    print(f"Exporting to {ONNX_PATH}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            args,
            ONNX_PATH,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
    print("Done. Server will use ONNX on next start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
