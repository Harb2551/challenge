"""
Upload the trained model (models/detector_v1) to Hugging Face Hub as a public model.

Prerequisites:
  - Model must exist: run training/train.py first.
  - Log in to HF: huggingface-cli login (or set HF_TOKEN).

Usage:
  python training/upload_model_to_hf.py [--repo-id REPO_ID]

Default repo-id: harshit2551/challenge-detector-v1 (must match HF_MODEL_ID in scaffold/server.py).
"""

import argparse
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/detector_v1"
DEFAULT_REPO_ID = "harshit2551/challenge-detector-v1"


def main():
    parser = argparse.ArgumentParser(description="Upload detector model to Hugging Face Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF repo (e.g. username/repo-name)")
    args = parser.parse_args()

    if not os.path.isdir(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run training/train.py first.")
        return 1

    print(f"Loading from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1, use_safetensors=True)

    print(f"Pushing to https://huggingface.co/{args.repo_id} ...")
    tokenizer.push_to_hub(args.repo_id)
    model.push_to_hub(args.repo_id, safe_serialization=True)

    print("Done. Set the repo to Public in Settings if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
