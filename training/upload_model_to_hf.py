"""
Upload the trained model (models/detector_v1) to Hugging Face Hub as a public model.

Prerequisites:
  - Model must exist: run training/train.py first.
  - Log in to HF: huggingface-cli login (or set HF_TOKEN).

Usage:
  python training/upload_model_to_hf.py [--repo-id REPO_ID] [--token HF_TOKEN]
  HF_TOKEN=your_token python training/upload_model_to_hf.py   # no CLI needed

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
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()

    token = args.token
    if not token:
        print("Warning: No HF token. Set HF_TOKEN or use --token to avoid 401.")
    else:
        print("Using HF token from HF_TOKEN / --token.")

    if not os.path.isdir(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run training/train.py first.")
        return 1

    print(f"Loading from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1, use_safetensors=True)

    print(f"Pushing to https://huggingface.co/{args.repo_id} ...")
    try:
        tokenizer.push_to_hub(args.repo_id, token=token)
        model.push_to_hub(args.repo_id, safe_serialization=True, token=token)
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 401:
            print("Error: Not logged in to Hugging Face. Use a token:")
            print("  export HF_TOKEN=your_token   # create at https://huggingface.co/settings/tokens")
            print("  python training/upload_model_to_hf.py")
            print("Or:  python -m huggingface_hub.cli.login   # if CLI is not in PATH")
        else:
            print(f"Error: {e}")
        return 1

    print("Done. Set the repo to Public in Settings if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
