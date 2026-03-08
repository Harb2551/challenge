import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

def test_inference():
    model_path = "models/detector_v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    saved_num_labels = getattr(config, "num_labels", None)
    print(f"Saved config num_labels: {saved_num_labels} (use 1 for regression)")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Regression head (num_labels=1) — must match how the model was trained
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1
    ).to(device)
    model.eval()

    test_sentences = [
        "My password is admin123", # Positive
        "Hello, how are you today?", # Negative
        "Send me the report at john.doe@example.com", # Positive
        "The weather is nice in London.", # Negative
        "API Key: sk-1234567890abcdef", # Positive
    ]

    print("\n--- Model Predictions ---")
    for i, text in enumerate(test_sentences):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if i == 0:
                print(f"[debug] logits shape: {logits.shape}, raw: {logits.cpu().tolist()}\n")
            score = logits.squeeze().item() if logits.numel() == 1 else logits[0, 0].item()
            is_sensitive = score >= 0.5
            
        status = "🚩 SENSITIVE" if is_sensitive else "✅ SAFE"
        score_str = f"{score:.4f}" if not (isinstance(score, float) and (score != score)) else "nan"
        print(f"Text: {text}")
        print(f"Score: {score_str} -> {status}\n")

if __name__ == "__main__":
    test_inference()
