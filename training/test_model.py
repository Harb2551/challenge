import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_inference():
    model_path = "models/detector_v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    test_sentences = [
        "My password is admin123", # Positive
        "Hello, how are you today?", # Negative
        "Send me the report at john.doe@example.com", # Positive
        "The weather is nice in London.", # Negative
        "API Key: sk-1234567890abcdef", # Positive
    ]

    print("\n--- Model Predictions ---")
    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits.item()
            is_sensitive = score >= 0.5
            
        status = "🚩 SENSITIVE" if is_sensitive else "✅ SAFE"
        print(f"Text: {text}")
        print(f"Score: {score:.4f} -> {status}\n")

if __name__ == "__main__":
    test_inference()
