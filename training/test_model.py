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
        model_path, num_labels=1, use_safetensors=True
    ).to(device)
    model.eval()
    # Force float32 to avoid bf16/fp16 NaN from precision
    model = model.float()

    # Trace where NaN first appears: embeddings -> backbone -> head
    one_text = "My password is admin123"
    inputs = tokenizer(one_text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    vocab_size = getattr(config, "vocab_size", None) or getattr(tokenizer, "vocab_size", 0)
    ids_ok = (input_ids >= 0).all().item() and (input_ids < vocab_size).all().item()
    print(f"[debug] input_ids in vocab [0, {vocab_size}): {ids_ok}")
    with torch.no_grad():
        base = getattr(model, "deberta", getattr(model, "roberta", model.base_model))
        emb = base.get_input_embeddings()(input_ids)
        emb_has_nan = torch.isnan(emb).any().item()
        print(f"[debug] Embedding output has NaN: {emb_has_nan}")
        base_out = base(**inputs)
        hidden = base_out.last_hidden_state
        pooled = hidden[:, 0, :]
        backbone_has_nan = torch.isnan(pooled).any().item()
        head_module = getattr(model, "classifier", getattr(model, "score", None))
        if head_module is not None:
            head_out = head_module(pooled)
            head_has_nan = torch.isnan(head_out).any().item()
            print(f"[debug] Backbone (CLS) has NaN: {backbone_has_nan}, Head output has NaN: {head_has_nan}")
        else:
            print(f"[debug] Backbone (CLS) has NaN: {backbone_has_nan}")

    test_sentences = [
        "My password is admin123", # Positive
        "Hello, how are you today?", # Negative
        "Send me the report at john.doe@example.com", # Positive
        "The weather is nice in London.", # Negative
        "API Key: sk-1234567890abcdef", # Positive
    ]

    print("\n--- Model Predictions ---")
    scores = []
    for i, text in enumerate(test_sentences):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if i == 0:
                print(f"[debug] logits shape: {logits.shape}, raw: {logits.cpu().tolist()}\n")
            raw = logits.squeeze().item() if logits.numel() == 1 else logits[0, 0].item()
            score = max(0.0, min(1.0, raw))  # head is linear (no sigmoid); clip to [0,1]
            scores.append(score)
            is_sensitive = score >= 0.5
            
        status = "🚩 SENSITIVE" if is_sensitive else "✅ SAFE"
        score_str = f"{score:.4f}" if not (isinstance(score, float) and (score != score)) else "nan"
        print(f"Text: {text}")
        print(f"Score: {score_str} -> {status}\n")

    if len(scores) > 1 and all(s == scores[0] for s in scores):
        print("⚠️  All scores identical — model may have collapsed during training (not using input). Consider retraining with more data or different hyperparameters.")

if __name__ == "__main__":
    test_inference()
