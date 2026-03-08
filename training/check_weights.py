import torch
from transformers import AutoModelForSequenceClassification

def check_weights():
    model_path = "models/detector_v1"
    print(f"Checking weights in {model_path}...")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        nan_layers = []
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_layers.append(name)
        
        if nan_layers:
            print("\n❌ CRITICAL: NaNs found in the following layers:")
            for layer in nan_layers[:10]:
                print(f" - {layer}")
            if len(nan_layers) > 10:
                print(f" ... and {len(nan_layers)-10} more.")
        else:
            print("\n✅ No NaNs found in weights. The issue might be in the tokenizer or input processing.")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_weights()
