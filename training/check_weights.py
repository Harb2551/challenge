import torch
from transformers import AutoModelForSequenceClassification

def check_weights():
    model_path = "models/detector_v1"
    print(f"Checking weights in {model_path}...")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        nan_param_layers = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_param_layers.append(name)
        nan_buf_layers = []
        for name, buf in model.named_buffers():
            if torch.isnan(buf).any():
                nan_buf_layers.append(name)
        if nan_param_layers:
            print("\n❌ NaNs in parameters:")
            for layer in nan_param_layers[:10]:
                print(f" - {layer}")
            if len(nan_param_layers) > 10:
                print(f" ... and {len(nan_param_layers)-10} more.")
        if nan_buf_layers:
            print("\n❌ NaNs in buffers (e.g. LayerNorm):")
            for layer in nan_buf_layers[:10]:
                print(f" - {layer}")
            if len(nan_buf_layers) > 10:
                print(f" ... and {len(nan_buf_layers)-10} more.")
        if not nan_param_layers and not nan_buf_layers:
            print("\n✅ No NaNs in parameters or buffers. NaN likely from forward (dtype/precision or inputs).")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_weights()
