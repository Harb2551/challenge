import os
import json
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score, accuracy_score

@dataclass
class TrainingConfig:
    """Configuration for the distillation training process."""
    model_name: str = "microsoft/deberta-v3-small"
    train_file: str = "datasets/train.json"
    val_file: str = "datasets/val.json"
    test_file: str = "datasets/test.json"
    output_dir: str = "models/detector_v1"
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PiiDataLoader:
    """Responsible for loading and formatting datasets for Transformers."""
    def __init__(self, config: TrainingConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def load_dataset(self, file_path: str) -> Dataset:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # HuggingFace Dataset requires internal 'label' key to be float for regression
        hf_data = [{"text": x["text"], "label": float(x.get("sensitivity_score", x.get("label", 0)))} for x in data]
        return Dataset.from_list(hf_data)

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"], 
            truncation=True, 
            padding=False, # Collator will handle padding
            max_length=self.config.max_length
        )

class DetectorModelManager:
    """Manages the lifecycle of the DeBERTa detector model."""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # num_labels=1 triggers Regression mode in Transformers
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, 
            num_labels=1 
        )
        # Ensure float32 so forward pass doesn't produce NaN (matches inference)
        self.model = self.model.float()

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # Squeeze because regression output is (batch, 1)
        predictions = np.asarray(predictions).squeeze()
        labels = np.asarray(labels)
        
        # Regression metrics (will raise if predictions contain NaN — no masking)
        mse = mean_squared_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        
        # Classification metrics (using 0.5 threshold)
        # Convert continuous labels/preds to binary
        bin_labels = (labels >= 0.5).astype(int)
        bin_preds = (predictions >= 0.5).astype(int)
        
        f1 = f1_score(bin_labels, bin_preds, zero_division=0)
        precision = precision_score(bin_labels, bin_preds, zero_division=0)
        recall = recall_score(bin_labels, bin_preds, zero_division=0)
        accuracy = accuracy_score(bin_labels, bin_preds)
        
        # FP Rate calculation: FP / (FP + TN)
        fps = np.sum((bin_preds == 1) & (bin_labels == 0))
        tns = np.sum((bin_preds == 0) & (bin_labels == 0))
        fp_rate = fps / (fps + tns) if (fps + tns) > 0 else 0
        
        return {
            "mse": mse, 
            "r2": r2,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "fp_rate": fp_rate
        }

    def train(self, train_ds: Dataset, val_ds: Dataset):
        # Use /workspace on RunPod (root is 20G and often full); keep default elsewhere
        output_dir = self.config.output_dir
        if os.path.isdir("/workspace"):
            output_dir = os.path.join("/workspace", "models", "detector_v1")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Using output_dir on /workspace (avoid full disk): {output_dir}")
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="mse",
            greater_is_better=False,
            # Train in float32 for stability (slower, more GPU memory than bf16)
            bf16=False,
            fp16=False,
            logging_steps=50,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics
        )

        print(f"Starting training on {self.config.device}...")
        trainer.train()
        
        # Save final artifacts
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"Model saved to {self.config.output_dir}")

class DistillationPipeline:
    """Orchestrates the entire training/distillation workflow."""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.manager = DetectorModelManager(config)
        self.loader = PiiDataLoader(config, self.manager.tokenizer)

    def run(self):
        print("--- Initiating Distillation Pipeline ---")
        
        # 1. Load data
        train_raw = self.loader.load_dataset(self.config.train_file)
        val_raw = self.loader.load_dataset(self.config.val_file)
        
        # 2. Tokenize
        train_ds = train_raw.map(self.loader.tokenize_function, batched=True)
        val_ds = val_raw.map(self.loader.tokenize_function, batched=True)
        
        # 3. Train
        self.manager.train(train_ds, val_ds)
        print("--- Distillation Complete ---")

if __name__ == "__main__":
    config = TrainingConfig()
    pipeline = DistillationPipeline(config)
    pipeline.run()
