import json
import os
import random
import glob
from typing import List, Dict, Tuple

class DataSplitter:
    """Handles the complex sampling and splitting logic for the training dataset."""
    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = data_dir
        random.seed(seed)

    def load_all_samples(self) -> Dict[str, List[Dict]]:
        """Loads all samples categorized by filename."""
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        return {os.path.splitext(os.path.basename(f))[0]: self._load_json(f) for f in files}

    def _load_json(self, path: str) -> List[Dict]:
        with open(path, 'r') as f:
            return json.load(f)

    def perform_split(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Implements the specific requirements:
        - 500 +ve from each of 4 categories (contact_info, credentials, identifiers, other_pii).
        - 2000 -ve from the remaining pool.
        - Out of 1000 left: 300 for Val, 700 for Test.
        """
        all_categories = self.load_all_samples()
        pos_categories = ["contact_info", "credentials", "identifiers", "other_pii"]
        
        train_pos_pool = []
        remainder_pool = []

        # 1. Take 500 positive from each of 4 categories
        for cat in pos_categories:
            data = all_categories.get(cat, [])
            pos_samples = [d for d in data if d.get("sensitivity_score", d.get("label", 0)) >= 0.5]
            neg_samples = [d for d in data if d.get("sensitivity_score", d.get("label", 0)) < 0.5]
            
            # Shuffle and pick 500
            random.shuffle(pos_samples)
            train_pos_pool.extend(pos_samples[:500])
            
            # Put rest in remainder pool
            remainder_pool.extend(pos_samples[500:])
            remainder_pool.extend(neg_samples)

        # Also add safe_cases to remainder (it's entirely negative)
        if "safe_cases" in all_categories:
            remainder_pool.extend(all_categories["safe_cases"])

        # 2. From negative examples in remainder, pick 2000
        neg_in_remainder = [d for d in remainder_pool if d.get("sensitivity_score", d.get("label", 0)) < 0.5]
        pos_in_remainder = [d for d in remainder_pool if d.get("sensitivity_score", d.get("label", 0)) >= 0.5]
        
        random.shuffle(neg_in_remainder)
        train_neg_pool = neg_in_remainder[:2000]
        
        # Everything not used in Train goes into the final split pool
        final_remnant = neg_in_remainder[2000:] + pos_in_remainder
        
        # 3. Combine Train
        train_set = train_pos_pool + train_neg_pool
        random.shuffle(train_set)

        # 4. Final Split: 300 Val, remainder (~700) Test
        random.shuffle(final_remnant)
        val_set = final_remnant[:300]
        test_set = final_remnant[300:]

        return train_set, val_set, test_set

class DatasetSaver:
    """Responsible for writing split data to disk."""
    @staticmethod
    def save(train: List[Dict], val: List[Dict], test: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for name, data in [("train.json", train), ("val.json", val), ("test.json", test)]:
            path = os.path.join(output_dir, name)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data)} items to {path}")

if __name__ == "__main__":
    project_root = os.getcwd()
    data_path = os.path.join(project_root, "data")
    output_path = os.path.join(project_root, "datasets")
    
    splitter = DataSplitter(data_path)
    train, val, test = splitter.perform_split()
    
    saver = DatasetSaver()
    saver.save(train, val, test, output_path)
    
    print("\nSplit Summary:")
    print(f"Train: {len(train)} (Target: 4000)")
    print(f"Val:   {len(val)} (Target: 300)")
    print(f"Test:  {len(test)} (Target: 700)")
