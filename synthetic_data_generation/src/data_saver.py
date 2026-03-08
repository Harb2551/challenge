import os
import json
from typing import List, Dict
from .config import Config

class DataSaver:
    """Handles saving the generated data to disk."""
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(self.config.data_dir, exist_ok=True)

    def save_category_data(self, category_name: str, data: List[Dict]):
        if not data:
            print(f"WARNING: No data to save for {category_name}.")
            return
            
        out_path = os.path.join(self.config.data_dir, f"{category_name}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f" -> Saved {len(data)} items to {out_path}")
