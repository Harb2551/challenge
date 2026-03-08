import os
import glob
from typing import List, Optional
from .config import Config

class PromptManager:
    """Handles loading of prompts and few-shot examples."""
    def __init__(self, config: Config):
        self.config = config

    def get_prompt_files(self, category: Optional[str] = None) -> List[str]:
        if category:
            target_file = os.path.join(self.config.prompts_dir, f"{category}.txt")
            if not os.path.exists(target_file):
                raise FileNotFoundError(f"Category file {target_file} not found.")
            return [target_file]
        return sorted(glob.glob(os.path.join(self.config.prompts_dir, "*.txt")))

    def load_prompt_content(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            return f.read()

    def load_few_shots(self, category_name: str) -> str:
        shot_path = os.path.join(self.config.few_shots_dir, f"{category_name}.json")
        if os.path.exists(shot_path):
            with open(shot_path, "r") as f:
                return f.read()
        return "[]"
