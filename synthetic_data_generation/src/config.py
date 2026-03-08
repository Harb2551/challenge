import os
from dotenv import load_dotenv

class Config:
    """Manages application configuration and paths."""
    def __init__(self):
        # We assume this script is loaded from src/
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = os.path.dirname(self.script_dir)
        self.prompts_dir = os.path.join(self.script_dir, "prompts")
        self.few_shots_dir = os.path.join(self.script_dir, "few_shot_examples")
        self.data_dir = os.path.join(self.project_root, "data")
        self.model_name = "claude-haiku-4-5-20251001"
        self.max_tokens = 8192
        
        self._load_env()
        self.api_key = self._get_api_key()

    def _load_env(self):
        dotenv_path = os.path.join(self.project_root, '.env')
        load_dotenv(dotenv_path)

    def _get_api_key(self) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        return api_key
