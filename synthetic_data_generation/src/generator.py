import json
from typing import List, Dict
from anthropic import AsyncAnthropic
from .config import Config

class ClaudeGenerator:
    """Responsible for interacting with the Anthropic API to generate data."""
    def __init__(self, config: Config):
        self.config = config
        self.client = AsyncAnthropic(api_key=config.api_key)
        self.system_prompt = (
            "You are a teacher model for a distillation task. "
            "Your goal is to generate diverse training examples for a 'Sensitive Content Detection' API. "
            "CRITICAL INSTRUCTION: Do not just output raw secrets or raw JSON templates. Embed the secrets/text in realistic contexts where possible (e.g., conversational messages, code comments, log snippets, customer support tickets). We need a mix of raw strings and embedded context. "
            "Return ONLY a JSON array of objects. Never include conversational text outside the JSON."
        )

    async def generate(self, user_prompt: str, target_count: int, previous_samples: List[Dict]) -> List[Dict]:
        try:
            # We enforce context so it doesn't repeat what it already made in this chunk chain.
            if previous_samples:
                # We limit the context to text strings so it doesn't blow up the prompt size immediately.
                prev_texts = [s["text"] for s in previous_samples]
                # To prevent massive token explosion, we only feed the last 150 generated texts as context
                context_texts = prev_texts[-150:]
                context_str = json.dumps(context_texts, indent=2)
                
                user_prompt += f"\n\nIMPORTANT: You have already generated {len(previous_samples)} samples in this session.\n"
                user_prompt += f"DO NOT REPEAT ANY OF THE FOLLOWING TEXTS:\n{context_str}\n"
                user_prompt += f"\nPlease generate exactly {target_count} MORE entirely NEW and DIFFERENT examples to reach our target."
            
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return self._parse_json_response(response.content[0].text)
        except Exception as e:
            print(f"API Error during generation: {e}")
            return []

    def _parse_json_response(self, text: str) -> List[Dict]:
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end <= 0:
                raise ValueError("No JSON array found in response")
            return json.loads(text[start:end])
        except Exception as e:
            print(f"Error parsing JSON from response: {e}")
            return []
