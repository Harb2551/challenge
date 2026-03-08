import os
from typing import Optional
from .config import Config
from .prompt_manager import PromptManager
from .generator import ClaudeGenerator
from .data_saver import DataSaver

class DataGenerationPipeline:
    """Orchestrates the data generation process."""
    def __init__(self, config: Config):
        self.prompt_manager = PromptManager(config)
        self.generator = ClaudeGenerator(config)
        self.saver = DataSaver(config)

    async def process_category(self, file_path: str):
        category_name = os.path.splitext(os.path.basename(file_path))[0]
        prompt_content = self.prompt_manager.load_prompt_content(file_path)
        few_shots = self.prompt_manager.load_few_shots(category_name)
        
        # Extract quantity natively from prompt text
        total_to_generate = self._extract_quantity(prompt_content)
        
        print(f"Generating exactly {total_to_generate} examples for '{category_name}' in one go...")

        user_prompt = f"""{prompt_content}

Few-shot examples (DO NOT REPEAT THESE):
{few_shots}

TASK: We need a total of {total_to_generate} examples.
Return ONLY a valid JSON array.
"""
        
        all_category_data = []
        chunk_size = 50
        chunks_needed = (total_to_generate + chunk_size - 1) // chunk_size
        
        for i in range(chunks_needed):
            current_chunk_size = min(chunk_size, total_to_generate - len(all_category_data))
            if current_chunk_size <= 0: break
            
            print(f"  -> {category_name}: Generating chunk {i+1}/{chunks_needed} ({current_chunk_size} samples)...")
            
            # Calculate current distribution
            pos_count = sum(1 for d in all_category_data if d.get("sensitivity_score", d.get("label", 0)) >= 0.5)
            neg_count = len(all_category_data) - pos_count
            
            # Request specific chunk amount and enforce balance
            chunk_prompt = user_prompt + f"\n\nCURRENT TASK: Generate exactly {current_chunk_size} NEW examples now."
            if len(all_category_data) > 0:
                chunk_prompt += f"\n\nCURRENT PROGRESS SUMMARY:\n"
                chunk_prompt += f"- Total generated so far: {len(all_category_data)}\n"
                chunk_prompt += f"- Positive/Flagged (score >= 0.5): {pos_count}\n"
                chunk_prompt += f"- Negative/Safe (score < 0.5): {neg_count}\n"
                chunk_prompt += f"-> Please adjust the score distribution in this next batch of {current_chunk_size} to ensure the overall dataset matches the target balance specified in the guidelines."
            
            chunk_data = await self.generator.generate(
                user_prompt=chunk_prompt, 
                target_count=current_chunk_size,
                previous_samples=all_category_data
            )
            all_category_data.extend(chunk_data)
            
        self.saver.save_category_data(category_name, all_category_data)

    def _extract_quantity(self, prompt_content: str, default: int = 250) -> int:
        for line in prompt_content.split("\n"):
            if "Quantity:" in line:
                try:
                    return int(line.split(":")[1].strip())
                except ValueError:
                    pass
        return default

    async def run(self, specific_category: Optional[str] = None):
        try:
            prompt_files = self.prompt_manager.get_prompt_files(specific_category)
            if not prompt_files:
                print("No active prompt files found. Exiting.")
                return

            # Processes one category at a time to prevent rate limits or timeout
            for p_file in prompt_files:
                await self.process_category(p_file)
                
        except Exception as e:
            print(f"Pipeline error: {e}")
