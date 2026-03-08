# Synthetic Data Generation Pipeline

This directory contains the complete pipeline for generating synthetic distillation data using Claude 4.5 Haiku as a teacher model. The data generated here is used to fine-tune a smaller student model (`DeBERTa-v3-small`) for the Sensitive Content Detection task.

## Evolution of the Strategy

Our data generation strategy evolved significantly to address various challenges and ensure high-quality distillation data. Here is the progression:

### 1. The Initial Simplistic Approach
Initially, we started with a single monolithic Python script (`generate_data.py`). 
- It used a single over-arching prompt to ask Claude to generate "sensitive and non-sensitive" examples.
- All generation was done in one giant API call.
- **Problem**: This led to highly skewed data, generic examples, and mostly binary labels (0.0 or 1.0) without the nuanced "gray area" soft labels needed for effective distillation. The model also hit output token limits when trying to generate 1,250 examples at once.

### 2. Category-Specific Prompts
To fix the lack of diversity and generic outputs, we broke the generation down by category.
- We created a `prompts/` directory.
- We wrote 5 highly specific text files (`credentials.txt`, `identifiers.txt`, `contact_info.txt`, `other_pii.txt`, `safe_cases.txt`).
- Each prompt was tailored to explain exactly what that category looks like, what edge cases exist, and how to score them (e.g., test credentials = 0.88, real credentials = 1.0).

### 3. Implementing Decision Thresholds & Soft Labels
We realized the student model needed to learn exactly where the "boundary" was.
- We updated the prompt instructions to explicitly center around a `0.5` decision threshold.
- We instructed Claude to generate varying shades of sensitivity (e.g., 0.4 for a masked credit card, 0.7 for a real address).

### 4. Static Category-Wise Few-Shot Examples
To ground Claude's logic, we needed few-shot examples.
- Initially, the script tried to auto-extract these from a general `sample_data.json` on the fly.
- We refined this by creating a dedicated `few_shot_examples/` directory.
- We manually curated exactly 10 high-quality examples (both positive and negative) for *each* of the 5 categories and saved them as static JSON files. These are now explicitly injected into the category prompts.

### 5. SOLID Refactoring
As the `generate_data.py` script grew complex, we refactored it using Object-Oriented Programming and SOLID principles.
- The monolithic script was broken down into a `src/` modular structure (`config.py`, `prompt_manager.py`, `generator.py`, `data_saver.py`, `pipeline.py`).
- This allowed for robust error handling, easier testing, and clearer separation of concerns.

### 6. Context-Aware Chunking & Balanced Generation
The most recent and advanced optimization addressed token limits and dataset balance.
- **Chunking**: Generating 250 items at once exceeded Claude's max output tokens, causing JSON truncation. We implemented a chunking system (batching in sizes of 50).
- **Context-Aware Deduplication**: To prevent the model from repeating the same examples across different chunks, the pipeline now feeds the *previously generated texts* back into the system prompt for the next chunk, instructing it: "Do not repeat these."
- **Dynamic Balance Adjustment**: We require a 50/50 mix of positive (>=0.5) and negative (<0.5) examples within each category. The script now calculates the *running total* of positive and negative distributions after every chunk and explicitly passes these stats to the model in the next chunk, forcing it to auto-correct its internal balance to maintain the target ratio.

## Directory Structure

- `generate_data.py`: The main CLI entrypoint.
- `prompts/`: Text files containing the instructions for each category.
- `few_shot_examples/`: JSON files containing the 10-shot examples for each category.
- `src/`: The modular engine.
    - `config.py`: Environment and path setup.
    - `prompt_manager.py`: Loads prompts and few-shots.
    - `generator.py`: Manages the AsyncAnthropic API calls.
    - `data_saver.py`: Handles writing the final chunked arrays back to the root `data/` directory.
    - `pipeline.py`: Orchestrates the chunking, context-injection, and balance-tracking logic.

## Usage

Generate all 5 categories (1,250 examples total):
```bash
python3 synthetic_data_generation/generate_data.py
```

Generate a single category (useful for iteration or fixing a specific set):
```bash
python3 synthetic_data_generation/generate_data.py --category contact_info
```
