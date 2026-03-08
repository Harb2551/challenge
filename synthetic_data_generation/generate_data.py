import argparse
import asyncio
from src.config import Config
from src.pipeline import DataGenerationPipeline

def main():
    parser = argparse.ArgumentParser(description="Synthetic Distillation Data Generator")
    parser.add_argument(
        "--category", 
        type=str, 
        help="Specify a single category to regenerate (e.g., 'credentials'). Overwrites existing file for that category."
    )
    args = parser.parse_args()

    config = Config()
    pipeline = DataGenerationPipeline(config)
    asyncio.run(pipeline.run(specific_category=args.category))

if __name__ == "__main__":
    main()
