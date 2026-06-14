"""Ghilli Pipeline — main entry point.

Usage:
    python pipeline/pipeline.py --langs ta
    python pipeline/pipeline.py --config pipeline/config.yaml --langs ta
"""

import argparse
import os
import sys

import yaml

# Add project root and pipeline dir to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, PIPELINE_DIR)

from stages import extract, clean, train


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Ghilli tokenizer pipeline")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "pipeline", "config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--langs", nargs="*", default=None,
                        help="Language codes to process (default: all in config)")
    args = parser.parse_args()

    config = load_config(args.config)
    languages = config["languages"]
    pipeline_config = config["pipeline"]
    training_config = config["training"]

    # Filter languages if specified
    if args.langs:
        languages = [l for l in languages if l["code"] in args.langs]
        if not languages:
            print(f"No matching languages found for: {args.langs}")
            sys.exit(1)

    for lang in languages:
        code = lang["code"]
        print(f"\n{'='*60}")
        print(f"  Processing: {lang['name']} ({code})")
        print(f"{'='*60}\n")

        # Stage 1: Extract
        extract.run(lang, pipeline_config)

        # Stage 2: Clean
        clean.run(lang, pipeline_config)

        # Stage 3: Train
        train.run(lang, pipeline_config, training_config)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
