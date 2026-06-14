"""Stage 1: Extract Tamil sentences from HuggingFace dataset to raw text file."""

import os
from datasets import load_dataset
from tqdm import tqdm


def run(lang_config: dict, pipeline_config: dict) -> str:
    code = lang_config["code"]
    corpus_dir = pipeline_config["corpus_dir"]
    output_path = os.path.join(corpus_dir, f"raw_{code}.txt")

    if os.path.exists(output_path):
        print(f"[extract] {output_path} already exists, skipping.")
        return output_path

    os.makedirs(corpus_dir, exist_ok=True)

    # Check for pre-extracted corpus file in project root
    legacy_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "corpus_ta_raw.txt")
    if code == "ta" and os.path.exists(legacy_path):
        print(f"[extract] Found pre-extracted corpus at {legacy_path}, copying...")
        import shutil
        shutil.copy2(legacy_path, output_path)
        print(f"[extract] Done → {output_path}")
        return output_path

    print(f"[extract] Loading {lang_config['hf_dataset']} ({code})...")
    ds = load_dataset(lang_config["hf_dataset"], code, split=lang_config["hf_split"])
    column = lang_config["hf_column"]

    with open(output_path, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"Extracting {lang_config['name']}"):
            line = (row.get(column) or "").strip()
            if line:
                f.write(line + "\n")

    print(f"[extract] Done → {output_path}")
    return output_path
