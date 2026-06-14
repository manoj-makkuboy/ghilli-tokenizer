"""Stage 2: Clean and normalize raw corpus — NFC, dedup, script ratio filter."""

import os
import re
import unicodedata
from tqdm import tqdm


def _in_unicode_block(char: str, block_start: int, block_end: int) -> bool:
    return block_start <= ord(char) <= block_end


def _script_ratio(line: str, block_start: int, block_end: int) -> float:
    chars = [c for c in line if not c.isspace()]
    if not chars:
        return 0.0
    count = sum(1 for c in chars if _in_unicode_block(c, block_start, block_end))
    return count / len(chars)


def run(lang_config: dict, pipeline_config: dict) -> str:
    code = lang_config["code"]
    input_path = os.path.join(pipeline_config["corpus_dir"], f"raw_{code}.txt")
    output_dir = pipeline_config["clean_dir"]
    output_path = os.path.join(output_dir, f"clean_{code}.txt")

    if os.path.exists(output_path):
        print(f"[clean] {output_path} already exists, skipping.")
        return output_path

    os.makedirs(output_dir, exist_ok=True)

    block = lang_config["unicode_block"]
    block_start = int(block.split("-")[0], 16)
    block_end = int(block.split("-")[1], 16)
    min_length = pipeline_config["min_line_length"]
    min_ratio = lang_config["min_ratio"]

    seen = set()
    kept = 0
    total = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"Cleaning {lang_config['name']}"):
            total += 1
            # NFC normalize first — critical for Tamil
            line = unicodedata.normalize("NFC", line.strip())
            # Collapse whitespace
            line = re.sub(r"\s+", " ", line)
            # Length filter
            if len(line) < min_length:
                continue
            # Script ratio filter
            if _script_ratio(line, block_start, block_end) < min_ratio:
                continue
            # Dedup
            h = hash(line)
            if h in seen:
                continue
            seen.add(h)
            fout.write(line + "\n")
            kept += 1

    print(f"[clean] {lang_config['name']}: {kept:,}/{total:,} lines kept → {output_path}")
    return output_path
