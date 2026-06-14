# extract_corpus.py
from datasets import load_dataset
from tqdm import tqdm

# Reads from your local HF cache — no re-download
ds = load_dataset("ai4bharat/samanantar", "ta", split="train")
print(f"Loaded {len(ds):,} sentence pairs")
print("Columns:", ds.column_names)   # sanity check the field names

# Samanantar columns are typically: idx, src (English), tgt (Tamil)
with open("corpus_ta_raw.txt", "w", encoding="utf-8") as f:
    for row in tqdm(ds, desc="Extracting Tamil"):
        line = (row.get("tgt") or "").strip()
        if line:
            f.write(line + "\n")

print("Done → corpus_ta_raw.txt")