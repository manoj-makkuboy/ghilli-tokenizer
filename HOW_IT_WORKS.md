# Ghilli — How It Works

A detailed technical walkthrough of the Ghilli tokenizer: why it exists, how each component works, and how the pieces fit together.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Core Insight](#2-the-core-insight)
3. [Architecture Overview](#3-architecture-overview)
4. [Pipeline: Data to Weights](#4-pipeline-data-to-weights)
   - [Stage 1: Extract](#stage-1-extract)
   - [Stage 2: Clean](#stage-2-clean)
   - [Stage 3: Train](#stage-3-train)
5. [Core Library: ghilli/](#5-core-library-ghilli)
   - [Grapheme Extraction](#grapheme-extraction)
   - [GPE-BPE Algorithm](#gpe-bpe-algorithm)
   - [Unigram Algorithm](#unigram-algorithm)
   - [Unified API](#unified-api)
6. [Pre-tokenizer Deep Dive](#6-pre-tokenizer-deep-dive)
7. [How BPE and Unigram Differ](#7-how-bpe-and-unigram-differ)
8. [Decode: Getting Text Back](#8-decode-getting-text-back)
9. [Configuration](#9-configuration)
10. [Testing](#10-testing)
11. [End-to-End Example](#11-end-to-end-example)
12. [File Reference](#12-file-reference)

---

## 1. The Problem

Tamil uses an **Abugida** writing system. A single visible character on screen is often composed of multiple Unicode codepoints:

```
க + ா = கா    (consonant க + vowel modifier ா = syllable "kaa")
க + ் = க்    (consonant க + pulli ் = pure consonant "k")
```

The vowel modifier `ா` (Unicode category: **Mark**) cannot exist independently — it must be attached to its preceding consonant. Together, `க` + `ா` form a single **grapheme cluster**: the smallest unit a human perceives as one character.

Most modern tokenizers (GPT-2, GPT-4, Llama 3) use a **regex-based pre-tokenizer** originally designed for English. GPT-2's regex pattern treats Unicode Mark characters as punctuation and tears them away from their consonants. The result:

| Tokenizer | Tamil Compression Ratio | Tokenization Parity |
|-----------|------------------------|---------------------|
| GPT-2     | 1.36x                 | 4.54 (needs 4.54x more tokens than English) |
| GPT-4     | 2.13x                 | 2.89 |
| mT5       | 9.21x                 | 0.78 |

Tamil text tokenized by GPT-2 produces **4.54x more tokens** than the equivalent English text, which means Tamil users get 1/4.5th of the effective context window.

---

## 2. The Core Insight

This project is based on the research paper:

> **"Egalitarian Language Representation in Language Models: It All Begins with Tokenizers"**
> Velayuthan & Sarveswaran, COLING 2025, University of Jaffna

The paper's central finding: **the pre-tokenizer matters more than the tokenization algorithm**.

```
GPT-2 regex pre-tokenizer + BPE   → Compression Ratio 1.36x  (Tamil)
Whitespace pre-tokenizer   + BPE   → Compression Ratio 4.32x  (Tamil)
```

Switching **only** the pre-tokenizer from GPT-2's regex to simple whitespace splitting yields a **3x improvement**. The algorithm choice (BPE vs Unigram vs WordPiece) only contributes ~0.04x difference when the pre-tokenizer is already correct.

Ghilli implements this insight as a usable library:

1. **Never use GPT-2 regex** — use whitespace-based pre-tokenization only
2. **Use grapheme clusters as atomic units** — the `grapheme` Python library correctly identifies multi-codepoint sequences that form single visual characters
3. **Seed the vocabulary with graphemes** — pass the full set of unique grapheme clusters as `initial_alphabet` to the tokenizer trainer, so the model can never split a vowel modifier from its consonant

---

## 3. Architecture Overview

```
ghilli-tokenizer/
├── pipeline/                    # Data pipeline (extract → clean → train)
│   ├── pipeline.py              # Orchestrator: python pipeline/pipeline.py --langs ta
│   ├── config.yaml              # Single source of truth for all parameters
│   └── stages/
│       ├── extract.py           # Stage 1: HuggingFace dataset → raw text
│       ├── clean.py             # Stage 2: NFC normalize, filter, dedup
│       └── train.py             # Stage 3: Grapheme extraction + tokenizer training
│
├── ghilli/                      # Core library (pip-installable)
│   ├── __init__.py              # Public API: GhilliTokenizer, GPETokenizer, UnigramTokenizer
│   ├── tokenizer.py             # Unified GhilliTokenizer wrapper
│   ├── algorithms/
│   │   ├── base.py              # Abstract BaseTokenizer (train/encode/decode/save/load)
│   │   ├── gpe.py               # GPE-BPE: grapheme-aware BPE
│   │   └── unigram.py           # Grapheme-seeded Unigram
│   └── pretokenizer/
│       └── grapheme.py          # Grapheme cluster extraction utility
│
├── data/                        # Auto-created by pipeline, gitignored
│   ├── raw/raw_ta.txt           # 5.26M raw Tamil sentences
│   ├── clean/clean_ta.txt       # 4.99M cleaned sentences
│   └── weights/ta/              # 6 trained tokenizer files
│       ├── ghilli-ta-bpe-16k.json
│       ├── ghilli-ta-bpe-32k.json
│       ├── ghilli-ta-bpe-48k.json
│       ├── ghilli-ta-unigram-16k.json
│       ├── ghilli-ta-unigram-32k.json
│       └── ghilli-ta-unigram-48k.json
│
└── tests/
    ├── test_gpe.py              # 8 tests: roundtrip, save/load, grapheme checks
    └── fixtures/sample_ta.txt   # 20 Tamil sentences for testing
```

The system has two halves:
- **Pipeline** — a three-stage data pipeline that goes from a HuggingFace dataset to trained tokenizer weight files
- **Library** — a Python package that wraps HuggingFace `tokenizers` with grapheme-aware configuration, usable independently of the pipeline

---

## 4. Pipeline: Data to Weights

The pipeline is invoked with a single command:

```bash
python pipeline/pipeline.py --langs ta
```

It reads `pipeline/config.yaml` for all parameters and runs three stages sequentially. Every stage is **resumable** — if the output file already exists, the stage is skipped entirely. This means you can re-run the pipeline after a crash without re-doing completed work.

### Stage 1: Extract

**File:** `pipeline/stages/extract.py`
**Input:** HuggingFace dataset `ai4bharat/samanantar`, Tamil (`ta`) config, `train` split
**Output:** `data/raw/raw_ta.txt` — one sentence per line

The Samanantar dataset is a parallel corpus of English-Tamil sentence pairs. The extract stage reads the `tgt` column (Tamil side) and writes each non-empty sentence as a line in a plain text file.

```python
ds = load_dataset("ai4bharat/samanantar", "ta", split="train")
# Samanantar columns: 'idx', 'src' (English), 'tgt' (Tamil)
for row in ds:
    line = row.get("tgt", "").strip()
    if line:
        f.write(line + "\n")
```

There is also a shortcut: if a pre-extracted `corpus_ta_raw.txt` file exists in the project root (from a previous manual extraction), it is copied directly to `data/raw/raw_ta.txt` instead of re-downloading.

**Result:** 5,264,867 Tamil sentences (~1 GB).

### Stage 2: Clean

**File:** `pipeline/stages/clean.py`
**Input:** `data/raw/raw_ta.txt`
**Output:** `data/clean/clean_ta.txt`

Five cleaning operations, applied in order to every line:

#### 1. NFC Normalization

```python
line = unicodedata.normalize("NFC", line)
```

This is the most critical step. Tamil text in the wild mixes **composed** and **decomposed** Unicode forms. The same visual character can be encoded two different ways:

- Composed (NFC): `கா` = U+0B95 U+0BBE (2 codepoints — consonant + dependent vowel sign)
- Decomposed (NFD): `கா` = U+0B95 U+0BC6 U+0BBE (3 codepoints — consonant + decomposed vowel components)

Tamil doesn't have many NFC/NFD differences in practice, but mixed-encoding corpora do appear in the wild (copy-paste from PDFs, web scraping, different editor encodings). Without NFC normalization, the tokenizer would treat identical-looking text as different tokens. NFC ensures every character is in its canonical composed form.

#### 2. Whitespace Collapse

```python
line = re.sub(r"\s+", " ", line)
```

Reduces runs of spaces, tabs, or newlines to a single space.

#### 3. Length Filter

```python
if len(line) < 10:
    continue
```

Drops lines shorter than 10 characters (configurable via `min_line_length`). Very short lines are typically noise — numbers, fragments, or formatting artifacts.

#### 4. Script Ratio Filter

```python
def _script_ratio(line, block_start, block_end):
    chars = [c for c in line if not c.isspace()]
    count = sum(1 for c in chars if block_start <= ord(c) <= block_end)
    return count / len(chars)
```

For Tamil, the Unicode block is U+0B80–U+0BFF. The filter drops any line where less than 50% of non-whitespace characters fall within this block. This removes:
- Mostly-English sentences that slipped through
- Sentences in other scripts (Hindi, Malayalam, etc.)
- Lines that are mostly numbers or punctuation

#### 5. Deduplication

```python
h = hash(line)
if h in seen:
    continue
seen.add(h)
```

Exact dedup using Python's built-in `hash()`. All hashes are held in a `set` in RAM. For 5M sentences on a 16GB machine, this uses approximately 200–400 MB — well within limits.

**Result:** 4,997,033 lines kept out of 5,264,867 (95% retention).

### Stage 3: Train

**File:** `pipeline/stages/train.py`
**Input:** `data/clean/clean_ta.txt`
**Output:** `data/weights/ta/ghilli-ta-{algo}-{size}k.json`

Training happens in two passes:

#### Pass 1: Grapheme Extraction

```python
from ghilli.pretokenizer.grapheme import extract_graphemes

initial_alphabet = extract_graphemes(clean_path)
# Result: 5,142 unique grapheme clusters
```

This reads every line of the clean corpus and splits it into grapheme clusters using the `grapheme` Python library. Every unique cluster is collected into a sorted list. This list becomes the **initial alphabet** for the tokenizer trainer.

Why this matters: Standard BPE starts with individual bytes (256 entries) or Unicode codepoints. For Tamil, this means the base vocabulary contains `க` and `ா` as separate entries. During merge training, the tokenizer *might* learn to merge them — but it also might not, or might merge them inconsistently. By seeding with grapheme clusters, we guarantee that `கா` is a single atomic unit from the start. The tokenizer can never tear it apart.

The 5,142 grapheme clusters found in the Tamil corpus include:
- All Tamil consonant-vowel combinations (e.g., `கா`, `கி`, `கு`, `கே`)
- Pure consonants with pulli (e.g., `க்`, `ங்`, `ச்`)
- Standalone vowels (e.g., `அ`, `ஆ`, `இ`)
- Numbers, punctuation, and Latin characters present in the corpus

#### Pass 2: Tokenizer Training

The training stage iterates over all configured algorithms (`bpe`, `unigram`) and vocab sizes (`16000`, `32000`, `48000`), training a separate tokenizer for each combination:

```python
for algo_name in ["bpe", "unigram"]:
    for size in [16000, 32000, 48000]:
        tok = ALGORITHM_MAP[algo_name](vocab_size=size, special_tokens=special_tokens)
        tok.train(clean_path, initial_alphabet=initial_alphabet)
        tok.save(f"data/weights/ta/ghilli-ta-{algo_name}-{size // 1000}k.json")
```

This produces 6 weight files (2 algorithms x 3 sizes). Each file is a self-contained JSON that includes the full vocabulary, merge rules (for BPE) or token scores (for Unigram), pre-tokenizer configuration, and decoder configuration.

---

## 5. Core Library: ghilli/

### Grapheme Extraction

**File:** `ghilli/pretokenizer/grapheme.py`

```python
import grapheme as _grapheme

def extract_graphemes(corpus_path: str) -> list[str]:
    unique = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            unique.update(_grapheme.graphemes(line.strip()))
    unique.discard("")
    unique.discard(" ")
    return sorted(unique)
```

The `grapheme` library implements the Unicode Text Segmentation algorithm (UAX #29). It correctly identifies extended grapheme cluster boundaries, handling:

- Tamil consonant + vowel sign combinations
- Consonant + pulli sequences
- Conjunct consonants (e.g., `க்ஷ`)
- ZWJ/ZWNJ sequences

Example:
```python
import grapheme
list(grapheme.graphemes("தமிழ்"))
# ['த', 'மி', 'ழ்']
# Note: 'மி' is TWO codepoints (ம + ி) kept as ONE grapheme
# Note: 'ழ்' is TWO codepoints (ழ + ்) kept as ONE grapheme
```

Compare with naive codepoint splitting:
```python
list("தமிழ்")
# ['த', 'ம', 'ி', 'ழ', '்']
# WRONG: vowel sign ி and pulli ் are torn away from their consonants
```

### GPE-BPE Algorithm

**File:** `ghilli/algorithms/gpe.py`

GPE (Grapheme Pair Encoding) is standard BPE with two modifications:

1. **Initial alphabet = grapheme clusters** (not bytes or codepoints)
2. **Pre-tokenizer = Whitespace** (not GPT-2 regex)

```python
class GPETokenizer(BaseTokenizer):
    def train(self, corpus_path, initial_alphabet=None):
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=initial_alphabet or [],
            special_tokens=self.special_tokens,
        )
        self.tokenizer.train(files=[corpus_path], trainer=trainer)
```

**How BPE training works:**

1. Start with every grapheme cluster as an individual token (5,142 entries)
2. Add special tokens (`<unk>`, `<s>`, `</s>`, `<pad>`, `<mask>`)
3. Count every adjacent pair of tokens in the corpus
4. Merge the most frequent pair into a new single token
5. Repeat step 3-4 until `vocab_size` is reached

For example, if `தமி` and `ழ்` appear adjacent 50,000 times, they merge into `தமிழ்` ("Tamil"). The trainer keeps merging common pairs until the vocabulary reaches the target size (e.g., 32,000 tokens).

The `Whitespace` pre-tokenizer ensures that merges never cross word boundaries. The text `"தமிழ் மொழி"` is first split into `["தமிழ்", "மொழி"]` by the pre-tokenizer, and BPE merges only happen within each word.

**No explicit decoder is set** for BPE. The HuggingFace `tokenizers` library uses offset tracking internally — each token remembers its start and end position in the original text. During decoding, the library reconstructs the original text including whitespace by using these offsets. This means spaces between words are preserved perfectly.

### Unigram Algorithm

**File:** `ghilli/algorithms/unigram.py`

The Unigram tokenizer uses the same grapheme-seeded initial alphabet but a different model and training algorithm:

```python
class UnigramTokenizer(BaseTokenizer):
    def train(self, corpus_path, initial_alphabet=None):
        self.tokenizer = Tokenizer(models.Unigram())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        self.tokenizer.decoder = decoders.Metaspace()

        trainer = trainers.UnigramTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=initial_alphabet or [],
            special_tokens=self.special_tokens,
            unk_token="<unk>",
        )
        self.tokenizer.train(files=[corpus_path], trainer=trainer)
```

**How Unigram training works:**

Unlike BPE which builds up from small units by merging, Unigram works top-down:

1. Start with a large initial vocabulary (all grapheme clusters + substrings up to a certain length)
2. Assign a probability score to each token using EM (Expectation-Maximization)
3. For each token, calculate how much the total corpus likelihood would decrease if that token were removed
4. Remove the tokens with the smallest impact (keeping the vocabulary within the target size)
5. Repeat steps 2-4 until `vocab_size` is reached

The result is a vocabulary where each token has a log-probability score. During encoding, the Viterbi algorithm finds the tokenization that maximizes the total probability.

**Metaspace pre-tokenizer and decoder:** The Unigram model uses `Metaspace` instead of plain `Whitespace`. Metaspace splits on whitespace like `Whitespace`, but additionally prepends a `▁` (U+2581, "lower one eighth block") character to the first token of each word:

```
Input:  "தமிழ் மொழி"
Tokens: ["▁தமிழ்", "▁மொழி"]
```

The `▁` marker tells the decoder where word boundaries are. During decoding, `▁` is stripped and replaced with a space (except at the start of the text). This is the same approach used by SentencePiece and Llama tokenizers.

Why not use `Whitespace` for Unigram too? Because the Unigram model can produce sub-word tokens (e.g., `["செ", "ன்ன", "ை"]` for `"சென்னை"`). Without the `▁` marker, the decoder would insert spaces between every token, producing `"செ ன்ன ை"` instead of `"சென்னை"`. The `▁` marker distinguishes "this is the start of a new word" from "this is a continuation of the current word."

### Unified API

**File:** `ghilli/tokenizer.py`

`GhilliTokenizer` provides a simple interface for loading and using trained models:

```python
from ghilli import GhilliTokenizer

tok = GhilliTokenizer("data/weights/ta/ghilli-ta-bpe-32k.json")
encoded = tok.encode("தமிழ் மொழி")
decoded = tok.decode(encoded.ids)
# decoded == "தமிழ் மொழி"
```

It also provides static factory methods for training:

```python
tok = GhilliTokenizer.train_bpe(corpus_path="data/clean/clean_ta.txt", vocab_size=32000,
                                 initial_alphabet=graphemes)
tok = GhilliTokenizer.train_unigram(corpus_path="data/clean/clean_ta.txt", vocab_size=32000,
                                     initial_alphabet=graphemes)
```

---

## 6. Pre-tokenizer Deep Dive

The pre-tokenizer is the component that splits raw text into chunks **before** the tokenization algorithm runs. This is the single most important design decision in the entire system.

### What GPT-2's regex does wrong

GPT-2 uses this regex pattern:
```
r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

The pattern `\p{L}` matches Unicode "Letter" characters. Tamil vowel signs (like `ா`, `ி`, `ு`) are Unicode category **Mark** (specifically `Mc` — "Spacing Combining Mark"), not **Letter**. So the regex treats them as "not a letter" and they match the `[^\s\p{L}\p{N}]+` pattern (non-letter, non-number characters) — the same bucket as punctuation.

This means GPT-2 splits `"கா"` into `["க", "ா"]` — the consonant and its vowel modifier become separate pre-tokens, and the tokenizer can never put them back together.

### What Ghilli does instead

**BPE:** Uses `pre_tokenizers.Whitespace()`, which simply splits on whitespace characters (`\s+`). The entire word `"கார்"` stays as one pre-token. The BPE algorithm then operates on grapheme clusters within this word, never splitting a vowel modifier from its consonant because the initial alphabet already treats them as atomic units.

**Unigram:** Uses `pre_tokenizers.Metaspace()`, which also splits on whitespace but adds a `▁` prefix to mark word starts. This achieves the same effect — no splitting within words — while also providing the boundary markers that the Unigram decoder needs.

Both approaches are whitespace-based. Neither uses any regex that could misclassify Tamil characters.

---

## 7. How BPE and Unigram Differ

| Aspect | GPE-BPE | Unigram |
|--------|---------|---------|
| Training direction | Bottom-up (merge pairs) | Top-down (prune vocabulary) |
| Encoding algorithm | Greedy left-to-right merge | Viterbi (optimal segmentation) |
| Pre-tokenizer | `Whitespace` | `Metaspace` |
| Decoder | Offset-based (implicit) | Metaspace (explicit `▁` markers) |
| Deterministic? | Yes | Can be probabilistic (we use deterministic) |
| Sub-word handling | Merges happen left-to-right | Finds globally optimal split |

For Tamil at 32k vocab, both produce similar compression ratios (within ~0.04x of each other), confirming the paper's finding that the pre-tokenizer matters far more than the algorithm.

**BPE** tends to produce longer, more common tokens because it greedily merges the most frequent pairs. A common word like `"தமிழ்"` will likely be a single token.

**Unigram** may split words differently based on probability scores. It can sometimes find more linguistically meaningful splits because it optimizes globally rather than greedily.

---

## 8. Decode: Getting Text Back

### BPE Decoding

The BPE tokenizer doesn't set an explicit decoder. Instead, it relies on the HuggingFace `tokenizers` library's **offset tracking**:

1. During encoding, each token records its byte offsets in the original text: `(start, end)`
2. During decoding, the library reconstructs the original text by concatenating characters from offset 0 through the end, using the offsets to reinsert any characters (like spaces) that were consumed by the pre-tokenizer

This means `decode(encode("தமிழ் மொழி"))` returns `"தமிழ் மொழி"` — the space between words is preserved even though it's not part of any token.

### Unigram Decoding

The Unigram tokenizer uses the `Metaspace` decoder:

1. Every `▁` character at the start of a token is replaced with a space
2. The leading space of the very first token is stripped (so the decoded text doesn't start with a space)

```
Tokens: ["▁தமிழ்", "▁மொழி"]
Step 1: [" தமிழ்", " மொழி"]
Step 2: ["தமிழ்", " மொழி"]
Result: "தமிழ் மொழி"
```

---

## 9. Configuration

**File:** `pipeline/config.yaml`

All pipeline parameters live in a single YAML file:

```yaml
languages:
  - code: ta                        # ISO 639-1 language code
    name: Tamil                     # Human-readable name
    hf_dataset: ai4bharat/samanantar # HuggingFace dataset ID
    hf_split: train                 # Dataset split
    hf_column: tgt                  # Column containing Tamil text
    unicode_block: "0B80-0BFF"      # Tamil Unicode block (for script ratio filter)
    flores_code: tam_Taml           # FLORES+ code (for future benchmarking)
    min_ratio: 0.5                  # Minimum fraction of chars in Tamil script

pipeline:
  corpus_dir: data/raw              # Raw extracted text
  clean_dir: data/clean             # Cleaned text
  weights_dir: data/weights         # Trained tokenizer weights
  min_line_length: 10               # Drop lines shorter than this
  vocab_sizes: [16000, 32000, 48000] # Train at these vocabulary sizes

training:
  algorithms: [bpe, unigram]        # Which algorithms to train
  pretokenizer: whitespace          # Pre-tokenizer type
  special_tokens:                   # Reserved tokens
    - "<unk>"
    - "<s>"
    - "</s>"
    - "<pad>"
    - "<mask>"
```

To add a new language, add an entry to `languages` with its HuggingFace dataset info and Unicode block, then run:

```bash
python pipeline/pipeline.py --langs NEW_CODE
```

---

## 10. Testing

**File:** `tests/test_gpe.py`

The test suite has 8 tests across 3 test classes:

### TestGPEBPE (3 tests)
- **test_encode_returns_ids** — Verifies that encoding Tamil text produces a non-empty list of token IDs
- **test_roundtrip** — Verifies `decode(encode(text)) == text` for `"தமிழ் மொழி"`
- **test_save_load_roundtrip** — Trains a tokenizer, saves it to JSON, loads it via `GhilliTokenizer`, and verifies the roundtrip still works

### TestUnigram (3 tests)
- Same three tests as GPE-BPE, but using the `UnigramTokenizer`

### TestGraphemeExtraction (2 tests)
- **test_extracts_graphemes** — Verifies that grapheme extraction from the Tamil fixture file produces results containing Tamil characters (U+0B80–U+0BFF range)
- **test_no_whitespace_in_alphabet** — Verifies that no whitespace-only strings appear in the extracted grapheme alphabet

All tests use a small fixture file (`tests/fixtures/sample_ta.txt`) with 20 Tamil sentences and train at vocab_size=500 for speed.

Run with:
```bash
.venv/bin/python -m pytest tests/test_gpe.py -v
```

---

## 11. End-to-End Example

### Training (via pipeline)

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the full pipeline for Tamil
python pipeline/pipeline.py --langs ta

# Output:
# [extract] data/raw/raw_ta.txt already exists, skipping.
# [clean] data/clean/clean_ta.txt already exists, skipping.
# [train] Extracting grapheme clusters from data/clean/clean_ta.txt...
# [train] Found 5142 unique grapheme clusters
# [train] Training BPE vocab_size=16,000 for Tamil...
# [train] Saved → data/weights/ta/ghilli-ta-bpe-16k.json
# ... (6 models total)
# Pipeline complete.
```

### Using a trained tokenizer

```python
from ghilli import GhilliTokenizer

# Load a trained model
tok = GhilliTokenizer("data/weights/ta/ghilli-ta-bpe-32k.json")

# Encode Tamil text
text = "தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்று"
encoded = tok.encode(text)

print(encoded.tokens)
# ['தமிழ்', 'மொழி', 'உலகின்', 'மிகப்', 'பழமையான', 'மொழிகளில்', 'ஒன்று']
# Note: 7 tokens for a full Tamil sentence. Each token is a complete word.

print(encoded.ids)
# [6291, 4934, 2755, 8620, 15234, 12878, 1654]

# Decode back to text
decoded = tok.decode(encoded.ids)
print(decoded)
# "தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்று"

print(decoded == text)
# True
```

### Training programmatically (without the pipeline)

```python
from ghilli import GhilliTokenizer
from ghilli.pretokenizer.grapheme import extract_graphemes

# Extract grapheme clusters from your corpus
alphabet = extract_graphemes("my_tamil_corpus.txt")

# Train a BPE tokenizer
tok = GhilliTokenizer.train_bpe(
    corpus_path="my_tamil_corpus.txt",
    vocab_size=32000,
    initial_alphabet=alphabet
)

# Save and reload
tok.save("my_tokenizer.json")
loaded = GhilliTokenizer("my_tokenizer.json")
```

---

## 12. File Reference

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata and dependencies for `pip install` |
| `requirements.txt` | Runtime dependencies: `datasets`, `tokenizers`, `grapheme`, `tqdm`, `pyyaml` |
| `pipeline/config.yaml` | All pipeline parameters (languages, paths, vocab sizes, algorithms) |
| `pipeline/pipeline.py` | Orchestrator — parses args, loads config, runs stages in order |
| `pipeline/stages/extract.py` | Stage 1: HuggingFace dataset to `data/raw/raw_ta.txt` |
| `pipeline/stages/clean.py` | Stage 2: NFC normalize, filter, dedup to `data/clean/clean_ta.txt` |
| `pipeline/stages/train.py` | Stage 3: Grapheme extraction + train BPE/Unigram at 16k/32k/48k |
| `ghilli/__init__.py` | Public API exports: `GhilliTokenizer`, `GPETokenizer`, `UnigramTokenizer` |
| `ghilli/tokenizer.py` | Unified `GhilliTokenizer` class — load, encode, decode, train |
| `ghilli/algorithms/base.py` | Abstract `BaseTokenizer` with shared encode/decode/save/load |
| `ghilli/algorithms/gpe.py` | `GPETokenizer` — BPE + Whitespace pre-tokenizer |
| `ghilli/algorithms/unigram.py` | `UnigramTokenizer` — Unigram + Metaspace pre-tokenizer/decoder |
| `ghilli/pretokenizer/grapheme.py` | `extract_graphemes()` — reads corpus, returns sorted unique grapheme clusters |
| `tests/test_gpe.py` | 8 tests: encode, roundtrip, save/load for both algorithms + grapheme checks |
| `tests/fixtures/sample_ta.txt` | 20 Tamil sentences for fast test runs |
| `.gitignore` | Ignores `data/`, `.venv/`, `__pycache__/`, etc. |
