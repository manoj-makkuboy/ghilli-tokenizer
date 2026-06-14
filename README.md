# Ghilli (கிள்ளி)

**Grapheme-aware tokenizer for Tamil and Indian Abugida scripts.**

> *"Strike at the right boundary."*

Named after the [Tamil street sport](https://en.wikipedia.org/wiki/Gilli-danda) where a small stick (gilli) is struck at exactly the right point and sent flying — a metaphor for grapheme-level tokenization that splits text at linguistically correct boundaries.

---

## The Problem

Tamil uses an **Abugida** writing system where a single visible character is often composed of multiple Unicode codepoints:

```
க + ா = கா    (consonant + vowel modifier = one visual character)
க + ் = க்    (consonant + pulli = pure consonant)
```

GPT-2's regex pre-tokenizer treats Tamil vowel modifiers (Unicode category: Mark) as punctuation, tearing them away from their consonants. The result: Tamil needs **4.54x more tokens** than English for equivalent text.

```
"தமிழ்" (Tamil)

  GPT-2 sees:    ['த', 'ம', 'ி', 'ழ', '்']     → 5 codepoints, split apart
  Ghilli sees:   ['த', 'மி', 'ழ்']               → 3 graphemes, kept intact
```

## The Fix

Based on [Velayuthan & Sarveswaran (COLING 2025)](https://aclanthology.org/2025.coling-main.400.pdf):

1. **Whitespace pre-tokenizer** — never GPT-2 regex
2. **Grapheme clusters as atomic units** — vowel modifiers stay attached to consonants
3. **Grapheme-seeded initial alphabet** — the tokenizer can never tear apart what belongs together

The pre-tokenizer alone accounts for a **3x improvement**. Algorithm choice (BPE vs Unigram) only contributes ~0.04x.

## Quick Start

### Install

```bash
git clone https://github.com/ghilli/ghilli.git
cd ghilli
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Use a Trained Tokenizer

```python
from ghilli import GhilliTokenizer

tok = GhilliTokenizer("data/weights/ta/ghilli-ta-bpe-32k.json")

text = "தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்று"
encoded = tok.encode(text)

print(encoded.tokens)
# ['தமிழ்', 'மொழி', 'உலகின்', 'மிகப்', 'பழமையான', 'மொழிகளில்', 'ஒன்று']
# 7 tokens — each one is a complete Tamil word

decoded = tok.decode(encoded.ids)
assert decoded == text  # Perfect roundtrip
```

### Train from Scratch

```bash
# Run the full pipeline: extract → clean → train
python pipeline/pipeline.py --langs ta
```

This will:
1. Extract 5.26M Tamil sentences from `ai4bharat/samanantar`
2. Clean and deduplicate (NFC normalize, script ratio filter) → 4.99M sentences
3. Extract 5,142 unique grapheme clusters
4. Train BPE and Unigram tokenizers at 16k, 32k, and 48k vocab sizes

Output: 6 tokenizer files in `data/weights/ta/`

### Train Programmatically

```python
from ghilli import GhilliTokenizer
from ghilli.pretokenizer.grapheme import extract_graphemes

# Extract grapheme clusters from your corpus
alphabet = extract_graphemes("my_corpus.txt")

# Train BPE
bpe = GhilliTokenizer.train_bpe("my_corpus.txt", vocab_size=32000, initial_alphabet=alphabet)
bpe.save("my_bpe_tokenizer.json")

# Train Unigram
uni = GhilliTokenizer.train_unigram("my_corpus.txt", vocab_size=32000, initial_alphabet=alphabet)
uni.save("my_unigram_tokenizer.json")
```

## Trained Models

| Model | Vocab Size | File Size |
|-------|-----------|-----------|
| `ghilli-ta-bpe-16k.json` | 16,000 | 1.3 MB |
| `ghilli-ta-bpe-32k.json` | 32,000 | 2.9 MB |
| `ghilli-ta-bpe-48k.json` | 48,000 | 4.5 MB |
| `ghilli-ta-unigram-16k.json` | 16,000 | 1.2 MB |
| `ghilli-ta-unigram-32k.json` | 32,000 | 2.4 MB |
| `ghilli-ta-unigram-48k.json` | 48,000 | 3.7 MB |

All models were trained on 4.99M cleaned Tamil sentences from the [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) parallel corpus.

## How It Works

### Pipeline

```
ai4bharat/samanantar (HuggingFace)
        │
        ▼
┌─────────────────┐
│  Stage 1: Extract │ → data/raw/raw_ta.txt (5.26M sentences)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Stage 2: Clean   │ → data/clean/clean_ta.txt (4.99M sentences)
│  - NFC normalize  │   NFC normalization is critical — Tamil text
│  - Whitespace      │   in the wild mixes composed and decomposed
│  - Script filter   │   Unicode forms for the same character.
│  - Dedup           │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Stage 3: Train   │ → data/weights/ta/ghilli-ta-{algo}-{size}k.json
│  Pass 1: Extract  │   5,142 unique grapheme clusters found
│    graphemes       │
│  Pass 2: Train     │   BPE and Unigram at 16k / 32k / 48k
│    tokenizer       │
└─────────────────┘
```

Every stage is **resumable** — if the output file exists, the stage is skipped.

### Two Algorithms

**GPE-BPE** — Standard BPE with two modifications:
- Initial alphabet = grapheme clusters (not bytes)
- Pre-tokenizer = `Whitespace` (not GPT-2 regex)
- Decoder: offset-based (spaces reconstructed from position tracking)

**Unigram** — Same grapheme-seeded alphabet, different model:
- Trains top-down (prune vocabulary) instead of bottom-up (merge pairs)
- Pre-tokenizer = `Metaspace` (whitespace + `▁` word boundary markers)
- Decoder: `Metaspace` (strips `▁` markers, reinserts spaces)

Both produce nearly identical compression ratios for Tamil. The pre-tokenizer is what matters — not the algorithm.

### Grapheme Extraction

The key innovation. The `grapheme` Python library implements Unicode Text Segmentation (UAX #29) to correctly identify character boundaries:

```python
import grapheme

# Grapheme-aware splitting (correct)
list(grapheme.graphemes("மொழிகளில்"))
# ['மொ', 'ழி', 'க', 'ளி', 'ல்']  →  5 graphemes

# Naive codepoint splitting (wrong)
list("மொழிகளில்")
# ['ம', 'ொ', 'ழ', 'ி', 'க', 'ள', 'ி', 'ல', '்']  →  9 codepoints
```

The 5,142 unique grapheme clusters extracted from the Tamil corpus become the `initial_alphabet` for the tokenizer trainer, ensuring no merge or split can ever break apart a vowel modifier from its consonant.

## Project Structure

```
ghilli-tokenizer/
├── ghilli/                      # Core library
│   ├── __init__.py              # Public API
│   ├── tokenizer.py             # GhilliTokenizer (unified interface)
│   ├── algorithms/
│   │   ├── base.py              # Abstract BaseTokenizer
│   │   ├── gpe.py               # GPE-BPE tokenizer
│   │   └── unigram.py           # Unigram tokenizer
│   └── pretokenizer/
│       └── grapheme.py          # Grapheme cluster extraction
├── pipeline/                    # Data pipeline
│   ├── pipeline.py              # Entry point
│   ├── config.yaml              # All parameters
│   └── stages/
│       ├── extract.py           # HuggingFace → raw text
│       ├── clean.py             # NFC, filter, dedup
│       └── train.py             # Grapheme extraction + training
├── tests/
│   ├── test_gpe.py              # 8 tests
│   └── fixtures/sample_ta.txt   # Test data
├── data/                        # Generated (gitignored)
├── pyproject.toml
├── requirements.txt
├── LICENSE                      # Apache 2.0
└── HOW_IT_WORKS.md              # Detailed technical documentation
```

## Testing

```bash
source .venv/bin/activate
pip install pytest
pytest tests/ -v
```

```
tests/test_gpe.py::TestGPEBPE::test_encode_returns_ids       PASSED
tests/test_gpe.py::TestGPEBPE::test_roundtrip                PASSED
tests/test_gpe.py::TestGPEBPE::test_save_load_roundtrip      PASSED
tests/test_gpe.py::TestUnigram::test_encode_returns_ids       PASSED
tests/test_gpe.py::TestUnigram::test_roundtrip                PASSED
tests/test_gpe.py::TestUnigram::test_save_load_roundtrip      PASSED
tests/test_gpe.py::TestGraphemeExtraction::test_extracts_graphemes      PASSED
tests/test_gpe.py::TestGraphemeExtraction::test_no_whitespace  PASSED
```

## Configuration

All parameters are in `pipeline/config.yaml`:

```yaml
languages:
  - code: ta
    name: Tamil
    hf_dataset: ai4bharat/samanantar
    unicode_block: "0B80-0BFF"
    min_ratio: 0.5

pipeline:
  vocab_sizes: [16000, 32000, 48000]

training:
  algorithms: [bpe, unigram]
```

To add a new language, add an entry to `languages` and run:

```bash
python pipeline/pipeline.py --langs NEW_CODE
```

## Research

Based on:

> **"Egalitarian Language Representation in Language Models: It All Begins with Tokenizers"**
> Velayuthan & Sarveswaran, COLING 2025, University of Jaffna, Sri Lanka
> [Paper](https://aclanthology.org/2025.coling-main.400.pdf)

Key finding: Switching only the pre-tokenizer from GPT-2 regex to whitespace improves Tamil compression ratio from 1.36x to 4.32x — a 3x improvement with zero algorithm changes.

| Tokenizer | Tamil CR | Tamil TP |
|-----------|----------|----------|
| GPT-2 | 1.36x | 4.54 |
| GPT-4 | 2.13x | 2.89 |
| mT5 | 9.21x | 0.78 |
| **Ghilli GPE 32k** | **~4.3x** | **~0.9** |

**CR** = Compression Ratio (higher is better). **TP** = Tokenization Parity vs English (1.0 is ideal).

## Dependencies

- [tokenizers](https://github.com/huggingface/tokenizers) >= 0.15.0 — HuggingFace tokenizer library
- [grapheme](https://github.com/alvinlindstam/grapheme) >= 0.6.0 — Unicode grapheme cluster segmentation
- [datasets](https://github.com/huggingface/datasets) >= 2.14.0 — HuggingFace dataset loading
- [pyyaml](https://github.com/yaml/pyyaml) >= 6.0 — config parsing
- [tqdm](https://github.com/tqdm/tqdm) >= 4.65.0 — progress bars

## License

Apache 2.0 — see [LICENSE](LICENSE).
