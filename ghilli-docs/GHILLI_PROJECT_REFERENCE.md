# Ghilli — Project Reference for Cursor

> Grapheme-aware tokenizer suite for Indian Abugida scripts.
> Named after the Vijay film and the Tamil/South Asian street sport Gilli-danda,
> where a small stick (gilli) is struck at the right point and sent flying —
> a perfect metaphor for grapheme-level tokenization.

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Research Foundation](#2-research-foundation)
3. [Repository Structure](#3-repository-structure)
4. [config.yaml — Full Spec](#4-configyaml--full-spec)
5. [Pipeline Stages](#5-pipeline-stages)
6. [Core Algorithm — GPE](#6-core-algorithm--gpe)
7. [Benchmark Metrics](#7-benchmark-metrics)
8. [Dependencies](#8-dependencies)
9. [Reference Repos](#9-reference-repos)
10. [Roadmap](#10-roadmap)
11. [Naming & Branding](#11-naming--branding)

---

## 1. Project Vision

**One-line pitch:**
> Ghilli is the first grapheme-aware, benchmarked, pip-installable tokenizer library
> for Indian Abugida scripts, with HuggingFace-compatible weights and a public leaderboard.

**The gap Ghilli fills:**
Nobody has shipped a *Tamil-first, grapheme-aware, properly benchmarked, pip-installable*
tokenizer as a standalone community library. Existing projects (AI4Bharat, Sarvam, Tamil-Llama)
ship tokenizers as a side effect of training models — not as the primary artifact.

**Phases:**
- Phase 1 (Hackathon): Tamil + Hindi + Sinhala + Malayalam, GPE-BPE at 32k, benchmarked, on PyPI
- Phase 2 (v0.2): All 12 Abugida languages in Samanantar, morpheme alignment metric, docs site
- Phase 3 (v1.0): Multilingual tokenizer, transformers.js browser support, live leaderboard
- Phase 4 (long term): Downstream evaluation, community contributors, LangChain integration

---

## 2. Research Foundation

**Paper:** "Egalitarian Language Representation in Language Models: It All Begins with Tokenizers"
Velayuthan & Sarveswaran, COLING 2025, University of Jaffna, Sri Lanka.
https://aclanthology.org/2025.coling-main.400.pdf

### Core Finding

Pre-tokenization has MORE impact than the tokenization algorithm itself.

```
GPT-2 pre-tokenizer + BPE   → Compression Ratio 1.36×  (Tamil)
Whitespace pre-tokenizer + BPE → Compression Ratio 4.32×  (Tamil)
```

Switching only the pre-tokenizer = 3× improvement. Algorithm choice (BPE vs Unigram
vs WordPiece) = only ~0.2× difference when pre-tokenizer is already correct.

### Why Tamil (and Abugida Scripts) Are Hard

Tamil uses the Abugida writing system. A single visible character is composed of
multiple Unicode codepoints. For example:

- க + ா = கா  (consonant + vowel modifier = compound character)
- க + ் = க்  (consonant + pulli = pure consonant)

These multi-codepoint sequences are called **graphemes**.

GPT-2's regex pre-tokenizer treats Tamil vowel signs (Unicode category: Mark) as
punctuation, tearing them away from their consonants. This is the root cause of
poor tokenization for Tamil.

### Key Metrics (from paper)

**Compression Ratio (CR):**
```
CR = Original Sequence Length / Tokenized Sequence Length
Higher is better. English EC models achieve ~5×. Tamil with GPT-2: only 1.36×.
```

**Tokenization Parity (TP):**
```
TP = |t(Tamil sentence)| / |t(English sentence)|
Ideal = 1.0. GPT-2 Tamil TP = 4.54 (needs 4.54× more tokens than English).
```

**Published results (Table 3 & 4 in paper):**

| Tokenizer | Tamil CR | Tamil TP |
|-----------|----------|----------|
| GPT-2     | 1.36×    | 4.54     |
| GPT-4     | 2.13×    | 2.89     |
| Llama 3   | 2.13×    | 2.89     |
| FLAN-T5   | 9.21×    | 0.78     |
| Gemma 2   | 9.21×    | 0.78     |
| mT5       | 9.21×    | 0.78     |

**GPE vs other algorithms (Table 7 in paper — all with whitespace pre-tokenizer):**

| BPE  | Unigram | WordPiece | GPE  |
|------|---------|-----------|------|
| 4.32 | 4.31    | 4.12      | 4.36 |

Note: The tiny difference (0.04) CONFIRMS the paper's thesis — pre-tokenizer matters
more than algorithm. GPE wins but only marginally. The real win is fixing the pre-tokenizer.

### GPE Algorithm (Grapheme Pair Encoding)

BPE modified to use graphemes as atomic units instead of bytes:

1. Extract all grapheme clusters from training corpus using `grapheme` Python library
2. Initialize vocabulary with unique graphemes (not bytes)
3. Run standard BPE merge algorithm on grapheme sequences

This means the tokenizer can never tear apart a vowel modifier from its consonant,
because the grapheme extractor has already joined them into a single unit.

---

## 3. Repository Structure

Inspired by:
- **HuggingFace tokenizers** → stage-based folder layout (models/, trainers/, decoders/)
- **OpenAI tiktoken** → flat, no over-engineering, fast
- **Google SentencePiece** → algorithm-per-file pattern (bpe_model.cc / unigram_model.cc)
- **Karpathy minBPE** → base class + variant pattern (base.py, basic.py, regex.py)

```
ghilli/
│
├── ghilli/                          # Main Python package
│   ├── __init__.py                  # Public API: GhilliTokenizer, encode, decode
│   ├── tokenizer.py                 # Unified tokenizer class (like HF Tokenizer)
│   │
│   ├── algorithms/                  # One file per algorithm (like SentencePiece)
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract BaseTokenizer class (like minBPE)
│   │   ├── gpe.py                   # Grapheme Pair Encoding (BPE variant) — PRIMARY
│   │   ├── unigram.py               # Grapheme-seeded Unigram
│   │   └── wordpiece.py             # Grapheme-seeded WordPiece
│   │
│   ├── pretokenizer/                # Pre-tokenization stage (like HF pre_tokenizers/)
│   │   ├── __init__.py
│   │   ├── grapheme.py              # Grapheme cluster extractor — core innovation
│   │   ├── whitespace.py            # Simple whitespace splitter
│   │   └── regex.py                 # GPT-2 / GPT-4 style regex (for comparison)
│   │
│   └── scripts/                     # Per-script Unicode rules
│       ├── __init__.py
│       ├── tamil.py                 # Tamil Unicode block U+0B80–U+0BFF
│       ├── malayalam.py             # U+0D00–U+0D7F
│       ├── sinhala.py               # U+0D80–U+0DFF
│       ├── hindi.py                 # Devanagari U+0900–U+097F
│       └── base_script.py           # Abstract script class
│
├── pipeline/                        # Data pipeline (single entry point)
│   ├── pipeline.py                  # Main entry: python pipeline.py --langs ta hi
│   ├── config.yaml                  # ALL parameters live here
│   └── stages/
│       ├── __init__.py
│       ├── extract.py               # Load from HF cache → raw .txt
│       ├── clean.py                 # NFC normalize, filter, dedup
│       └── train.py                 # Grapheme extraction + tokenizer training
│
├── benchmark/                       # Reproducible evaluation
│   ├── benchmark.py                 # Main benchmark runner
│   ├── metrics/
│   │   ├── compression_ratio.py     # CR metric (Equation 1 from paper)
│   │   ├── tokenization_parity.py   # TP metric (Equation 2 from paper)
│   │   └── morpheme_alignment.py    # Novel metric — Ghilli's original contribution
│   ├── baselines/
│   │   ├── tiktoken_baseline.py     # GPT-4 / GPT-2 comparison
│   │   └── hf_baseline.py           # Llama 3 / mT5 / Gemma 2 comparison
│   └── results/
│       └── leaderboard.json         # Live results — updated on each benchmark run
│
├── data/                            # Auto-created, gitignored
│   ├── raw/                         # raw_{lang}.txt files
│   ├── clean/                       # clean_{lang}.txt files
│   └── weights/                     # ghilli-{lang}-{size}k.json files
│       ├── ta/
│       │   ├── ghilli-ta-16k.json
│       │   ├── ghilli-ta-32k.json
│       │   └── ghilli-ta-48k.json
│       ├── hi/
│       ├── ml/
│       └── si/
│
├── playground/                      # Web playground UI (already built)
│   └── tamil-tokenizer-playground.jsx
│
├── tests/
│   ├── test_gpe.py
│   ├── test_pretokenizer.py
│   ├── test_pipeline.py
│   └── fixtures/
│       └── sample_ta.txt
│
├── docs/                            # Documentation site (mkdocs or docusaurus)
│   ├── index.md
│   ├── quickstart.md
│   └── benchmark.md
│
├── .gitignore
├── pyproject.toml                   # Package config (replaces setup.py)
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 4. config.yaml — Full Spec

```yaml
# config.yaml — single source of truth for the entire pipeline

languages:
  - code: ta
    name: Tamil
    hf_dataset: ai4bharat/samanantar
    hf_split: train
    hf_column: tgt              # Tamil side of parallel corpus
    unicode_block: "0B80-0BFF"  # Tamil Unicode block
    flores_code: tam_Taml       # FLORES+ language code for evaluation
    min_ratio: 0.5              # min fraction of chars that must be Tamil

  - code: hi
    name: Hindi
    hf_dataset: ai4bharat/samanantar
    hf_split: train
    hf_column: tgt
    unicode_block: "0900-097F"  # Devanagari
    flores_code: hin_Deva
    min_ratio: 0.5

  - code: ml
    name: Malayalam
    hf_dataset: ai4bharat/samanantar
    hf_split: train
    hf_column: tgt
    unicode_block: "0D00-0D7F"
    flores_code: mal_Mlym
    min_ratio: 0.5

  - code: si
    name: Sinhala
    hf_dataset: ai4bharat/samanantar
    hf_split: train
    hf_column: tgt
    unicode_block: "0D80-0DFF"
    flores_code: sin_Sinh
    min_ratio: 0.5

  - code: kn
    name: Kannada
    hf_dataset: ai4bharat/samanantar
    hf_split: train
    hf_column: tgt
    unicode_block: "0C80-0CFF"
    flores_code: kan_Knda
    min_ratio: 0.5

  - code: te
    name: Telugu
    hf_dataset: ai4bharat/samanantar
    hf_split: train
    hf_column: tgt
    unicode_block: "0C00-0C7F"
    flores_code: tel_Telu
    min_ratio: 0.5

  - code: bn
    name: Bengali
    hf_dataset: ai4bharat/samanantar
    hf_split: train
    hf_column: tgt
    unicode_block: "0980-09FF"
    flores_code: ben_Beng
    min_ratio: 0.5

pipeline:
  corpus_dir: data/raw
  clean_dir: data/clean
  weights_dir: data/weights
  min_line_length: 10
  vocab_sizes: [16000, 32000, 48000]

training:
  algorithm: bpe                # bpe | unigram | wordpiece
  pretokenizer: whitespace      # whitespace | grapheme | regex_gpt2 | regex_gpt4
  special_tokens:
    - <unk>
    - <s>
    - </s>
    - <pad>
    - <mask>

benchmark:
  eval_dataset: facebook/flores
  eval_split: devtest
  baselines:
    - gpt2
    - gpt4
    - llama3
    - mt5
    - gemma2
  output: benchmark/results/leaderboard.json
```

---

## 5. Pipeline Stages

### How to run

```bash
# Single language
python pipeline/pipeline.py --langs ta

# Ring 1 (hackathon scope)
python pipeline/pipeline.py --langs ta hi si ml

# All configured languages
python pipeline/pipeline.py

# Custom config
python pipeline/pipeline.py --config my_config.yaml --langs ta
```

### Stage 1 — extract.py

**Input:** HuggingFace datasets cache (already downloaded)
**Output:** `data/raw/raw_{lang}.txt` — one sentence per line

```python
# Key logic
ds = load_dataset(lang_config["hf_dataset"], lang_config["code"], split="train")
# ds.column_names → check what the Tamil column is called
# Samanantar v1: columns are 'idx', 'src' (English), 'tgt' (Tamil)
# Some versions use 'tamil' or 'sentence_ta' — check column_names output
```

**Resumable:** Skips if output file already exists.

### Stage 2 — clean.py

**Input:** `data/raw/raw_{lang}.txt`
**Output:** `data/clean/clean_{lang}.txt`

**Operations (in order):**
1. `unicodedata.normalize("NFC", line)` — CRITICAL for Tamil. Normalizes composed/decomposed forms.
2. `re.sub(r"\s+", " ", line)` — collapse whitespace
3. Length filter: drop lines < `min_line_length` chars
4. Native script ratio filter: drop lines where < `min_ratio` chars are in the target Unicode block
5. Exact dedup via `hash(line)` → set (holds all hashes in RAM, fine for 5M sentences on 16GB)

**Resumable:** Skips if output file already exists.

### Stage 3 — train.py

**Input:** `data/clean/clean_{lang}.txt`
**Output:** `data/weights/{lang}/ghilli-{lang}-{size}k.json`

**Operations:**
1. Pass 1: Extract unique grapheme clusters using `grapheme.graphemes(line)` — builds initial alphabet
2. Pass 2: Train BPE/Unigram/WordPiece with `initial_alphabet=sorted(unique_graphemes)`
3. Pre-tokenizer: `pre_tokenizers.Whitespace()` — NOT GPT-2 regex
4. Trains at each vocab size in `config.pipeline.vocab_sizes`

**Resumable:** Skips individual weight files if they already exist.

**Expected training time on M1 Pro 16GB:**
- Extract graphemes from 5M sentences: ~2 min
- Train 32k BPE: ~5 min
- Total for Tamil at 3 vocab sizes: ~20 min

---

## 6. Core Algorithm — GPE

```python
# ghilli/algorithms/gpe.py

import grapheme
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class GPETokenizer:
    """
    Grapheme Pair Encoding — BPE variant that uses grapheme clusters
    as atomic units instead of bytes.

    Key insight from Velayuthan & Sarveswaran (COLING 2025):
    Tamil vowel modifiers (vowel signs) are Unicode category Mark and
    cannot stand alone. They must be processed with their consonant.
    The grapheme library handles this correctly; BPE on bytes does not.

    Example:
        ka + aa_modifier = கா  ← one grapheme, two Unicode codepoints
        GPT-2 regex splits these apart → meaningless tokens
        GPE keeps them together → linguistically valid tokens
    """

    def __init__(self, vocab_size: int = 32_000, special_tokens: list = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
        self.tokenizer = None

    def _extract_graphemes(self, corpus_path: str) -> set:
        """Pass 1: collect unique grapheme clusters from corpus."""
        unique = set()
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                unique.update(grapheme.graphemes(line))
        return unique

    def train(self, corpus_path: str) -> None:
        """Train GPE tokenizer on corpus file."""
        unique_graphemes = self._extract_graphemes(corpus_path)

        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # NOT GPT-2 regex
        self.tokenizer.decoder = decoders.BPEDecoder()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=sorted(unique_graphemes),  # graphemes, not bytes
            special_tokens=self.special_tokens,
            show_progress=True,
        )
        self.tokenizer.train(files=[corpus_path], trainer=trainer)

    def save(self, path: str) -> None:
        self.tokenizer.save(path)

    def load(self, path: str) -> None:
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def decode(self, ids: list) -> str:
        return self.tokenizer.decode(ids)
```

---

## 7. Benchmark Metrics

### Compression Ratio (CR) — from paper Equation 1

```python
# benchmark/metrics/compression_ratio.py
import grapheme as grapheme_lib

def compression_ratio(text: str, tokenizer) -> float:
    """
    CR = grapheme_count / token_count
    Higher is better. English EC models: ~5×. Tamil GPT-2: 1.36×.
    """
    grapheme_count = len(list(grapheme_lib.graphemes(text)))
    token_count = len(tokenizer.encode(text).tokens)
    return grapheme_count / token_count if token_count > 0 else 0.0
```

### Tokenization Parity (TP) — from paper Equation 2

```python
# benchmark/metrics/tokenization_parity.py

def tokenization_parity(native_text: str, english_text: str, tokenizer) -> float:
    """
    TP = |t(native)| / |t(english)|
    Ideal = 1.0. GPT-2 Tamil TP = 4.54 (needs 4.54× more tokens).
    Can be read as: context window size relative to English.
    """
    native_tokens = len(tokenizer.encode(native_text).tokens)
    english_tokens = len(tokenizer.encode(english_text).tokens)
    return native_tokens / english_tokens if english_tokens > 0 else 0.0
```

### Morpheme Alignment — Ghilli's original contribution

```python
# benchmark/metrics/morpheme_alignment.py
# Novel metric not in the paper — Ghilli's unique addition.
# Tests whether token boundaries align with true morpheme boundaries.
# Tamil is agglutinative: வீடுகளிலிருந்து = வீடு + கள் + இல் + இருந்து

def morpheme_alignment_score(word: str, true_morphemes: list, tokenizer) -> float:
    """
    Score = fraction of token boundaries that land on true morpheme boundaries.
    Expects true_morphemes as a list of strings (e.g. ['வீடு', 'கள்', 'இல்', 'இருந்து'])
    """
    tokens = tokenizer.encode(word).tokens
    true_boundaries = set()
    pos = 0
    for m in true_morphemes[:-1]:
        pos += len(m)
        true_boundaries.add(pos)

    token_boundaries = set()
    pos = 0
    for t in tokens[:-1]:
        pos += len(t.lstrip('▁Ġ'))   # strip HF space markers
        token_boundaries.add(pos)

    if not token_boundaries:
        return 0.0
    hits = len(token_boundaries & true_boundaries)
    return hits / len(token_boundaries)
```

### Expected Benchmark Results (based on paper)

| Tokenizer | Tamil CR | Tamil TP | Notes |
|-----------|----------|----------|-------|
| GPT-2 | 1.36× | 4.54 | Worst — regex tears Tamil apart |
| GPT-4 | 2.13× | 2.89 | Better regex, still English-centric |
| Llama 3 | 2.13× | 2.89 | Same as GPT-4 |
| mT5 | 9.21× | 0.78 | Best existing — multilingual |
| Ghilli GPE 32k | ~4.3× | ~0.9 | Expected based on paper Figure 1 |

Note: Ghilli won't beat mT5 on raw CR because mT5 has a much larger vocab dedicated
to multilingual coverage. Ghilli's value is being a *standalone Tamil-first library*
with transparent benchmarks, not beating every model in existence.

---

## 8. Dependencies

### requirements.txt

```txt
datasets>=2.14.0
huggingface_hub>=0.17.0
tokenizers>=0.15.0
transformers>=4.35.0
sentencepiece>=0.1.99
grapheme>=0.6.0
tqdm>=4.65.0
pyyaml>=6.0
indic-nlp-library>=0.92
```

### requirements-dev.txt

```txt
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
ruff>=0.1.0
mypy>=1.5.0
```

### Installation (M1 Mac)

```bash
conda create -n ghilli python=3.11 -y
conda activate ghilli
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .    # editable install for development
```

### pyproject.toml skeleton

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "ghilli"
version = "0.1.0"
description = "Grapheme-aware tokenizer for Indian Abugida scripts"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.9"
keywords = ["nlp", "tokenizer", "tamil", "indic", "abugida", "bpe"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3",
]
dependencies = [
    "tokenizers>=0.15.0",
    "grapheme>=0.6.0",
    "datasets>=2.14.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
]

[project.urls]
Homepage = "https://github.com/[your-username]/ghilli"
"HuggingFace Hub" = "https://huggingface.co/ghilli"

[project.scripts]
ghilli-train = "ghilli.cli:train"
ghilli-benchmark = "ghilli.cli:benchmark"
```

---

## 9. Reference Repos

Study these before writing any code:

| Repo | URL | What to study |
|------|-----|---------------|
| HuggingFace tokenizers | https://github.com/huggingface/tokenizers | Stage-based folder layout, pre_tokenizers/, decoders/ separation |
| OpenAI tiktoken | https://github.com/openai/tiktoken | How minimal a tokenizer lib can be. core.py is the whole thing |
| Google SentencePiece | https://github.com/google/sentencepiece | Algorithm-per-file: bpe_model.cc / unigram_model.cc side by side |
| Karpathy minBPE | https://github.com/karpathy/minbpe | Base class pattern. Read base.py → basic.py → regex.py in order |
| GPE paper code | https://github.com/sarves/ghilli (TBC) | The paper's reference implementation, 5k vocab, no packaging |

**Study order:** minBPE first (understand BPE), then HF tokenizers (understand production structure),
then tiktoken (understand what to cut), then SentencePiece (understand Unigram).

---

## 10. Roadmap

### Phase 1 — Hackathon (TOSSHack '26, 2–3 days)

- [ ] `pipeline/pipeline.py` running end-to-end for Tamil
- [ ] `ghilli/algorithms/gpe.py` — GPE-BPE implementation
- [ ] `benchmark/benchmark.py` — CR and TP vs GPT-2, GPT-4, Llama 3, mT5
- [ ] `pip install ghilli` on PyPI (v0.1.0)
- [ ] HuggingFace Hub: `ghilli/ta-32k` weights published
- [ ] Playground live (already built in React)
- [ ] README with benchmark table

### Phase 2 — v0.2 (1 month post hackathon)

- [ ] All 12 Abugida languages in Samanantar
- [ ] Unigram and WordPiece variants alongside GPE-BPE
- [ ] Morpheme alignment benchmark (novel metric)
- [ ] Vocab size sweep: 16k / 32k / 48k, with knee-of-curve analysis
- [ ] Documentation site (mkdocs)
- [ ] Submit resource paper to ACL/EMNLP Findings

### Phase 3 — v1.0 (3–6 months)

- [ ] `ghilli-multilingual` — one tokenizer for all 12 scripts
- [ ] transformers.js integration — load in browser
- [ ] Live benchmark leaderboard website
- [ ] Integration guide for Llama/Mistral fine-tuning on Indian languages

---

## 11. Naming & Branding

**Name:** Ghilli (கிள்ளி)

**Three layers of meaning:**
1. Vijay's 2004 blockbuster Tamil film — instant cultural recognition
2. Gilli-danda — ancient Tamil/South Asian street sport where a small stick (gilli)
   is struck at the right point and sent flying
3. The gilli stick itself — the small atomic unit that travels far — maps directly
   to grapheme-level tokenization

**Tagline:** "Strike at the right boundary."

**Package names (all available as of June 2026):**
- PyPI: `ghilli` ✅
- npm: `ghilli` ✅
- GitHub: `ghilli` ✅
- HuggingFace org: `ghilli` (check availability)
- Domain: `ghilli.dev` / `ghilli.ai` (likely available)

**HuggingFace model naming convention:**
```
ghilli/ta-gpe-16k
ghilli/ta-gpe-32k
ghilli/ta-gpe-48k
ghilli/hi-gpe-32k
ghilli/ml-gpe-32k
ghilli/indic-multilingual-32k
```

---

## Appendix A — Corpus Sources

| Source | HF Dataset ID | Size | Quality | Use |
|--------|---------------|------|---------|-----|
| Samanantar | ai4bharat/samanantar | ~700MB Tamil | ⭐⭐⭐⭐⭐ | Training |
| Tamil Wikipedia | wikimedia dump | ~200MB | ⭐⭐⭐⭐ | Training |
| FLORES+ | facebook/flores | ~1MB | ⭐⭐⭐⭐⭐ | Eval ONLY — never train |

**FLORES+ language codes:**
```
ta → tam_Taml
hi → hin_Deva
ml → mal_Mlym
si → sin_Sinh
kn → kan_Knda
te → tel_Telu
bn → ben_Beng
```

## Appendix B — Unicode Blocks for Abugida Scripts

| Language | Script | Unicode Block |
|----------|--------|---------------|
| Tamil | Tamil | U+0B80–U+0BFF |
| Malayalam | Malayalam | U+0D00–U+0D7F |
| Sinhala | Sinhala | U+0D80–U+0DFF |
| Hindi/Marathi/Nepali | Devanagari | U+0900–U+097F |
| Kannada | Kannada | U+0C80–U+0CFF |
| Telugu | Telugu | U+0C00–U+0C7F |
| Bengali/Assamese | Bengali | U+0980–U+09FF |
| Gujarati | Gujarati | U+0A80–U+0AFF |
| Punjabi (Gurmukhi) | Gurmukhi | U+0A00–U+0A7F |
| Odia | Odia | U+0B00–U+0B7F |

## Appendix C — Cursor Instructions

When working in Cursor on this project:

1. **Start with `config.yaml`** — all parameters live there. Never hardcode language
   codes, paths, or vocab sizes in Python files.

2. **One stage at a time** — build `extract.py` fully before touching `clean.py`.
   Each stage is independently testable.

3. **Resumability is non-negotiable** — every stage must check if output already
   exists and skip if so. You will run the pipeline many times.

4. **NFC normalization is always first** — before any filtering or processing,
   always call `unicodedata.normalize("NFC", text)`. Tamil text in the wild mixes
   composed and decomposed Unicode forms.

5. **Never use GPT-2 regex as your pre-tokenizer** — `pre_tokenizers.Whitespace()`
   only. The entire point of Ghilli is to fix what GPT-2's regex breaks.

6. **The grapheme library is your ground truth** — `grapheme.graphemes(text)` is
   the correct way to split Tamil text. Unicode codepoint splitting is wrong.

7. **Test on FLORES+ only** — never mix FLORES+ sentences into training data.
   It is the benchmark. Keep it clean.
