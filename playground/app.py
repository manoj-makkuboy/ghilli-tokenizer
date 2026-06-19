"""Ghilli Tokenizer Playground — compare Tamil tokenization across models."""

import os

import grapheme
import pandas as pd
import streamlit as st
import tiktoken
from ghilli import GhilliTokenizer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ghilli Tokenizer Playground",
    page_icon="🏏",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "weights", "ta")

SAMPLE_TEXTS = {
    "Simple greeting": "வணக்கம் உலகம்",
    "Tamil language (short)": "தமிழ் மொழி",
    "Tamil language (full sentence)": "தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்று",
    "AI and Tamil": "செயற்கை நுண்ணறிவு தமிழ் மொழியை மேம்படுத்த உதவுகிறது",
    "Thirukkural (Kural 391)": "கற்க கசடறக் கற்பவை கற்றபின் நிற்க அதற்குத் தக",
    "Mixed Tamil and English": "Chennai என்பது Tamil Nadu மாநிலத்தின் தலைநகரம்",
    "Long agglutinative word": "வீடுகளிலிருந்து புறப்பட்டார்கள்",
}

# Token colors — alternating palette for visual distinction
TOKEN_COLORS = [
    "#DBEAFE", "#DCFCE7", "#FEF9C3", "#FCE7F3", "#E0E7FF",
    "#CFFAFE", "#FEE2E2", "#F3E8FF", "#D1FAE5", "#FFF7ED",
]

# ---------------------------------------------------------------------------
# Tokenizer loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_ghilli_tokenizer(path: str) -> GhilliTokenizer:
    return GhilliTokenizer(path)


@st.cache_resource
def load_tiktoken(encoding_name: str):
    return tiktoken.get_encoding(encoding_name)


def get_available_ghilli_models() -> dict[str, str]:
    """Discover trained Ghilli models from data/weights/ta/."""
    models = {}
    if not os.path.isdir(WEIGHTS_DIR):
        return models
    for f in sorted(os.listdir(WEIGHTS_DIR)):
        if f.endswith(".json"):
            label = f.replace(".json", "").replace("ghilli-ta-", "").replace("-", " ").upper()
            models[f"Ghilli {label}"] = os.path.join(WEIGHTS_DIR, f)
    return models


# ---------------------------------------------------------------------------
# Tokenize helpers
# ---------------------------------------------------------------------------
def tokenize_ghilli(text: str, path: str) -> list[str]:
    tok = load_ghilli_tokenizer(path)
    enc = tok.encode(text)
    return enc.tokens


def tokenize_tiktoken(text: str, encoding_name: str) -> list[str]:
    enc = load_tiktoken(encoding_name)
    ids = enc.encode(text)
    tokens = []
    for tid in ids:
        b = enc.decode_single_token_bytes(tid)
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            s = b.hex()
        tokens.append(s)
    return tokens


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_tokens_html(tokens: list[str], colors: list[str] = TOKEN_COLORS) -> str:
    """Render tokens as colored inline spans."""
    spans = []
    for i, token in enumerate(tokens):
        color = colors[i % len(colors)]
        display = token.replace("▁", "·")  # Make metaspace visible
        # Escape HTML
        display = display.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Show raw bytes as dimmed
        if all(c in "0123456789abcdef" for c in display.lower()) and len(display) > 2:
            style = f"background-color:{color};color:#999;padding:2px 5px;margin:1px;border-radius:4px;border:1px solid #ddd;display:inline-block;font-size:14px;font-family:monospace"
        else:
            style = f"background-color:{color};padding:2px 5px;margin:1px;border-radius:4px;border:1px solid #ddd;display:inline-block;font-size:16px;font-family:sans-serif"
        spans.append(f'<span style="{style}">{display}</span>')
    return " ".join(spans)


def render_comparison_card(name: str, tokens: list[str], text: str):
    """Render a single tokenizer result card."""
    grapheme_count = len(list(grapheme.graphemes(text.replace(" ", ""))))
    token_count = len(tokens)
    cr = grapheme_count / token_count if token_count > 0 else 0

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Tokens", token_count)
    col2.metric("Characters", len(text))
    col3.metric("Compression Ratio", f"{cr:.2f}x")

    # Colored tokens
    html = render_tokens_html(tokens)
    st.markdown(f'<div style="line-height:2.2;padding:12px 0">{html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Available tokenizers
# ---------------------------------------------------------------------------
TIKTOKEN_MODELS = {
    "GPT-2": "gpt2",
    "GPT-4 (cl100k)": "cl100k_base",
    "GPT-4o (o200k)": "o200k_base",
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.title("🏏 Ghilli Tokenizer Playground")
    st.markdown(
        "Compare **grapheme-aware Tamil tokenization** against GPT-2, GPT-4, and GPT-4o. "
        "See how many tokens each model needs for the same Tamil text."
    )

    st.divider()

    # --- Input ---
    col_input, col_sample = st.columns([3, 1])
    with col_sample:
        sample = st.selectbox("Sample texts", ["Custom"] + list(SAMPLE_TEXTS.keys()))
    with col_input:
        default = SAMPLE_TEXTS.get(sample, "") if sample != "Custom" else ""
        text = st.text_area(
            "Enter Tamil text",
            value=default,
            height=100,
            placeholder="தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்று",
        )

    if not text.strip():
        st.info("Enter some Tamil text above to see how different tokenizers handle it.")
        return

    # --- Tokenizer selection ---
    ghilli_models = get_available_ghilli_models()
    all_tokenizers = list(ghilli_models.keys()) + list(TIKTOKEN_MODELS.keys())

    selected = st.multiselect(
        "Tokenizers to compare",
        all_tokenizers,
        default=["Ghilli BPE 32K", "GPT-2", "GPT-4o (o200k)"] if "Ghilli BPE 32K" in all_tokenizers else all_tokenizers[:3],
    )

    if not selected:
        st.warning("Select at least one tokenizer.")
        return

    st.divider()

    # --- Results ---
    # Collect results for summary table
    summary = []

    for name in selected:
        if name in ghilli_models:
            tokens = tokenize_ghilli(text, ghilli_models[name])
        elif name in TIKTOKEN_MODELS:
            tokens = tokenize_tiktoken(text, TIKTOKEN_MODELS[name])
        else:
            continue

        grapheme_count = len(list(grapheme.graphemes(text.replace(" ", ""))))
        token_count = len(tokens)
        cr = grapheme_count / token_count if token_count > 0 else 0
        summary.append({"Tokenizer": name, "Tokens": token_count, "CR": cr})

        is_ghilli = name.startswith("Ghilli")
        icon = "✅" if is_ghilli else "📊"
        with st.expander(f"{icon} **{name}** — {token_count} tokens", expanded=True):
            render_comparison_card(name, tokens, text)

    # --- Summary bar chart ---
    if len(summary) > 1:
        st.divider()
        st.subheader("Token Count Comparison")

        df = pd.DataFrame(summary)
        # Color Ghilli bars differently
        st.bar_chart(df.set_index("Tokenizer")["Tokens"], horizontal=True)

        st.caption(
            "**Fewer tokens = better.** "
            "Ghilli's grapheme-aware approach keeps Tamil characters intact, "
            "while GPT-2 tears them into byte-level subwords (129 tokens for a single sentence). "
            "Compression Ratio (CR) = grapheme count / token count — higher is better."
        )

    # --- How it works sidebar ---
    with st.sidebar:
        st.header("How Ghilli Works")
        st.markdown("""
**The problem:** GPT-2's regex pre-tokenizer treats Tamil vowel
modifiers as punctuation and tears them from their consonants.

**The fix:**
1. Use **whitespace** pre-tokenizer (not GPT-2 regex)
2. Use **grapheme clusters** as atomic units
3. Seed vocabulary with all unique graphemes

This alone yields a **3x improvement** in compression ratio.

---

**Grapheme vs Codepoint:**
```
"தமிழ்" (Tamil)

Graphemes:  ['த', 'மி', 'ழ்']     → 3
Codepoints: ['த','ம','ி','ழ','்'] → 5
```

The vowel sign `ி` and pulli `்` must stay
attached to their consonants. The `grapheme`
library handles this correctly.

---

**Based on:**
[Velayuthan & Sarveswaran, COLING 2025](https://aclanthology.org/2025.coling-main.400.pdf)

*"Egalitarian Language Representation
in Language Models: It All Begins
with Tokenizers"*
        """)

        st.divider()
        st.markdown(
            "**Ghilli** (கிள்ளி) — *Strike at the right boundary.*\n\n"
            "[GitHub](https://github.com/manoj-makkuboy/ghilli-tokenizer) · "
            "Apache 2.0"
        )


if __name__ == "__main__":
    main()
