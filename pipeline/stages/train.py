"""Stage 3: Train BPE and Unigram tokenizers at multiple vocab sizes."""

import os
import sys

# Add project root to path so ghilli package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ghilli.pretokenizer.grapheme import extract_graphemes
from ghilli.algorithms.gpe import GPETokenizer
from ghilli.algorithms.unigram import UnigramTokenizer


ALGORITHM_MAP = {
    "bpe": GPETokenizer,
    "unigram": UnigramTokenizer,
}


def run(lang_config: dict, pipeline_config: dict, training_config: dict) -> list[str]:
    code = lang_config["code"]
    clean_path = os.path.join(pipeline_config["clean_dir"], f"clean_{code}.txt")
    weights_dir = os.path.join(pipeline_config["weights_dir"], code)
    os.makedirs(weights_dir, exist_ok=True)

    vocab_sizes = pipeline_config["vocab_sizes"]
    algorithms = training_config["algorithms"]
    special_tokens = training_config["special_tokens"]

    # Pass 1: Extract grapheme clusters for initial alphabet
    print(f"[train] Extracting grapheme clusters from {clean_path}...")
    initial_alphabet = extract_graphemes(clean_path)
    print(f"[train] Found {len(initial_alphabet)} unique grapheme clusters")

    output_paths = []

    for algo_name in algorithms:
        tokenizer_cls = ALGORITHM_MAP.get(algo_name)
        if tokenizer_cls is None:
            print(f"[train] Unknown algorithm '{algo_name}', skipping.")
            continue

        for size in vocab_sizes:
            size_k = size // 1000
            output_path = os.path.join(weights_dir, f"ghilli-{code}-{algo_name}-{size_k}k.json")

            if os.path.exists(output_path):
                print(f"[train] {output_path} already exists, skipping.")
                output_paths.append(output_path)
                continue

            print(f"[train] Training {algo_name.upper()} vocab_size={size:,} for {lang_config['name']}...")
            tok = tokenizer_cls(vocab_size=size, special_tokens=special_tokens)
            tok.train(clean_path, initial_alphabet=initial_alphabet)
            tok.save(output_path)
            print(f"[train] Saved → {output_path}")
            output_paths.append(output_path)

    return output_paths
