"""Grapheme cluster extraction using the `grapheme` library."""

import grapheme as _grapheme
from tqdm import tqdm


def extract_graphemes(corpus_path: str) -> list[str]:
    """Extract sorted unique grapheme clusters from a corpus file.

    This is the key insight from Velayuthan & Sarveswaran (COLING 2025):
    Tamil vowel modifiers cannot stand alone — they must stay attached to
    their consonant. The grapheme library handles this correctly.
    """
    unique = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Extracting graphemes"):
            unique.update(_grapheme.graphemes(line.strip()))
    # Remove empty strings and whitespace-only entries
    unique.discard("")
    unique.discard(" ")
    unique.discard("\n")
    unique.discard("\t")
    return sorted(unique)
