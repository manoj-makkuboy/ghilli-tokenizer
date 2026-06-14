"""Tests for GPE-BPE and Unigram tokenizers — roundtrip and basic sanity checks."""

import os
import tempfile

import pytest

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ghilli.algorithms.gpe import GPETokenizer
from ghilli.algorithms.unigram import UnigramTokenizer
from ghilli.pretokenizer.grapheme import extract_graphemes
from ghilli.tokenizer import GhilliTokenizer

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_ta.txt")


@pytest.fixture(scope="module")
def grapheme_alphabet():
    return extract_graphemes(FIXTURE_PATH)


@pytest.fixture(scope="module")
def trained_bpe(grapheme_alphabet, tmp_path_factory):
    tok = GPETokenizer(vocab_size=500, special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])
    tok.train(FIXTURE_PATH, initial_alphabet=grapheme_alphabet)
    path = str(tmp_path_factory.mktemp("bpe") / "test_bpe.json")
    tok.save(path)
    return tok, path


@pytest.fixture(scope="module")
def trained_unigram(grapheme_alphabet, tmp_path_factory):
    tok = UnigramTokenizer(vocab_size=500, special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])
    tok.train(FIXTURE_PATH, initial_alphabet=grapheme_alphabet)
    path = str(tmp_path_factory.mktemp("unigram") / "test_unigram.json")
    tok.save(path)
    return tok, path


class TestGPEBPE:
    def test_encode_returns_ids(self, trained_bpe):
        tok, _ = trained_bpe
        result = tok.encode("தமிழ் மொழி")
        assert len(result.ids) > 0

    def test_roundtrip(self, trained_bpe):
        tok, _ = trained_bpe
        text = "தமிழ் மொழி"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded.ids)
        assert decoded == text

    def test_save_load_roundtrip(self, trained_bpe):
        _, path = trained_bpe
        loaded = GhilliTokenizer(path)
        text = "கணினி அறிவியல்"
        encoded = loaded.encode(text)
        decoded = loaded.decode(encoded.ids)
        assert decoded == text


class TestUnigram:
    def test_encode_returns_ids(self, trained_unigram):
        tok, _ = trained_unigram
        result = tok.encode("தமிழ் மொழி")
        assert len(result.ids) > 0

    def test_roundtrip(self, trained_unigram):
        tok, _ = trained_unigram
        text = "தமிழ் மொழி"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded.ids)
        assert decoded == text

    def test_save_load_roundtrip(self, trained_unigram):
        _, path = trained_unigram
        loaded = GhilliTokenizer(path)
        text = "சென்னை தமிழ்நாடு"
        encoded = loaded.encode(text)
        decoded = loaded.decode(encoded.ids)
        assert decoded == text


class TestGraphemeExtraction:
    def test_extracts_graphemes(self, grapheme_alphabet):
        assert len(grapheme_alphabet) > 0
        # Tamil characters should be present
        assert any("\u0B80" <= c[0] <= "\u0BFF" for c in grapheme_alphabet if c)

    def test_no_whitespace_in_alphabet(self, grapheme_alphabet):
        for g in grapheme_alphabet:
            assert g.strip() == g
            assert g != ""
