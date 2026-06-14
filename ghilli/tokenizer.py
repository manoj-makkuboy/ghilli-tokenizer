"""Unified GhilliTokenizer — wraps GPE-BPE and Unigram tokenizers."""

from tokenizers import Tokenizer

from .algorithms.gpe import GPETokenizer
from .algorithms.unigram import UnigramTokenizer


class GhilliTokenizer:
    """Unified interface to load and use trained Ghilli tokenizers."""

    def __init__(self, path: str):
        self._tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str):
        return self._tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @staticmethod
    def train_bpe(corpus_path: str, vocab_size: int = 32_000,
                  initial_alphabet: list[str] | None = None,
                  special_tokens: list[str] | None = None) -> GPETokenizer:
        tok = GPETokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
        tok.train(corpus_path, initial_alphabet=initial_alphabet)
        return tok

    @staticmethod
    def train_unigram(corpus_path: str, vocab_size: int = 32_000,
                      initial_alphabet: list[str] | None = None,
                      special_tokens: list[str] | None = None) -> UnigramTokenizer:
        tok = UnigramTokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
        tok.train(corpus_path, initial_alphabet=initial_alphabet)
        return tok
