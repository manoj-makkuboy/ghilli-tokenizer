"""Abstract base class for Ghilli tokenizers."""

from abc import ABC, abstractmethod
from tokenizers import Tokenizer


class BaseTokenizer(ABC):
    def __init__(self, vocab_size: int = 32_000, special_tokens: list[str] | None = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
        self.tokenizer: Tokenizer | None = None

    @abstractmethod
    def train(self, corpus_path: str, initial_alphabet: list[str] | None = None) -> None:
        ...

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def save(self, path: str) -> None:
        self.tokenizer.save(path)

    def load(self, path: str) -> None:
        self.tokenizer = Tokenizer.from_file(path)
