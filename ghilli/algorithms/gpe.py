"""GPE-BPE: Grapheme Pair Encoding — BPE variant using grapheme clusters as atomic units."""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from .base import BaseTokenizer


class GPETokenizer(BaseTokenizer):
    """BPE tokenizer with grapheme-aware initial alphabet and whitespace pre-tokenizer.

    Key insight: Using grapheme clusters (not bytes) as the initial alphabet
    ensures vowel modifiers stay attached to consonants in Abugida scripts.
    """

    def train(self, corpus_path: str, initial_alphabet: list[str] | None = None) -> None:
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=initial_alphabet or [],
            special_tokens=self.special_tokens,
            show_progress=True,
        )
        self.tokenizer.train(files=[corpus_path], trainer=trainer)
