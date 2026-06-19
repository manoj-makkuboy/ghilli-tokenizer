"""Grapheme-seeded Unigram tokenizer."""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from .base import BaseTokenizer


class UnigramTokenizer(BaseTokenizer):
    """Unigram tokenizer with grapheme-aware initial alphabet and Metaspace pre-tokenizer.

    Same grapheme-seeded alphabet as GPE, but uses Unigram model instead of BPE.
    Uses Metaspace pre-tokenizer/decoder for correct whitespace reconstruction.
    """

    def train(self, corpus_path: str, initial_alphabet: list[str] | None = None) -> None:
        self.tokenizer = Tokenizer(models.Unigram())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        self.tokenizer.decoder = decoders.Metaspace()

        trainer = trainers.UnigramTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=initial_alphabet or [],
            special_tokens=self.special_tokens,
            unk_token="<unk>",
            show_progress=True,
        )
        self.tokenizer.train(files=[corpus_path], trainer=trainer)
