"""Tokenizer used by ArcNeuron training, tuning, and generation.

The tokenizer deliberately does not contain semantic rules.  It learns byte-pair
pieces only to compress text into a shorter reversible token stream.  Byte
fallback means text outside the training alphabet can still be represented.

The serialized SentencePiece model is stored inside the ArcNeuron checkpoint, so
training does not leave extra vocabulary files beside the repository sources.
"""

from pathlib import Path  # Path keeps all file handling explicit and platform-independent.
from tempfile import TemporaryDirectory  # SentencePiece's trainer writes files, so a temporary directory keeps them out of the repo.

import sentencepiece as spm  # SentencePiece provides a mature BPE implementation with byte fallback.


class ArcTokenizer:
    """Small wrapper around one serialized SentencePiece BPE model."""

    def __init__(self, model_bytes: bytes) -> None:
        if not model_bytes:  # An empty tokenizer model cannot map text to ArcNeuron's embedding IDs.
            raise ValueError("model_bytes cannot be empty")
        self._model_bytes = bytes(model_bytes)  # Make an owned immutable copy so checkpoints cannot mutate tokenizer state accidentally.
        self._processor = spm.SentencePieceProcessor(model_proto=self._model_bytes)  # Load the complete tokenizer directly from memory.

    @classmethod
    def train(cls, corpus_path: str | Path, vocab_size: int = 8192, character_coverage: float = 1.0) -> "ArcTokenizer":
        corpus_path = Path(corpus_path)  # Normalize strings and Path objects to one representation.
        if not corpus_path.is_file():  # Tokenizer training must fail loudly when the corpus path is wrong.
            raise FileNotFoundError(corpus_path)
        if vocab_size < 512:  # Byte fallback itself consumes 256 byte symbols plus SentencePiece meta symbols.
            raise ValueError("vocab_size must be at least 512 when byte fallback is enabled")
        if not 0.0 < character_coverage <= 1.0:  # Coverage must reserve some characters; byte fallback covers the rest.
            raise ValueError("character_coverage must be in the interval (0, 1]")
        with TemporaryDirectory(prefix="arcneuron-tokenizer-") as temp_dir:  # All intermediate SentencePiece files disappear when this block ends.
            model_prefix = Path(temp_dir) / "arcneuron"  # SentencePiece derives .model and .vocab paths from this temporary prefix.
            spm.SentencePieceTrainer.train(  # Train only statistical text pieces; no dictionary or semantic annotation is supplied.
                input=str(corpus_path),  # The raw natural-language corpus is the only tokenizer training input.
                model_prefix=str(model_prefix),  # Write the temporary model beside the temporary vocabulary file.
                vocab_size=vocab_size,  # Cap the number of subword pieces used by ArcNeuron's embedding table.
                model_type="bpe",  # BPE keeps tokenization simple, fast, deterministic, and widely understood.
                character_coverage=character_coverage,  # Keep at least this fraction of characters; byte fallback represents the rest so a small vocab can survive a multilingual corpus.
                byte_fallback=True,  # Any unseen Unicode text remains losslessly representable through UTF-8 byte pieces.
                normalization_rule_name="identity",  # Preserve the user's exact text instead of silently rewriting Unicode forms.
                remove_extra_whitespaces=False,  # Whitespace itself can carry formatting information and therefore must not be collapsed.
                add_dummy_prefix=False,  # Do not inject a synthetic leading space that was absent from the original text.
                split_by_whitespace=True,  # Whitespace boundaries make BPE training cheaper without assigning semantic meaning.
                bos_id=1,  # Reserve a beginning-of-sequence token for complete training documents and prompts.
                eos_id=2,  # Reserve an end-of-sequence token so generation can learn when text naturally ends.
                pad_id=3,  # Reserve a padding token for future batching strategies even though the current trainer uses packed chunks.
                unk_id=0,  # SentencePiece requires an unknown token even though byte fallback should make it effectively unnecessary.
                hard_vocab_limit=False,  # Tiny demo corpora may not contain enough distinct pairs to fill a large requested vocabulary.
                minloglevel=2,  # Keep Colab output focused on ArcNeuron training rather than thousands of tokenizer trainer messages.
            )
            model_bytes = (model_prefix.with_suffix(".model")).read_bytes()  # Read the complete trained tokenizer before the temporary directory is deleted.
        return cls(model_bytes)  # Return a normal in-memory tokenizer ready to encode training text immediately.

    @classmethod
    def from_bytes(cls, model_bytes: bytes) -> "ArcTokenizer":
        return cls(model_bytes)  # Checkpoints restore the exact tokenizer that produced the embedding IDs used during training.

    @property
    def vocab_size(self) -> int:
        return int(self._processor.get_piece_size())  # ArcNeuron's embedding and LM head must have exactly this many rows.

    @property
    def bos_id(self) -> int:
        return int(self._processor.bos_id())  # Expose the tokenizer's learned-model metadata without duplicating the numeric value elsewhere.

    @property
    def eos_id(self) -> int:
        return int(self._processor.eos_id())  # Generation uses this ID only as a sequence boundary, never as a semantic rule.

    @property
    def pad_id(self) -> int:
        return int(self._processor.pad_id())  # Future padded batches can use the same checkpoint-stable special token.

    def to_bytes(self) -> bytes:
        return self._model_bytes  # Store this byte string inside the PyTorch checkpoint so no sidecar tokenizer file is required.

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids = list(self._processor.encode(text, out_type=int))  # SentencePiece turns raw text into subword IDs using only its learned compression model.
        if add_bos:  # Complete sequences can explicitly tell the neural model where a document or prompt begins.
            token_ids.insert(0, self.bos_id)  # Insert BOS after tokenization so it can never be confused with literal user text.
        if add_eos:  # Complete documents can also teach a natural stopping boundary.
            token_ids.append(self.eos_id)  # Append EOS after all text pieces so next-token training can learn to terminate.
        return token_ids  # The returned integers are the only tokenizer information ArcNeuron receives.

    def decode(self, token_ids: list[int]) -> str:
        return self._processor.decode(token_ids)  # Convert generated IDs back to text without adding interpretation or post-processing rules.
