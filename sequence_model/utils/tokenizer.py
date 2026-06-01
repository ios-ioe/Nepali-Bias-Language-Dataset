import re
import json
from collections import Counter
from typing import List, Dict, Optional
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PAD_TOKEN, UNK_TOKEN, MAX_VOCAB_SIZE, MAX_SEQ_LEN


def nepali_tokenize(text: str) -> List[str]:
    # Normalise dandas and common punctuation
    text = text.replace("।", " । ").replace("?", " ? ").replace("!", " ! ")
    # Split on whitespace, drop empty strings
    tokens = [t.strip() for t in text.split() if t.strip()]
    return tokens


class NepaliTokenizer:

    def __init__(
        self,
        max_vocab_size: int = MAX_VOCAB_SIZE,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len    = max_seq_len

        # Special tokens always occupy the first two indices
        self.word2idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.vocab_size: int = 2


    def build_vocab(self, texts: List[str]) -> None:
        """Count tokens across all texts and keep the top-N."""
        counter: Counter = Counter()
        for text in texts:
            counter.update(nepali_tokenize(text))

        # Most-common tokens (excluding specials)
        most_common = counter.most_common(self.max_vocab_size - 2)
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word

        self.vocab_size = len(self.word2idx)
        print(f"[Tokenizer] Vocabulary built: {self.vocab_size} tokens")

    def encode(
        self,
        text: str,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """
        Tokenise ``text`` → integer indices, padded/truncated to ``max_len``.
        Returns a list of length ``max_len`` with trailing PAD tokens.
        """
        max_len = max_len or self.max_seq_len
        tokens  = nepali_tokenize(text)
        ids     = [self.word2idx.get(t, self.word2idx[UNK_TOKEN]) for t in tokens]

        # Truncate
        ids = ids[:max_len]
        # Pad
        ids += [self.word2idx[PAD_TOKEN]] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert integer indices back to a space-joined string."""
        return " ".join(
            self.idx2word.get(i, UNK_TOKEN)
            for i in ids
            if i != self.word2idx[PAD_TOKEN]
        )

    def get_tokens(self, text: str) -> List[str]:
        """Return raw tokens (no padding) for a given text."""
        return nepali_tokenize(text)

    def save(self, path: str) -> None:
        """Serialise vocabulary to a JSON file."""
        data = {
            "max_vocab_size": self.max_vocab_size,
            "max_seq_len":    self.max_seq_len,
            "word2idx":       self.word2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "NepaliTokenizer":
        """Restore a tokeniser from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(
            max_vocab_size=data["max_vocab_size"],
            max_seq_len=data["max_seq_len"],
        )
        tok.word2idx  = {k: int(v) for k, v in data["word2idx"].items()}
        tok.idx2word  = {int(v): k for k, v in data["word2idx"].items()}
        tok.vocab_size = len(tok.word2idx)
        print(f"[Tokenizer] Loaded from {path} ({tok.vocab_size} tokens)")
        return tok
