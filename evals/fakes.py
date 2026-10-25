"""Deterministic stand-ins for parts that need a model download."""

from __future__ import annotations

import re
import zlib

import numpy as np

from harness.memory import unit


class WordEmbedder:
    """Bag of words: texts that share words score high. No model download."""

    min_score = 0.3

    def __init__(self, name: str = "words") -> None:
        self.name = name
        self.calls = 0
        self.broken = False

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.broken:
            raise RuntimeError("embedding model unavailable")
        self.calls += 1
        out = np.zeros((len(texts), 1024), dtype=np.float32)
        for row, t in enumerate(texts):
            for word in re.findall(r"[a-z]+", t.lower()):
                out[row, zlib.crc32(word.encode()) % 1024] += 1
        return unit(out)
