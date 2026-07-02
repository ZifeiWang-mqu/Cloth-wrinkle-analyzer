"""Text embeddings for review-memory retrieval.

This pass ships a LOCAL, deterministic hashed bag-of-words embedder — no
external APIs, no keys, fully offline and testable. It sits behind a tiny
interface so a real semantic embedder (e.g. OpenAI, server-side only and
optional) can replace it later without touching the search endpoint:
subclass :class:`BaseEmbedder` and swap :func:`get_embedder`.

Determinism note: token hashing uses md5, NOT Python's builtin ``hash()``
(which is randomized per process) — vectors are stable across runs/machines.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BaseEmbedder:
    name = "base"
    dim: int = 0

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalHashEmbedder(BaseEmbedder):
    """Hashed bag-of-words, L2-normalized. Lexical, cheap, deterministic."""

    name = "local_hash_bow"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            vec[int(digest[:8], 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [round(v / norm, 6) for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


_EMBEDDER: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    """Factory + cache. Swap the constructed class here to upgrade later."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = LocalHashEmbedder()
    return _EMBEDDER
