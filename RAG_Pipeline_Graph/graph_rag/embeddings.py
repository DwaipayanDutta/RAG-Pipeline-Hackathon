"""
Embedding backends.

An embedder is any callable `texts -> list of float vectors` with a stable name (the name
is part of the index fingerprint, so switching models triggers a rebuild).

    sentence-transformers  all-MiniLM-L6-v2 (default when installed / cached)
    hash                   deterministic hashed bag-of-words; no downloads, for offline use and tests
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Callable, Dict, List, Sequence

from .text import content_tokens

logger = logging.getLogger("graph_rag.embeddings")

Embedder = Callable[[Sequence[str]], List[List[float]]]


class HashEmbedder:
    """Hashed bag of stemmed content words (+ bigrams), L2-normalised. Lexical, not semantic."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-bow-{dim}"

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            toks = content_tokens(text)
            for tok in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v] if n else v)
        return out


class SentenceTransformerEmbedder:
    """Lazy wrapper around sentence-transformers (model loaded on first call)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.name = f"st:{model_name}"
        self._model = None

    def load(self) -> "SentenceTransformerEmbedder":
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        self.load()
        vecs = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]


def embedder_name(embedder: Any) -> str:
    """Stable name for any embedder (also accepts ChromaDB-style EmbeddingFunctions)."""
    name = getattr(embedder, "name", None)
    if callable(name):
        try:
            name = name()
        except TypeError:
            name = None
    return str(name or type(embedder).__name__)


def make_embedder(config: Dict[str, Any]) -> Embedder:
    """Pick an embedder from `embedding_backend` (auto falls back to hash if the model is unavailable)."""
    backend = config.get("embedding_backend", "auto")
    if backend == "hash":
        return HashEmbedder()
    try:
        emb = SentenceTransformerEmbedder(config["embedding_model"]).load()
        logger.info("Embeddings: %s", config["embedding_model"])
        return emb
    except Exception as e:
        if backend == "sentence-transformers":
            raise RuntimeError(f"Could not load embedding model {config['embedding_model']}: {e}") from e
        logger.warning("sentence-transformers unavailable (%s); using offline hash embeddings. "
                       "Semantic recall will be lexical only.", type(e).__name__)
        return HashEmbedder()
