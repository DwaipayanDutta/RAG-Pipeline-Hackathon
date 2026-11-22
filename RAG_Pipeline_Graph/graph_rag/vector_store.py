"""
Vector stores for wiki passages (tier wiki) and PDF chunks (tier pdf).

    ChromaVectorStore     persistent ChromaDB collection (cosine space)
    InMemoryVectorStore   pure-Python cosine search persisted to JSON (no extra dependency)

Both compute embeddings with the configured embedder and pass vectors explicitly, so they do
not depend on ChromaDB's (version-specific) embedding-function API.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .embeddings import Embedder
from .models import Chunk
from .text import jaccard, token_set

logger = logging.getLogger("graph_rag.vector_store")


def _chunk_id(c: Chunk) -> str:
    return f"{c.metadata.get('source', 'doc')}:{c.page}:{c.chunk_id}"


def _meta(c: Chunk) -> Dict[str, Any]:
    m = {"chunk_id": c.chunk_id, "page": c.page, "section": c.section or "", "source_type": c.source_type}
    for k in ("tier", "source", "title", "path", "timestamp", "subject", "doc_id"):
        if c.metadata.get(k) is not None:
            m[k] = str(c.metadata[k])
    return m


def _hit(text: str, meta: Dict[str, Any], dist: float) -> Dict[str, Any]:
    return {"text": text, "tier": meta.get("tier", "pdf"), "source": meta.get("source", ""),
            "title": meta.get("title", ""), "section": meta.get("section", ""),
            "page": int(meta.get("page", 0) or 0), "chunk_id": int(meta.get("chunk_id", 0) or 0),
            "path": meta.get("path", ""), "timestamp": meta.get("timestamp", ""),
            "distance": round(float(dist), 4), "score": round(1.0 - float(dist), 4)}


def postprocess_hits(query: str, hits: List[Dict[str, Any]], k: int, max_distance: Optional[float],
                     min_overlap: int = 0, dedupe_jaccard: float = 1.01) -> List[Dict[str, Any]]:
    """Distance cut-off, lexical-overlap guard and near-duplicate removal, keeping rank order."""
    q_tokens = token_set(query)
    kept: List[Dict[str, Any]] = []
    kept_tokens: List[set] = []
    for h in sorted(hits, key=lambda h: h["distance"]):
        if max_distance is not None and h["distance"] > max_distance:
            continue
        toks = token_set(h["text"])
        if min_overlap and len(q_tokens & toks) < min_overlap:
            continue
        if any(jaccard(toks, t) >= dedupe_jaccard for t in kept_tokens):
            continue
        kept.append(h)
        kept_tokens.append(toks)
        if len(kept) >= k:
            break
    return kept


class VectorStore:
    """Interface shared by the backends."""
    backend = "base"

    def count(self) -> int:
        raise NotImplementedError

    def add_chunks(self, chunks: List[Chunk], source: Optional[str] = None) -> None:
        raise NotImplementedError

    def _raw_search(self, query: str, n: int, tiers: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def search(self, query: str, k: int = 3, max_distance: Optional[float] = None,
               tiers: Optional[Sequence[str]] = None, min_overlap: int = 0,
               dedupe_jaccard: float = 1.01) -> List[Dict[str, Any]]:
        if self.count() == 0 or k <= 0:
            return []
        raw = self._raw_search(query, min(self.count(), max(k * 4, 10)), tiers)
        return postprocess_hits(query, raw, k, max_distance, min_overlap, dedupe_jaccard)

    def persist(self) -> None:
        """Flush to disk (no-op for backends that persist on write)."""


class ChromaVectorStore(VectorStore):
    backend = "chroma"

    def __init__(self, path, embedding_fn: Embedder, collection: str = "graph_rag_passages",
                 reset: bool = False):
        import chromadb
        self.embed = embedding_fn
        self.client = chromadb.PersistentClient(path=str(path))
        if reset:
            try:
                self.client.delete_collection(collection)
            except Exception:
                pass
        self.col = self.client.get_or_create_collection(
            name=collection, embedding_function=None, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.col.count()

    def add_chunks(self, chunks: List[Chunk], source: Optional[str] = None) -> None:
        if not chunks:
            return
        for c in chunks:
            if source:
                c.metadata.setdefault("source", source)
        self.col.upsert(ids=[_chunk_id(c) for c in chunks], documents=[c.text for c in chunks],
                        metadatas=[_meta(c) for c in chunks], embeddings=self.embed([c.text for c in chunks]))

    def _raw_search(self, query, n, tiers):
        where = {"tier": {"$in": list(tiers)}} if tiers else None
        res = self.col.query(query_embeddings=self.embed([query]), n_results=n, where=where)
        return [_hit(d, m, dist) for d, m, dist in
                zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


class InMemoryVectorStore(VectorStore):
    backend = "memory"

    def __init__(self, path, embedding_fn: Embedder, reset: bool = False):
        self.embed = embedding_fn
        self.path = Path(path)
        self.items: Dict[str, Dict[str, Any]] = {}
        if not reset and self.path.is_file():
            self.items = json.loads(self.path.read_text(encoding="utf-8"))

    def count(self) -> int:
        return len(self.items)

    def add_chunks(self, chunks: List[Chunk], source: Optional[str] = None) -> None:
        if not chunks:
            return
        for c in chunks:
            if source:
                c.metadata.setdefault("source", source)
        for c, v in zip(chunks, self.embed([c.text for c in chunks])):
            self.items[_chunk_id(c)] = {"text": c.text, "meta": _meta(c), "vec": list(map(float, v))}

    def _raw_search(self, query, n, tiers):
        q = self.embed([query])[0]
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        scored = []
        for item in self.items.values():
            if tiers and item["meta"].get("tier") not in tiers:
                continue
            v = item["vec"]
            vn = math.sqrt(sum(x * x for x in v)) or 1.0
            cos = sum(a * b for a, b in zip(q, v)) / (qn * vn)
            scored.append(_hit(item["text"], item["meta"], 1.0 - cos))
        return sorted(scored, key=lambda h: h["distance"])[:n]

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items), encoding="utf-8")


def chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401
        return True
    except Exception:
        return False


def make_vector_store(backend: str, index_dir: Path, embedder: Embedder, reset: bool = False) -> VectorStore:
    """`auto` uses ChromaDB when importable, otherwise the in-memory JSON store."""
    if backend == "chroma" or (backend == "auto" and chroma_available()):
        return ChromaVectorStore(index_dir / "chroma", embedder, reset=reset)
    if backend == "auto":
        logger.info("chromadb not installed; using the in-memory vector store")
    return InMemoryVectorStore(index_dir / "vectors.json", embedder, reset=reset)


def resolved_backend(backend: str) -> str:
    return "chroma" if backend == "chroma" or (backend == "auto" and chroma_available()) else "memory"
