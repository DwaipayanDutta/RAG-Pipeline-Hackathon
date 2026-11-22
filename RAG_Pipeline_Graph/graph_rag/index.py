"""
Offline phase: ingest all three tiers into the quad store + vector store, with a
fingerprint-based cache (like IndexManager in rag_pipeline.py): the index is rebuilt only when a
source file or an index-affecting setting changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import embedder_name, make_embedder
from .graph_store import QuadStore
from .models import Chunk
from .sources.database import DatabaseSource, ingest_database, open_database, table_specs
from .sources.pdf import load_pdf, pdf_facts
from .sources.wiki import ingest_wiki_dir, load_wiki_dir, wiki_passages
from .text import compute_sha256
from .vector_store import VectorStore, make_vector_store, resolved_backend

logger = logging.getLogger("graph_rag.index")

#: Settings that change what is stored in the index.
FINGERPRINT_KEYS = ("chunk_size", "chunk_overlap", "entity_extraction_model", "use_llm_wiki_extraction",
                    "db_tables", "db_table", "db_key_column", "extract_pdf_facts", "index_wiki_passages",
                    "known_entities")


@dataclass
class GraphIndex:
    store: QuadStore
    vstore: VectorStore
    db: Optional[DatabaseSource] = None
    db_specs: List[Dict[str, Any]] = field(default_factory=list)
    from_cache: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)


class GraphIndexManager:
    SCHEMA_VERSION = "2.0"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dir = Path(config["index_dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path = self.dir / "chroma"

    def source_files(self) -> List[Path]:
        c = self.config
        files: List[Path] = []
        db = c.get("db_path") or ""
        db = db[len("sqlite:///"):] if db.startswith("sqlite:///") else db
        if db and Path(db).is_file():
            files.append(Path(db))
        if c.get("wiki_dir") and Path(c["wiki_dir"]).is_dir():
            files += sorted(Path(c["wiki_dir"]).glob("*.md"))
        if c.get("pdf_dir") and Path(c["pdf_dir"]).is_dir():
            files += sorted(Path(c["pdf_dir"]).glob("*.pdf"))
        return files

    def fingerprint(self, embedder_id: str) -> str:
        h = hashlib.sha256()
        for f in self.source_files():
            h.update(f.name.encode())
            h.update(compute_sha256(str(f)).encode())
        h.update(json.dumps({k: self.config.get(k) for k in FINGERPRINT_KEYS}, sort_keys=True, default=str).encode())
        h.update(embedder_id.encode())
        h.update(resolved_backend(self.config.get("vector_backend", "auto")).encode())
        h.update(self.SCHEMA_VERSION.encode())
        return h.hexdigest()

    def _vectors_present(self) -> bool:
        return self.chroma_path.exists() or (self.dir / "vectors.json").exists()

    def load(self, fingerprint: str) -> Optional[QuadStore]:
        meta_path, quad_path = self.dir / "meta.json", self.dir / "quads.json"
        if not (meta_path.exists() and quad_path.exists() and self._vectors_present()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            logger.info("Graph index is stale (sources or settings changed); rebuilding.")
            return None
        return QuadStore.load(quad_path)

    def save(self, store: QuadStore, fingerprint: str, stats: Dict[str, Any]) -> None:
        store.save(self.dir / "quads.json")
        (self.dir / "meta.json").write_text(json.dumps(
            {"fingerprint": fingerprint, "schema_version": self.SCHEMA_VERSION, "num_quads": len(store),
             "stats": stats}, indent=2, default=str), encoding="utf-8")
        logger.info("Saved graph index to %s", self.dir)

    def read_meta(self) -> Dict[str, Any]:
        p = self.dir / "meta.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _open_db(config: Dict[str, Any]) -> Optional[DatabaseSource]:
    if not config.get("db_path"):
        return None
    try:
        return open_database(config["db_path"])
    except FileNotFoundError:
        logger.warning("Priority-1 database not found at %s; continuing without it", config["db_path"])
        return None


def build_graph_index(config: Dict[str, Any], rebuild: bool = False, embedding_fn=None,
                      wiki_llm=None) -> GraphIndex:
    """Ingest the database, wiki and PDFs (or load the cached index if nothing changed)."""
    mgr = GraphIndexManager(config)
    emb = embedding_fn if embedding_fn is not None else make_embedder(config)
    fp = mgr.fingerprint(embedder_name(emb))
    db = _open_db(config)
    specs = table_specs(config)
    backend = config.get("vector_backend", "auto")

    if not rebuild:
        store = mgr.load(fp)
        if store is not None:
            logger.info("Loaded cached graph index (%d quads)", len(store))
            return GraphIndex(store, make_vector_store(backend, mgr.dir, emb, reset=False), db, specs,
                              True, mgr.read_meta().get("stats", {}))

    store = QuadStore()
    stats: Dict[str, Any] = {"embedder": embedder_name(emb), "vector_backend": resolved_backend(backend)}
    if db is not None:
        stats["db_quads"] = ingest_database(config["db_path"], store, tables=specs)

    passages: List[Chunk] = []
    wiki_dir = config.get("wiki_dir")
    if wiki_dir and Path(wiki_dir).is_dir():
        pages = load_wiki_dir(wiki_dir)
        getter = None
        if config.get("use_llm_wiki_extraction"):
            from .generation import get_llm
            getter = (lambda: wiki_llm) if wiki_llm is not None else \
                     (lambda: get_llm(config["entity_extraction_model"], config))
        stats["wiki_quads"] = ingest_wiki_dir(wiki_dir, store, getter, config.get("known_entities") or [], pages)
        stats["wiki_pages"] = len(pages)
        if config.get("index_wiki_passages", True):
            for page in pages:
                for i, c in enumerate(wiki_passages(page)):
                    c.chunk_id = i
                    passages.append(c)
    elif wiki_dir:
        logger.warning("Wiki directory not found: %s", wiki_dir)

    pdf_dir = config.get("pdf_dir")
    pdf_chunks = 0
    if pdf_dir and Path(pdf_dir).is_dir():
        known = store.node_names() | set(config.get("known_entities") or [])
        pdf_quads = 0
        for f in sorted(Path(pdf_dir).glob("*.pdf")):
            try:
                chunks, _, doc_date = load_pdf(f, int(config["chunk_size"]), int(config["chunk_overlap"]))
            except Exception as e:
                logger.error("Skipping unreadable PDF %s: %s", f.name, e)
                continue
            passages += chunks
            pdf_chunks += len(chunks)
            if config.get("extract_pdf_facts", True):
                pdf_quads += store.add_many(pdf_facts(chunks, f.name, known, doc_date))
        stats.update({"pdf_chunks": pdf_chunks, "pdf_quads": pdf_quads})
    elif pdf_dir:
        logger.warning("PDF directory not found: %s", pdf_dir)

    vstore = make_vector_store(backend, mgr.dir, emb, reset=True)
    vstore.add_chunks(passages)
    vstore.persist()
    stats.update({"passages_indexed": len(passages), "graph": store.stats()})
    mgr.save(store, fp, stats)
    return GraphIndex(store, vstore, db, specs, False, stats)


def build_index(config: Dict[str, Any], rebuild: bool = False, embedding_fn=None,
                wiki_llm=None) -> Tuple[QuadStore, VectorStore]:
    """Backward-compatible wrapper returning (quad store, vector store)."""
    idx = build_graph_index(config, rebuild, embedding_fn, wiki_llm)
    return idx.store, idx.vstore
