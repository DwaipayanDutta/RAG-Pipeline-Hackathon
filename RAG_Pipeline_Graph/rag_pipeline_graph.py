#!/usr/bin/env python3
"""
Graph-RAG Pipeline: priority-federated retrieval with prompt-enforced conflict resolution.

A third, standalone variant next to rag_pipeline.py and rag_pipeline_jev.py. It follows the
same conventions (single file, DEFAULT_CONFIG + load_config, Chunk/QueryResult dataclasses,
"I don't know from the document." refusal, cached index, JSON output) but targets the
enterprise scenario from the Graph-RAG tutorial instead of the Titan annual report:

  Priority 1  live database (SQLite `services` table)   -> SPOC quads, context="cmdb"
  Priority 2  Markdown wiki pages (stale but useful)    -> SPOC quads, context="wiki"
  Priority 3  PDFs (narrative context, vector fallback) -> ChromaDB chunks

The graph lives in-process (QuadStore, no server). Conflicts between tiers are resolved by
explicit priority rules in the prompt, then re-checked after generation.

Heavy dependencies (transformers, torch, sentence-transformers, chromadb, spacy, pdfplumber)
are imported lazily so the module can be imported and unit-tested without them.
"""
import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Iterable, Set

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rag_pipeline_graph")

REFUSAL = "I don't know from the document."
CTX_CMDB, CTX_WIKI, CTX_PDF = "cmdb", "wiki", "pdf"


# -------------------------------------------------------------------------
# Data Structures
# -------------------------------------------------------------------------
@dataclass
class Chunk:
    """A single chunk with metadata (same shape as rag_pipeline.py)."""
    chunk_id: int
    page: int
    section: Optional[str]
    source_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Quad:
    """Subject-Predicate-Object-Context fact."""
    subject: str
    predicate: str
    object: str
    context: str
    source: str = ""

    def as_text(self) -> str:
        return f"({self.context}) {self.subject} {self.predicate}: {self.object}."


@dataclass
class QueryResult:
    """Same fields as rag_pipeline.QueryResult, plus graph-specific audit fields."""
    query: str
    query_type: str
    answer: str
    grounded: bool
    grounding_status: str
    confidence: float
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    validation: Dict[str, bool]
    refusal_reason: Optional[str] = None
    # graph-specific
    priority_tier: str = "none"          # highest tier that had retrieved evidence
    vector_fallback_used: bool = False   # True when no graph fact matched but PDF chunks did
    entities: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, str]] = field(default_factory=list)
    prompt: Optional[str] = None


@dataclass
class Retrieval:
    entities: List[str]
    p1: List[Quad] = field(default_factory=list)
    p2: List[Quad] = field(default_factory=list)
    related: List[Quad] = field(default_factory=list)
    p3: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Tuple[Quad, Quad]] = field(default_factory=list)

    @property
    def vector_fallback_used(self) -> bool:
        return not self.p1 and not self.p2 and bool(self.p3)


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # chunking (same defaults as rag_pipeline.py)
    "chunk_size": 400,
    "chunk_overlap": 80,
    # models
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "generation_model": "google/flan-t5-base",
    "entity_extraction_model": "google/flan-t5-base",
    "spacy_model": "en_core_web_sm",
    "max_context_tokens": 512,
    "max_answer_tokens": 150,
    "index_dir": "graph_index_cache",
    # sources
    "db_path": "demo_graph_data/cmdb.sqlite",
    "db_table": "services",
    "db_key_column": "service_name",
    "wiki_dir": "demo_graph_data/wiki",
    "pdf_dir": "demo_graph_data/pdfs",
    "use_llm_wiki_extraction": True,     # only used for wiki pages the regex cannot parse
    # retrieval
    "known_entities": ["payments-service", "checkout-service"],
    "graph_top_k": 10,                   # max quads per tier
    "graph_hops": 1,                     # graph-aware expansion depth (0 disables)
    "traversal_predicates": ["on_call_engineer", "owner_team", "depends_on", "member_of"],
    "vector_fallback_top_k": 2,
    "vector_max_distance": 0.65,         # cosine distance cutoff; NOT tuned, adjust on your data
    # priority
    "priority_weights": {"cmdb": 1.0, "wiki": 0.6, "pdf": 0.3},
    # validation
    "grounding_min_coverage": 0.8,
    "include_prompt": False,
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config.update(json.load(f))
    return config


# -------------------------------------------------------------------------
# Utilities (normalize_text / compute_sha256 as in rag_pipeline.py)
# -------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Lowercase, remove extra spaces, keep numbers and punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def compute_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _flat(s: str) -> str:
    """Case/hyphen/underscore/punctuation-insensitive form used for matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.strip().lower()).strip("-")


STOP = set("""the a an is are was were of for to in on at by and or what who why how which when does did do
this that it with from as be been i me my your their his her we you they there here""".split())


# -------------------------------------------------------------------------
# QuadStore (in-process knowledge graph)
# -------------------------------------------------------------------------
class QuadStore:
    """Lightweight in-memory quad store with JSON persistence and BFS expansion."""

    def __init__(self) -> None:
        self._quads: List[Quad] = []
        self._seen: Set[Quad] = set()

    def __len__(self) -> int:
        return len(self._quads)

    def add(self, quad: Quad) -> bool:
        if quad in self._seen:
            return False
        self._seen.add(quad)
        self._quads.append(quad)
        return True

    def add_many(self, quads: Iterable[Quad]) -> int:
        return sum(1 for q in quads if self.add(q))

    def all(self, context: Optional[str] = None) -> List[Quad]:
        return [q for q in self._quads if context is None or q.context == context]

    def subjects(self) -> Set[str]:
        return {q.subject for q in self._quads}

    def query(self, entities: Iterable[str], context: Optional[str] = None) -> List[Quad]:
        """Quads where any entity is the subject OR the object."""
        wanted = {_flat(e) for e in entities}
        return [q for q in self._quads
                if (context is None or q.context == context)
                and (_flat(q.subject) in wanted or _flat(q.object) in wanted)]

    def expand(self, seeds: Iterable[str], hops: int,
               traversal_predicates: Iterable[str] = ()) -> List[Quad]:
        """
        Facts about entities reached by walking the graph away from the seeds.

        Layer 0 (facts that touch a seed directly) is NOT returned; `query` covers that.
        Each hop moves to the subjects of the previous layer's facts and, for predicates in
        `traversal_predicates`, to their objects too (e.g. the on-call person), then returns
        the facts that touch those new nodes. Plain values like "degraded" are not walked.
        """
        trav = set(traversal_predicates)
        visited = {_flat(s) for s in seeds}
        layer = self.query(visited)
        out: List[Quad] = []
        for _ in range(max(0, hops)):
            nodes: Set[str] = set()
            for q in layer:
                nodes.add(_flat(q.subject))
                if q.predicate in trav:
                    nodes.add(_flat(q.object))
            nodes -= visited
            if not nodes:
                break
            visited |= nodes
            layer = [q for q in self.query(nodes) if q not in out]
            out.extend(layer)
        direct = set(self.query([s for s in seeds]))
        return [q for q in out if q not in direct]

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(q) for q in self._quads], f, indent=1)

    @classmethod
    def load(cls, path) -> "QuadStore":
        store = cls()
        with open(path, "r") as f:
            store.add_many(Quad(**d) for d in json.load(f))
        return store


# -------------------------------------------------------------------------
# LLM (FLAN-T5, loaded once and reused)
# -------------------------------------------------------------------------
class FlanT5Generator:
    """Same generation call as rag_pipeline.generate_answer, but the model is loaded once."""

    def __init__(self, model_name: str, max_input_tokens: int = 712, max_new_tokens: int = 150):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self._tok = self._model = self._device = None

    def _load(self) -> None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self._device)

    def generate(self, prompt: str) -> str:
        if self._model is None:
            self._load()
        inputs = self._tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self._tok.decode(out[0], skip_special_tokens=True).strip()


_LLM_CACHE: Dict[str, FlanT5Generator] = {}


def get_llm(model_name: str, config: Dict[str, Any]) -> FlanT5Generator:
    if model_name not in _LLM_CACHE:
        _LLM_CACHE[model_name] = FlanT5Generator(
            model_name, config["max_context_tokens"] + 200, config["max_answer_tokens"])
    return _LLM_CACHE[model_name]


# -------------------------------------------------------------------------
# Priority 1: live database -> quads (context="cmdb")
# -------------------------------------------------------------------------
def ingest_database(db_path: str, store: QuadStore, table: str = "services",
                    key_column: str = "service_name") -> int:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"Unsafe table name: {table}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    finally:
        conn.close()
    added = 0
    for row in rows:
        data = dict(row)
        subject = str(data.pop(key_column))
        for col, val in data.items():
            if val is not None:
                added += store.add(Quad(subject, col, str(val), CTX_CMDB, f"{Path(db_path).name}:{table}"))
    logger.info(f"cmdb: {added} quads from {len(rows)} rows")
    return added


# -------------------------------------------------------------------------
# Priority 2: wiki pages -> quads (context="wiki")
# -------------------------------------------------------------------------
KEY_ALIASES = {
    "owner": "owner_team", "owning_team": "owner_team", "team": "owner_team",
    "oncall": "on_call_engineer", "on_call": "on_call_engineer", "oncall_engineer": "on_call_engineer",
}
# "Key: value", "- Key: value", "**Key:** value"
WIKI_KV_RE = re.compile(r"^\s*(?:[-*]\s+)?\**([A-Za-z][A-Za-z \-/]{1,40}?)\s*:\s*\**\s*(.+?)\s*\**\s*$")


def normalize_key(key: str) -> str:
    k = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    return KEY_ALIASES.get(k, k)


def extract_wiki_facts_llm(text: str, filename: str, llm) -> List[Quad]:
    """Fallback for pages with no key/value lines: ask the extraction model for triples."""
    subject_default = _slug(Path(filename).stem)
    prompt = ("Extract facts from the text. Output one fact per line as: subject | predicate | object\n"
              f"Text: {text[:1500]}\nFacts:")
    quads = []
    for line in llm.generate(prompt).splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            quads.append(Quad(_slug(parts[0]) or subject_default, normalize_key(parts[1]),
                              parts[2], CTX_WIKI, filename))
    return quads


def extract_wiki_facts(text: str, filename: str, llm=None) -> List[Quad]:
    """Regex key/value extraction. Subject = page filename stem. LLM only if regex finds nothing."""
    subject = _slug(Path(filename).stem)
    quads = []
    for line in text.splitlines():
        m = WIKI_KV_RE.match(line)
        if not m or re.match(r"[a-z][a-z0-9+.-]*://", m.group(2), re.I):   # skip URL lines ("See: https://...")
            continue
        quads.append(Quad(subject, normalize_key(m.group(1)), m.group(2).strip(), CTX_WIKI, filename))
    if not quads and llm is not None:
        quads = extract_wiki_facts_llm(text, filename, llm)
    return quads


def ingest_wiki_dir(wiki_dir: str, store: QuadStore,
                    llm_getter: Optional[Callable[[], Any]] = None) -> int:
    added, llm, tried = 0, None, False
    for f in sorted(Path(wiki_dir).glob("*.md")):
        text = f.read_text(encoding="utf-8")
        facts = extract_wiki_facts(text, f.name)
        if not facts and llm_getter is not None:
            if not tried:                               # load the extraction model lazily, once
                tried = True
                try:
                    llm = llm_getter()
                except Exception as e:
                    logger.warning(f"LLM wiki extraction unavailable: {e}")
            if llm is not None:
                try:
                    facts = extract_wiki_facts_llm(text, f.name, llm)
                except Exception as e:
                    logger.warning(f"LLM extraction failed for {f.name}: {e}")
        added += store.add_many(facts)
    logger.info(f"wiki: {added} quads")
    return added


# -------------------------------------------------------------------------
# Priority 3: PDFs -> ChromaDB (vector fallback)
# -------------------------------------------------------------------------
def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    """Page text extraction with pdfplumber (same output shape as rag_pipeline.extract_pdf_content)."""
    import pdfplumber
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            pages_text.append({"page": page_num, "text": page.extract_text() or ""})
    return pages_text


def chunk_text(pages_text: List[Dict], chunk_size: int = 400, overlap: int = 80) -> List[Chunk]:
    """Split text into overlapping chunks preserving paragraphs (copied from rag_pipeline.py)."""
    chunks = []
    chunk_id = 0
    for page_info in pages_text:
        page = page_info["page"]
        text = page_info["text"]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n"
            else:
                if current_chunk:
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        page=page,
                        section=None,
                        source_type="text",
                        text=current_chunk.strip(),
                        metadata={"page": page}
                    ))
                    chunk_id += 1
                    overlap_text = current_chunk[-overlap:] if overlap else ""
                    current_chunk = overlap_text + para + "\n"
                else:
                    current_chunk = para + "\n"
        if current_chunk:
            chunks.append(Chunk(
                chunk_id=chunk_id,
                page=page,
                section=None,
                source_type="text",
                text=current_chunk.strip(),
                metadata={"page": page}
            ))
            chunk_id += 1
    return chunks


def make_embedding_function(config: Dict[str, Any]):
    """ChromaDB's SentenceTransformer wrapper around the repo's embedding model."""
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config["embedding_model"])


class ChromaVectorStore:
    def __init__(self, path, embedding_fn, collection: str = "pdf_chunks", reset: bool = False):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(path))
        if reset:
            try:
                self.client.delete_collection(collection)
            except Exception:
                pass
        self.col = self.client.get_or_create_collection(
            name=collection, embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.col.count()

    def add_chunks(self, chunks: List[Chunk], source: str) -> None:
        if not chunks:
            return
        self.col.upsert(
            ids=[f"{source}:{c.page}:{c.chunk_id}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": source, "page": c.page, "chunk_id": c.chunk_id} for c in chunks],
        )

    def search(self, query: str, k: int = 2, max_distance: Optional[float] = None) -> List[Dict[str, Any]]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=min(k, n))
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if max_distance is None or dist <= max_distance:
                hits.append({"text": doc, "source": meta.get("source", ""), "page": meta.get("page", 0),
                             "chunk_id": meta.get("chunk_id", 0), "distance": float(dist)})
        return hits


def ingest_pdfs(pdf_dir: str, vstore: ChromaVectorStore, chunk_size: int, overlap: int) -> int:
    total = 0
    for f in sorted(Path(pdf_dir).glob("*.pdf")):
        chunks = chunk_text(extract_pdf_pages(str(f)), chunk_size, overlap)
        vstore.add_chunks(chunks, f.name)
        total += len(chunks)
    logger.info(f"pdf: {total} chunks in ChromaDB")
    return total


# -------------------------------------------------------------------------
# Index management (fingerprint-based cache, like IndexManager in rag_pipeline.py)
# -------------------------------------------------------------------------
class GraphIndexManager:
    SCHEMA_VERSION = "1.0"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dir = Path(config["index_dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path = self.dir / "chroma"

    def _source_files(self) -> List[Path]:
        c = self.config
        files = []
        if c.get("db_path") and Path(c["db_path"]).is_file():
            files.append(Path(c["db_path"]))
        if c.get("wiki_dir") and Path(c["wiki_dir"]).is_dir():
            files += sorted(Path(c["wiki_dir"]).glob("*.md"))
        if c.get("pdf_dir") and Path(c["pdf_dir"]).is_dir():
            files += sorted(Path(c["pdf_dir"]).glob("*.pdf"))
        return files

    def fingerprint(self) -> str:
        c = self.config
        h = hashlib.sha256()
        for f in self._source_files():
            h.update(str(f).encode())
            h.update(compute_sha256(str(f)).encode())
        h.update(json.dumps({k: c.get(k) for k in (
            "chunk_size", "chunk_overlap", "embedding_model", "entity_extraction_model",
            "use_llm_wiki_extraction", "db_table", "db_key_column")}, sort_keys=True).encode())
        h.update(self.SCHEMA_VERSION.encode())
        return h.hexdigest()

    def load(self, fingerprint: str) -> Optional[QuadStore]:
        meta_path, quad_path = self.dir / "meta.json", self.dir / "quads.json"
        if not (meta_path.exists() and quad_path.exists() and self.chroma_path.exists()):
            return None
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            logger.warning("Graph index mismatch or outdated. Rebuilding.")
            return None
        return QuadStore.load(quad_path)

    def save(self, store: QuadStore, fingerprint: str) -> None:
        store.save(self.dir / "quads.json")
        (self.dir / "meta.json").write_text(json.dumps(
            {"fingerprint": fingerprint, "schema_version": self.SCHEMA_VERSION, "num_quads": len(store)}, indent=2))
        logger.info(f"Saved graph index to {self.dir}")


def build_index(config: Dict[str, Any], rebuild: bool = False, embedding_fn=None,
                wiki_llm=None) -> Tuple[QuadStore, ChromaVectorStore]:
    """Offline phase: ingest all three tiers (or load the cached index if sources are unchanged)."""
    mgr = GraphIndexManager(config)
    emb = embedding_fn if embedding_fn is not None else make_embedding_function(config)
    fp = mgr.fingerprint()
    if not rebuild:
        store = mgr.load(fp)
        if store is not None:
            logger.info(f"Loaded existing graph index ({len(store)} quads)")
            return store, ChromaVectorStore(mgr.chroma_path, emb, reset=False)

    store = QuadStore()
    if config.get("db_path") and Path(config["db_path"]).is_file():
        ingest_database(config["db_path"], store, config["db_table"], config["db_key_column"])
    if config.get("wiki_dir") and Path(config["wiki_dir"]).is_dir():
        getter = None
        if config.get("use_llm_wiki_extraction"):
            getter = (lambda: wiki_llm) if wiki_llm is not None else \
                     (lambda: get_llm(config["entity_extraction_model"], config))
        ingest_wiki_dir(config["wiki_dir"], store, getter)
    vstore = ChromaVectorStore(mgr.chroma_path, emb, reset=True)
    if config.get("pdf_dir") and Path(config["pdf_dir"]).is_dir():
        ingest_pdfs(config["pdf_dir"], vstore, config["chunk_size"], config["chunk_overlap"])
    mgr.save(store, fp)
    return store, vstore


# -------------------------------------------------------------------------
# Query-time retrieval
# -------------------------------------------------------------------------
_NLP = None


def load_spacy(model: str = "en_core_web_sm"):
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load(model)
        except Exception as e:
            logger.warning(f"spaCy unavailable ({e}); using known-entity matching only")
            _NLP = False
    return _NLP or None


def extract_query_entities(query: str, known_entities: Iterable[str] = (), nlp=None) -> List[str]:
    """spaCy NER plus a known-entity list (boosts recall for internal jargon)."""
    found = []
    qflat = f" {_flat(query)} "
    for ent in sorted(known_entities):
        if f" {_flat(ent)} " in qflat:
            found.append(ent)
    if nlp is not None:
        found += [e.text for e in nlp(query).ents]
    seen, out = set(), []
    for e in found:
        if _flat(e) not in seen:
            seen.add(_flat(e))
            out.append(e)
    return out


def find_conflicts(authoritative: List[Quad], lower: List[Quad]) -> List[Tuple[Quad, Quad]]:
    """(authoritative, overridden) pairs: same subject + predicate, different object."""
    return [(a, b) for a in authoritative for b in lower
            if _flat(a.subject) == _flat(b.subject) and a.predicate == b.predicate
            and _flat(a.object) != _flat(b.object)]


def retrieve(query: str, store: QuadStore, vstore: Optional[ChromaVectorStore],
             config: Dict[str, Any], nlp=None) -> Retrieval:
    top_k = int(config["graph_top_k"])
    known = set(config.get("known_entities", [])) | store.subjects()
    entities = extract_query_entities(query, known, nlp)
    r = Retrieval(entities=entities)
    if entities:
        r.p1 = store.query(entities, CTX_CMDB)[:top_k]
        r.p2 = store.query(entities, CTX_WIKI)[:top_k]
        if config.get("graph_hops", 0) > 0:
            direct = set(r.p1) | set(r.p2)
            weights = config["priority_weights"]
            rel = [q for q in store.expand(entities, int(config["graph_hops"]), config["traversal_predicates"])
                   if q not in direct and q.context in (CTX_CMDB, CTX_WIKI)]
            r.related = sorted(rel, key=lambda q: -weights.get(q.context, 0))[:top_k]
        auth = r.p1 + [q for q in r.related if q.context == CTX_CMDB]
        lower = r.p2 + [q for q in r.related if q.context == CTX_WIKI]
        r.conflicts = find_conflicts(auth, lower)
    if vstore is not None:
        r.p3 = vstore.search(query, int(config["vector_fallback_top_k"]), config.get("vector_max_distance"))
    return r


# -------------------------------------------------------------------------
# Prompt engineering: priority rules enforced in the prompt
# -------------------------------------------------------------------------
PRIORITY_RULES = f"""You are an enterprise assistant. Answer ONLY from the context below.

SOURCE PRIORITY RULES:
- Priority 1 (CMDB) is absolute and always current. If it conflicts with any other source, use Priority 1.
- Priority 2 (Wiki) is supplementary. Use it only if Priority 1 does not cover the question.
- Priority 3 (PDFs) is last-resort narrative context.

If the context does not contain enough information to answer, say exactly: "{REFUSAL}\""""


def _section(title: str, lines: List[str]) -> str:
    return f"=== {title} ===\n" + ("\n".join(lines) if lines else "(none)")


def build_system_prompt(question: str, p1: List[Quad], p2: List[Quad],
                        p3: List[Dict[str, Any]], related: List[Quad] = ()) -> str:
    parts = [PRIORITY_RULES,
             _section("PRIORITY 1: LIVE DATABASE FACTS", [q.as_text() for q in p1]),
             _section("PRIORITY 2: WIKI FACTS (may be stale)", [q.as_text() for q in p2])]
    if related:
        parts.append(_section("RELATED GRAPH FACTS", [q.as_text() for q in related]))
    parts.append(_section("PRIORITY 3: PDF NARRATIVE CONTEXT",
                          [f"[{c.get('source', 'pdf')} p.{c.get('page', '?')}] {c['text']}" for c in p3]))
    parts.append(f"=== QUESTION ===\n{question}\n\nGive a concise answer.\nAnswer:")
    return "\n\n".join(parts)


def fit_prompt(question: str, r: Retrieval, config: Dict[str, Any]) -> str:
    """
    Keep the prompt inside the model's input window (the question is last, so truncation would
    cut it first). Drops lowest-priority material first: related facts, PDF chunks, wiki facts.
    Priority 1 facts are never dropped. Budget uses the same len//4 token estimate as
    rag_pipeline.generate_answer.
    """
    budget = (config["max_context_tokens"] + 200) * 4
    p1, p2, rel, p3 = list(r.p1), list(r.p2), list(r.related), list(r.p3)
    prompt = build_system_prompt(question, p1, p2, p3, rel)
    while len(prompt) > budget and (rel or p3 or p2):
        (rel or p3 or p2).pop()
        prompt = build_system_prompt(question, p1, p2, p3, rel)
    return prompt


# -------------------------------------------------------------------------
# Grounding validation (adapted from rag_pipeline.validate_answer)
# -------------------------------------------------------------------------
_NAME_RE = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+|\d[\d,.]*")


def validate_graph_answer(answer: str, r: Retrieval,
                          min_coverage: float = 0.8) -> Tuple[bool, Dict[str, bool], str]:
    """
    Returns (grounded, validation, reason), same shape as rag_pipeline.validate_answer.
      no_priority_conflict - answer must not assert a lower-tier value that Priority 1 overrides
      entity_grounding     - every name / hyphenated id / number in the answer appears in the evidence
      token_coverage       - share of the answer's content words found in the evidence >= min_coverage
    """
    validation = {"no_priority_conflict": True, "entity_grounding": False, "token_coverage": False}
    a = _flat(answer)

    for auth, stale in r.conflicts:
        if _flat(stale.object) in a and _flat(auth.object) not in a:
            validation["no_priority_conflict"] = False
            return False, validation, (f"Contradicts Priority 1: {auth.subject} {auth.predicate} "
                                       f"is {auth.object}, not {stale.object}")

    facts = r.p1 + r.p2 + r.related
    evidence = " ".join([f"{q.subject} {q.predicate} {q.object}" for q in facts] + [c["text"] for c in r.p3])
    eflat = _flat(evidence)
    etokens = set(eflat.split())

    missing = [c for c in _NAME_RE.findall(answer) if _flat(c) and _flat(c) not in eflat]
    if missing:
        return False, validation, f"Entity not found in retrieved evidence: {missing[0]}"
    validation["entity_grounding"] = True

    content = [t for t in a.split() if t not in STOP]
    coverage = (sum(t in etokens for t in content) / len(content)) if content else 0.0
    if coverage < min_coverage:
        return False, validation, f"Answer not grounded in evidence (coverage {coverage:.2f})"
    validation["token_coverage"] = True
    return True, validation, ""


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------
def _refuse(query: str, reason: str, r: Retrieval, qtype: str, prompt: Optional[str],
            config: Dict[str, Any], validation: Optional[Dict[str, bool]] = None) -> QueryResult:
    return QueryResult(
        query=query, query_type=qtype, answer=REFUSAL, grounded=False, grounding_status="refused",
        confidence=0.0, citations=[], retrieved_chunks=_chunk_dicts(r), validation=validation or {},
        refusal_reason=reason, priority_tier=_tier(r), vector_fallback_used=r.vector_fallback_used,
        entities=r.entities, conflicts=_conflict_dicts(r),
        prompt=prompt if config.get("include_prompt") else None)


def _tier(r: Retrieval) -> str:
    return CTX_CMDB if r.p1 else CTX_WIKI if r.p2 else CTX_PDF if r.p3 else "none"


def _chunk_dicts(r: Retrieval) -> List[Dict[str, Any]]:
    return [{"chunk_id": c["chunk_id"], "page": c["page"], "source_type": CTX_PDF,
             "source": c["source"], "distance": c["distance"]} for c in r.p3]


def _conflict_dicts(r: Retrieval) -> List[Dict[str, str]]:
    return [{"authoritative": a.as_text(), "overridden": b.as_text()} for a, b in r.conflicts]


def answer_query(query: str, store: QuadStore, vstore: Optional[ChromaVectorStore],
                 config: Dict[str, Any], llm, nlp=None) -> QueryResult:
    r = retrieve(query, store, vstore, config, nlp)
    qtype = "GRAPH_FACT" if (r.p1 or r.p2) else "NARRATIVE" if r.p3 else "UNKNOWN"
    if not (r.p1 or r.p2 or r.related or r.p3):
        return _refuse(query, "No evidence retrieved from any priority tier.", r, qtype, None, config)

    prompt = fit_prompt(query, r, config)
    raw = llm.generate(prompt)
    if "don't know" in raw.lower() or "not enough information" in raw.lower() or not raw.strip():
        return _refuse(query, "Model reported insufficient evidence.", r, qtype, prompt, config)

    grounded, validation, reason = validate_graph_answer(raw, r, config["grounding_min_coverage"])
    if not grounded:
        return _refuse(query, reason, r, qtype, prompt, config, validation)

    tier = _tier(r)
    citations = [{"tier": q.context, "fact": q.as_text(), "source": q.source} for q in r.p1 + r.p2]
    citations += [{"tier": CTX_PDF, "source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]} for c in r.p3]
    return QueryResult(
        query=query, query_type=qtype, answer=raw, grounded=True, grounding_status="grounded",
        confidence=float(config["priority_weights"].get(tier, 0.0)),  # tier-based, not calibrated
        citations=citations, retrieved_chunks=_chunk_dicts(r), validation=validation,
        priority_tier=tier, vector_fallback_used=r.vector_fallback_used, entities=r.entities,
        conflicts=_conflict_dicts(r), prompt=prompt if config.get("include_prompt") else None)


def run_pipeline(query: str, config: Dict[str, Any], rebuild: bool = False) -> QueryResult:
    """Convenience wrapper mirroring rag_pipeline.run_pipeline (builds/loads the index every call)."""
    store, vstore = build_index(config, rebuild)
    llm = get_llm(config["generation_model"], config)
    return answer_query(query, store, vstore, config, llm, load_spacy(config["spacy_model"]))


# -------------------------------------------------------------------------
# Demo data: the Graph-RAG tutorial's stale-wiki conflict
# -------------------------------------------------------------------------
DEMO_PDF_PAGES = [
    "March incident post-mortem. During the March incident, the on-call team rolled back deploy 4412 "
    "because a misconfigured connection pool exhausted database connections. Latency returned to normal "
    "within ten minutes of the rollback.",
    "Checkout migration notes. The migration of checkout traffic to the new payment gateway completed in "
    "April after two weeks of shadow testing and a staged percentage rollout.",
    "Capacity review. The quarterly capacity review recommended doubling the read replicas before the "
    "holiday season and adding autoscaling alarms for queue depth.",
]


def seed_demo_data(root: str) -> Dict[str, str]:
    """
    Create demo_graph_data: CMDB says on_call_engineer=Amara Chen, the stale wiki says Jordan Blake.
    The PDF is written with reportlab (only needed for this demo; not a pipeline dependency).
    """
    r = Path(root)
    (r / "wiki").mkdir(parents=True, exist_ok=True)
    (r / "pdfs").mkdir(parents=True, exist_ok=True)
    db = r / "cmdb.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE IF EXISTS services")
    conn.execute("CREATE TABLE services (service_name TEXT, on_call_engineer TEXT, status TEXT)")
    conn.executemany("INSERT INTO services VALUES (?,?,?)", [
        ("payments-service", "Amara Chen", "degraded"),
        ("checkout-service", "Luis Ortega", "healthy"),
        ("ledger-service", "Amara Chen", "healthy"),
    ])
    conn.commit()
    conn.close()
    (r / "wiki" / "payments-service.md").write_text(
        "# Payments Service\n\n- Owner team: Payments Platform\n- On-call engineer: Jordan Blake\n- Tier: 1\n")
    (r / "wiki" / "checkout-service.md").write_text(
        "# Checkout Service\n\n**Owner team:** Web Commerce\n**On-call engineer:** Luis Ortega\n")
    try:
        import textwrap
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(r / "pdfs" / "incident_reports.pdf"))
        for page in DEMO_PDF_PAGES:
            y = 800
            for line in textwrap.wrap(page, 90):
                c.drawString(60, y, line)
                y -= 16
            c.showPage()
        c.save()
    except ImportError:
        logger.warning("reportlab not installed: demo PDF not created (pip install reportlab)")
    return {"db_path": str(db), "wiki_dir": str(r / "wiki"), "pdf_dir": str(r / "pdfs")}


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Graph-RAG Pipeline (CMDB > wiki > PDF priority fusion)")
    parser.add_argument("--query", type=str, help="Single query string")
    parser.add_argument("--queries", type=str, help="Path to JSON file with list of queries")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--config", type=str, help="Path to config JSON (e.g. config_graph.json)")
    parser.add_argument("--db", type=str, help="SQLite database path (overrides config)")
    parser.add_argument("--wiki-dir", type=str, help="Wiki markdown directory (overrides config)")
    parser.add_argument("--pdf-dir", type=str, help="PDF directory (overrides config)")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of graph + vector index")
    parser.add_argument("--show-prompt", action="store_true", help="Include the fused prompt in the output")
    parser.add_argument("--seed-demo", type=str, metavar="DIR", help="Create demo data in DIR and exit")
    args = parser.parse_args()

    if args.seed_demo:
        print(json.dumps(seed_demo_data(args.seed_demo), indent=2))
        return

    config = load_config(args.config)
    for arg, key in ((args.db, "db_path"), (args.wiki_dir, "wiki_dir"), (args.pdf_dir, "pdf_dir")):
        if arg:
            config[key] = arg
    if args.show_prompt:
        config["include_prompt"] = True

    queries = []
    if args.query:
        queries = [args.query]
    elif args.queries:
        with open(args.queries, "r") as f:
            queries = json.load(f)
    else:
        print("Enter your query (type 'exit' to quit):")
        while True:
            q = input("> ")
            if q.lower() in ("exit", "quit"):
                break
            queries.append(q)

    store, vstore = build_index(config, args.rebuild_index)      # build/load once for all queries
    llm = get_llm(config["generation_model"], config)
    nlp = load_spacy(config["spacy_model"])

    results = []
    for q in queries:
        res = answer_query(q, store, vstore, config, llm, nlp)
        results.append(asdict(res))
        logger.info(f"Query: {q} -> {res.answer[:100]}...")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
