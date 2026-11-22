"""Vector indexing and semantic retrieval (both backends)."""
import pytest

from graph_rag.embeddings import HashEmbedder, embedder_name
from graph_rag.models import Chunk
from graph_rag.vector_store import ChromaVectorStore, InMemoryVectorStore, postprocess_hits

from conftest import HAS_CHROMA

DOCS = [
    ("incident.pdf", 1, "pdf", "During the March incident the on-call team rolled back deploy 4412 after the connection pool was exhausted."),
    ("incident.pdf", 2, "pdf", "During the March incident the on-call team rolled back deploy 4412 after the connection pool was exhausted!"),
    ("capacity.pdf", 1, "pdf", "The capacity review recommended doubling read replicas before the holiday season."),
    ("runbook.md", 0, "wiki", "To roll back a bad deploy, freeze deploys and run the rollback job."),
]


def _chunks():
    out = []
    for i, (src, page, tier, text) in enumerate(DOCS):
        out.append(Chunk(i, page, "s" if tier == "wiki" else None, "text" if tier == "pdf" else "wiki", text,
                         {"tier": tier, "source": src, "title": src.split(".")[0].title(), "page": page}))
    return out


def _stores(tmp_path):
    stores = [InMemoryVectorStore(tmp_path / "v.json", HashEmbedder(), reset=True)]
    if HAS_CHROMA:
        stores.append(ChromaVectorStore(tmp_path / "chroma", HashEmbedder(), reset=True))
    return stores


@pytest.mark.parametrize("which", [0, 1])
def test_search_returns_metadata_dedupes_and_filters(tmp_path, which):
    stores = _stores(tmp_path)
    if which >= len(stores):
        pytest.skip("chromadb not installed")
    vs = stores[which]
    vs.add_chunks(_chunks())
    vs.persist()
    assert vs.count() == 4
    hits = vs.search("why was deploy 4412 rolled back in the March incident", k=3, max_distance=0.9,
                     dedupe_jaccard=0.8)
    assert hits[0]["source"] == "incident.pdf" and hits[0]["page"] in (1, 2) and hits[0]["tier"] == "pdf"
    assert sum(h["source"] == "incident.pdf" for h in hits) == 1            # near-duplicate page dropped
    assert {"text", "title", "chunk_id", "distance", "score", "path"} <= set(hits[0])
    assert all(h["tier"] == "wiki" for h in vs.search("roll back deploy", k=3, tiers=["wiki"]))
    assert vs.search("capital of Mars", k=3, max_distance=0.9) == []        # nothing similar enough


def test_memory_store_persists(tmp_path):
    vs = InMemoryVectorStore(tmp_path / "v.json", HashEmbedder(), reset=True)
    vs.add_chunks(_chunks())
    vs.persist()
    assert InMemoryVectorStore(tmp_path / "v.json", HashEmbedder()).count() == 4
    assert InMemoryVectorStore(tmp_path / "v.json", HashEmbedder(), reset=True).count() == 0


def test_lexical_overlap_guard():
    hits = [{"text": "totally unrelated words", "distance": 0.1}, {"text": "connection pool", "distance": 0.2}]
    assert [h["text"] for h in postprocess_hits("connection pool size", hits, 5, None, min_overlap=1)] == [
        "connection pool"]


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(64)
    a, b = e(["rolled back deploy"]), e(["rolled back deploy"])
    assert a == b and abs(sum(x * x for x in a[0]) - 1) < 1e-9 and embedder_name(e) == "hash-bow-64"
