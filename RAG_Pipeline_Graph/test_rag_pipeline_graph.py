"""
Tests for rag_pipeline_graph.py.   Run:  pytest test_rag_pipeline_graph.py -v

Offline by design: the generator is a scripted stand-in and ChromaDB uses a hashing embedder, so no
model weights are downloaded. That means these tests verify retrieval, prompt construction,
validation and wiring, NOT how well FLAN-T5 follows the priority rules. The real-model conflict test
(test_real_flan_t5_prefers_cmdb) is skipped unless RUN_LLM_TESTS=1 and the weights are available.
"""
import hashlib
import json
import os
import re

import numpy as np
import pytest

import rag_pipeline_graph as g
from rag_pipeline_graph import Quad, QuadStore

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("pdfplumber")
pytest.importorskip("reportlab")


# ------------------------------------------------------------------ test doubles
class HashEmbedding(chromadb.EmbeddingFunction):
    """Deterministic bag-of-words embedder (no downloads)."""
    DIM = 256

    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "hash_embedding_test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()

    def __call__(self, input):
        out = []
        for text in input:
            v = np.zeros(self.DIM, dtype=np.float32)
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok in g.STOP:
                    continue
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


class ScriptedLLM:
    """Returns a fixed answer and records the prompts it was given."""
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture()
def env(tmp_path):
    cfg = g.load_config()
    cfg.update(g.seed_demo_data(str(tmp_path / "data")))
    cfg.update({"index_dir": str(tmp_path / "idx"), "use_llm_wiki_extraction": False,
                "vector_max_distance": 0.85, "graph_hops": 1})
    store, vstore = g.build_index(cfg, embedding_fn=HashEmbedding())
    return cfg, store, vstore


def ask(env, question, reply):
    cfg, store, vstore = env
    llm = ScriptedLLM(reply)
    return g.answer_query(question, store, vstore, cfg, llm), llm


# ------------------------------------------------------------------ unit: QuadStore
def test_quadstore_add_dedupe_query_context_and_roundtrip(tmp_path):
    s = QuadStore()
    q = Quad("payments-service", "status", "degraded", "cmdb", "db")
    assert s.add(q) and not s.add(q) and len(s) == 1
    s.add(Quad("payments-service", "member_of", "Platform", "wiki", "p.md"))
    assert s.query(["Payments Service"]) != []                         # case / hyphen insensitive
    assert [x.object for x in s.query(["payments-service"], "wiki")] == ["Platform"]
    assert s.query(["degraded"], "cmdb") == [q]                        # object match
    assert s.query(["degraded"], "wiki") == []                         # context filter
    s.save(tmp_path / "q.json")
    assert len(QuadStore.load(tmp_path / "q.json")) == 2


def test_quad_format_matches_spec():
    assert Quad("payments-service", "on_call_engineer", "Amara Chen", "cmdb").as_text() == \
        "(cmdb) payments-service on_call_engineer: Amara Chen."


def test_expand_walks_to_other_services_via_on_call_person():
    s = QuadStore()
    s.add(Quad("payments-service", "on_call_engineer", "Amara Chen", "cmdb"))
    s.add(Quad("ledger-service", "on_call_engineer", "Amara Chen", "cmdb"))
    s.add(Quad("ledger-service", "status", "healthy", "cmdb"))
    s.add(Quad("checkout-service", "on_call_engineer", "Luis Ortega", "cmdb"))
    rel = s.expand(["payments-service"], 1, ["on_call_engineer"])
    assert {q.subject for q in rel} == {"ledger-service"}              # reached via Amara Chen, not checkout
    assert s.expand(["payments-service"], 1, []) == []                 # no traversal predicates -> nothing


def test_database_ingestion_tags_cmdb(env):
    _, store, _ = env
    db = store.query(["payments-service"], "cmdb")
    assert {(q.predicate, q.object) for q in db} == {("on_call_engineer", "Amara Chen"), ("status", "degraded")}
    assert all(q.source == "cmdb.sqlite:services" for q in db)


# ------------------------------------------------------------------ unit: wiki extraction
def test_wiki_regex_extraction_formats_and_source_tag():
    text = "# Payments Service\n- Owner team: Payments Platform\n**On-call engineer:** Jordan Blake\nSee: https://x.io/a\n"
    qs = g.extract_wiki_facts(text, "payments-service.md")
    assert {(q.subject, q.predicate, q.object) for q in qs} == {
        ("payments-service", "owner_team", "Payments Platform"),
        ("payments-service", "on_call_engineer", "Jordan Blake")}
    assert all(q.context == "wiki" and q.source == "payments-service.md" for q in qs)


def test_wiki_llm_fallback_only_when_regex_finds_nothing():
    class Extractor:
        def generate(self, prompt): return "Ledger Service | Owner | Finance Platform\ngarbage line"
    prose = "The ledger is run by the finance platform team."
    qs = g.extract_wiki_facts(prose, "ledger-service.md", Extractor())
    assert qs == [Quad("ledger-service", "owner_team", "Finance Platform", "wiki", "ledger-service.md")]
    assert g.extract_wiki_facts("Owner team: X", "a.md", Extractor())[0].object == "X"   # regex wins
    assert g.extract_wiki_facts(prose, "ledger-service.md") == []                         # no LLM, no facts


# ------------------------------------------------------------------ unit: entities / chunking / prompt
def test_query_entity_extraction_known_list():
    known = ["payments-service", "checkout-service"]
    assert g.extract_query_entities("Who is on-call for payments-service?", known) == ["payments-service"]
    assert set(g.extract_query_entities("owner of Checkout Service and payments service?", known)) == set(known)
    assert g.extract_query_entities("What is the capital of Mars?", known) == []


def test_query_entity_extraction_with_spacy_finds_people():
    nlp = g.load_spacy("en_core_web_sm")
    if nlp is None:
        pytest.skip("spaCy model en_core_web_sm not installed")
    ents = g.extract_query_entities("What does Amara Chen cover besides payments-service?", ["payments-service"], nlp)
    assert "payments-service" in ents and any("Amara" in e for e in ents)


def test_chunk_text_is_the_repo_implementation():
    pages = [{"page": 1, "text": "Para one.\n\nPara two."}, {"page": 2, "text": ""}]
    chunks = g.chunk_text(pages, 400, 80)
    assert len(chunks) == 1 and chunks[0].page == 1 and "Para two." in chunks[0].text


def test_prompt_has_required_rules_headers_and_quad_format(env):
    res, llm = ask(env, "Who is on-call for payments-service?", "Amara Chen")
    p = llm.prompts[0]
    for s in ("Priority 1 (CMDB) is absolute and always current. If it conflicts with any other source, use Priority 1.",
              "Priority 2 (Wiki) is supplementary. Use it only if Priority 1 does not cover the question.",
              "Priority 3 (PDFs) is last-resort narrative context.",
              "=== PRIORITY 1: LIVE DATABASE FACTS ===", "=== PRIORITY 2: WIKI FACTS (may be stale) ===",
              "=== PRIORITY 3: PDF NARRATIVE CONTEXT ===",
              "(cmdb) payments-service on_call_engineer: Amara Chen.",
              "(wiki) payments-service on_call_engineer: Jordan Blake."):
        assert s in p
    assert p.index("PRIORITY 1: LIVE") < p.index("PRIORITY 2: WIKI") < p.index("PRIORITY 3: PDF")


def test_fit_prompt_never_drops_priority_1():
    r = g.Retrieval(entities=["x"], p1=[Quad("x", "status", "up", "cmdb")],
                    p2=[Quad("x", "note", "w" * 200, "wiki")] * 50,
                    p3=[{"text": "t" * 500, "source": "a.pdf", "page": 1, "chunk_id": 0, "distance": .1}] * 5)
    cfg = g.load_config(); cfg["max_context_tokens"] = 100
    prompt = g.fit_prompt("Q?", r, cfg)
    assert "(cmdb) x status: up." in prompt and prompt.rstrip().endswith("Answer:") and len(prompt) <= 300 * 4 + 400


# ------------------------------------------------------------------ integration: stale wiki
def test_stale_wiki_conflict_is_detected_and_cmdb_value_returned(env):
    res, _ = ask(env, "Who is on-call for payments-service?", "Amara Chen")
    assert res.grounded and "Amara Chen" in res.answer and "Jordan Blake" not in res.answer
    assert res.priority_tier == "cmdb" and not res.vector_fallback_used
    assert res.conflicts == [{"authoritative": "(cmdb) payments-service on_call_engineer: Amara Chen.",
                              "overridden": "(wiki) payments-service on_call_engineer: Jordan Blake."}]


def test_answer_siding_with_stale_wiki_is_rejected(env):
    res, _ = ask(env, "Who is on-call for payments-service?", "Jordan Blake")
    assert not res.grounded and res.answer == g.REFUSAL
    assert "Contradicts Priority 1" in res.refusal_reason


def test_hallucinated_entity_is_rejected(env):
    res, _ = ask(env, "Who is on-call for payments-service?", "Maria Lopez")
    assert not res.grounded and "Maria Lopez" in res.refusal_reason


def test_wiki_used_when_cmdb_does_not_cover_the_question(env):
    res, _ = ask(env, "Which team owns payments-service?", "Payments Platform")
    assert res.grounded and res.answer == "Payments Platform"


def test_graph_expansion_surfaces_related_service_in_prompt(env):
    _, llm = ask(env, "Who is on-call for payments-service?", "Amara Chen")
    assert "(cmdb) ledger-service on_call_engineer: Amara Chen." in llm.prompts[0]


# ------------------------------------------------------------------ fallback: PDF narrative
def test_vector_fallback_used_for_pdf_only_question(env):
    res, llm = ask(env, "Why was the deploy rolled back during the March incident?",
                   "a misconfigured connection pool exhausted database connections")
    assert res.vector_fallback_used and res.priority_tier == "pdf" and res.query_type == "NARRATIVE"
    assert res.grounded and res.retrieved_chunks and res.retrieved_chunks[0]["source"] == "incident_reports.pdf"
    assert "rolled back deploy 4412" in re.sub(r"\s+", " ", llm.prompts[0])      # PDF lines wrap
    assert "PRIORITY 1: LIVE DATABASE FACTS ===\n(none)" in llm.prompts[0]


def test_no_evidence_refuses_without_calling_the_model(env):
    res, llm = ask(env, "What is the capital of Mars?", "Olympus")
    assert res.answer == g.REFUSAL and llm.prompts == [] and res.priority_tier == "none"


def test_model_abstention_is_passed_through(env):
    res, _ = ask(env, "Who is on-call for payments-service?", "I don't know from the document.")
    assert res.answer == g.REFUSAL and not res.grounded


# ------------------------------------------------------------------ index cache + CLI output shape
def test_index_cache_reused_and_invalidated_on_change(tmp_path, env):
    cfg, store, _ = env
    calls = []
    orig = g.ingest_database
    g.ingest_database = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        g.build_index(cfg, embedding_fn=HashEmbedding())               # unchanged -> cached
        assert calls == []
        with open(os.path.join(cfg["wiki_dir"], "payments-service.md"), "a") as f:
            f.write("- Pager: 1234\n")
        g.build_index(cfg, embedding_fn=HashEmbedding())               # changed -> rebuilt
        assert calls == [1]
        g.build_index(cfg, rebuild=True, embedding_fn=HashEmbedding())  # forced
        assert calls == [1, 1]
    finally:
        g.ingest_database = orig


def test_query_result_is_json_serialisable_with_repo_fields(env):
    res, _ = ask(env, "Who is on-call for payments-service?", "Amara Chen")
    d = json.loads(json.dumps(g.asdict(res)))
    for k in ("query", "query_type", "answer", "grounded", "grounding_status", "confidence",
              "citations", "retrieved_chunks", "validation", "refusal_reason"):
        assert k in d


# ------------------------------------------------------------------ real model (opt-in)
@pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1", reason="set RUN_LLM_TESTS=1 to download/run FLAN-T5")
def test_real_flan_t5_prefers_cmdb(env):
    cfg, store, vstore = env
    llm = g.get_llm(cfg["generation_model"], cfg)
    res = g.answer_query("Who is on-call for payments-service?", store, vstore, cfg, llm)
    assert "Jordan Blake" not in res.answer
    assert res.answer == g.REFUSAL or "Amara Chen" in res.answer   # FLAN-T5-base may abstain; it must not pick the wiki
