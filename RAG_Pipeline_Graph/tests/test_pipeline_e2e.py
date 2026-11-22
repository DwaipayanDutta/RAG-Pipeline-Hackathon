"""End-to-end query flow, index caching, CLI and evaluation harness."""
import json
import os
import re

import pytest

from graph_rag import GraphRAGPipeline, REFUSAL, answer_query
from graph_rag import index as index_mod
from graph_rag.demo_data import SAMPLE_DIR
from graph_rag.embeddings import HashEmbedder
from graph_rag.evaluation import evaluate, load_queries
from graph_rag.generation import ExtractiveGenerator

from conftest import HAS_CHROMA, ScriptedLLM, make_config


def ask(env, question, reply=None):
    cfg, store, vstore = env
    llm = ScriptedLLM(reply) if reply is not None else ExtractiveGenerator()
    return answer_query(question, store, vstore, cfg, llm), llm


# ------------------------------------------------------------------ stale wiki / priority rules
def test_stale_wiki_conflict_is_detected_and_cmdb_value_returned(env):
    res, _ = ask(env, "Who is on-call for payments-service?")
    assert res.grounded and "Amara Chen" in res.answer and "Jordan Blake" not in res.answer
    assert res.priority_tier == "cmdb" and not res.vector_fallback_used and res.confidence == 1.0
    used = [c for c in res.conflicts if c["affects_answer"]]
    assert {(c["authoritative"], c["overridden"]) for c in used} >= {
        ("(cmdb) payments-service on_call_engineer: Amara Chen.", "(wiki) payments-service on_call_engineer: Jordan Blake."),
        ("(cmdb) payments-service on_call_engineer: Amara Chen.", "(pdf) payments-service on_call_engineer: Jordan Blake.")}
    assert res.citations[0]["tier"] == "cmdb" and res.citations[0]["source"] == "cmdb.sqlite:services"
    assert "Priority 1 (live database)" in res.source_explanation and res.evidence[0]["resolution"]


def test_answer_siding_with_stale_wiki_is_rejected(env):
    res, _ = ask(env, "Who is on-call for payments-service?", "Jordan Blake")
    assert not res.grounded and res.answer == REFUSAL and "Contradicts Priority 1" in res.refusal_reason


def test_hallucinated_entity_is_rejected(env):
    res, _ = ask(env, "Who is on-call for payments-service?", "Maria Lopez and Amara Chen")
    assert not res.grounded and "Maria Lopez" in res.refusal_reason


def test_wiki_used_when_cmdb_does_not_cover_the_question(env):
    res, _ = ask(env, "Who is the escalation contact for payments-service?")
    assert res.grounded and "Priya Nair" in res.answer and res.priority_tier == "wiki" and res.confidence == 0.6


def test_scripted_llm_sees_fused_prompt_and_graph_expansion(env):
    env[0]["max_context_tokens"] = 2000          # roomy window: nothing is trimmed
    res, llm = ask(env, "Who is on-call for payments-service?", "Amara Chen")
    assert res.grounded and res.answer == "Amara Chen"
    assert "(cmdb) ledger-service on_call_engineer: Amara Chen." in llm.prompts[0]


def test_pdf_conflict_with_database_is_reported(env):
    res, _ = ask(env, "Which team owns payments-service?")
    assert res.answer == "The owner team for payments-service is Payments Platform."
    assert any(c["overridden"] == "(pdf) payments-service owner_team: Core Payments." and c["affects_answer"]
               for c in res.conflicts)


# ------------------------------------------------------------------ passages / fallback / refusal
def test_vector_fallback_used_for_pdf_only_question(env):
    res, llm = ask(env, "Why was the deploy rolled back during the March incident?",
                   "a misconfigured connection pool exhausted database connections")
    assert res.vector_fallback_used and res.priority_tier == "pdf" and res.query_type == "NARRATIVE"
    assert res.grounded and any(c["source"] == "incident_reports.pdf" for c in res.retrieved_chunks)
    assert "rolled back deploy 4412" in re.sub(r"\s+", " ", llm.prompts[0])
    assert "PRIORITY 1: LIVE DATABASE FACTS ===\n(none)" in llm.prompts[0]


def test_runbook_question_answered_from_wiki_passage(env):
    res, _ = ask(env, "How do I roll back a bad deploy of payments-service?")
    assert res.grounded and "rollback job" in res.answer and res.priority_tier == "wiki"
    assert res.citations[0]["section"] == "Runbook: rolling back a bad deploy"


def test_no_evidence_refuses_without_calling_the_model(env):
    res, llm = ask(env, "What is the capital of Mars?", "Olympus")
    assert res.answer == REFUSAL and llm.prompts == [] and res.priority_tier == "none"


def test_unknown_service_is_refused(env):
    res, _ = ask(env, "Who is on-call for inventory-service?")
    assert res.answer == REFUSAL and res.refusal_reason.startswith("No evidence")


def test_model_abstention_is_passed_through(env):
    res, _ = ask(env, "Who is on-call for payments-service?", "I don't know from the document.")
    assert res.answer == REFUSAL and not res.grounded


# ------------------------------------------------------------------ index cache + API + CLI
def test_index_cache_reused_and_invalidated_on_change(cfg, index, monkeypatch):
    calls = []
    orig = index_mod.ingest_database
    monkeypatch.setattr(index_mod, "ingest_database", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    assert index_mod.build_graph_index(cfg, embedding_fn=HashEmbedder()).from_cache and calls == []
    with open(os.path.join(cfg["wiki_dir"], "payments-service.md"), "a", encoding="utf-8") as f:
        f.write("- Pager: 1234\n")
    assert not index_mod.build_graph_index(cfg, embedding_fn=HashEmbedder()).from_cache and calls == [1]
    index_mod.build_graph_index(cfg, rebuild=True, embedding_fn=HashEmbedder())
    assert calls == [1, 1]
    assert index_mod.build_graph_index(dict(cfg, chunk_size=50), embedding_fn=HashEmbedder()).from_cache is False


def test_query_result_is_json_serialisable_with_repo_fields(env):
    res, _ = ask(env, "Who is on-call for payments-service?")
    d = json.loads(json.dumps(res.to_dict()))
    for k in ("query", "query_type", "answer", "grounded", "grounding_status", "confidence", "citations",
              "retrieved_chunks", "validation", "refusal_reason", "priority_tier", "conflicts", "evidence",
              "source_explanation", "trace"):
        assert k in d


@pytest.mark.parametrize("backend", ["memory", "chroma"])
def test_python_api_and_sample_evaluation_passes(tmp_path, backend):
    if backend == "chroma" and not HAS_CHROMA:
        pytest.skip("chromadb not installed")
    pipe = GraphRAGPipeline(make_config(tmp_path, vector_backend=backend), embedding_fn=HashEmbedder()).build()
    report = evaluate(pipe, load_queries(SAMPLE_DIR / "queries_graph.json"))
    failures = [(r["query"], r["answer"], r["problems"]) for r in report["rows"] if not r["passed"]]
    assert failures == [] and report["scored"] == 15
    s = pipe.stats()
    assert s["graph"]["by_tier"].keys() == {"cmdb", "wiki", "pdf"} and s["vector_backend"] == backend


def test_load_queries_accepts_repo_formats(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(["q1", {"query": "q2"}]), encoding="utf-8")
    assert [q["query"] for q in load_queries(tmp_path / "a.json")] == ["q1", "q2"]
    assert len(load_queries(SAMPLE_DIR.parents[1] / "Data and Question" / "queries.json")) == 11


def test_cli_query_json_and_text(tmp_path, capsys):
    from graph_rag.cli import main
    cfg = make_config(tmp_path)
    args = ["--db", cfg["db_path"], "--wiki-dir", cfg["wiki_dir"], "--pdf-dir", cfg["pdf_dir"],
            "--index-dir", cfg["index_dir"], "--embedding-backend", "hash", "--vector-backend", "memory",
            "--log-level", "ERROR"]
    assert main(args + ["--query", "Who is on-call for payments-service?"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["answer"] == "The on-call engineer for payments-service is Amara Chen."
    assert main(args + ["--query", "Which team owns payments-service?", "--format", "text"]) == 0
    text = capsys.readouterr().out
    assert "A: The owner team for payments-service is Payments Platform." in text and "WINS OVER" in text
    out_file = tmp_path / "res.json"
    assert main(args + ["--queries", str(SAMPLE_DIR / "queries_graph.json"), "--output", str(out_file)]) == 0
    assert len(json.loads(out_file.read_text(encoding="utf-8"))) == 15


# ------------------------------------------------------------------ real model (opt-in)
@pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1", reason="set RUN_LLM_TESTS=1 to download/run FLAN-T5")
def test_real_flan_t5_prefers_cmdb(env):
    from graph_rag.generation import get_llm
    cfg, store, vstore = env
    res = answer_query("Who is on-call for payments-service?", store, vstore, cfg, get_llm(cfg["generation_model"], cfg))
    assert "Jordan Blake" not in res.answer
    assert res.answer == REFUSAL or "Amara Chen" in res.answer   # FLAN-T5 may abstain; it must not pick the wiki
