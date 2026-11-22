"""Config, validation, citations, routing and observability of the structured package."""
import json

import pytest

from RAG_Pipeline_Structured.citations import generate_citations
from RAG_Pipeline_Structured.config import RAGConfig
from RAG_Pipeline_Structured.models import Chunk, RetrievalCandidate
from RAG_Pipeline_Structured.numerical import extract_numbers, normalize_crore
from RAG_Pipeline_Structured.observability import RequestContext
from RAG_Pipeline_Structured.query_router import classify_query
from RAG_Pipeline_Structured.validation import validate_answer


def _cand(text, page=1, cid="chunk_0"):
    return RetrievalCandidate(chunk=Chunk(cid, "doc", page, page, None, None, "text", None, text), rrf_score=0.5)


def test_config_defaults_point_at_repo_data_and_json_overrides(tmp_path):
    cfg = RAGConfig()
    assert cfg.pdf_path.endswith("Titan AR 2026_0.pdf") and cfg.device in ("cpu", "cuda")
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"final_top_k": 3, "use_reranker": False}), encoding="utf-8")
    assert RAGConfig.from_json(str(p)).final_top_k == 3
    p.write_text(json.dumps({"no_such_key": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        RAGConfig.from_json(str(p))
    with pytest.raises(ValueError):
        RAGConfig(chunk_size=10, chunk_overlap=10, device="cpu").validate()


def test_numeric_validation_and_refusal():
    cands = [_cand("Revenue was 10,000 crore")]
    assert validate_answer("Revenue was 10,000 crore", cands).status == "grounded"
    assert validate_answer("Revenue was 5,000 crore", cands).status == "unsupported"
    assert validate_answer("I don't know from the document.", cands).status == "refused"


def test_extract_numbers_and_units():
    assert extract_numbers("Revenue was 10,000 crore") == [{"value": 10000, "unit": "crore"}]
    assert extract_numbers("growth 15.5%") == [{"value": 15.5, "unit": "%"}]
    assert normalize_crore(2, "crore") == 2e7


def test_citations_and_routing():
    assert generate_citations([_cand("x", page=7, cid="chunk_9")]) == [
        {"page": 7, "chunk_id": "chunk_9", "source_type": "text"}]
    assert classify_query("Which businesses had revenue exceeding INR 5,000 crore?") == "threshold"
    assert classify_query("What are the key processes in the risk management framework?") == "policy"


def test_request_context_records_stage_durations():
    ctx = RequestContext()
    ctx.stage_start("retrieval")
    ctx.stage_end("retrieval")
    assert ctx.stages["retrieval"]["duration_ms"] >= 0 and len(ctx.request_id) == 36


def test_package_imports_without_heavy_dependencies_and_cli_help():
    import RAG_Pipeline_Structured
    from RAG_Pipeline_Structured.cli import main
    assert RAG_Pipeline_Structured.RAGConfig is RAGConfig
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
