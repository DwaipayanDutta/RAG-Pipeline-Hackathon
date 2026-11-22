"""Priority 2: wiki / runbook ingestion."""
from pathlib import Path

from graph_rag import Quad, QuadStore
from graph_rag.demo_data import SAMPLE_DIR
from graph_rag.sources.relations import extract_relations
from graph_rag.sources.wiki import (extract_wiki_facts, ingest_wiki_dir, load_wiki_page, parse_frontmatter,
                                    wiki_passages)


def test_wiki_regex_extraction_formats_and_source_tag():
    text = "# Payments Service\n- Owner team: Payments Platform\n**On-call engineer:** Jordan Blake\nSee: https://x.io/a\n"
    qs = extract_wiki_facts(text, "payments-service.md")
    assert {(q.subject, q.predicate, q.object) for q in qs} == {
        ("payments-service", "owner_team", "Payments Platform"),
        ("payments-service", "on_call_engineer", "Jordan Blake")}
    assert all(q.context == "wiki" and q.source == "payments-service.md" for q in qs)


def test_wiki_llm_fallback_only_when_nothing_else_matches():
    class Extractor:
        def generate(self, prompt): return "Ledger Service | Owner | Finance Platform\ngarbage line"
    prose = "The ledger is run by the finance platform team."
    qs = extract_wiki_facts(prose, "ledger-service.md", Extractor())
    assert qs == [Quad("ledger-service", "owner_team", "Finance Platform", "wiki", "ledger-service.md")]
    assert qs[0].confidence == 0.5
    assert extract_wiki_facts("Owner team: X", "a.md", Extractor())[0].object == "X"   # regex wins
    assert extract_wiki_facts(prose, "ledger-service.md") == []                         # no LLM, no facts


def test_frontmatter_sets_subject_and_timestamp_and_lists_are_split():
    text = "---\nentity: Payments Platform\nlast_updated: 2026-09-15\n---\n- Members: Ann Lee, Bo Wu and Cy Ma\n"
    meta, body = parse_frontmatter(text)
    assert meta["entity"] == "Payments Platform" and "last_updated" not in body
    qs = extract_wiki_facts(text, "team.md")
    assert {(q.subject, q.predicate, q.object) for q in qs} == {
        ("Payments Platform", "member", n) for n in ("Ann Lee", "Bo Wu", "Cy Ma")}
    assert all(q.timestamp == "2026-09-15" for q in qs)


def test_prose_page_uses_relation_patterns():
    page = load_wiki_page(SAMPLE_DIR / "wiki" / "ledger-service.md")
    qs = extract_wiki_facts(page.body, page.filename, known_entities={"ledger-service", "fraud-service"},
                            subject=page.subject, timestamp=page.timestamp)
    assert {(q.predicate, q.object) for q in qs} == {
        ("owner_team", "Core Ledger"), ("on_call_engineer", "Amara Chen"), ("depends_on", "fraud-service")}
    assert all(q.confidence == 0.7 and q.evidence for q in qs)


def test_relation_patterns():
    k = ["payments-service", "Core Ledger"]
    rel = lambda t: [(s, p, o) for s, p, o, _ in extract_relations(t, k)]   # noqa: E731
    assert rel("payments-service is owned by the Core Payments team since 2023.") == [
        ("payments-service", "owner_team", "Core Payments")]
    assert rel("The on-call engineer for Payments Service was Amara Chen.") == [
        ("payments-service", "on_call_engineer", "Amara Chen")]
    assert rel("payments-service depends on a-svc, b-svc and c-svc.") == [
        ("payments-service", "depends_on", x) for x in ("a-svc", "b-svc", "c-svc")]
    assert rel("Core Ledger team is managed by Sofia Rossi.") == [("Core Ledger", "manager", "Sofia Rossi")]
    assert rel("amara chen is on-call for payments-service.") == []    # names must be capitalised
    assert rel("unknown-service is owned by Team X.") == []             # subject must be a known entity


def test_passages_exclude_fact_lines_and_keep_sections():
    page = load_wiki_page(SAMPLE_DIR / "wiki" / "payments-service.md")
    chunks = wiki_passages(page)
    sections = [c.section for c in chunks]
    assert "Runbook: rolling back a bad deploy" in sections
    assert all("On-call engineer: Jordan Blake" not in c.text for c in chunks)
    assert all(c.metadata["tier"] == "wiki" and c.metadata["timestamp"] == "2025-01-10" for c in chunks)


def test_ingest_wiki_dir_counts_and_skips_fact_free_pages(tmp_path):
    (tmp_path / "a.md").write_text("Owner: Team A\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("Just prose, nothing structured here.\n", encoding="utf-8")
    s = QuadStore()
    assert ingest_wiki_dir(str(tmp_path), s) == 1
    assert s.all()[0] == Quad("a", "owner_team", "Team A", "wiki", "a.md")
    assert Path(tmp_path / "b.md").exists()
