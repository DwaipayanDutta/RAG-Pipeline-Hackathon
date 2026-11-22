"""Source priority rules and conflict resolution."""
from graph_rag import Quad
from graph_rag.fusion import conflict_dicts, explain_sources, find_conflicts, rank_candidates, resolve_facts

DB = Quad("payments-service", "on_call_engineer", "Amara Chen", "cmdb", "cmdb:services", "2026-11-21")
WIKI = Quad("payments-service", "on_call_engineer", "Jordan Blake", "wiki", "payments-service.md", "2025-01-10")
PDF = Quad("payments-service", "on_call_engineer", "Jordan Blake", "pdf", "review.pdf p.1", "2024-06-02", 0.6)


def test_database_beats_wiki_beats_pdf_and_nothing_is_silently_dropped():
    resolved, conflicts = resolve_facts([PDF, WIKI, DB])
    (rf,) = resolved
    assert rf.value == "Amara Chen" and rf.winner == DB
    assert rf.overridden == [WIKI, PDF] and conflicts == [(DB, WIKI), (DB, PDF)]
    assert "Priority 1 (live database)" in rf.reason and "outranks" in rf.reason and "Jordan Blake" in rf.reason


def test_wiki_wins_when_database_is_silent():
    (rf,), conflicts = resolve_facts([PDF, WIKI])
    assert rf.winner == WIKI and rf.overridden == [] and rf.agreeing == [PDF] and conflicts == []
    assert "corroborated" in rf.reason


def test_same_tier_newer_timestamp_then_confidence_wins():
    old = Quad("svc", "version", "1.0", "wiki", "old.md", "2025-01-01")
    new = Quad("svc", "version", "2.0", "wiki", "new.md", "2026-01-01")
    (rf,), _ = resolve_facts([old, new])
    assert rf.value == "2.0" and "newer timestamp" in rf.reason
    low = Quad("svc", "status", "up", "pdf", "a.pdf", confidence=0.4)
    high = Quad("svc", "status", "down", "pdf", "b.pdf", confidence=0.6)
    assert rank_candidates([low, high])[0] == high


def test_multi_valued_predicates_are_unioned_not_conflicts():
    deps = [Quad("svc", "depends_on", "a-svc", "cmdb"), Quad("svc", "depends_on", "b-svc", "wiki"),
            Quad("svc", "depends_on", "a-svc", "pdf")]
    resolved, conflicts = resolve_facts(deps)
    assert [(rf.value, rf.winner.context, len(rf.agreeing)) for rf in resolved] == [("a-svc", "cmdb", 1), ("b-svc", "wiki", 0)]
    assert conflicts == []


def test_mentions_are_evidence_not_claims_and_matching_is_normalised():
    resolved, _ = resolve_facts([Quad("svc", "mentioned_in", "a.pdf p.1", "pdf"),
                                 Quad("Payments Service", "status", "UP", "cmdb"),
                                 Quad("payments-service", "status", "up", "wiki")])
    assert len(resolved) == 1 and resolved[0].agreeing and not resolved[0].overridden


def test_conflict_report_and_explanation():
    resolved, _ = resolve_facts([DB, WIKI])
    (c,) = conflict_dicts(resolved)
    assert c["authoritative"] == DB.as_text() and c["overridden"] == WIKI.as_text()
    assert c["winner_tier"] == "cmdb" and c["overridden_source"] == "payments-service.md"
    text = explain_sources(resolved, [], [c])
    assert "Priority 1 (live database)" in text and "1 conflicting statement" in text
    assert explain_sources([], [], []).startswith("No source")


def test_legacy_find_conflicts():
    assert find_conflicts([DB], [WIKI]) == [(DB, WIKI)]
    assert find_conflicts([DB], [Quad("payments-service", "on_call_engineer", "amara chen", "wiki")]) == []
