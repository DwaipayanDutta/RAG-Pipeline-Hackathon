"""Knowledge graph: SPOC quads, QuadStore indexing, traversal, persistence."""
from graph_rag import Quad, QuadStore


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


def test_quad_format_and_provenance_fields():
    q = Quad("payments-service", "on_call_engineer", "Amara Chen", "cmdb", timestamp="2026-11-21", confidence=1.0)
    assert q.as_text() == "(cmdb) payments-service on_call_engineer: Amara Chen."
    assert q.priority == 1 and Quad("a", "b", "c", "wiki").priority == 2 and Quad("a", "b", "c", "pdf").priority == 3
    # provenance metadata does not affect identity: same statement from the same source is one fact
    assert q == Quad("payments-service", "on_call_engineer", "Amara Chen", "cmdb", timestamp="other")


def test_metadata_survives_persistence(tmp_path):
    s = QuadStore()
    s.add(Quad("x", "status", "up", "wiki", "x.md", timestamp="2025-01-10", confidence=0.9, evidence="Status: up"))
    s.save(tmp_path / "q.json")
    q = QuadStore.load(tmp_path / "q.json").all()[0]
    assert (q.timestamp, q.confidence, q.evidence) == ("2025-01-10", 0.9, "Status: up")


def test_expand_walks_to_other_services_via_on_call_person():
    s = QuadStore()
    s.add(Quad("payments-service", "on_call_engineer", "Amara Chen", "cmdb"))
    s.add(Quad("ledger-service", "on_call_engineer", "Amara Chen", "cmdb"))
    s.add(Quad("ledger-service", "status", "healthy", "cmdb"))
    s.add(Quad("checkout-service", "on_call_engineer", "Luis Ortega", "cmdb"))
    rel = s.expand(["payments-service"], 1, ["on_call_engineer"])
    assert {q.subject for q in rel} == {"ledger-service"}              # reached via Amara Chen, not checkout
    assert s.expand(["payments-service"], 1, []) == []                 # no traversal predicates -> nothing


def test_multi_hop_service_to_team_to_manager():
    s = QuadStore()
    s.add(Quad("payments-service", "owner_team", "Payments Platform", "cmdb"))
    s.add(Quad("Payments Platform", "manager", "Priya Nair", "cmdb"))
    s.add(Quad("Priya Nair", "located_in", "London", "wiki"))
    hops = s.expand_with_hops(["payments-service"], 2, ["owner_team", "manager"])
    assert (1, Quad("Payments Platform", "manager", "Priya Nair", "cmdb")) in hops
    assert (2, Quad("Priya Nair", "located_in", "London", "wiki")) in hops
    assert all(h <= 1 for h, _ in s.expand_with_hops(["payments-service"], 1, ["owner_team", "manager"]))


def test_literal_values_are_not_traversed():
    s = QuadStore()
    s.add(Quad("a-service", "status", "healthy", "cmdb"))
    s.add(Quad("b-service", "status", "healthy", "cmdb"))
    assert s.expand(["a-service"], 3, ["owner_team"]) == []


def test_replace_subject_facts_and_node_names():
    s = QuadStore()
    s.add_many([Quad("svc", "status", "up", "cmdb"), Quad("svc", "status", "down", "wiki"),
                Quad("svc", "owner_team", "Team A", "cmdb")])
    s.replace_subject_facts("SVC", "cmdb", [Quad("svc", "status", "degraded", "cmdb")])
    assert {(q.context, q.object) for q in s.about("svc")} == {("wiki", "down"), ("cmdb", "degraded")}
    assert "Team A" not in s.node_names() and "svc" in s.node_names()
    assert "degraded" not in s.node_names()                            # literal value, not a node
    assert s.stats()["by_tier"] == {"wiki": 1, "cmdb": 1}
