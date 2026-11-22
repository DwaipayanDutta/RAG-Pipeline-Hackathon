"""Entity linking, predicate intent, tiered retrieval, live refresh and traversal."""
import sqlite3

import pytest

from graph_rag.retrieval import detect_predicates, extract_query_entities, is_narrative_question, load_spacy, retrieve


def test_query_entity_extraction_known_list():
    known = ["payments-service", "checkout-service"]
    assert extract_query_entities("Who is on-call for payments-service?", known) == ["payments-service"]
    assert set(extract_query_entities("owner of Checkout Service and payments service?", known)) == set(known)
    assert extract_query_entities("What is the capital of Mars?", known) == []


def test_longest_match_wins_and_order_of_appearance():
    known = ["ledger", "ledger-service", "Amara Chen"]
    assert extract_query_entities("Is Amara Chen on call for ledger-service?", known) == ["Amara Chen", "ledger-service"]


def test_query_entity_extraction_with_spacy_finds_people():
    nlp = load_spacy("en_core_web_sm")
    if nlp is None:
        pytest.skip("spaCy model en_core_web_sm not installed")
    ents = extract_query_entities("What does Amara Chen cover besides payments-service?", ["payments-service"], nlp)
    assert "payments-service" in ents and any("Amara" in e for e in ents)


def test_predicate_intents():
    assert detect_predicates("Who is on-call for payments-service?") == ["on_call_engineer"]
    assert detect_predicates("Who manages the team that owns payments-service?") == ["owner_team", "manager"]
    assert detect_predicates("Which services depend on ledger-service?") == ["depends_on"]
    assert detect_predicates("What is the escalation contact?", ["escalation_contact"]) == ["escalation_contact"]
    assert detect_predicates("How do I roll back a bad deploy?") == []
    assert is_narrative_question("How do I roll back a bad deploy?") and not is_narrative_question("Who owns x?")


def test_retrieval_groups_by_tier_and_resolves_conflicts(env):
    cfg, store, vstore = env
    r = retrieve("Who is on-call for payments-service?", store, vstore, cfg)
    assert r.entities == ["payments-service"] and r.predicates == ["on_call_engineer"]
    assert r.p1[0].predicate == "on_call_engineer" and r.p1[0].context == "cmdb"      # asked predicate first
    assert any(q.object == "Jordan Blake" for q in r.p2)
    on_call = next(rf for rf in r.resolved if rf.predicate == "on_call_engineer" and rf.subject == "payments-service")
    assert on_call.value == "Amara Chen" and {q.context for q in on_call.overridden} == {"wiki", "pdf"}
    assert r.trace["counts"]["conflicts"] >= 2 and "retrieval_ms" in r.trace


def test_multi_hop_trace_reaches_team_manager(env):
    cfg, store, vstore = env
    r = retrieve("Who manages the team that owns payments-service?", store, vstore, cfg)
    assert any(q.subject == "Payments Platform" and q.predicate == "manager" for q in r.related)
    assert any(h["hop"] == 1 for h in r.trace["hops"])


def test_unknown_entity_gets_no_passages(env):
    cfg, store, vstore = env
    r = retrieve("Who is on-call for inventory-service?", store, vstore, cfg)
    assert r.unknown_entities == ["inventory-service"] and r.is_empty


def test_live_refresh_picks_up_database_changes_without_rebuild(env, index):
    cfg, store, vstore = env
    conn = sqlite3.connect(cfg["db_path"])
    conn.execute("UPDATE services SET on_call_engineer='Nia Patel' WHERE service_name='payments-service'")
    conn.commit()
    conn.close()
    r = retrieve("Who is on-call for payments-service?", store, vstore, cfg, db=index.db, db_specs=index.db_specs)
    assert r.trace["live_refresh"]["payments-service"] > 0
    winner = next(rf for rf in r.resolved if rf.predicate == "on_call_engineer" and rf.subject == "payments-service")
    assert winner.value == "Nia Patel" and winner.winner.context == "cmdb"
    cfg2 = dict(cfg, live_db_refresh=False)                                            # flag is honoured
    assert "live_refresh" not in retrieve("Who is on-call for payments-service?", store, vstore, cfg2,
                                          db=index.db, db_specs=index.db_specs).trace
