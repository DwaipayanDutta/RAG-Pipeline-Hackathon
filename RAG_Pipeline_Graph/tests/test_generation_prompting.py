"""Prompt construction, extractive generation and grounding validation."""
from graph_rag import Quad, REFUSAL, Retrieval, load_config
from graph_rag.fusion import resolve_facts
from graph_rag.generation import ExtractiveGenerator, make_generator
from graph_rag.prompting import fit_prompt
from graph_rag.retrieval import retrieve
from graph_rag.validation import validate_graph_answer


def _r(facts, entities, predicates, passages=()):
    r = Retrieval(entities=entities, predicates=predicates,
                  p1=[q for q in facts if q.context == "cmdb"], p2=[q for q in facts if q.context == "wiki"])
    r.p3 = list(passages)
    r.resolved, r.conflicts = resolve_facts(facts)
    return r


def test_prompt_has_required_rules_headers_and_quad_format(env):
    cfg, store, vstore = env
    r = retrieve("Who is on-call for payments-service?", store, vstore, cfg)
    p = fit_prompt("Who is on-call for payments-service?", r, cfg)
    for s in ("Priority 1 (CMDB) is absolute and always current. If it conflicts with any other source, use Priority 1.",
              "Priority 2 (Wiki) is supplementary. Use it only if Priority 1 does not cover the question.",
              "Priority 3 (PDFs) is last-resort narrative context.",
              "=== PRIORITY 1: LIVE DATABASE FACTS ===", "=== PRIORITY 2: WIKI FACTS (may be stale) ===",
              "=== PRIORITY 3: PDF NARRATIVE CONTEXT ===", "=== RESOLVED CONFLICTS ===",
              "(cmdb) payments-service on_call_engineer: Amara Chen.",
              "(wiki) payments-service on_call_engineer: Jordan Blake.",
              "payments-service on call engineer: Amara Chen (Priority 1"):
        assert s in p, s
    assert p.index("PRIORITY 1: LIVE") < p.index("PRIORITY 2: WIKI") < p.index("PRIORITY 3: PDF")
    assert p.index("KEY FACTS FOR THIS QUESTION") < p.index("=== QUESTION ===")


def test_fit_prompt_never_drops_priority_1():
    r = Retrieval(entities=["x"], p1=[Quad("x", "status", "up", "cmdb")],
                  p2=[Quad("x", "note", "w" * 200, "wiki")] * 50,
                  p3=[{"text": "t" * 500, "source": "a.pdf", "page": 1, "chunk_id": 0, "distance": .1}] * 5)
    cfg = load_config(use_env=False)
    cfg["max_context_tokens"] = 100
    prompt = fit_prompt("Q?", r, cfg)
    assert "(cmdb) x status: up." in prompt and prompt.rstrip().endswith("Answer:") and len(prompt) <= 300 * 4 + 400
    assert len(fit_prompt("Q?", r, cfg, count_tokens=lambda s: len(s.split()))) > len(prompt)   # tokenizer-aware


def test_extractive_forward_reverse_and_multi_hop():
    gen = ExtractiveGenerator()
    facts = [Quad("payments-service", "on_call_engineer", "Amara Chen", "cmdb"),
             Quad("payments-service", "on_call_engineer", "Jordan Blake", "wiki"),
             Quad("ledger-service", "on_call_engineer", "Amara Chen", "cmdb"),
             Quad("payments-service", "owner_team", "Payments Platform", "cmdb"),
             Quad("Payments Platform", "manager", "Priya Nair", "cmdb"),
             Quad("payments-service", "depends_on", "ledger-service", "cmdb")]
    out = gen.answer(_r(facts, ["payments-service"], ["on_call_engineer"]), "Who is on-call for payments-service?")
    assert out.text == "The on-call engineer for payments-service is Amara Chen." and out.mode == "facts"
    assert out.used_facts[0].overridden[0].object == "Jordan Blake"
    out = gen.answer(_r(facts, ["Amara Chen"], ["on_call_engineer"]), "What is Amara Chen on call for?")
    assert out.text == "Amara Chen is the on-call engineer for payments-service and ledger-service."
    out = gen.answer(_r(facts, ["payments-service"], ["owner_team", "manager"]), "Who manages the team that owns payments-service?")
    assert "Payments Platform" in out.text and "The manager for Payments Platform is Priya Nair." in out.text
    out = gen.answer(_r(facts, ["ledger-service"], ["depends_on"]), "Which services depend on ledger-service?")
    assert out.text == "payments-service depends on ledger-service."


def test_extractive_passages_and_refusal():
    gen = ExtractiveGenerator()
    p = {"text": "Capacity review. The review recommended doubling read replicas before the holiday season. "
                 "It also proposed queue alarms.", "tier": "pdf", "source": "cap.pdf", "page": 1, "chunk_id": 0,
         "distance": 0.3}
    out = gen.answer(_r([], [], [], [p]), "Why did the review recommend doubling read replicas?")
    assert out.text.startswith("The review recommended doubling read replicas") and out.used_passages == [p]
    assert gen.answer(_r([], [], [], [p]), "What is the capital of Mars?").text == REFUSAL
    assert gen.generate("bare prompt") == REFUSAL
    assert make_generator({"llm_backend": "extractive"}).name == "extractive"


def test_validation_rejects_stale_hallucinated_and_off_topic_answers():
    facts = [Quad("checkout-service", "version", "5.3.0", "cmdb"), Quad("checkout-service", "version", "5.2.0", "wiki"),
             Quad("checkout-service", "owner_team", "Web Commerce", "cmdb")]
    r = _r(facts, ["checkout-service"], ["version"])
    assert validate_graph_answer("5.3.0", r)[0]
    ok, v, reason = validate_graph_answer("5.2.0", r)
    assert not ok and not v["no_priority_conflict"] and "Contradicts Priority 1" in reason
    ok, v, reason = validate_graph_answer("Web Commerce", r)
    assert not ok and "asked attribute" in reason                       # grounded text, wrong attribute
    ok, _, reason = validate_graph_answer("Maria Lopez runs checkout-service 5.3.0", r)
    assert not ok and "Maria Lopez" in reason
    rr = _r([], [], [], [{"text": "To declare a SEV1 incident page the on-call engineer.", "tier": "wiki",
                          "source": "r.md", "page": 0, "chunk_id": 0, "distance": .2}])
    assert validate_graph_answer("To declare a SEV1 incident page the on-call engineer.", rr)[0]   # "1" in SEV1 is not a number claim
