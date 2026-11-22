"""
Source fusion and priority-based conflict resolution.

Rules (applied per (subject, predicate)):

1. Single-valued predicates (owner_team, on_call_engineer, status, version, ...) can have one
   true value. Candidates are ranked by
       tier priority (cmdb=1 > wiki=2 > pdf=3)  ->  newer timestamp  ->  higher confidence
   The top candidate wins; every source stating a *different* value is recorded as
   overridden (a conflict) with the reason, never silently dropped.
2. Multi-valued predicates (depends_on, member, mentioned_in, ...) are unioned: each distinct
   value becomes its own resolved fact, attributed to its most authoritative source.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import TIER_LABEL, TIER_PRIORITY, Quad, ResolvedFact
from .text import flat

DEFAULT_SINGLE_VALUED = (
    "owner_team", "on_call_engineer", "status", "version", "environment", "sla_tier",
    "last_deployed_at", "manager", "slack_channel", "escalation_contact", "tier", "type",
)


def rank_candidates(quads: Iterable[Quad]) -> List[Quad]:
    """
    Most authoritative first: tier priority, then newer timestamp (ISO strings compare
    chronologically; missing timestamps lose), then higher confidence. Stable multi-pass sort.
    """
    ranked = sorted(quads, key=lambda q: -q.confidence)
    ranked = sorted(ranked, key=lambda q: q.timestamp or "", reverse=True)
    return sorted(ranked, key=lambda q: q.priority)


def _describe(q: Quad) -> str:
    when = f", updated {q.timestamp}" if q.timestamp else ""
    return f"{TIER_LABEL.get(q.context, q.context)} [{q.source}{when}]"


def _reason(winner: Quad, overridden: Sequence[Quad], agreeing: Sequence[Quad]) -> str:
    if not overridden:
        if agreeing:
            return (f"{_describe(winner)} states this; corroborated by "
                    + ", ".join(_describe(q) for q in agreeing) + ".")
        return f"Only {_describe(winner)} states this."
    parts = [f"{_describe(q)} says '{q.object}' ({_why(winner, q)})" for q in overridden]
    return f"{_describe(winner)} says '{winner.object}' and wins. Overridden: " + "; ".join(parts) + "."


def _why(winner: Quad, loser: Quad) -> str:
    """Which rule made `winner` beat `loser`."""
    if loser.priority > winner.priority:
        return f"{TIER_LABEL[winner.context].split(' (')[0]} outranks {TIER_LABEL[loser.context].split(' (')[0]}"
    if loser.timestamp and winner.timestamp and winner.timestamp > loser.timestamp:
        return "same tier, newer timestamp wins"
    return "same tier, higher confidence wins"


def resolve_facts(facts: Iterable[Quad], single_valued: Iterable[str] = DEFAULT_SINGLE_VALUED
                  ) -> Tuple[List[ResolvedFact], List[Tuple[Quad, Quad]]]:
    """
    Resolve candidate facts from all tiers.

    Returns (resolved facts in first-seen order, conflict pairs (winner, overridden)).
    """
    single = set(single_valued)
    groups: "OrderedDict[Tuple[str, str], List[Quad]]" = OrderedDict()
    for q in facts:
        if q.predicate == "mentioned_in":
            continue                            # document references are evidence, not claims
        groups.setdefault((flat(q.subject), q.predicate), []).append(q)

    resolved: List[ResolvedFact] = []
    conflicts: List[Tuple[Quad, Quad]] = []
    for (_, predicate), quads in groups.items():
        if predicate in single:
            ranked = rank_candidates(quads)
            winner = ranked[0]
            agreeing = [q for q in ranked[1:] if flat(q.object) == flat(winner.object)]
            overridden = [q for q in ranked[1:] if flat(q.object) != flat(winner.object)]
            # one entry per (tier, source, value) so repeated PDF pages don't spam the output
            overridden = list(OrderedDict(((q.context, q.source.split(" p.")[0], flat(q.object)), q)
                                          for q in overridden).values())
            resolved.append(ResolvedFact(winner.subject, predicate, winner.object, winner, agreeing,
                                         overridden, _reason(winner, overridden, agreeing)))
            conflicts.extend((winner, q) for q in overridden)
        else:
            by_value: "OrderedDict[str, List[Quad]]" = OrderedDict()
            for q in quads:
                by_value.setdefault(flat(q.object), []).append(q)
            for vals in by_value.values():
                ranked = rank_candidates(vals)
                resolved.append(ResolvedFact(ranked[0].subject, predicate, ranked[0].object, ranked[0],
                                             ranked[1:], [], _reason(ranked[0], [], ranked[1:])))
    return resolved, conflicts


def find_conflicts(authoritative: List[Quad], lower: List[Quad]) -> List[Tuple[Quad, Quad]]:
    """(authoritative, overridden) pairs: same subject + predicate, different object (legacy helper)."""
    return [(a, b) for a in authoritative for b in lower
            if flat(a.subject) == flat(b.subject) and a.predicate == b.predicate
            and flat(a.object) != flat(b.object)]


def conflict_dicts(resolved: List[ResolvedFact]) -> List[Dict[str, object]]:
    """JSON-friendly conflict report: one entry per (winner, overridden) pair."""
    out = []
    for rf in resolved:
        for q in rf.overridden:
            out.append({
                "subject": rf.subject, "predicate": rf.predicate,
                "authoritative": rf.winner.as_text(), "overridden": q.as_text(),
                "winner_tier": rf.winner.context, "overridden_tier": q.context,
                "winner_source": rf.winner.source, "overridden_source": q.source,
                "reason": (f"{_describe(rf.winner)} says '{rf.value}'; {_describe(q)} says '{q.object}'. "
                           f"Winner: {rf.winner.context} ({_why(rf.winner, q)})."),
                "affects_answer": False,      # set by the pipeline for facts used in the answer
            })
    return out


def explain_sources(used: Sequence[ResolvedFact], passages: Sequence[Dict[str, object]],
                    conflicts: Sequence[Dict[str, object]]) -> str:
    """One-paragraph provenance explanation: which tiers answered and how conflicts were settled."""
    if not used and not passages:
        return "No source contained evidence for this question."
    lines = []
    tiers = sorted({rf.winner.context for rf in used}, key=lambda t: TIER_PRIORITY.get(t, 99))
    if tiers:
        lines.append("Answer facts come from " + " and ".join(TIER_LABEL[t] for t in tiers) + ".")
    if passages:
        ptiers = sorted({str(p.get("tier")) for p in passages}, key=lambda t: TIER_PRIORITY.get(t, 99))
        lines.append("Supporting passages: " + ", ".join(TIER_LABEL.get(t, t) for t in ptiers) + ".")
    if conflicts:
        lines.append(f"{len(conflicts)} conflicting statement(s) were overridden by higher-priority "
                     "or newer sources (see `conflicts`).")
    else:
        lines.append("No conflicts between sources were found for the facts used.")
    return " ".join(lines)
