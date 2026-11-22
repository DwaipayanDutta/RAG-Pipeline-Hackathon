"""
Deterministic relation extraction from prose (wiki pages without key/value lines, PDF text).

Patterns are anchored on *known entities* (names already in the graph) so free text can only
attach facts to nodes the system knows about; that keeps extraction precise and testable.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from ..text import split_sentences

_NAME = r"(?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"          # "Amara Chen" (case-sensitive)
_ID = r"[a-z0-9]+(?:-[a-z0-9]+)+"                          # "ledger-service"
# Team names are capitalised words; (?-i:...) keeps that case-sensitive even inside re.I patterns,
# so "owned by the Core Payments team since 2023" captures "Core Payments".
_TEAM = r"(?:the\s+)?(?P<team>(?-i:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*))(?:\s+team)?"


def _entity_alternation(entities: Iterable[str]) -> str:
    """Regex alternation that matches each entity with flexible hyphen/space separators."""
    alts = []
    for e in sorted({e for e in entities if e}, key=len, reverse=True):
        parts = [re.escape(p) for p in re.split(r"[\s\-_]+", e.strip()) if p]
        if parts:
            alts.append(r"[\s\-_]+".join(parts))
    return "|".join(alts)


def extract_relations(text: str, known_entities: Iterable[str]) -> List[Tuple[str, str, str, str]]:
    """
    Return (subject, predicate, object, evidence_sentence) tuples found in `text`.

    Supported statements (case-insensitive on the entity, the subject must be a known entity):
        <entity> is owned by [the] <Team> [team]          -> owner_team
        <Team> [team] owns <entity>                       -> owner_team
        <entity> depends on <id>[, <id> and <id>]         -> depends_on
        <Name> is [the] on-call [engineer] for <entity>   -> on_call_engineer
        [the] on-call [engineer] for <entity> is/was <Name>  -> on_call_engineer
        <Team> [team] is managed by <Name>                -> manager
    """
    ents = _entity_alternation(known_entities)
    if not ents:
        return []
    e = rf"(?P<ent>{ents})"
    # All patterns are case-insensitive except the person/team name groups (see _NAME/_TEAM).
    patterns = [(pred, re.compile(rx, re.I)) for pred, rx in (
        ("owner_team", rf"\b{e}\s+(?:service\s+)?is\s+owned\s+by\s+{_TEAM}(?=[\s.,;]|$)"),
        ("owner_team", rf"\b{_TEAM}\s+owns\s+(?:the\s+)?{e}\b"),
        ("depends_on", rf"\b{e}\s+depends\s+on\s+(?P<deps>{_ID}(?:\s*(?:,|and)\s*{_ID})*)"),
        ("on_call_engineer",
         rf"\b(?P<name>{_NAME})\s+is\s+(?:the\s+)?on[\s-]call(?:\s+engineer)?\s+for\s+(?:the\s+)?{e}\b"),
        ("on_call_engineer",
         rf"\bon[\s-]call(?:\s+engineer)?\s+for\s+(?:the\s+)?{e}\s+(?:is|was)\s+(?P<name>{_NAME})\b"),
        ("manager", rf"\b{e}\s+(?:team\s+)?is\s+managed\s+by\s+(?P<name>{_NAME})\b"),
    )]
    canon = {re.sub(r"[\s\-_]+", " ", k.lower()): k for k in known_entities}
    out: List[Tuple[str, str, str, str]] = []
    for sentence in split_sentences(text):
        for predicate, rx in patterns:
            for m in rx.finditer(sentence):
                subject = canon.get(re.sub(r"[\s\-_]+", " ", m.group("ent").lower()), m.group("ent"))
                gd = m.groupdict()
                if gd.get("deps"):
                    objects = re.split(r"\s*(?:,|\band\b)\s*", gd["deps"])
                else:
                    objects = [gd.get("team") or gd.get("name")]
                for obj in objects:
                    if obj and obj.strip():
                        out.append((subject, predicate, obj.strip(), sentence))
    return out
