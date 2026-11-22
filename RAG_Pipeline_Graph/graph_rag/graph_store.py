"""In-process knowledge graph: a quad store with subject/object indexes, traversal and JSON persistence."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .models import Quad
from .text import flat

#: Predicates whose objects are literal values or document references, never graph nodes.
NON_NODE_PREDICATES = {"type", "status", "version", "environment", "sla_tier", "last_deployed_at",
                       "mentioned_in", "slack_channel", "tier"}


class QuadStore:
    """
    Lightweight in-memory SPOC quad store.

    Matching is case/hyphen/punctuation-insensitive (`payments-service` == `Payments Service`).
    No server is needed; the store is persisted as a JSON list of quads.
    """

    def __init__(self) -> None:
        self._quads: List[Quad] = []
        self._seen: Set[Quad] = set()
        self._by_subject: Dict[str, List[Quad]] = defaultdict(list)
        self._by_object: Dict[str, List[Quad]] = defaultdict(list)

    # ---- mutation ---------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._quads)

    def __iter__(self):
        return iter(self._quads)

    def add(self, quad: Quad) -> bool:
        """Add a quad; returns False if an identical statement from the same source exists."""
        if quad in self._seen:
            return False
        self._seen.add(quad)
        self._quads.append(quad)
        self._by_subject[flat(quad.subject)].append(quad)
        self._by_object[flat(quad.object)].append(quad)
        return True

    def add_many(self, quads: Iterable[Quad]) -> int:
        return sum(1 for q in quads if self.add(q))

    def remove(self, predicate_fn) -> int:
        """Remove every quad for which predicate_fn(quad) is true; returns the number removed."""
        keep = [q for q in self._quads if not predicate_fn(q)]
        removed = len(self._quads) - len(keep)
        if removed:
            self.__init__()
            self.add_many(keep)
        return removed

    def replace_subject_facts(self, subject: str, context: str, quads: Iterable[Quad]) -> int:
        """Swap all `context` facts about `subject` for `quads` (used by live DB refresh)."""
        s = flat(subject)
        self.remove(lambda q: q.context == context and flat(q.subject) == s)
        return self.add_many(quads)

    # ---- lookup ----------------------------------------------------------------------------
    def all(self, context: Optional[str] = None) -> List[Quad]:
        return [q for q in self._quads if context is None or q.context == context]

    def subjects(self) -> Set[str]:
        return {q.subject for q in self._quads}

    def node_names(self) -> Set[str]:
        """Subjects plus objects that are graph nodes (people, teams, services)."""
        names = self.subjects()
        names |= {q.object for q in self._quads if q.predicate not in NON_NODE_PREDICATES}
        return names

    def query(self, entities: Iterable[str], context: Optional[str] = None) -> List[Quad]:
        """Quads where any entity is the subject OR the object, in insertion order."""
        wanted = {flat(e) for e in entities}
        hits: Set[Quad] = set()
        for key in wanted:
            hits.update(self._by_subject.get(key, ()))
            hits.update(self._by_object.get(key, ()))
        return [q for q in self._quads if q in hits and (context is None or q.context == context)]

    def about(self, subject: str, predicate: Optional[str] = None,
              context: Optional[str] = None) -> List[Quad]:
        """Facts whose subject is `subject` (optionally filtered by predicate / tier)."""
        return [q for q in self._by_subject.get(flat(subject), ())
                if (predicate is None or q.predicate == predicate)
                and (context is None or q.context == context)]

    # ---- traversal -------------------------------------------------------------------------
    def expand_with_hops(self, seeds: Iterable[str], hops: int,
                         traversal_predicates: Iterable[str] = ()) -> List[Tuple[int, Quad]]:
        """
        Facts about entities reached by walking the graph away from the seeds, tagged with
        the hop at which they were reached.

        Layer 0 (facts that touch a seed directly) is NOT returned; `query` covers that.
        Each hop moves to the subjects of the previous layer's facts and, for predicates in
        `traversal_predicates`, to their objects too (e.g. the on-call person, the owning
        team), then returns the facts that touch those new nodes. Literal values such as
        "degraded" are never walked.
        """
        seeds = list(seeds)
        trav = set(traversal_predicates)
        visited = {flat(s) for s in seeds}
        layer = self.query(visited)
        direct = set(layer)
        out: List[Tuple[int, Quad]] = []
        emitted: Set[Quad] = set()
        for hop in range(1, max(0, hops) + 1):
            nodes: Set[str] = set()
            for q in layer:
                nodes.add(flat(q.subject))
                if q.predicate in trav:
                    nodes.add(flat(q.object))
            nodes -= visited
            if not nodes:
                break
            visited |= nodes
            layer = [q for q in self.query(nodes) if q not in emitted and q not in direct]
            emitted.update(layer)
            out.extend((hop, q) for q in layer)
        return out

    def expand(self, seeds: Iterable[str], hops: int,
               traversal_predicates: Iterable[str] = ()) -> List[Quad]:
        return [q for _, q in self.expand_with_hops(seeds, hops, traversal_predicates)]

    # ---- persistence / stats ---------------------------------------------------------------
    def stats(self) -> Dict[str, object]:
        return {
            "quads": len(self._quads),
            "subjects": len(self.subjects()),
            "by_tier": dict(Counter(q.context for q in self._quads)),
            "by_predicate": dict(Counter(q.predicate for q in self._quads).most_common()),
        }

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([q.to_dict() for q in self._quads], f, indent=1)

    @classmethod
    def load(cls, path) -> "QuadStore":
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            store.add_many(Quad(**d) for d in json.load(f))
        return store
