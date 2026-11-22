"""
Priority 2: wiki pages and runbooks (Markdown) -> SPOC quads with context="wiki",
plus section passages for semantic search.

Page format (all parts optional):

    ---
    entity: payments-service        # graph subject (default: file name stem)
    last_updated: 2025-01-10        # provenance timestamp used for tie-breaks / staleness
    title: Payments Service
    ---
    # Payments Service
    - Owner team: Payments Platform          <- "Key: value" lines become facts
    **On-call engineer:** Jordan Blake
    Depends on: ledger-service, fraud-service   <- list keys are split into several facts

    ## Runbook: rolling back a deploy        <- sections become searchable passages
    1. Freeze deploys ...

Pages with no key/value lines fall back to relation patterns over the prose (see
relations.py) and, if enabled, to FLAN-T5 triple extraction.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..graph_store import QuadStore
from ..models import CTX_WIKI, Chunk, Quad
from ..text import slug
from .relations import extract_relations

logger = logging.getLogger("graph_rag.wiki")

KEY_ALIASES = {
    "owner": "owner_team", "owning_team": "owner_team", "team": "owner_team", "owned_by": "owner_team",
    "oncall": "on_call_engineer", "on_call": "on_call_engineer", "oncall_engineer": "on_call_engineer",
    "on_call_owner": "on_call_engineer",
    "dependencies": "depends_on", "depends": "depends_on", "upstream_dependencies": "depends_on",
    "members": "member", "team_members": "member",
    "escalation": "escalation_contact", "escalate_to": "escalation_contact",
    "slack": "slack_channel", "channel": "slack_channel",
    "sla": "sla_tier", "env": "environment", "health": "status", "state": "status",
    "current_version": "version", "deployed_version": "version",
}
#: Keys whose values are comma-separated lists of graph nodes.
MULTI_VALUED_KEYS = {"depends_on", "member", "member_of", "tags"}

# "Key: value", "- Key: value", "**Key:** value"
WIKI_KV_RE = re.compile(r"^\s*(?:[-*]\s+)?\**([A-Za-z][A-Za-z \-/]{1,40}?)\s*:\s*\**\s*(.+?)\s*\**\s*$")
_URL_RE = re.compile(r"[a-z][a-z0-9+.-]*://", re.I)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def normalize_key(key: str) -> str:
    k = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    return KEY_ALIASES.get(k, k)


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Split a simple `key: value` front-matter block (no YAML dependency) from the body."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip("'\"")
    return meta, text[m.end():]


@dataclass
class WikiPage:
    path: str
    filename: str
    subject: str
    title: str
    timestamp: str
    body: str
    meta: Dict[str, str] = field(default_factory=dict)


def load_wiki_page(path: Path) -> WikiPage:
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    heading = next((m.group(2) for m in map(_HEADING_RE.match, body.splitlines()) if m), path.stem)
    return WikiPage(path=str(path), filename=path.name,
                    subject=meta.get("entity") or slug(path.stem),
                    title=meta.get("title") or heading,
                    timestamp=meta.get("last_updated") or meta.get("updated") or "",
                    body=body, meta=meta)


def _kv_lines(text: str):
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = WIKI_KV_RE.match(line)
        if m and not _URL_RE.match(m.group(2)):        # skip URL lines ("See: https://...")
            yield line, m.group(1), m.group(2).strip()


def extract_wiki_facts_llm(text: str, filename: str, llm, timestamp: str = "") -> List[Quad]:
    """Fallback for pages with no parsable facts: ask the extraction model for triples."""
    subject_default = slug(Path(filename).stem)
    prompt = ("Extract facts from the text. Output one fact per line as: subject | predicate | object\n"
              f"Text: {text[:1500]}\nFacts:")
    quads = []
    for line in llm.generate(prompt).splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            quads.append(Quad(slug(parts[0]) or subject_default, normalize_key(parts[1]), parts[2],
                              CTX_WIKI, filename, timestamp=timestamp, confidence=0.5,
                              evidence="llm-extracted"))
    return quads


def extract_wiki_facts(text: str, filename: str, llm=None, known_entities: Iterable[str] = (),
                       subject: Optional[str] = None, timestamp: str = "") -> List[Quad]:
    """
    Facts from one page: key/value lines (subject = front-matter `entity` or file stem), else
    relation patterns over the prose (needs `known_entities`), else the LLM (if given).
    """
    meta, body = parse_frontmatter(text)
    subject = subject or meta.get("entity") or slug(Path(filename).stem)
    timestamp = timestamp or meta.get("last_updated", "")
    quads: List[Quad] = []
    for line, key, value in _kv_lines(body):
        pred = normalize_key(key)
        values = [v.strip() for v in re.split(r",|\band\b", value)] if pred in MULTI_VALUED_KEYS else [value]
        for v in filter(None, values):
            quads.append(Quad(subject, pred, v, CTX_WIKI, filename, timestamp=timestamp,
                              confidence=0.9, evidence=line.strip()))
    if not quads:
        known = set(known_entities) | {subject}
        for s, p, o, sentence in extract_relations(body, known):
            quads.append(Quad(s, p, o, CTX_WIKI, filename, timestamp=timestamp, confidence=0.7,
                              evidence=sentence))
    if not quads and llm is not None:
        quads = extract_wiki_facts_llm(body, filename, llm, timestamp)
    return quads


def wiki_passages(page: WikiPage, max_words: int = 200, min_words: int = 8) -> List[Chunk]:
    """Split a page into heading-delimited sections (key/value lines removed) for vector search."""
    sections: List[Tuple[str, List[str]]] = [(page.title, [])]
    kv = {line for line, _, _ in _kv_lines(page.body)}
    for line in page.body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append((m.group(2).strip(), []))
        elif line not in kv:
            sections[-1][1].append(line)
    chunks: List[Chunk] = []
    for heading, lines in sections:
        text = "\n".join(lines).strip()
        words = text.split()
        if len(words) < min_words:
            continue
        for start in range(0, len(words), max_words):
            piece = " ".join(words[start:start + max_words]) if len(words) > max_words else text
            chunks.append(Chunk(chunk_id=0, page=0, section=heading, source_type="wiki",
                                text=f"{page.title} - {heading}\n{piece}",
                                metadata={"tier": CTX_WIKI, "source": page.filename, "title": page.title,
                                          "subject": page.subject, "timestamp": page.timestamp,
                                          "path": page.path}))
            if len(words) <= max_words:
                break
    return chunks


def load_wiki_dir(wiki_dir: str) -> List[WikiPage]:
    return [load_wiki_page(f) for f in sorted(Path(wiki_dir).glob("*.md"))]


def ingest_wiki_dir(wiki_dir: str, store: QuadStore, llm_getter: Optional[Callable[[], Any]] = None,
                    known_entities: Iterable[str] = (), pages: Optional[List[WikiPage]] = None) -> int:
    """Extract facts from every `*.md` page; the LLM is loaded lazily, once, only if needed."""
    added, llm, tried = 0, None, False
    known = set(known_entities) | store.node_names()
    pages = pages if pages is not None else load_wiki_dir(wiki_dir)
    known |= {p.subject for p in pages}
    for page in pages:
        facts = extract_wiki_facts(page.body, page.filename, known_entities=known,
                                   subject=page.subject, timestamp=page.timestamp)
        if not facts and llm_getter is not None:
            if not tried:
                tried = True
                try:
                    llm = llm_getter()
                except Exception as e:
                    logger.warning("LLM wiki extraction unavailable: %s", e)
            if llm is not None:
                try:
                    facts = extract_wiki_facts_llm(page.body, page.filename, llm, page.timestamp)
                except Exception as e:
                    logger.warning("LLM extraction failed for %s: %s", page.filename, e)
        if not facts:
            logger.info("wiki: no facts extracted from %s (passages are still indexed)", page.filename)
        added += store.add_many(facts)
    logger.info("wiki: %d quads from %d pages", added, len(pages))
    return added
