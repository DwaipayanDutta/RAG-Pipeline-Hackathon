"""Text utilities shared by ingestion, retrieval and validation."""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Set

STOP = set("""the a an is are was were of for to in on at by and or what who whom why how which when does did do
this that it its with from as be been i me my your their his her we you they there here can should would
could will shall has have had any all about into than then so if not no yes our us them these those""".split())

_CID_RE = re.compile(r"\(cid:\d+\)")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w+)")
# Word endings that indicate typographic hyphenation ("connec-\ntion"); anything else is treated
# as a real hyphenated term split across lines ("payments-\nservice") and keeps its hyphen.
_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ing", "ed", "ly", "able", "ible", "ance", "ence",
             "ure", "ous", "ive", "al", "er", "est", "ies", "es", "ty", "ry", "cy")
# Sentence boundary: before a capital/digit/quote, or before a lowercase hyphenated id.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(]|[a-z0-9]+-[a-z0-9])")
_ID_RE = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+)+\b")

#: Hyphenated words that are vocabulary, not entity ids.
COMMON_HYPHENATED = {"on-call", "post-mortem", "follow-up", "read-only", "roll-back", "self-service",
                     "real-time", "end-to-end", "sign-off", "e-mail", "fy2025-26", "fy2024-25"}


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace (kept for parity with the sibling pipelines)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _rejoin_hyphenated(m: "re.Match") -> str:
    """'connec-\\ntion' / 'ex-\\nhausted' -> joined word; 'payments-\\nservice' -> 'payments-service'."""
    start = re.search(r"\w+$", m.string[:m.start(1) + 1]).group(0)
    rest = m.group(2)
    typographic = rest.lower().startswith(_SUFFIXES) or len(start) <= 3 or len(rest) <= 3
    return m.group(1) + ("" if typographic else "-") + rest


def clean_text(text: str) -> str:
    """
    Normalise extracted document text while keeping paragraph breaks:
    removes pdfplumber `(cid:N)` glyph markers, re-joins words hyphenated across line breaks,
    collapses runs of spaces/tabs and limits blank lines to one.
    """
    text = _CID_RE.sub("", text or "")
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK_RE.sub(_rejoin_hyphenated, text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def flat(s: str) -> str:
    """Case/hyphen/underscore/punctuation-insensitive form used for matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.strip().lower()).strip("-")


def stem(token: str) -> str:
    """Very light suffix stripping so 'rolled'/'roll' and 'services'/'service' match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_tokens(text: str) -> List[str]:
    """Stemmed, stop-word-free tokens."""
    return [stem(t) for t in flat(text).split() if t not in STOP and len(t) > 1]


def token_set(text: str) -> Set[str]:
    return set(content_tokens(text))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a and b else 0.0


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences; list items / numbered steps become their own sentence."""
    out: List[str] = []
    for block in re.split(r"\n(?=\s*(?:[-*]|\d+[.)])\s)|\n{2,}", text):
        block = re.sub(r"\s+", " ", block).strip()
        if block:
            out.extend(s.strip() for s in _SENT_RE.split(block) if s.strip())
    return out


def find_ids(text: str) -> List[str]:
    """Hyphenated identifiers such as `inventory-service` (common hyphenated words excluded)."""
    return [m for m in _ID_RE.findall(text.lower()) if m not in COMMON_HYPHENATED]


def compute_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()
