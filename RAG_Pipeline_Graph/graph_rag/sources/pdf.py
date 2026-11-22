"""
Priority 3: PDF documents -> cleaned page text -> overlapping word chunks (for the vector store)
and low-confidence graph quads (entity mentions + relation patterns) with context="pdf".
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..models import CTX_PDF, Chunk, Quad
from ..text import clean_text, flat, slug, split_sentences
from .relations import extract_relations

logger = logging.getLogger("graph_rag.pdf")

_DATE_RE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")


def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    """
    Page text with pdfplumber. Returns [{"page": n, "text": cleaned_text}, ...] (1-based pages),
    the same shape as rag_pipeline.extract_pdf_content's page list. Unreadable pages yield "".
    """
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:                       # malformed page: keep going
                logger.warning("%s p.%d: text extraction failed: %s", pdf_path, page_num, e)
                text = ""
            pages.append({"page": page_num, "text": clean_text(text)})
    if not any(p["text"] for p in pages):
        logger.warning("%s: no extractable text (scanned PDF? OCR is not implemented)", pdf_path)
    return pages


def pdf_title(pdf_path: str, pages: List[Dict]) -> str:
    """PDF metadata title, else the first non-empty line, else the file name."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            title = (pdf.metadata or {}).get("Title")
            if title and str(title).strip() and str(title).strip().lower() != "untitled":
                return str(title).strip()
    except Exception:
        pass
    for p in pages:
        for line in p["text"].splitlines():
            if line.strip():
                return line.strip()[:120]
    return Path(pdf_path).stem


def _windows(words: List[str], size: int, overlap: int) -> Iterable[List[str]]:
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        yield words[start:start + size]
        if start + size >= len(words):
            break


def chunk_text(pages_text: List[Dict], chunk_size: int = 120, overlap: int = 30) -> List[Chunk]:
    """
    Fixed-size *word* chunks with overlap, per page (chunks never span pages, so every chunk
    has exactly one page reference). Paragraphs are packed greedily; a chunk that would exceed
    `chunk_size` words is closed and the next one starts with its last `overlap` words.
    """
    chunks: List[Chunk] = []
    chunk_id = 0

    def emit(page: int, words: List[str]) -> None:
        nonlocal chunk_id
        chunks.append(Chunk(chunk_id=chunk_id, page=page, section=None, source_type="text",
                            text=" ".join(words), metadata={"page": page}))
        chunk_id += 1

    for info in pages_text:
        page, text = info["page"], info.get("text") or ""
        current: List[str] = []
        for para in (p for p in re.split(r"\n\s*\n", text) if p.strip()):
            words = para.split()
            if len(current) + len(words) <= chunk_size:
                current += words
                continue
            if current:
                emit(page, current)
                current = current[-overlap:] if overlap else []
            if len(current) + len(words) <= chunk_size:
                current += words
            else:                                       # paragraph longer than a chunk: slide over it
                windows = list(_windows(current + words, chunk_size, overlap))
                for w in windows[:-1]:
                    emit(page, w)
                current = windows[-1]
        if current:
            emit(page, current)
    return chunks


def pdf_facts(chunks: List[Chunk], source: str, known_entities: Iterable[str],
              doc_timestamp: str = "") -> List[Quad]:
    """
    Low-confidence graph facts from PDF text:
      * `mentioned_in` quads linking known entities to the page that mentions them
      * relation-pattern facts (owner_team, depends_on, on_call_engineer, ...) - these are what
        let the fusion layer detect PDF-vs-database/wiki conflicts.
    """
    known = [e for e in known_entities if e and len(flat(e)) > 2]
    out: List[Quad] = []
    for c in chunks:
        ref = f"{source} p.{c.page}"
        text_flat = f" {flat(c.text)} "
        for ent in known:
            if f" {flat(ent)} " in text_flat:
                sentence = next((s for s in split_sentences(c.text) if f" {flat(ent)} " in f" {flat(s)} "), "")
                out.append(Quad(ent, "mentioned_in", ref, CTX_PDF, source, timestamp=doc_timestamp,
                                confidence=0.4, evidence=sentence[:300]))
        for s, p, o, sentence in extract_relations(c.text, known):
            out.append(Quad(s, p, o, CTX_PDF, ref, timestamp=doc_timestamp, confidence=0.6,
                            evidence=sentence[:300]))
    return out


def load_pdf(pdf_path: Path, chunk_size: int, overlap: int) -> Tuple[List[Chunk], str, str]:
    """Extract + chunk one PDF. Returns (chunks with provenance metadata, title, document date)."""
    pages = extract_pdf_pages(str(pdf_path))
    title = pdf_title(str(pdf_path), pages)
    date_match = next((m for p in pages for m in [_DATE_RE.search(p["text"])] if m), None)
    doc_date = date_match.group(1) if date_match else ""
    chunks = chunk_text(pages, chunk_size, overlap)
    for c in chunks:
        c.metadata.update({"tier": CTX_PDF, "source": pdf_path.name, "title": title,
                           "doc_id": slug(pdf_path.stem), "path": str(pdf_path), "timestamp": doc_date})
    logger.info("pdf: %s -> %d pages, %d chunks", pdf_path.name, len(pages), len(chunks))
    return chunks, title, doc_date
