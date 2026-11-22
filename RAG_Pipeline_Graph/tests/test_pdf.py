"""Priority 3: PDF extraction, cleaning, chunking and PDF-derived facts."""
import pytest

from graph_rag.demo_data import SAMPLE_DIR
from graph_rag.models import Chunk
from graph_rag.sources.pdf import chunk_text, extract_pdf_pages, load_pdf, pdf_facts
from graph_rag.text import clean_text

from conftest import HAS_PDFPLUMBER


def test_clean_text_removes_cid_markers_and_rejoins_hyphenation():
    raw = "Connec-\ntion pool (cid:12) ex-\nhausted.\n\n\n\nNext   para"
    assert clean_text(raw) == "Connection pool exhausted.\n\nNext para"
    assert clean_text("on-call for payments-\nservice today") == "on-call for payments-service today"


def test_chunk_text_packs_paragraphs_and_skips_empty_pages():
    pages = [{"page": 1, "text": "Para one.\n\nPara two."}, {"page": 2, "text": ""}]
    chunks = chunk_text(pages, 400, 80)
    assert len(chunks) == 1 and chunks[0].page == 1 and "Para two." in chunks[0].text


def test_chunk_text_is_word_based_with_overlap_and_page_refs():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_text([{"page": 7, "text": " ".join(words)}], chunk_size=100, overlap=20)
    assert [len(c.text.split()) for c in chunks] == [100, 100, 90]
    assert chunks[1].text.split()[:20] == chunks[0].text.split()[-20:]       # overlap
    assert all(c.page == 7 for c in chunks) and [c.chunk_id for c in chunks] == [0, 1, 2]
    assert chunks[-1].text.split()[-1] == "w249"                             # nothing lost


def test_chunks_never_span_pages():
    pages = [{"page": 1, "text": "alpha " * 30}, {"page": 2, "text": "beta " * 30}]
    chunks = chunk_text(pages, 100, 10)
    assert [(c.page, set(c.text.split())) for c in chunks] == [(1, {"alpha"}), (2, {"beta"})]


@pytest.mark.skipif(not HAS_PDFPLUMBER, reason="pdfplumber not installed")
def test_extract_sample_pdf_with_metadata():
    pages = extract_pdf_pages(str(SAMPLE_DIR / "pdfs" / "incident_reports.pdf"))
    assert len(pages) == 3 and "rolled back deploy 4412" in " ".join(pages[0]["text"].split())
    chunks, title, date = load_pdf(SAMPLE_DIR / "pdfs" / "incident_reports.pdf", 120, 30)
    assert date == "2026-03-14" and title
    m = chunks[0].metadata
    assert m["tier"] == "pdf" and m["source"] == "incident_reports.pdf" and m["page"] == 1 and m["doc_id"]


def test_pdf_facts_mentions_and_relations():
    c = Chunk(0, 3, None, "text", "Review. payments-service is owned by the Core Payments team. "
                                  "payments-service depends on ledger-service.")
    facts = pdf_facts([c], "review.pdf", ["payments-service", "ledger-service"], "2024-06-02")
    triples = {(q.subject, q.predicate, q.object) for q in facts}
    assert ("payments-service", "mentioned_in", "review.pdf p.3") in triples
    assert ("ledger-service", "mentioned_in", "review.pdf p.3") in triples
    assert ("payments-service", "owner_team", "Core Payments") in triples
    assert ("payments-service", "depends_on", "ledger-service") in triples
    assert all(q.context == "pdf" and q.timestamp == "2024-06-02" and q.confidence < 1 for q in facts)
