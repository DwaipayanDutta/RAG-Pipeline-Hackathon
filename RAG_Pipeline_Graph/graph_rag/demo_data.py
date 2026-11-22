"""
Sample data: the committed dataset in RAG_Pipeline_Graph/sample_data/ and helpers to (re)generate
its PDFs or copy it elsewhere.

The data reproduces the Graph-RAG article's stale-wiki scenario and adds PDF-level conflicts:

    on-call for payments-service   CMDB: Amara Chen      wiki (2025-01-10): Jordan Blake   PDF (2024): Jordan Blake
    status of payments-service     CMDB: degraded        wiki: healthy
    version of checkout-service    CMDB: 5.3.0           wiki: 5.2.0
    owner of payments-service      CMDB: Payments Platform  wiki: Payments Platform  PDF (2024): Core Payments
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
import textwrap
from pathlib import Path
from typing import Dict, List

from .config import PACKAGE_ROOT

logger = logging.getLogger("graph_rag.demo")

SAMPLE_DIR = PACKAGE_ROOT / "sample_data"

#: PDF name -> list of pages (each page is a list of paragraphs).
SAMPLE_PDFS: Dict[str, List[List[str]]] = {
    "incident_reports.pdf": [
        ["Incident report INC-2291. Date: 2026-03-14.",
         "March incident post-mortem. Incident INC-2291 affected payments-service. During the March incident, "
         "the on-call team rolled back deploy 4412 because a misconfigured connection pool exhausted database "
         "connections. Latency returned to normal within ten minutes of the rollback.",
         "Follow-up actions: add an alert on connection pool saturation and require a canary stage for "
         "configuration changes."],
        ["Checkout migration notes.",
         "The migration of checkout traffic to the new payment gateway completed in April after two weeks of "
         "shadow testing and a staged percentage rollout. checkout-service depends on payments-service for "
         "authorisation."],
        ["Capacity review.",
         "The quarterly capacity review recommended doubling the read replicas before the holiday season and "
         "adding autoscaling alarms for queue depth."],
    ],
    "architecture_review_2024.pdf": [
        ["Payments architecture review. Date: 2024-06-02.",
         "This review describes the payments platform as it was in mid 2024. payments-service is owned by the "
         "Core Payments team. Jordan Blake is the on-call engineer for payments-service.",
         "payments-service depends on ledger-service for balance updates."],
        ["Ledger design.",
         "The ledger-service uses double-entry event sourcing so that every balance change is auditable and can "
         "be replayed after an outage. Event sourcing was chosen over in-place updates because regulators "
         "require a complete audit trail."],
    ],
}


def write_pdf(path: Path, pages: List[List[str]]) -> None:
    """Render simple text pages with reportlab (only needed to regenerate the sample PDFs)."""
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    c.setTitle(path.stem.replace("_", " ").title())
    for paragraphs in pages:
        y = 800
        for para in paragraphs:
            for line in textwrap.wrap(para, 90, break_on_hyphens=False):
                c.drawString(60, y, line)
                y -= 16
            y -= 16                                   # blank line between paragraphs
        c.showPage()
    c.save()


def generate_sample_pdfs(pdf_dir: Path = SAMPLE_DIR / "pdfs") -> List[Path]:
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, pages in SAMPLE_PDFS.items():
        write_pdf(pdf_dir / name, pages)
        out.append(pdf_dir / name)
    return out


def seed_demo_data(root: str) -> Dict[str, str]:
    """
    Copy the sample dataset into `root`, materialising the CMDB as a real SQLite file:
    root/cmdb.sqlite, root/wiki/*.md, root/pdfs/*.pdf. Returns config overrides for those paths.
    PDFs are copied when present in sample_data/, otherwise generated (requires reportlab).
    """
    r = Path(root)
    (r / "wiki").mkdir(parents=True, exist_ok=True)
    (r / "pdfs").mkdir(parents=True, exist_ok=True)
    db = r / "cmdb.sqlite"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    try:
        conn.executescript((SAMPLE_DIR / "cmdb.sql").read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()
    for f in sorted((SAMPLE_DIR / "wiki").glob("*.md")):
        shutil.copy2(f, r / "wiki" / f.name)
    missing = [n for n in SAMPLE_PDFS if not (SAMPLE_DIR / "pdfs" / n).is_file()]
    if missing:
        try:
            generate_sample_pdfs(r / "pdfs")
        except ImportError:
            logger.warning("reportlab not installed and sample PDFs missing: PDF tier will be empty")
    else:
        for n in SAMPLE_PDFS:
            shutil.copy2(SAMPLE_DIR / "pdfs" / n, r / "pdfs" / n)
    return {"db_path": str(db), "wiki_dir": str(r / "wiki"), "pdf_dir": str(r / "pdfs")}
