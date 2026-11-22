import pdfplumber
import re
from typing import List, Dict
from .models import Chunk, Table
from .config import RAGConfig

def extract_pages(pdf_path: str) -> List[Dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            pages.append({
                "page": page_num,
                "raw_text": text,
                "tables": tables
            })
    return pages

def normalize_text(text: str) -> str:
    # Remove CID markers
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()