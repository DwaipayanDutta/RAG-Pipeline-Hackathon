from typing import List
from .models import Chunk, Table
from .ingestion import normalize_text
import re

def chunk_document(pages: List[Dict], document_id: str, config) -> List[Chunk]:
    chunks = []
    chunk_counter = 0
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["raw_text"]
        tables = page_data["tables"]

        # 1. Process tables
        for tbl_idx, table in enumerate(tables):
            if not table or len(table) < 2:
                continue
            header = [str(h).strip() if h else "" for h in table[0]]
            table_id = f"tbl_{page_num}_{tbl_idx}"
            # Also create a table object for structured storage (not shown here)
            for row in table[1:]:
                row_text = " | ".join(
                    f"{h}: {v}" for h, v in zip(header, row) if h and v
                )
                chunk = Chunk(
                    chunk_id=f"chunk_{chunk_counter}",
                    document_id=document_id,
                    page_start=page_num,
                    page_end=page_num,
                    section=None,
                    subsection=None,
                    source_type="table",
                    table_id=table_id,
                    text=row_text
                )
                chunks.append(chunk)
                chunk_counter += 1

        # 2. Process text – split by paragraphs (double newline)
        if text.strip():
            paragraphs = re.split(r"\n\s*\n", text)
            for para in paragraphs:
                para = normalize_text(para)
                if not para:
                    continue
                words = para.split()
                if len(words) <= config.chunk_size:
                    chunk = Chunk(
                        chunk_id=f"chunk_{chunk_counter}",
                        document_id=document_id,
                        page_start=page_num,
                        page_end=page_num,
                        section=None,
                        subsection=None,
                        source_type="text",
                        table_id=None,
                        text=para
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                else:
                    # Sliding window within paragraph
                    step = config.chunk_size - config.chunk_overlap
                    for start in range(0, len(words), step):
                        piece = words[start:start+config.chunk_size]
                        if not piece:
                            break
                        text_chunk = " ".join(piece)
                        chunk = Chunk(
                            chunk_id=f"chunk_{chunk_counter}",
                            document_id=document_id,
                            page_start=page_num,
                            page_end=page_num,
                            section=None,
                            subsection=None,
                            source_type="text",
                            table_id=None,
                            text=text_chunk
                        )
                        chunks.append(chunk)
                        chunk_counter += 1

    return chunks