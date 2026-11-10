#!/usr/bin/env python3
"""
Production-Grade RAG Pipeline for Financial Annual Reports with Jev Gating.

This variant adds an optional Jev reranking and gating step after retrieval,
inspired by the "Jev + RAG" architecture. When enabled, retrieved candidates
are scored by TypeSafe's Jev model; those below `jev_threshold` are discarded
before generation.
"""
import argparse
import json
import hashlib
import logging
import os
import re
import pickle
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------------------------------------------------------------------------
# Jev Reranker (optional)
# -------------------------------------------------------------------------
try:
    from jev_reranker import JevReranker
    JEV_AVAILABLE = True
except ImportError:
    JEV_AVAILABLE = False
    JevReranker = None

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rag_pipeline_jev")

# -------------------------------------------------------------------------
# Data Structures
# -------------------------------------------------------------------------
@dataclass
class EvidenceRecord:
    """A structured financial piece of evidence extracted from the document."""
    entity: Optional[str]
    metric: Optional[str]
    fiscal_year: Optional[str]
    value: Optional[float]
    unit: Optional[str]
    page: int
    chunk_id: int
    source_text: str = ""
    source_type: str = "text"
    is_percentage: bool = False


@dataclass
class Chunk:
    """A single chunk with metadata and embeddings."""
    chunk_id: int
    page: int
    section: Optional[str]
    source_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    evidence_records: List[EvidenceRecord] = field(default_factory=list)


@dataclass
class QueryResult:
    query: str
    query_type: str
    answer: str
    grounded: bool
    grounding_status: str
    confidence: float
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    validation: Dict[str, bool]
    refusal_reason: Optional[str] = None
    jev_scores: Optional[List[Dict[str, Any]]] = None


# -------------------------------------------------------------------------
# Configuration (added Jev parameters)
# -------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "chunk_size": 400,
    "chunk_overlap": 80,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "dense_top_k": 20,
    "bm25_top_k": 20,
    "rrf_k": 60,
    "rerank_top_k": 10,
    "final_top_k": 5,
    "use_reranker": False,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "generation_model": "google/flan-t5-base",
    "max_context_tokens": 512,
    "max_answer_tokens": 150,
    "index_dir": "index_cache",
    # Jev-specific
    "use_jev": False,
    "jev_threshold": 0.2,          # passages below this are dropped
    "jev_api_key_env": "TYPESAFE_API_KEY",
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = json.load(f)
            config.update(user_config)
    return config


# -------------------------------------------------------------------------
# Utilities (unchanged)
# -------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def compute_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def safe_float(value_str: str) -> Optional[float]:
    if not value_str:
        return None
    cleaned = re.sub(r"[₹$,€£\s]", "", str(value_str))
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_fiscal_years(text: str) -> List[str]:
    patterns = [
        r"FY\s*(\d{4})\s*[-–]\s*(\d{2})",
        r"(\d{4})\s*[-–]\s*(\d{2})",
        r"FY\s*(\d{2})\s*[-–]\s*(\d{2})",
    ]
    years = []
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            if len(match.groups()) == 2:
                y1, y2 = match.groups()
                if len(y1) == 4:
                    y1_full = y1
                else:
                    y1_full = f"20{y1}" if int(y1) >= 20 else f"19{y1}"
                y2_full = f"{y1_full[:2]}{y2}"
                fy = f"FY{y1_full}-{y2_full[2:]}"
                years.append(fy)
    return list(set(years))


def extract_entities(text: str) -> List[str]:
    known = [
        "Titan Company", "Titan", "Jewellery", "Tanishq", "CaratLane",
        "Watches", "Eyewear", "Emerging Businesses", "Zoya", "Mia",
        "Fastrack", "Sonata", "Skinn", "Titan Eye+"
    ]
    found = []
    for ent in known:
        if re.search(rf"\b{re.escape(ent)}\b", text, re.IGNORECASE):
            found.append(ent)
    return list(set(found))


def extract_metrics(text: str) -> List[str]:
    metric_map = {
        "revenue": ["revenue", "turnover", "sales"],
        "profit_before_tax": ["profit before tax", "pbt", "profit-before-tax", "profit before taxation"],
        "profit_after_tax": ["profit after tax", "pat", "net profit"],
        "ebitda": ["ebitda"],
        "growth": ["growth", "increase", "decrease"],
        "margin": ["margin", "operating margin"],
        "investment": ["investment", "invested"],
        "stake": ["stake", "ownership"],
    }
    found = []
    text_lower = text.lower()
    for key, aliases in metric_map.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text_lower):
                found.append(key)
                break
    return found


def parse_financial_value(text: str) -> Tuple[Optional[float], Optional[str], bool]:
    text = str(text).strip()
    if not text:
        return None, None, False
    is_pct = "%" in text
    cleaned = re.sub(r"[₹$,€£]", "", text)
    match = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", cleaned)
    if not match:
        return None, None, False
    val = safe_float(match.group(1))
    if val is None:
        return None, None, False
    unit = None
    if is_pct:
        unit = "percent"
    elif "crore" in text.lower():
        unit = "INR crore"
    elif "million" in text.lower():
        unit = "INR million"
    elif "billion" in text.lower():
        unit = "INR billion"
    elif "lakh" in text.lower():
        unit = "INR lakh"
    elif "₹" in text or "rs" in text.lower():
        unit = "INR"
    return val, unit, is_pct


def parse_threshold(query: str) -> Optional[Dict[str, Any]]:
    op_pattern = r"\b(exceeding|greater than|above|more than|at least|below|less than|at most)\b"
    op_match = re.search(op_pattern, query, re.IGNORECASE)
    if not op_match:
        return None
    op = op_match.group(1).lower()
    op_map = {
        "exceeding": ">", "greater than": ">", "above": ">", "more than": ">",
        "at least": ">=", "below": "<", "less than": "<", "at most": "<=",
    }
    operator = op_map.get(op, ">")
    num_pattern = r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:%|crore|million|billion|INR|₹)?"
    num_match = re.search(num_pattern, query)
    if not num_match:
        return None
    threshold = safe_float(num_match.group(1))
    if threshold is None:
        return None
    unit = None
    if "crore" in query:
        unit = "INR crore"
    elif "%" in query:
        unit = "percent"
    elif "million" in query:
        unit = "INR million"
    elif "billion" in query:
        unit = "INR billion"
    fiscal_years = extract_fiscal_years(query)
    fy = fiscal_years[0] if fiscal_years else None
    metrics = extract_metrics(query)
    metric = metrics[0] if metrics else None
    return {
        "operator": operator, "threshold": threshold, "unit": unit,
        "fiscal_year": fy, "metric": metric,
    }


# -------------------------------------------------------------------------
# PDF Ingestion & Chunking (unchanged)
# -------------------------------------------------------------------------
def extract_pdf_content(pdf_path: str) -> Tuple[List[Dict], List[Dict]]:
    pages_text, tables = [], []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages_text.append({"page": page_num, "text": text})
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables):
                    if not table:
                        continue
                    headers = table[0] if table else []
                    rows = []
                    for row in table[1:]:
                        if any(cell is not None and str(cell).strip() for cell in row):
                            rows.append(row)
                    if rows:
                        tables.append({
                            "page": page_num,
                            "table_id": f"table_{page_num}_{table_idx}",
                            "headers": headers,
                            "rows": rows,
                            "raw": table,
                        })
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}")
        raise
    return pages_text, tables


def normalize_table(table_dict: Dict) -> List[Dict[str, Any]]:
    page = table_dict["page"]
    table_id = table_dict["table_id"]
    headers = table_dict["headers"]
    rows = table_dict["rows"]
    clean_headers = [str(h).strip() if h else "" for h in headers]

    header_fys, header_metrics = [], []
    for h in clean_headers:
        fys = extract_fiscal_years(h)
        if fys:
            header_fys.extend(fys)
        metrics = extract_metrics(h)
        if metrics:
            header_metrics.extend(metrics)

    col_mapping = []
    for col_idx, h in enumerate(clean_headers):
        fys = extract_fiscal_years(h)
        metrics = extract_metrics(h)
        if fys and metrics:
            for fy in fys:
                for met in metrics:
                    col_mapping.append((col_idx, fy, met))
        elif fys:
            for fy in fys:
                col_mapping.append((col_idx, fy, None))
        elif metrics:
            for met in metrics:
                col_mapping.append((col_idx, None, met))
        else:
            col_mapping.append((col_idx, None, None))

    if not any(cm[1] or cm[2] for cm in col_mapping):
        all_text = " ".join(str(cell) for row in rows for cell in row if cell)
        fiscal_years = extract_fiscal_years(all_text)
        structured_rows = []
        for row_idx, row in enumerate(rows):
            row_data = {}
            for col_idx, cell in enumerate(row):
                col_name = clean_headers[col_idx] if col_idx < len(clean_headers) else f"col_{col_idx}"
                row_data[col_name] = cell
            entity = None
            if row_data:
                first_key = list(row_data.keys())[0]
                entity_val = row_data.get(first_key)
                if entity_val and isinstance(entity_val, str):
                    entity = entity_val.strip()
            numeric_values = []
            for key, val in row_data.items():
                if val and isinstance(val, str):
                    num, unit, is_pct = parse_financial_value(val)
                    if num is not None:
                        numeric_values.append((key, num, val, unit, is_pct))
            structured_rows.append({
                "page": page, "table_id": table_id, "headers": clean_headers,
                "row_data": row_data, "entity": entity,
                "numeric_values": numeric_values, "fiscal_years": fiscal_years,
                "col_mapping": col_mapping, "raw_row": row,
            })
        return structured_rows

    structured_rows = []
    for row_idx, row in enumerate(rows):
        row_data = {}
        for col_idx, cell in enumerate(row):
            col_name = clean_headers[col_idx] if col_idx < len(clean_headers) else f"col_{col_idx}"
            row_data[col_name] = cell

        entity = None
        if row_data:
            first_key = list(row_data.keys())[0]
            entity_val = row_data.get(first_key)
            if entity_val and isinstance(entity_val, str):
                entity = entity_val.strip()

        numeric_values = []
        for col_idx, (_, fy, metric) in enumerate(col_mapping):
            if col_idx >= len(row):
                continue
            val_str = row[col_idx]
            if not val_str or not isinstance(val_str, str):
                continue
            num, unit, is_pct = parse_financial_value(val_str)
            if num is not None:
                if metric is None:
                    col_name = clean_headers[col_idx] if col_idx < len(clean_headers) else ""
                    metrics_in_col = extract_metrics(col_name)
                    metric = metrics_in_col[0] if metrics_in_col else None
                if fy is None:
                    col_name = clean_headers[col_idx] if col_idx < len(clean_headers) else ""
                    fys_in_col = extract_fiscal_years(col_name)
                    fy = fys_in_col[0] if fys_in_col else None
                numeric_values.append({
                    "col_idx": col_idx, "value": num, "unit": unit,
                    "is_percentage": is_pct, "fiscal_year": fy,
                    "metric": metric, "source_text": val_str,
                })

        row_text = " ".join(str(cell) for cell in row if cell)
        row_fys = extract_fiscal_years(row_text)

        structured_rows.append({
            "page": page, "table_id": table_id, "headers": clean_headers,
            "row_data": row_data, "entity": entity,
            "numeric_values": numeric_values, "fiscal_years": row_fys,
            "col_mapping": col_mapping, "raw_row": row,
        })
    return structured_rows


def chunk_text(pages_text: List[Dict], chunk_size: int = 400, overlap: int = 80) -> List[Chunk]:
    chunks = []
    chunk_id = 0
    for page_info in pages_text:
        page = page_info["page"]
        text = page_info["text"]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n"
            else:
                if current_chunk:
                    chunks.append(Chunk(
                        chunk_id=chunk_id, page=page, section=None,
                        source_type="text", text=current_chunk.strip(),
                        metadata={"page": page}
                    ))
                    chunk_id += 1
                    overlap_text = current_chunk[-overlap:] if overlap else ""
                    current_chunk = overlap_text + para + "\n"
                else:
                    current_chunk = para + "\n"
        if current_chunk:
            chunks.append(Chunk(
                chunk_id=chunk_id, page=page, section=None,
                source_type="text", text=current_chunk.strip(),
                metadata={"page": page}
            ))
            chunk_id += 1
    return chunks


def chunk_tables(tables: List[Dict], chunk_size: int = 400) -> List[Chunk]:
    chunks = []
    chunk_id = 0
    for table_dict in tables:
        structured_rows = normalize_table(table_dict)
        for row in structured_rows:
            entity = row["entity"] or "Unknown"
            fiscal_years = row["fiscal_years"] or []
            row_text = f"Table from page {row['page']}\n"
            row_text += f"Entity: {entity}\n"
            for key, val in row["row_data"].items():
                row_text += f"{key}: {val}\n"
            for nv in row["numeric_values"]:
                fy = nv.get("fiscal_year", "")
                metric = nv.get("metric", "")
                val = nv.get("value")
                unit = nv.get("unit", "")
                if fy:
                    row_text += f"{metric or 'Value'} ({fy}): {val} {unit}\n".strip()
                else:
                    row_text += f"{metric or 'Value'}: {val} {unit}\n".strip()

            chunks.append(Chunk(
                chunk_id=chunk_id, page=row["page"], section=None,
                source_type="table", text=row_text.strip(),
                metadata={
                    "page": row["page"], "table_id": row["table_id"],
                    "entity": entity, "fiscal_years": fiscal_years,
                    "headers": row["headers"], "row_data": row["row_data"],
                    "numeric_values": row["numeric_values"],
                }
            ))
            chunk_id += 1
    return chunks


# -------------------------------------------------------------------------
# Enrichment (unchanged)
# -------------------------------------------------------------------------
def enrich_chunks(chunks: List[Chunk]) -> List[Chunk]:
    for chunk in chunks:
        text = chunk.text
        entities = extract_entities(text)
        metrics = extract_metrics(text)
        fiscal_years = extract_fiscal_years(text)

        chunk.metadata["entities"] = entities
        chunk.metadata["metrics"] = metrics
        chunk.metadata["fiscal_years"] = fiscal_years

        if chunk.source_type == "text":
            evidence = []
            sentences = re.split(r"[.!?]", text)
            for sent in sentences:
                if len(sent.strip()) < 10:
                    continue
                val, unit, is_pct = parse_financial_value(sent)
                if val is not None:
                    ents = extract_entities(sent)
                    metrics2 = extract_metrics(sent)
                    fys = extract_fiscal_years(sent)
                    if ents and metrics2:
                        for ent in ents:
                            for met in metrics2:
                                for fy in (fys or [None]):
                                    evidence.append(EvidenceRecord(
                                        entity=ent, metric=met, fiscal_year=fy,
                                        value=val, unit=unit, page=chunk.page,
                                        chunk_id=chunk.chunk_id,
                                        source_text=sent.strip(),
                                        source_type="text",
                                        is_percentage=is_pct,
                                    ))
            chunk.evidence_records = evidence
        else:
            evidence = []
            meta = chunk.metadata
            entities = meta.get("entities", [])
            if meta.get("entity"):
                entities.append(meta["entity"])
            entities = list(set(entities))
            numeric_vals = meta.get("numeric_values", [])
            for nv in numeric_vals:
                val = nv.get("value")
                if val is None:
                    continue
                fy = nv.get("fiscal_year")
                metric = nv.get("metric")
                unit = nv.get("unit")
                is_pct = nv.get("is_percentage", False)
                source_text = nv.get("source_text", "")
                if not metric:
                    if meta.get("metrics"):
                        metric = meta["metrics"][0] if meta["metrics"] else None
                if not fy and meta.get("fiscal_years"):
                    fy = meta["fiscal_years"][0] if meta["fiscal_years"] else None
                for ent in entities:
                    evidence.append(EvidenceRecord(
                        entity=ent, metric=metric, fiscal_year=fy,
                        value=val, unit=unit, page=chunk.page,
                        chunk_id=chunk.chunk_id, source_text=source_text,
                        source_type="table", is_percentage=is_pct,
                    ))
            chunk.evidence_records = evidence
    return chunks


# -------------------------------------------------------------------------
# Index Management (unchanged)
# -------------------------------------------------------------------------
class IndexManager:
    def __init__(self, config: Dict[str, Any], pdf_path: str):
        self.config = config
        self.pdf_path = pdf_path
        self.index_dir = Path(config["index_dir"])
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.doc_hash = compute_sha256(pdf_path)
        self.schema_version = "2.0"
        self.model_name = config["embedding_model"]
        self.dim = None
        self.cache_prefix = f"{self.doc_hash}_{self.model_name.replace('/','_')}_{self.schema_version}"

    def _cache_path(self, suffix: str) -> Path:
        return self.index_dir / f"{self.cache_prefix}_{suffix}"

    def save(self, chunks: List[Chunk], index: faiss.Index, bm25: BM25Okapi, tokenized_corpus: List[List[str]]):
        faiss_path = self._cache_path("faiss.index")
        faiss.write_index(index, str(faiss_path))
        bm25_path = self._cache_path("bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
        corpus_path = self._cache_path("corpus.pkl")
        with open(corpus_path, "wb") as f:
            pickle.dump(tokenized_corpus, f)
        meta = {
            "chunks": [
                {
                    "chunk_id": c.chunk_id, "page": c.page, "section": c.section,
                    "source_type": c.source_type, "text": c.text,
                    "metadata": c.metadata,
                    "evidence_records": [asdict(e) for e in c.evidence_records],
                }
                for c in chunks
            ],
            "doc_hash": self.doc_hash, "model_name": self.model_name,
            "schema_version": self.schema_version,
            "num_chunks": len(chunks), "dim": self.dim,
        }
        meta_path = self._cache_path("meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved index to {self.index_dir}")

    def load(self) -> Tuple[List[Chunk], faiss.Index, BM25Okapi, List[List[str]]]:
        meta_path = self._cache_path("meta.json")
        if not meta_path.exists():
            return None, None, None, None
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if (meta.get("doc_hash") != self.doc_hash or
            meta.get("model_name") != self.model_name or
            meta.get("schema_version") != self.schema_version):
            logger.warning("Index mismatch or outdated. Rebuilding.")
            return None, None, None, None
        faiss_path = self._cache_path("faiss.index")
        if not faiss_path.exists():
            return None, None, None, None
        index = faiss.read_index(str(faiss_path))
        bm25_path = self._cache_path("bm25.pkl")
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        corpus_path = self._cache_path("corpus.pkl")
        tokenized_corpus = []
        if corpus_path.exists():
            with open(corpus_path, "rb") as f:
                tokenized_corpus = pickle.load(f)
        chunks = []
        for cdata in meta["chunks"]:
            evidence = [EvidenceRecord(**e) for e in cdata.get("evidence_records", [])]
            chunk = Chunk(
                chunk_id=cdata["chunk_id"], page=cdata["page"],
                section=cdata.get("section"), source_type=cdata["source_type"],
                text=cdata["text"], metadata=cdata.get("metadata", {}),
                evidence_records=evidence,
            )
            chunks.append(chunk)
        self.dim = meta.get("dim", index.d)
        logger.info(f"Loaded index from {self.index_dir} ({len(chunks)} chunks)")
        return chunks, index, bm25, tokenized_corpus


# -------------------------------------------------------------------------
# Retriever (with optional Jev gating)
# -------------------------------------------------------------------------
class Retriever:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(config["embedding_model"], device=self.device)
        self.dim = self.embedding_model.get_sentence_embedding_dimension()
        self.reranker = None
        if config.get("use_reranker", False):
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(config["reranker_model"], device=self.device)
                logger.info(f"Loaded reranker {config['reranker_model']}")
            except Exception as e:
                logger.warning(f"Reranker not available: {e}")

        self.jev_reranker = None
        if config.get("use_jev", False) and JEV_AVAILABLE:
            api_key = os.getenv(config.get("jev_api_key_env", "TYPESAFE_API_KEY"))
            if api_key:
                try:
                    self.jev_reranker = JevReranker(api_key=api_key)
                    logger.info("Jev reranker initialized.")
                except Exception as e:
                    logger.warning(f"Failed to initialize Jev reranker: {e}")
            else:
                logger.warning("Jev enabled but TYPESAFE_API_KEY not set. Jev gating skipped.")
        elif config.get("use_jev", False) and not JEV_AVAILABLE:
            logger.warning("jev-reranker package not installed. Install with: pip install jev-reranker")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.embedding_model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)
        return index

    def dense_retrieve(self, query: str, index: faiss.Index, chunks: List[Chunk], top_k: int) -> List[Tuple[int, float]]:
        q_emb = self.encode([query])
        scores, indices = index.search(q_emb, top_k)
        return [(int(idx), float(scores[0][i])) for i, idx in enumerate(indices[0]) if idx != -1]

    def bm25_retrieve(self, query: str, bm25: BM25Okapi, tokenized_corpus: List[List[str]], top_k: int) -> List[Tuple[int, float]]:
        tokenized_query = normalize_text(query).split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def rrf(self, dense_results: List[Tuple[int, float]], bm25_results: List[Tuple[int, float]], k: int = 60) -> List[Tuple[int, float]]:
        scores = defaultdict(float)
        for rank, (idx, _) in enumerate(dense_results, start=1):
            scores[idx] += 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(bm25_results, start=1):
            scores[idx] += 1.0 / (k + rank)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_items]

    def rerank(self, query: str, chunk_texts: List[str], chunk_ids: List[int]) -> List[Tuple[int, float]]:
        if self.reranker is None:
            return [(cid, 0.0) for cid in chunk_ids]
        pairs = [(query, text) for text in chunk_texts]
        scores = self.reranker.predict(pairs)
        sorted_pairs = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return sorted_pairs

    def jev_gate(self, query: str, chunks: List[Chunk]) -> Tuple[List[Chunk], List[Dict[str, Any]]]:
        """
        Apply Jev reranking and gating. Returns filtered chunks and Jev scores.
        If Jev is unavailable or fails, returns the original chunks with empty scores.
        """
        if not self.jev_reranker or not chunks:
            return chunks, []

        texts = [c.text for c in chunks]
        try:
            # relevance_rerank returns a dict with "results" list containing
            # "text", "score", "document_index", etc.
            result = self.jev_reranker.relevance_rerank(
                query=query,
                documents=texts,
                threshold=self.config.get("jev_threshold", 0.2),
            )
            kept_indices = [item["document_index"] for item in result.get("results", [])]
            kept_chunks = [chunks[i] for i in kept_indices]
            jev_scores = [
                {"chunk_id": chunks[i].chunk_id, "score": item["score"], "index": i}
                for item in result.get("results", [])
                for i in [item["document_index"]]
            ]
            logger.info(f"Jev kept {len(kept_chunks)}/{len(chunks)} chunks.")
            return kept_chunks, jev_scores
        except Exception as e:
            logger.error(f"Jev gating failed: {e}. Falling back to original chunks.")
            return chunks, []


# -------------------------------------------------------------------------
# Query Classification & Reasoning (unchanged)
# -------------------------------------------------------------------------
def classify_query(query: str) -> str:
    q_lower = query.lower()
    if re.search(r"\b(exceeding|greater than|above|more than|at least|below|less than|at most)\b", q_lower):
        return "THRESHOLD"
    if re.search(r"\b(compare|vs|versus|compared to)\b", q_lower):
        return "COMPARISON"
    if re.search(r"\b(highest|lowest|largest|smallest|top|bottom|rank|ranking)\b", q_lower):
        return "RANKING"
    if re.search(r"\b(sum|total|average|mean|maximum|minimum|aggregate)\b", q_lower):
        return "AGGREGATION"
    if re.search(r"\b(double.?digit growth)\b", q_lower):
        return "THRESHOLD"
    if re.search(r"\b\d+\b", q_lower):
        return "NUMERIC"
    metrics = extract_metrics(q_lower)
    if metrics:
        return "NUMERIC"
    return "FACTUAL"


def extract_structured_evidence_from_chunks(chunks: List[Chunk]) -> List[EvidenceRecord]:
    evidence = []
    for chunk in chunks:
        evidence.extend(chunk.evidence_records)
    return evidence


def handle_threshold_query(query: str, chunks: List[Chunk]) -> Tuple[Optional[str], List[Dict], List[EvidenceRecord]]:
    parsed = parse_threshold(query)
    if not parsed:
        return None, [], []
    operator, threshold, unit, fiscal_year, metric = (
        parsed["operator"], parsed["threshold"], parsed["unit"],
        parsed["fiscal_year"], parsed["metric"]
    )
    evidence = extract_structured_evidence_from_chunks(chunks)

    if "double-digit growth" in query.lower() or "double digit growth" in query.lower():
        filtered = [
            rec for rec in evidence
            if rec.metric == "growth" and rec.is_percentage and rec.value is not None and rec.value >= 10.0
        ]
        if not filtered:
            return "No entities with double-digit growth found.", [], []
        lines = [f"{rec.entity}: {rec.value}% (FY{rec.fiscal_year})" for rec in filtered]
        citations = [{"page": rec.page, "chunk_id": rec.chunk_id, "source_type": rec.source_type} for rec in filtered]
        return "\n".join(lines), citations, filtered

    filtered = []
    for rec in evidence:
        if rec.value is None:
            continue
        if fiscal_year and rec.fiscal_year != fiscal_year:
            continue
        if metric and rec.metric != metric:
            continue
        if unit and rec.unit != unit:
            continue
        if operator == ">" and rec.value > threshold:
            filtered.append(rec)
        elif operator == ">=" and rec.value >= threshold:
            filtered.append(rec)
        elif operator == "<" and rec.value < threshold:
            filtered.append(rec)
        elif operator == "<=" and rec.value <= threshold:
            filtered.append(rec)

    if not filtered:
        return "No entities met the threshold.", [], []
    lines = []
    for rec in filtered:
        unit_str = f" {rec.unit}" if rec.unit else ""
        fy_str = f" (FY{rec.fiscal_year})" if rec.fiscal_year else ""
        lines.append(f"{rec.entity} ({rec.metric}): {rec.value}{unit_str}{fy_str}")
    citations = [{"page": rec.page, "chunk_id": rec.chunk_id, "source_type": rec.source_type} for rec in filtered]
    return "\n".join(lines), citations, filtered


def handle_comparison_query(query: str, chunks: List[Chunk]) -> Tuple[Optional[str], List[Dict], List[EvidenceRecord]]:
    fiscal_years = extract_fiscal_years(query)
    if len(fiscal_years) < 2:
        return None, [], []
    fy1, fy2 = fiscal_years[:2]
    evidence = extract_structured_evidence_from_chunks(chunks)
    grouped = defaultdict(lambda: {})
    for rec in evidence:
        if rec.entity and rec.metric and rec.fiscal_year in (fy1, fy2):
            grouped[(rec.entity, rec.metric)][rec.fiscal_year] = rec.value
    lines, used_evidence = [], []
    for (entity, metric), values in grouped.items():
        val1, val2 = values.get(fy1), values.get(fy2)
        if val1 is not None and val2 is not None:
            diff = val1 - val2
            if val2 != 0:
                pct = (diff / val2) * 100
                lines.append(f"{entity} {metric}: {fy1} = {val1}, {fy2} = {val2}, change = {diff} ({pct:.1f}%)")
            else:
                lines.append(f"{entity} {metric}: {fy1} = {val1}, {fy2} = {val2}, change = {diff}")
            used_evidence.extend([r for r in evidence if r.entity == entity and r.metric == metric and r.fiscal_year in (fy1, fy2)])
    if not lines:
        return None, [], []
    citations = [{"page": r.page, "chunk_id": r.chunk_id} for r in used_evidence]
    return "\n".join(lines), citations, used_evidence


def handle_ranking_query(query: str, chunks: List[Chunk]) -> Tuple[Optional[str], List[Dict], List[EvidenceRecord]]:
    ascending = "lowest" in query.lower() or "smallest" in query.lower()
    metrics = extract_metrics(query)
    metric = metrics[0] if metrics else None
    fiscal_years = extract_fiscal_years(query)
    fy = fiscal_years[0] if fiscal_years else None
    evidence = extract_structured_evidence_from_chunks(chunks)
    filtered = [r for r in evidence if (not metric or r.metric == metric) and (not fy or r.fiscal_year == fy)]
    if not filtered:
        return None, [], []
    best = {}
    for r in filtered:
        if r.entity and r.metric:
            key = (r.entity, r.metric)
            if key not in best or r.value > best[key].value:
                best[key] = r
    sorted_items = sorted(best.values(), key=lambda x: x.value, reverse=not ascending)
    lines = []
    for r in sorted_items:
        unit_str = f" {r.unit}" if r.unit else ""
        fy_str = f" (FY{r.fiscal_year})" if r.fiscal_year else ""
        lines.append(f"{r.entity} ({r.metric}): {r.value}{unit_str}{fy_str}")
    citations = [{"page": r.page, "chunk_id": r.chunk_id} for r in sorted_items]
    return "\n".join(lines), citations, sorted_items


def handle_aggregation_query(query: str, chunks: List[Chunk]) -> Tuple[Optional[str], List[Dict], List[EvidenceRecord]]:
    q_lower = query.lower()
    if "sum" in q_lower or "total" in q_lower:
        op = "sum"
    elif "average" in q_lower or "mean" in q_lower:
        op = "avg"
    elif "maximum" in q_lower or "max" in q_lower:
        op = "max"
    elif "minimum" in q_lower or "min" in q_lower:
        op = "min"
    else:
        return None, [], []
    metrics = extract_metrics(query)
    metric = metrics[0] if metrics else None
    fy_list = extract_fiscal_years(query)
    fy = fy_list[0] if fy_list else None
    evidence = extract_structured_evidence_from_chunks(chunks)
    filtered = [r for r in evidence if (not metric or r.metric == metric) and (not fy or r.fiscal_year == fy)]
    if not filtered:
        return None, [], []
    values = [r.value for r in filtered if r.value is not None]
    if not values:
        return None, [], []
    if op == "sum":
        result, label = sum(values), "Sum"
    elif op == "avg":
        result, label = sum(values) / len(values), "Average"
    elif op == "max":
        result, label = max(values), "Maximum"
    elif op == "min":
        result, label = min(values), "Minimum"
    else:
        return None, [], []
    unit = filtered[0].unit or ""
    answer = f"{label}: {result} {unit}".strip()
    citations = [{"page": r.page, "chunk_id": r.chunk_id} for r in filtered]
    return answer, citations, filtered


# -------------------------------------------------------------------------
# LLM Generation (unchanged)
# -------------------------------------------------------------------------
def build_prompt(query: str, context: str) -> str:
    return f"""Answer the question using ONLY the context below.
The context is from an annual report.

Context:
{context}

Question: {query}

If the context does not contain enough information to answer, say exactly: "I don't know from the document."
Provide a concise, factual answer. Cite the page number if available.

Answer:"""


def generate_answer(query: str, chunks: List[Chunk], config: Dict[str, Any]) -> Tuple[str, List[Dict]]:
    context_parts = []
    total_len = 0
    for chunk in chunks:
        text = chunk.text
        est_len = len(text) // 4
        if total_len + est_len > config["max_context_tokens"]:
            break
        context_parts.append(text)
        total_len += est_len
    context = "\n\n".join(context_parts)
    if not context:
        return "I don't know from the document.", []
    prompt = build_prompt(query, context)
    model_name = config["generation_model"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config["max_context_tokens"] + 200)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=config["max_answer_tokens"], do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return "I don't know from the document.", []
    if "don't know" in answer.lower() or "not enough information" in answer.lower():
        return "I don't know from the document.", []
    citations = []
    page_matches = re.findall(r"Page\s*(\d+)", answer, re.IGNORECASE)
    for p in page_matches:
        citations.append({"page": int(p)})
    if not citations and chunks:
        citations.append({"page": chunks[0].page, "chunk_id": chunks[0].chunk_id})
    return answer, citations


# -------------------------------------------------------------------------
# Grounding Validation (unchanged)
# -------------------------------------------------------------------------
def validate_answer(answer: str, chunks: List[Chunk]) -> Tuple[bool, Dict[str, bool], str]:
    evidence = extract_structured_evidence_from_chunks(chunks)
    numbers = re.findall(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", answer)
    claims = []
    for num_str in numbers:
        val = safe_float(num_str)
        if val is not None:
            claims.append(val)
    if not claims:
        entities_in_answer = extract_entities(answer)
        if entities_in_answer:
            entity_found = any(any(e in ent for ent in entities_in_answer) for e in evidence if e.entity)
            if not entity_found:
                return False, {"entity_grounding": False}, "Entity not grounded"
            return True, {"entity_grounding": True}, ""
        return True, {}, ""
    validation = {"numeric_grounding": False, "entity_grounding": False, "metric_grounding": False, "fiscal_year_grounding": False}
    grounded_claims = 0
    for claim in claims:
        for rec in evidence:
            if rec.value is None:
                continue
            if abs(rec.value - claim) < 0.01:
                if rec.entity and rec.entity.lower() in answer.lower():
                    validation["entity_grounding"] = True
                if rec.metric and rec.metric.replace("_", " ") in answer.lower():
                    validation["metric_grounding"] = True
                if rec.fiscal_year and rec.fiscal_year.lower() in answer.lower():
                    validation["fiscal_year_grounding"] = True
                grounded_claims += 1
                break
    if grounded_claims == len(claims) and claims:
        validation["numeric_grounding"] = True
        all_grounded = all([
            validation["entity_grounding"] or not any(e.entity for e in evidence),
            validation["metric_grounding"] or not any(e.metric for e in evidence),
            validation["fiscal_year_grounding"] or not any(e.fiscal_year for e in evidence),
        ])
        if all_grounded:
            return True, validation, ""
        else:
            return False, validation, "Some dimensions not grounded"
    else:
        return False, validation, "Numeric claim not fully grounded"


# -------------------------------------------------------------------------
# Main Pipeline (Jev gating inserted)
# -------------------------------------------------------------------------
def run_pipeline(pdf_path: str, query: str, config: Dict[str, Any], rebuild: bool = False) -> QueryResult:
    logger.info(f"Processing query: {query}")

    pages_text, tables = extract_pdf_content(pdf_path)
    logger.info(f"Extracted {len(pages_text)} pages, {len(tables)} tables")

    text_chunks = chunk_text(pages_text, config["chunk_size"], config["chunk_overlap"])
    table_chunks = chunk_tables(tables, config["chunk_size"])
    all_chunks = text_chunks + table_chunks
    logger.info(f"Created {len(all_chunks)} chunks")

    all_chunks = enrich_chunks(all_chunks)

    index_manager = IndexManager(config, pdf_path)

    if rebuild:
        logger.info("Rebuilding index per request")
        chunks, index, bm25, tokenized_corpus = None, None, None, None
    else:
        chunks, index, bm25, tokenized_corpus = index_manager.load()

    if chunks is None:
        logger.info("Building indices from chunks")
        retriever = Retriever(config)
        texts = [c.text for c in all_chunks]
        embeddings = retriever.encode(texts)
        index = retriever.build_faiss_index(embeddings)
        tokenized_corpus = [normalize_text(c.text).split() for c in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        for i, emb in enumerate(embeddings):
            all_chunks[i].embedding = emb
        index_manager.save(all_chunks, index, bm25, tokenized_corpus)
        chunks = all_chunks
    else:
        logger.info("Loaded existing index")
        retriever = Retriever(config)

    # Retrieval
    dense_results = retriever.dense_retrieve(query, index, chunks, config["dense_top_k"])
    bm25_results = retriever.bm25_retrieve(query, bm25, tokenized_corpus, config["bm25_top_k"])
    rrf_results = retriever.rrf(dense_results, bm25_results, config["rrf_k"])

    candidate_ids = [idx for idx, _ in rrf_results[:config["rerank_top_k"]]]
    candidate_chunks = [chunks[idx] for idx in candidate_ids]

    # Optional cross-encoder reranking
    if config.get("use_reranker", False) and retriever.reranker is not None:
        chunk_texts = [c.text for c in candidate_chunks]
        reranked = retriever.rerank(query, chunk_texts, candidate_ids)
        final_ids = [idx for idx, _ in reranked[:config["final_top_k"]]]
        final_chunks = [chunks[idx] for idx in final_ids]
    else:
        final_chunks = candidate_chunks[:config["final_top_k"]]

    # --- Jev gating (new step) ---
    jev_scores = None
    if config.get("use_jev", False) and retriever.jev_reranker is not None:
        final_chunks, jev_scores = retriever.jev_gate(query, final_chunks)
        if not final_chunks:
            logger.warning("Jev filtered out all chunks. Refusing to answer.")
            return QueryResult(
                query=query, query_type="REFUSED", answer="I don't know from the document.",
                grounded=False, grounding_status="refused", confidence=0.0,
                citations=[], retrieved_chunks=[], validation={},
                refusal_reason="Jev gating removed all evidence.",
                jev_scores=jev_scores,
            )
    # -----------------------------

    query_type = classify_query(query)

    answer = None
    citations = []
    evidence_used = []
    refusal_reason = None

    if query_type == "THRESHOLD":
        answer, citations, evidence_used = handle_threshold_query(query, final_chunks)
    elif query_type == "COMPARISON":
        answer, citations, evidence_used = handle_comparison_query(query, final_chunks)
    elif query_type == "RANKING":
        answer, citations, evidence_used = handle_ranking_query(query, final_chunks)
    elif query_type == "AGGREGATION":
        answer, citations, evidence_used = handle_aggregation_query(query, final_chunks)
    else:
        answer, citations = generate_answer(query, final_chunks, config)
        if "don't know" in answer.lower():
            ans_t, cit_t, ev_t = handle_threshold_query(query, final_chunks)
            if ans_t and "No entities" not in ans_t:
                answer, citations, evidence_used = ans_t, cit_t, ev_t
            else:
                ans_r, cit_r, ev_r = handle_ranking_query(query, final_chunks)
                if ans_r:
                    answer, citations, evidence_used = ans_r, cit_r, ev_r
                else:
                    ans_a, cit_a, ev_a = handle_aggregation_query(query, final_chunks)
                    if ans_a:
                        answer, citations, evidence_used = ans_a, cit_a, ev_a

    if answer is None or "don't know" in answer.lower():
        answer = "I don't know from the document."
        grounded = False
        refusal_reason = refusal_reason or "No reasoning engine produced an answer."
    else:
        grounded, validation, refusal_reason = validate_answer(answer, final_chunks)
        if not grounded:
            answer = "I don't know from the document."
            refusal_reason = refusal_reason or "Answer not grounded in evidence."

    if grounded:
        confidence = 0.8 + 0.2 * (len(final_chunks) / config["final_top_k"])
        confidence = min(confidence, 1.0)
    else:
        confidence = 0.0

    result = QueryResult(
        query=query, query_type=query_type, answer=answer,
        grounded=grounded, grounding_status="grounded" if grounded else "refused",
        confidence=confidence, citations=citations if grounded else [],
        retrieved_chunks=[
            {"chunk_id": c.chunk_id, "page": c.page, "source_type": c.source_type, "score": 0.0}
            for c in final_chunks
        ],
        validation=validation if grounded else {},
        refusal_reason=refusal_reason, jev_scores=jev_scores,
    )
    return result


# -------------------------------------------------------------------------
# CLI (extended with --use-jev)
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Champion RAG Pipeline with optional Jev Gating")
    parser.add_argument("--pdf", default="Titan_AR_2026.pdf", help="Path to PDF file")
    parser.add_argument("--query", type=str, help="Single query string")
    parser.add_argument("--queries", type=str, help="Path to JSON file with list of queries")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of index")
    parser.add_argument("--use-jev", action="store_true", help="Enable Jev reranking and gating")
    parser.add_argument("--jev-threshold", type=float, default=0.2, help="Jev relevance threshold (default: 0.2)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.use_jev:
        config["use_jev"] = True
    if args.jev_threshold is not None:
        config["jev_threshold"] = args.jev_threshold

    queries = []
    if args.query:
        queries = [args.query]
    elif args.queries:
        with open(args.queries, "r") as f:
            queries = json.load(f)
    else:
        print("Enter your query (type 'exit' to quit):")
        while True:
            q = input("> ")
            if q.lower() in ("exit", "quit"):
                break
            queries.append(q)

    results = []
    for q in queries:
        res = run_pipeline(args.pdf, q, config, args.rebuild_index)
        results.append(asdict(res))
        logger.info(f"Query: {q} -> {res.answer[:100]}...")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()