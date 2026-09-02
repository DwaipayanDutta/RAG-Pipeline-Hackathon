#!/usr/bin/env python3
"""
RAG Pipeline for Titan Company Annual Report 2025-26.
Modular, configurable, evaluation-oriented.
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pdfplumber
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Optional BM25 reranker
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_pipeline")

# ---------- Configuration ----------
CONFIG_DEFAULT = {
    "pdf_path": "Titan AR 2026_0.pdf",
    "queries_path": "queries.json",
    "output_path": "results.json",
    "index_dir": "index",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "generation_model": "google/flan-t5-base",
    "chunk_size": 400,          # words
    "chunk_overlap": 80,
    "top_k_retrieval": 15,      # before reranking
    "top_k_final": 5,
    "use_reranker": True,       # BM25 lexical reranking
    "relevance_threshold": None, # disabled; use validation instead
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_prompt_tokens": 512,
    "seed": 42,
}

@dataclass
class Chunk:
    chunk_id: int
    page: int
    source_type: str          # "text" or "table_row"
    text: str
    table_id: Optional[str] = None
    section_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# ---------- Document Fingerprint ----------
def compute_document_fingerprint(pdf_path: str, config: Dict) -> str:
    """Create a fingerprint from the PDF content and configuration."""
    # Hash the first 10k characters of PDF text to detect content changes
    # This is a simple approach; can be improved by hashing the whole file.
    try:
        with pdfplumber.open(pdf_path) as pdf:
            sample = "".join(p.page.extract_text() or "" for p in pdf.pages[:5])
        content_hash = hashlib.md5(sample.encode("utf-8")).hexdigest()
    except Exception:
        content_hash = "unknown"
    # Include configuration parameters that affect index
    config_str = f"{config['embedding_model']}_{config['chunk_size']}_{config['chunk_overlap']}"
    fingerprint = f"{content_hash}_{config_str}"
    return fingerprint

# ---------- Extraction ----------
def extract_text_and_tables(pdf_path: str) -> List[Dict]:
    """Extract page text and tables with metadata."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            pages.append({
                "page": page_num,
                "text": clean_text(text),
                "tables": tables if tables else []
            })
    return pages

def clean_text(text: str) -> str:
    """Remove CID markers, normalise whitespace, strip repeated headers/footers."""
    # Remove (cid:xxx) patterns
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Optionally remove page numbers (if they appear as standalone)
    # This is heuristic; we keep page numbers in metadata.
    return text

# ---------- Chunking ----------
def chunk_document(pages: List[Dict], config: Dict) -> List[Chunk]:
    """Create structure-aware chunks from pages."""
    all_chunks = []
    chunk_id = 0
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        tables = page_data["tables"]

        # 1. Table chunks
        for table_idx, table in enumerate(tables):
            if not table or len(table) < 2:
                continue
            header = [str(h).strip() if h else "" for h in table[0]]
            table_id = f"table_{page_num}_{table_idx}"
            for row in table[1:]:
                clean_row = [str(cell).strip() if cell else "" for cell in row]
                if not any(clean_row):
                    continue
                # Pair header:value
                pairs = [f"{h}: {v}" for h, v in zip(header, clean_row) if h]
                row_text = " | ".join(pairs)
                chunk = Chunk(
                    chunk_id=chunk_id,
                    page=page_num,
                    source_type="table_row",
                    text=row_text,
                    table_id=table_id,
                )
                all_chunks.append(chunk)
                chunk_id += 1

        # 2. Text chunks: split on paragraphs (double newline) to preserve sections
        # If no paragraphs, fallback to word-based sliding window.
        if text.strip():
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if paragraphs:
                for para in paragraphs:
                    # If paragraph is very long, split further by sentences or fixed size.
                    words = para.split()
                    if len(words) <= config["chunk_size"]:
                        chunk = Chunk(
                            chunk_id=chunk_id,
                            page=page_num,
                            source_type="text",
                            text=" ".join(words),
                        )
                        all_chunks.append(chunk)
                        chunk_id += 1
                    else:
                        # Sliding window over this paragraph
                        step = config["chunk_size"] - config["chunk_overlap"]
                        for start in range(0, len(words), step):
                            piece = words[start:start+config["chunk_size"]]
                            if not piece:
                                break
                            chunk = Chunk(
                                chunk_id=chunk_id,
                                page=page_num,
                                source_type="text",
                                text=" ".join(piece),
                            )
                            all_chunks.append(chunk)
                            chunk_id += 1
            else:
                # Fallback: slide over whole page text
                words = text.split()
                step = config["chunk_size"] - config["chunk_overlap"]
                for start in range(0, len(words), step):
                    piece = words[start:start+config["chunk_size"]]
                    if not piece:
                        break
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        page=page_num,
                        source_type="text",
                        text=" ".join(piece),
                    )
                    all_chunks.append(chunk)
                    chunk_id += 1

    logger.info(f"Created {len(all_chunks)} chunks")
    return all_chunks

# ---------- Index Management ----------
def build_or_load_index(chunks: List[Chunk], config: Dict) -> Tuple[SentenceTransformer, faiss.Index, List[Chunk]]:
    """Build FAISS index or load if fingerprint matches."""
    index_dir = Path(config["index_dir"])
    index_dir.mkdir(exist_ok=True)

    fingerprint = compute_document_fingerprint(config["pdf_path"], config)
    index_path = index_dir / f"index_{fingerprint}.faiss"
    meta_path = index_dir / f"metadata_{fingerprint}.json"

    embedder = SentenceTransformer(config["embedding_model"])
    embedder.to(config["device"])

    # If index exists and metadata matches, load
    if index_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("fingerprint") == fingerprint and meta.get("num_chunks") == len(chunks):
            logger.info(f"Loading existing index from {index_path}")
            index = faiss.read_index(str(index_path))
            return embedder, index, chunks

    # Build new index
    logger.info("Building new index...")
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(index_path))
    metadata = {
        "fingerprint": fingerprint,
        "num_chunks": len(chunks),
        "embedding_model": config["embedding_model"],
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "created": str(Path(config["pdf_path"]).stat().st_mtime),  # approximate
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Index saved to {index_path}")
    return embedder, index, chunks

# ---------- Retrieval ----------
def retrieve(query: str, embedder: SentenceTransformer, index: faiss.Index,
             chunks: List[Chunk], config: Dict) -> List[Tuple[Chunk, float]]:
    """Retrieve top-k candidates, then optionally rerank with BM25."""
    # Normalise query (optional)
    query_clean = query.strip()

    # Dense retrieval
    q_emb = embedder.encode([query_clean], normalize_embeddings=True, convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, config["top_k_retrieval"])

    candidates = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        candidates.append((chunks[idx], float(score)))

    # Deduplicate by (page, text)
    seen = set()
    unique = []
    for chunk, score in candidates:
        key = (chunk.page, chunk.text)
        if key not in seen:
            seen.add(key)
            unique.append((chunk, score))

    # Optional lexical reranking using BM25
    if config.get("use_reranker", True) and BM25Okapi is not None:
        # Use chunk texts as corpus
        corpus = [c.text for c, _ in unique]
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = query_clean.split()
        bm25_scores = bm25.get_scores(query_tokens)
        # Combine scores (weighted average)
        # Normalise dense scores to [0,1] and BM25 scores to [0,1]
        if unique:
            dense_scores = np.array([s for _, s in unique])
            min_d, max_d = dense_scores.min(), dense_scores.max()
            if max_d > min_d:
                dense_norm = (dense_scores - min_d) / (max_d - min_d)
            else:
                dense_norm = np.ones_like(dense_scores)
            bm25_norm = bm25_scores / max(bm25_scores.max(), 1e-6)
            # Combine: 0.7 * dense + 0.3 * bm25
            combined = 0.7 * dense_norm + 0.3 * bm25_norm
            # Resort
            sorted_idx = np.argsort(-combined)
            reranked = [unique[i] for i in sorted_idx[:config["top_k_final"]]]
            return reranked

    # Fallback: return top_k_final from unique
    return unique[:config["top_k_final"]]

# ---------- Prompt Building ----------
def build_prompt(query: str, retrieved: List[Tuple[Chunk, float]], config: Dict) -> str:
    """Construct a concise prompt with context and clear instructions."""
    context_parts = []
    for chunk, score in retrieved:
        context_parts.append(f"[Page {chunk.page}] {chunk.text}")
    context = "\n---\n".join(context_parts)

    prompt = f"""You are an AI assistant. Your task is to answer the user's question using ONLY the provided context.
The context is extracted from an annual report. It may contain tables, financial numbers, and business information.
If the context does not contain enough information to answer, respond exactly with: "I don't know from the document."

Context:
{context}

Question: {query}

Answer:"""
    # Truncate to max_tokens (approx)
    # Simple truncation by tokens (may be improved with a tokenizer)
    max_tokens = config.get("max_prompt_tokens", 512)
    # Estimate tokens ~ words * 1.3
    prompt_words = prompt.split()
    if len(prompt_words) > max_tokens * 0.8:
        # Truncate context part (keep question and instructions)
        # We'll keep first and last part of context
        # This is simplistic; in production use tokenizer.
        pass
    return prompt

# ---------- Generation ----------
def generate_answer(prompt: str, config: Dict) -> str:
    """Generate answer using the configured model."""
    model_name = config["generation_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(config["device"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(config["device"]) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer

# ---------- Validation ----------
def validate_answer(answer: str, retrieved: List[Tuple[Chunk, float]], config: Dict) -> Tuple[str, bool]:
    """Check if answer is grounded in retrieved context. If unsupported, refuse."""
    # Simple validation: if answer contains numbers, ensure they appear in context.
    # For exact numbers, we could do a regex search.
    # This is a basic check; more robust validation would parse numbers.
    numbers = re.findall(r"\b\d+\.?\d*\b", answer)
    if numbers:
        context_text = " ".join([chunk.text for chunk, _ in retrieved])
        for num in numbers:
            if num not in context_text:
                logger.warning(f"Number {num} not found in context. Refusing.")
                return "I don't know from the document.", False
    # Also ensure the answer is not the refusal phrase already.
    if "I don't know" in answer:
        return answer, True
    return answer, True

# ---------- Main Evaluation ----------
def run_evaluation(config: Dict) -> List[Dict]:
    """Orchestrate the entire RAG pipeline."""
    # Extract
    pages = extract_text_and_tables(config["pdf_path"])
    chunks = chunk_document(pages, config)

    # Index
    embedder, index, _ = build_or_load_index(chunks, config)

    # Load queries
    with open(config["queries_path"], "r", encoding="utf-8") as f:
        queries_data = json.load(f)
    queries = [q["query"] for q in queries_data]

    results = []
    for q_idx, query in enumerate(queries, start=1):
        logger.info(f"Processing query {q_idx}: {query[:50]}...")
        # Retrieve
        retrieved = retrieve(query, embedder, index, chunks, config)
        # Build prompt
        prompt = build_prompt(query, retrieved, config)
        # Generate
        answer = generate_answer(prompt, config)
        # Validate
        final_answer, grounded = validate_answer(answer, retrieved, config)

        # Build output
        retrieved_info = [
            {
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "source_type": chunk.source_type,
                "score": round(score, 3),
            }
            for chunk, score in retrieved
        ]
        results.append({
            "id": q_idx,
            "query": query,
            "answer": final_answer,
            "retrieved_chunks": retrieved_info,
            "grounded": grounded,
        })

    return results

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Titan RAG Pipeline")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--pdf", type=str, default=CONFIG_DEFAULT["pdf_path"])
    parser.add_argument("--queries", type=str, default=CONFIG_DEFAULT["queries_path"])
    parser.add_argument("--output", type=str, default=CONFIG_DEFAULT["output_path"])
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild index")
    args = parser.parse_args()

    config = CONFIG_DEFAULT.copy()
    if args.config:
        with open(args.config, "r") as f:
            config.update(json.load(f))
    config["pdf_path"] = args.pdf
    config["queries_path"] = args.queries
    config["output_path"] = args.output

    # If rebuild, remove index dir
    if args.rebuild_index:
        import shutil
        index_dir = Path(config["index_dir"])
        if index_dir.exists():
            shutil.rmtree(index_dir)

    results = run_evaluation(config)
    with open(config["output_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {config['output_path']}")

if __name__ == "__main__":
    main()