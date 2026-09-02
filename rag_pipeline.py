
---

```python
import json
import re
import numpy as np
import pdfplumber
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# 
CONFIG = {
    "pdf_file": "Titan AR 2026_0.pdf",
    "queries_file": "queries.json",
    "output_file": "results.json",
    "chunk_size": 300,
    "chunk_overlap": 60,
    "relevance_threshold": 0.30,
    "top_k": 5,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "google/flan-t5-base",
    "faiss_index_path": "rag_index.faiss",
    "metadata_path": "rag_metadata.json",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CID_RE = re.compile(r"\(cid:\d+\)")


def clean_text(text: str) -> str:
    """Remove CID markers from PDF-extracted text."""
    return CID_RE.sub("", text).strip()


def chunk_words(text: str, page_num: int, chunk_id_start: int,
                chunk_size: int = 300, overlap: int = 60) -> tuple:
    """Fixed-size, overlapping word chunks."""
    words = text.split()
    chunks = []
    if not words:
        return chunks, chunk_id_start

    step = max(chunk_size - overlap, 1)
    chunk_id = chunk_id_start
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if not piece:
            continue
        chunks.append({
            "chunk_id": chunk_id,
            "page": page_num,
            "text": " ".join(piece),
            "type": "text",
        })
        chunk_id += 1
        if start + chunk_size >= len(words):
            break
    return chunks, chunk_id


def chunk_table(table: List[List], page_num: int, chunk_id_start: int) -> tuple:
    """Convert table rows to text chunks."""
    chunks = []
    if not table or len(table) < 2:
        return chunks, chunk_id_start

    header = [str(h).strip() if h else "" for h in table[0]]
    chunk_id = chunk_id_start
    for row in table[1:]:
        clean_row = [str(cell).strip() if cell else "" for cell in row]
        if not any(clean_row):
            continue
        pairs = [f"{h}: {v}" for h, v in zip(header, clean_row) if h]
        row_text = " | ".join(pairs)
        chunks.append({
            "chunk_id": chunk_id,
            "page": page_num,
            "text": row_text,
            "type": "table_row",
        })
        chunk_id += 1
    return chunks, chunk_id


def extract_and_chunk_pdf(pdf_path: str, chunk_size: int = 300, overlap: int = 60) -> List[Dict]:
    """Extract text and tables from PDF and return chunks."""
    chunks = []
    chunk_id = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                text_chunks, chunk_id = chunk_words(
                    clean_text(text), page_num, chunk_id, chunk_size, overlap
                )
                chunks.extend(text_chunks)

            for table in (page.extract_tables() or []):
                table_chunks, chunk_id = chunk_table(table, page_num, chunk_id)
                chunks.extend(table_chunks)

    logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
    return chunks


def build_vector_store(chunks: List[Dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                       index_path: Optional[str] = None, metadata_path: Optional[str] = None):
    """Build FAISS index from chunks, with optional save/load."""
    # Try loading from disk
    if index_path and Path(index_path).exists() and metadata_path and Path(metadata_path).exists():
        logger.info(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            saved_chunks = json.load(f)
        # Verify chunks match
        if len(saved_chunks) == len(chunks):
            embedder = SentenceTransformer(model_name)
            return embedder, index

    logger.info("Building new FAISS index...")
    embedder = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]

    embeddings = embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=32, show_progress_bar=len(texts) > 20
    )
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save to disk
    if index_path and metadata_path:
        faiss.write_index(index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        logger.info(f"Saved FAISS index to {index_path} and metadata to {metadata_path}")

    return embedder, index


def deduplicate_chunks(chunks: List[Dict], top_k: int) -> List[Dict]:
    """Remove duplicate chunks based on (page, text)."""
    seen = set()
    unique = []
    for chunk in chunks:
        key = (chunk["page"], chunk["text"])
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
        if len(unique) >= top_k:
            break
    return unique


def run_rag_pipeline(pdf_path: str, queries: List[str], config: Dict = None) -> List[Dict]:
    """Main RAG pipeline."""
    if config is None:
        config = CONFIG

    # 1. Extract and chunk
    chunks = extract_and_chunk_pdf(pdf_path, config["chunk_size"], config["chunk_overlap"])

    # 2. Build vector store
    embedder, index = build_vector_store(
        chunks,
        config["embedding_model"],
        config.get("faiss_index_path"),
        config.get("metadata_path")
    )

    # 3. Load LLM
    logger.info(f"Loading {config['llm_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["llm_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["llm_model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = []

    for query in queries:
        logger.info(f"Processing query: {query[:50]}...")

        # Embed query
        q_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        faiss.normalize_L2(q_embedding)

        # Search
        scores, indices = index.search(q_embedding, config["top_k"] * 2)  # Retrieve extra for dedup

        retrieved_chunks = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            retrieved_chunks.append({
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(np.round(score, 2)),
            })

        # Deduplicate
        unique_chunks = deduplicate_chunks(retrieved_chunks, config["top_k"])

        best_score = unique_chunks[0]["score"] if unique_chunks else 0.0

        if best_score < config["relevance_threshold"]:
            answer = "I don't know from the document."
            retrieved_info = [{"page": c["page"], "score": c["score"]} for c in unique_chunks]
        else:
            context_texts = [f"[Page {c['page']}] {c['text']}" for c in unique_chunks]
            context = "\n---\n".join(context_texts)

            prompt = (
                f"Context from a document:\n{context}\n\n"
                f"Using ONLY the context above, answer the question directly and "
                f"concisely. If the context includes a table, list each relevant "
                f"item with its value. If the context does not contain the answer, "
                f"respond exactly with: I don't know from the document.\n"
                f"Question: {query}\n"
                f"Answer:"
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if not answer:
                answer = "I don't know from the document."

            retrieved_info = [{"page": c["page"], "score": c["score"]} for c in unique_chunks]

        results.append({
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_info,
        })

    return results


if __name__ == "__main__":
    # Load queries
    queries_file = CONFIG.get("queries_file", "queries.json")
    try:
        with open(queries_file, "r", encoding="utf-8") as f:
            query_data = json.load(f)
            queries = [q["query"] for q in query_data]
    except Exception:
        logger.warning(f"Could not load {queries_file}, using default queries.")
        queries = [
            "What were the key factors that contributed to the growth of Titan's Jewellery Division during FY2025-26?",
            "List the key financial performance indicators for Titan Company on a consolidated basis for FY2025-26 and FY2024-25.",
            "Which Titan business divisions recorded double-digit growth during FY2025-26, and what were their respective growth rates?",
            "What were the turnover and profit-before-tax figures reported for CaratLane for FY2025-26?",
            "Which Titan businesses had revenue or turnover exceeding INR 5,000 crore during FY2025-26?",
            "What were the major components of Tanishq's Retail Transformation programme during FY2025-26?",
            "What initiatives did Titan undertake to improve transparency and consumer confidence in diamonds?",
            "What was Titan's investment in increasing its stake in CaratLane, and what was the resulting ownership level?",
            "How did Titan combine physical retail expansion with omnichannel capabilities across its Jewellery businesses?",
            "What are the key processes and controls described in Titan's risk management framework?",
            "What was Titan Company's total revenue in FY2027?",
        ]

    output = run_rag_pipeline(CONFIG["pdf_file"], queries, CONFIG)

    with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Pipeline done. See {CONFIG['output_file']}.")