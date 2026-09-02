import json
import re
import numpy as np
import pdfplumber
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CID_RE = re.compile(r"\(cid:\d+\)")

RELEVANCE_THRESHOLD = 0.30
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
PDF_FILE = "Titan AR 2026_0.pdf"
DEFAULT_TOP_K = 5


def _clean(text):
    return CID_RE.sub("", text).strip()


def _chunk_words(text, page_num, chunk_id_start, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Fixed-size, overlapping word chunks (spec section 2)."""
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


def _chunk_table(table, page_num, chunk_id_start):

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


def extract_and_chunk_pdf(pdf_path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    chunk_id = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                text_chunks, chunk_id = _chunk_words(
                    _clean(text), page_num, chunk_id, chunk_size, overlap
                )
                chunks.extend(text_chunks)

            for table in (page.extract_tables() or []):
                table_chunks, chunk_id = _chunk_table(table, page_num, chunk_id)
                chunks.extend(table_chunks)

    return chunks


def build_vector_store(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]

    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=len(texts) > 20)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return embedder, index


def run_rag_pipeline(pdf_path, queries, top_k=4):
    chunks = extract_and_chunk_pdf(pdf_path)
    embedder, index = build_vector_store(chunks)

    print("Loading google/flan-t5-base ...")
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = []

    for query in queries:
        q_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        faiss.normalize_L2(q_embedding)

        scores, indices = index.search(q_embedding, top_k)

        retrieved_chunks_info = []
        context_texts = []
        seen = set()

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            chunk = chunks[idx]
            key = (chunk["page"], chunk["text"])
            if key in seen:
                continue
            seen.add(key)
            context_texts.append(f"[Page {chunk['page']}] {chunk['text']}")
            retrieved_chunks_info.append({
                "page": chunk["page"],
                "score": float(np.round(score, 2)),
            })

        best_score = retrieved_chunks_info[0]["score"] if retrieved_chunks_info else 0.0

        if best_score < RELEVANCE_THRESHOLD:
            answer = "I don't know from the document."
        else:
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

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if not answer:
                answer = "I don't know from the document."

        results.append({
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks_info,
        })

    return results


if __name__ == "__main__":
    pdf_file = PDF_FILE

    try:
        with open("queries.json", "r", encoding="utf-8") as f:
            query_data = json.load(f)
            queries = [q["query"] for q in query_data]
    except Exception:
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

    output = run_rag_pipeline(pdf_file, queries)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Pipeline done. See results.json.")
