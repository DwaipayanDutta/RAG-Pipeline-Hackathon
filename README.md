# RAG Pipeline for Titan Company Annual Report 2025–26

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that extracts, indexes, and queries information from the Titan Company Limited Integrated Annual Report 2025–26. Built entirely in Python using open-source tools.

---

## 🚀 Features

- **PDF Extraction:** Extracts both text and tables using `pdfplumber`
- **Intelligent Chunking:** Fixed-size word-based chunks with configurable overlap (300 words, 60 overlap)
- **Semantic Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` for high-quality sentence embeddings
- **Vector Search:** FAISS `IndexFlatIP` with L2 normalization for fast cosine similarity search
- **LLM Generation:** `google/flan-t5-base` for grounded, context-aware answer generation
- **Hallucination Control:** Relevance threshold to refuse answers when evidence is insufficient
- **Traceable Output:** JSON output with retrieved chunk metadata (page, score) for auditability

---

## 📁 Project Structure

```text
Titan-RAG/
├── rag_pipeline.py        # Main pipeline implementation
├── queries.json            # Input queries (optional)
├── results.json            # Generated answers (output)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── Titan AR 2026_0.pdf     # Annual report PDF
├── rag_index.faiss         # FAISS index (generated)
└── rag_metadata.json       # Chunk metadata (generated)
```

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Titan-RAG.git
cd Titan-RAG
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```bat
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```text
pdfplumber>=0.10.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
```

> **Note:** For GPU support, replace `faiss-cpu` with `faiss-gpu`.

---

## 🧠 How It Works

```text
┌─────────────────┐
│  PDF Document   │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Extract Text   │
│  & Tables       │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Chunking       │  ← Fixed-size, overlapping
└────────┬────────┘
         ▼
┌─────────────────┐
│  Embeddings     │  ← sentence-transformers
└────────┬────────┘
         ▼
┌─────────────────┐
│  FAISS Index    │  ← Vector store
└────────┬────────┘
         ▼
┌─────────────────┐
│  Query          │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Top-K Retrieval│  ← Similarity search
└────────┬────────┘
         ▼
┌─────────────────┐
│  Context Build  │
└────────┬────────┘
         ▼
┌─────────────────┐
│  LLM Generation │  ← flan-t5-base
└────────┬────────┘
         ▼
┌─────────────────┐
│  JSON Output    │
└─────────────────┘
```

---

## 📊 Usage

### Run the Pipeline

```bash
python rag_pipeline.py
```

### Using Custom Queries

Create a `queries.json` file:

```json
[
  {
    "query": "What were the key factors that contributed to the growth of Titan's Jewellery Division during FY2025-26?"
  },
  {
    "query": "List the key financial performance indicators for Titan Company on a consolidated basis for FY2025-26 and FY2024-25."
  }
]
```

### Output Format (`results.json`)

```json
[
  {
    "query": "What were the key factors...?",
    "answer": "The Jewellery Division grew by 34% due to resilient consumer demand, continued premiumisation, and strong traction across festive and wedding-led occasions.",
    "retrieved_chunks": [
      {
        "page": 33,
        "score": 0.81
      },
      {
        "page": 34,
        "score": 0.76
      }
    ]
  }
]
```

---

## ⚙️ Configuration

You can modify the following constants in `rag_pipeline.py`:

| Constant | Default | Description |
|---|---:|---|
| `CHUNK_SIZE` | `300` | Number of words per chunk |
| `CHUNK_OVERLAP` | `60` | Overlap between consecutive chunks |
| `RELEVANCE_THRESHOLD` | `0.30` | Minimum similarity score to answer |
| `DEFAULT_TOP_K` | `5` | Number of chunks retrieved per query |
| `PDF_FILE` | `"Titan AR 2026_0.pdf"` | Path to the PDF document |

---

## 🧪 Evaluation Queries

The pipeline is tested against **11 queries** covering:

- **Causal/Semantic Retrieval:** Business growth factors
- **Financial Table Retrieval:** KPI extraction from financial statements
- **Numerical Filtering:** Double-digit growth divisions
- **Threshold Queries:** Revenue > INR 5,000 crore
- **Multi-section Retrieval:** Initiatives across sections
- **Out-of-Document Test:** FY2027 revenue (should return `"I don't know from the document."`)

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Out of Memory | Reduce `CHUNK_SIZE` or use `faiss-cpu` |
| Slow embedding generation | Use GPU or reduce batch size |
| Low answer quality | Adjust `RELEVANCE_THRESHOLD` or increase `DEFAULT_TOP_K` |

---

## 📝 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📧 Contact

For questions or feedback, please reach out via GitHub Issues.
