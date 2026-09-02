# Titan Annual Report RAG Pipeline

An evaluation-oriented Retrieval-Augmented Generation (RAG) pipeline for answering questions from the **Titan Company Limited Integrated Annual Report 2025–26**.

The project demonstrates the core engineering concerns of document question answering:

- PDF text and table extraction
- Structure-aware chunking with source metadata
- Semantic retrieval with Sentence Transformers and FAISS
- Lexical reranking with BM25
- Evidence-constrained generation with FLAN-T5
- Lightweight validation of generated numerical claims
- Reproducible evaluation over a fixed question set
- Explicit refusal when the document does not contain enough evidence

> **Project status:** This is an evaluation and portfolio implementation, not a production-ready enterprise RAG platform. Production use would require stronger observability, security, testing, model governance, latency and cost controls, and claim-level grounding evaluation.

## Problem statement

The pipeline answers a fixed set of questions about Titan's FY2025–26 annual report. The evaluation set covers several common RAG challenges:

| Question type | Example capability |
| --- | --- |
| Semantic and causal | Explain what drove Jewellery Division growth |
| Financial tables | Extract KPIs for FY2025–26 and FY2024–25 |
| Numerical filtering | Identify businesses with double-digit growth |
| Threshold comparisons | Find businesses above INR 5,000 crore |
| Cross-section reasoning | Combine evidence from multiple report sections |
| Strategy retrieval | Find initiatives and programmes |
| Out-of-document handling | Refuse unsupported FY2027 questions |

When the report does not provide sufficient evidence, the expected response is:

```text
I don't know from the document.
```

## Architecture

```text
Titan annual report PDF
        |
        v
PDF extraction with pdfplumber
        |
        v
Text cleaning and table conversion
        |
        v
Structure-aware page and paragraph chunks
        |
        +--> Sentence Transformer embeddings
        |          |
        |          v
        |     FAISS cosine-style index
        |
        +--> BM25 lexical index
                   |
                   v
        Candidate fusion and reranking
                   |
                   v
        Evidence context and question
                   |
                   v
        FLAN-T5 grounded generation
                   |
                   v
        Number validation and refusal logic
                   |
                   v
        results.json with answers and evidence
```

## How retrieval works

The pipeline uses a two-stage retrieval strategy:

1. Embed the question with `sentence-transformers/all-MiniLM-L6-v2`.
2. Retrieve the top 15 candidates from a normalized `FAISS IndexFlatIP` index.
3. Remove duplicate page and text entries.
4. Rerank the dense candidates with BM25 when `rank-bm25` is installed.
5. Combine normalized scores using a 70% dense and 30% lexical weighting.
6. Pass the top five chunks to the generation step.

Dense retrieval helps with paraphrased questions, while BM25 improves matching for exact business names, financial terms, fiscal years, percentages, and values.

## Document processing

The report is processed page by page. Each extracted chunk retains traceability metadata:

```json
{
  "chunk_id": 42,
  "page": 137,
  "source_type": "table_row",
  "table_id": "table_137_1",
  "text": "Revenue: INR ..."
}
```

Text chunks use approximately:

| Setting | Default |
| --- | ---: |
| Text chunk size | 400 words |
| Chunk overlap | 80 words |
| Dense retrieval candidates | 15 |
| Final context chunks | 5 |

Tables are represented as row-level chunks in the form:

```text
Header: Value | Header: Value | ...
```

This preserves useful financial values and source-page information, while the limitations below describe where table representation can be improved.

## Generation and grounding

Generation uses:

```text
google/flan-t5-base
```

The model is instructed to answer only from the retrieved context and to return the refusal phrase when the context is insufficient. Sampling is disabled for deterministic generation:

```text
do_sample=False
```

After generation, the validation layer extracts numerical claims and checks whether the values appear in the retrieved evidence. Answers containing unsupported numbers are rejected and replaced with:

```text
I don't know from the document.
```

This is a useful guardrail, but it is not a complete factuality validator. Numeric presence alone does not prove that a value belongs to the correct entity, year, unit, or calculation.

## Project structure

```text
Titan-RAG/
├── rag_pipeline.py
├── queries.json
├── results.json
├── requirements.txt
├── README.md
├── Titan AR 2026_0.pdf
└── index/
    ├── index_<fingerprint>.faiss
    └── metadata_<fingerprint>.json
```

The main pipeline is intentionally kept in one Python file so that the hackathon implementation is easy to run and inspect. A production implementation should separate ingestion, retrieval, generation, validation, and evaluation into modules.

## Requirements

- Python 3.9 or newer
- The Titan annual report PDF
- Sufficient disk space for downloaded model weights and the generated FAISS index

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The current dependency set includes:

```text
pdfplumber
sentence-transformers
faiss-cpu
torch
transformers
numpy
rank-bm25
pytest
```

## Running the pipeline

Run with the default PDF and evaluation questions:

```bash
python rag_pipeline.py
```

Provide custom input and output paths:

```bash
python rag_pipeline.py \
  --pdf "Titan AR 2026_0.pdf" \
  --queries queries.json \
  --output results.json
```

Force the retrieval index to be rebuilt:

```bash
python rag_pipeline.py --rebuild-index
```

Provide optional configuration through JSON:

```bash
python rag_pipeline.py --config config.json
```

## Configuration

The main configuration values are:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `chunk_size` | `400` | Text chunk size in words |
| `chunk_overlap` | `80` | Overlap between text chunks |
| `top_k_retrieval` | `15` | Number of dense candidates |
| `top_k_final` | `5` | Number of evidence chunks used for generation |
| `use_reranker` | `true` | Enable BM25 reranking |
| `relevance_threshold` | `null` | Optional relevance cutoff |
| `max_prompt_tokens` | `512` | Intended prompt budget |
| `seed` | `42` | Reserved deterministic seed |
| `embedding_model` | MiniLM-L6-v2 | Sentence embedding model |
| `generation_model` | FLAN-T5-base | Text generation model |

## Output format

The pipeline writes structured results to `results.json`:

```json
[
  {
    "id": 1,
    "query": "What were the key factors that contributed to the growth of Titan's Jewellery Division during FY2025-26?",
    "answer": "...",
    "retrieved_chunks": [
      {
        "page": 33,
        "chunk_id": 42,
        "source_type": "text",
        "score": 0.812
      }
    ],
    "grounded": true
  }
]
```

The evidence metadata makes it possible to audit answers, debug retrieval, and compare pipeline versions.

## Evaluation principles

Keep `queries.json` unchanged when comparing pipeline versions. Separating evaluation questions from pipeline code helps ensure that improvements are measured against the same test set.

The current output format is useful for manual inspection. A fuller evaluation harness should add:

### Retrieval metrics

- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- nDCG

### Generation metrics

- Answer correctness
- Faithfulness or groundedness
- Completeness

### Safety metrics

- Out-of-document refusal accuracy
- Unsupported-number rate
- Unsupported-claim rate

## Known limitations

The current implementation is intentionally lightweight. The main limitations are:

1. **The document fingerprint is partial.** It samples early PDF content rather than hashing the complete file.
2. **Cleaning can remove structure.** Normalizing whitespace before chunking can weaken paragraph detection.
3. **Table context is incomplete.** Row chunks may omit table titles, section names, units, multi-row headers, and nearby explanations.
4. **BM25 cannot recover dense misses.** Lexical reranking currently runs only over dense candidates.
5. **Score weighting is query-relative.** Dense and BM25 scores are normalized within the current candidate set.
6. **Numerical reasoning is implicit.** Retrieval and generation do not guarantee correct unit normalization, filtering, or arithmetic.
7. **Grounding validation is primarily numeric.** Qualitative claims and entity-year relationships are not fully checked.
8. **Prompt budgeting is approximate.** A robust implementation should budget context using the selected model tokenizer.
9. **Model loading can be inefficient.** The generation model should be loaded once and reused across questions.
10. **Reproducibility metadata is incomplete.** Results should record full document hashes, model versions, library versions, and configuration.
11. **`grounded` is a coarse status.** It should eventually distinguish retrieval support, answer validation, full grounding, and refusal.
12. **Reference answers are missing.** Without golden answers, answer correctness and regression detection remain limited.

## Development roadmap

### Phase 1: Baseline

- PDF extraction
- Text and table chunking
- MiniLM embeddings
- FAISS retrieval
- BM25 reranking
- FLAN-T5 generation
- Basic numerical validation

### Phase 2: Retrieval optimization

- Full-file SHA-256 fingerprints
- Better structural chunking
- Context-rich table representations
- Independent dense and BM25 retrieval
- Reciprocal Rank Fusion
- Query-type routing

### Phase 3: Financial reasoning

- Entity and number extraction
- Fiscal-year normalization
- Unit normalization
- Threshold filtering
- Arithmetic validation

### Phase 4: Grounding

- Claim extraction
- Evidence-to-claim mapping
- Entailment validation
- Unsupported-claim detection
- More precise refusal logic

### Phase 5: Evaluation

- Golden answers
- Retrieval metrics
- Faithfulness and completeness scoring
- Refusal accuracy
- Baseline-versus-improved regression tests

### Phase 6: Productionization

- Modular package structure
- API layer
- Observability
- Performance testing
- Security controls
- Model and version tracking
- Model governance

## Engineering philosophy

This project avoids adding frameworks such as LangChain, LangGraph, MCP, or multi-agent orchestration without a demonstrated need. For a single annual report, the highest-value improvements are:

```text
retrieval recall
→ ranking precision
→ numerical reasoning
→ grounding
→ evaluation
→ reproducibility
```

## License

MIT License.

## Author

**Dwaipayan Dutta**

[GitHub repository](https://github.com/DwaipayanDutta/RAG-Pipeline-Hackathon)