# Titan Annual Report RAG Pipeline

An evaluation-oriented Retrieval-Augmented Generation (RAG) pipeline for answering questions from the **Titan Company Limited Integrated Annual Report 2025–26**.

The project is intentionally implemented in Python with open-source components and is designed to demonstrate the core engineering concerns of document RAG: document extraction, chunking, semantic retrieval, lexical reranking, grounded generation, evidence metadata, and reproducible evaluation.

> **Important:** This project should be considered an evaluation/portfolio RAG implementation, not a fully production-ready enterprise RAG platform. Production deployment would require additional observability, security, testing, model governance, latency/cost controls, and stronger grounding evaluation.

---

## 1. Problem Statement

The system answers a fixed evaluation set of questions against the Titan Annual Report 2025–26.

The evaluation set deliberately contains different RAG problem types:

- Semantic/causal questions
- Financial table questions
- Numerical filtering questions
- Threshold/comparison questions
- Cross-section questions
- Business-performance questions
- Out-of-document questions

A key design objective is to **avoid hallucination when the annual report does not contain sufficient evidence**.

For unsupported questions, the expected refusal is:

```text
I don't know from the document.
```

---

## 2. Current Architecture

```text
                    Titan Annual Report PDF
                              |
                              v
                    +--------------------+
                    | PDF Extraction     |
                    | pdfplumber          |
                    | Text + Tables       |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | Text Cleaning      |
                    | CID removal         |
                    | Whitespace cleanup  |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | Structure-aware     |
                    | Page / paragraph    |
                    | chunking            |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | Sentence           |
                    | Transformers       |
                    | MiniLM embeddings  |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | FAISS              |
                    | IndexFlatIP        |
                    | cosine-style       |
                    | similarity         |
                    +---------+----------+
                              |
                     User / Evaluation
                         Question
                              |
                              v
                    +--------------------+
                    | Dense Retrieval    |
                    | top 15 candidates   |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | BM25 Lexical       |
                    | Reranking          |
                    | top 5              |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | Prompt Builder     |
                    | Evidence + Query   |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | FLAN-T5-base       |
                    | Open-source LLM     |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | Grounding          |
                    | Number validation  |
                    +---------+----------+
                              |
                              v
                    +--------------------+
                    | results.json       |
                    | Answer + Evidence  |
                    +--------------------+
```

---

## 3. Key Design Decisions

### 3.1 Open-source models

The current implementation uses:

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generation: `google/flan-t5-base`

This keeps the project free from paid LLM API dependencies and makes local execution possible.

The models are configurable through the pipeline configuration.

### 3.2 Dense retrieval

Embeddings are normalized and stored in:

```text
FAISS IndexFlatIP
```

Because normalized vectors are used, inner-product similarity behaves as cosine similarity.

### 3.3 Two-stage retrieval

The pipeline retrieves a larger candidate set first:

```text
Dense retrieval: top 15
        |
        v
BM25 lexical reranking
        |
        v
Final context: top 5
```

This is useful for annual reports because queries frequently contain exact entities, financial terms, fiscal years, percentages, and business names.

### 3.4 Grounding

The generation prompt instructs the model to answer only from retrieved context.

A lightweight validation layer additionally checks numerical claims in the generated answer against retrieved evidence.

This is a useful guardrail, but it is **not a complete factuality or entailment validator**.

---

## 4. Document Processing

The PDF is processed page-by-page.

Each page retains:

```text
page number
text
tables
```

Tables are converted into row-level chunks using:

```text
Header: Value | Header: Value | ...
```

Text is chunked using approximately:

```text
Chunk size:     400 words
Chunk overlap:   80 words
```

Chunk metadata includes:

```json
{
  "chunk_id": 42,
  "page": 137,
  "source_type": "table_row",
  "table_id": "table_137_1",
  "text": "..."
}
```

This metadata makes retrieval results traceable back to the source PDF.

---

## 5. Retrieval Pipeline

### Stage 1 — Dense Retrieval

The query is embedded with the configured sentence-transformer model.

FAISS returns the top 15 candidates.

### Stage 2 — Deduplication

Duplicate `(page, text)` entries are removed.

### Stage 3 — BM25 Reranking

The dense candidates are reranked using BM25 when `rank-bm25` is installed.

The current combined score uses:

```text
70% dense similarity
30% BM25 score
```

The final five chunks are passed to the generation stage.

### Why not use BM25 alone?

Semantic retrieval helps with paraphrased questions:

```text
"What drove jewellery growth?"
```

versus:

```text
"factors contributing to growth of the Jewellery Division"
```

Lexical retrieval is particularly useful for exact terms such as:

```text
CaratLane
FY2025-26
PBT
turnover
5,000 crore
```

The combination provides a simple hybrid retrieval strategy without introducing a separate vector database or external service.

---

## 6. Generation

The current generator is:

```text
google/flan-t5-base
```

Generation is deterministic:

```text
do_sample=False
```

The prompt establishes a strict evidence boundary:

```text
Answer using ONLY the provided context.

If the context does not contain enough information:
I don't know from the document.
```

The model is not expected to use external knowledge.

---

## 7. Grounding and Validation

A lightweight post-generation validation step checks numeric claims.

For example, if the model generates:

```text
Revenue was INR 12,500 crore.
```

the validator checks whether the numeric value appears in retrieved context.

If a generated number is not found, the answer is rejected and replaced by:

```text
I don't know from the document.
```

### Current limitation

This validation is intentionally simple.

It does **not** yet prove that:

- every qualitative claim is supported;
- a number is associated with the correct company/division;
- a number belongs to the correct fiscal year;
- units are equivalent;
- a calculation is correct;
- two retrieved statements are logically consistent.

A future production implementation should use structured numerical extraction and/or an entailment-based grounding evaluator.

---

## 8. Evaluation Questions

The evaluation set is stored separately in:

```text
queries.json
```

This is preferable to hard-coding evaluation questions in Python because it separates:

```text
pipeline logic
```

from:

```text
evaluation data
```

The current evaluation set covers:

| Category | Example capability |
|---|---|
| Causal / semantic | Explain business growth |
| Financial table | Extract FY2025-26 and FY2024-25 KPIs |
| Numerical filtering | Identify double-digit growth |
| Threshold | Identify businesses above INR 5,000 crore |
| Cross-section | Combine information from multiple report sections |
| Strategy | Retrieve initiatives/programmes |
| Out-of-document | Refuse unsupported FY2027 questions |

The evaluation questions should **not be modified** when comparing pipeline versions.

---

## 9. Output Format

The pipeline writes:

```text
results.json
```

Example:

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

The evidence metadata enables retrieval debugging and auditability.

---

## 10. Project Structure

```text
Titan-RAG/
│
├── rag_pipeline.py
├── queries.json
├── results.json
├── requirements.txt
├── README.md
│
├── Titan AR 2026_0.pdf
│
└── index/
    ├── index_<fingerprint>.faiss
    └── metadata_<fingerprint>.json
```

The current implementation keeps the main pipeline in one Python file intentionally to make the hackathon submission easy to run.

For a larger production system, ingestion, retrieval, generation, validation, and evaluation should be separated into modules.

---

## 11. Installation

### Requirements

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Current dependencies include:

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

---

## 12. Running the Pipeline

Run using the default PDF and evaluation questions:

```bash
python rag_pipeline.py
```

Specify custom inputs:

```bash
python rag_pipeline.py \
    --pdf "Titan AR 2026_0.pdf" \
    --queries queries.json \
    --output results.json
```

Force an index rebuild:

```bash
python rag_pipeline.py --rebuild-index
```

Optional configuration can be supplied through JSON:

```bash
python rag_pipeline.py --config config.json
```

---

## 13. Configuration

The current defaults are approximately:

| Parameter | Default | Purpose |
|---|---:|---|
| `chunk_size` | 400 | Text chunk size in words |
| `chunk_overlap` | 80 | Chunk overlap |
| `top_k_retrieval` | 15 | Dense retrieval candidates |
| `top_k_final` | 5 | Final context chunks |
| `use_reranker` | true | Enable BM25 reranking |
| `relevance_threshold` | null | Disabled in favor of validation |
| `max_prompt_tokens` | 512 | Intended prompt budget |
| `seed` | 42 | Reserved deterministic seed setting |
| `embedding_model` | MiniLM-L6-v2 | Embedding model |
| `generation_model` | FLAN-T5-base | Generation model |

---

# 14. Important Known Limitations

The current implementation is substantially stronger than a basic single-stage RAG pipeline, but there are still important limitations.

## 14.1 Document fingerprint is not a full-file hash

The current fingerprint samples extracted content from the first five PDF pages.

That means a change later in the PDF could theoretically go undetected.

### Recommended improvement

Use:

```text
SHA-256(full PDF bytes)
+
embedding model
+
chunk configuration
+
extraction version
```

---

## 14.2 Paragraph detection is partially defeated by text cleaning

The cleaning step normalizes whitespace before chunking.

Because newline structure is flattened, the intended paragraph-based splitting can become less effective.

### Recommended improvement

Preserve structural newlines during extraction and clean only after identifying blocks.

---

## 14.3 Table representation can be improved

Row-level table chunks are useful, but they can lose:

- table title
- section title
- units
- multi-row headers
- fiscal-year relationships
- neighbouring explanatory text

### Recommended improvement

Represent a table as:

```text
TABLE
Title: ...
Section: ...
Page: ...

Columns:
Metric | FY2024-25 | FY2025-26

Row:
Revenue | ... | ...
```

This is particularly important for financial questions.

---

## 14.4 BM25 is applied only after dense retrieval

The current BM25 stage reranks the dense top-15 candidates.

Therefore:

```text
If dense retrieval misses the correct chunk,
BM25 cannot recover it.
```

A stronger hybrid retriever would independently retrieve candidates using both:

```text
Dense top-N
+
BM25 top-N
```

and merge/rerank the union.

---

## 14.5 BM25 scoring normalization is query-dependent

The dense and BM25 scores are normalized over the current candidate set.

Consequently, the combined score is relative to the retrieved candidates rather than globally calibrated.

A stronger implementation could use reciprocal rank fusion (RRF), which avoids directly comparing incompatible score scales.

---

## 14.6 Numerical reasoning is not yet explicit

Questions such as:

```text
Which businesses exceeded INR 5,000 crore?
```

require:

```text
retrieve
→ extract entities and numbers
→ normalize units
→ compare
→ filter
```

Semantic retrieval + generation alone does not guarantee this behavior.

A stronger implementation should add a numerical reasoning/extraction layer.

---

## 14.7 Grounding validation is only numeric

The validator primarily checks whether numbers generated by the LLM occur in the retrieved context.

It does not perform full claim-level entailment.

Therefore an answer can theoretically contain a correct number attached to the wrong entity or year.

---

## 14.8 Prompt truncation should use the actual tokenizer

The pipeline has a configured prompt-token limit, but robust context budgeting should use the selected generation tokenizer rather than a rough word/token approximation.

A better implementation would:

1. tokenize each evidence chunk;
2. rank chunks;
3. add chunks until the context budget is reached;
4. always reserve space for instructions and the question.

---

## 14.9 Model loading is inefficient

The current generation function loads the tokenizer and model when generating an answer.

For 11 questions, this creates unnecessary repeated model initialization.

### Recommended improvement

Load the model once:

```text
Pipeline startup
      ↓
Load embedding model
Load generation model
      ↓
Process all questions
```

This substantially improves evaluation runtime.

---

## 14.10 Reproducibility can be stronger

A `seed` configuration exists, but deterministic reproducibility should explicitly set seeds for:

```text
Python
NumPy
PyTorch
```

and record:

```text
model versions
library versions
configuration
document hash
```

in the evaluation output.

---

## 14.11 `grounded=true` needs more precise semantics

The current output uses:

```json
"grounded": true
```

for accepted answers.

This should eventually distinguish:

```text
retrieval_supported
answer_validated
fully_grounded
refused
```

because a simple numeric-presence check is not equivalent to complete factual grounding.

---

## 14.12 Evaluation needs reference answers and retrieval metrics

The current evaluation executes the questions and generates results, but the repository would benefit from an explicit evaluation harness measuring:

### Retrieval

- Recall@K
- Precision@K
- MRR
- nDCG

### Generation

- Answer correctness
- Faithfulness / groundedness
- Completeness

### Safety

- Out-of-document refusal accuracy
- Unsupported-number rate
- Unsupported-claim rate

A baseline-vs-improved comparison would make the architectural improvements measurable.

---

# 15. Recommended Next-Generation Architecture

For a stronger RAG architecture, the next version should look like:

```text
                         Question
                            |
                            v
                   Query Classification
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
       Dense             BM25             Numerical
     Retrieval          Retrieval          Retrieval
          |                 |                  |
          +-----------------+------------------+
                            |
                            v
                    Candidate Fusion
                         (RRF)
                            |
                            v
                       Reranking
                            |
                            v
                    Context Builder
                            |
                            v
                    Grounded LLM
                            |
                            v
                  Claim / Number Check
                            |
                   +--------+--------+
                   |                 |
                Supported        Unsupported
                   |                 |
                   v                 v
             Final Answer       Refusal
```

For this annual-report use case, this architecture is more valuable than adding agents unnecessarily.

---

# 16. Why This Is a RAG Architecture Project

The project demonstrates several important RAG engineering concepts:

### Ingestion

PDF → text + tables

### Representation

Text/table content → chunks + metadata

### Retrieval

Dense embeddings + FAISS

### Hybrid retrieval

Dense candidates + BM25 reranking

### Generation

Evidence-constrained open-source LLM

### Grounding

Post-generation evidence validation

### Evaluation

Fixed question set + structured JSON results

The next maturity level is not simply adding more frameworks. It is improving:

```text
retrieval recall
→ ranking precision
→ numerical reasoning
→ grounding
→ evaluation
→ reproducibility
```

---

# 17. Suggested Development Roadmap

## Phase 1 — Current baseline

- PDF extraction
- Chunking
- MiniLM
- FAISS
- BM25 reranking
- FLAN-T5
- Basic validation

## Phase 2 — Retrieval optimization

- Full-document SHA-256 fingerprint
- Better structural chunking
- Table-aware chunks
- Independent BM25 retrieval
- Reciprocal Rank Fusion
- Better reranking
- Query-type routing

## Phase 3 — Financial reasoning

- Numeric/entity extraction
- Fiscal-year normalization
- Unit normalization
- Threshold filtering
- Arithmetic validation

## Phase 4 — Grounding

- Claim extraction
- Evidence mapping
- Entailment validation
- Unsupported-claim detection
- Better refusal logic

## Phase 5 — Evaluation

- Golden answers
- Retrieval Recall@K
- MRR / nDCG
- Faithfulness
- Answer correctness
- Refusal accuracy
- Regression testing

## Phase 6 — Productionization

- Modular package structure
- Observability
- Model/version tracking
- CI/CD
- Containerization
- API layer
- Security
- Performance testing
- Model governance

---

# 18. Engineering Philosophy

The project deliberately avoids introducing frameworks such as LangChain, LangGraph, MCP, or multi-agent orchestration unless they solve a demonstrated problem.

For a document-question-answering system over one annual report, architectural quality is better demonstrated through:

```text
better retrieval
+
better evidence representation
+
better numerical handling
+
better grounding
+
better evaluation
```

rather than framework complexity.

---

# 19. License

MIT License.

---

## Author

**Dwaipayan Dutta**

GitHub:

https://github.com/DwaipayanDutta/RAG-Pipeline-Hackathon
