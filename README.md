# News RAG (PGVector + Hybrid Retrieval Benchmarks)

A small end-to-end Retrieval-Augmented Generation (RAG) prototype for a news-style corpus:
**ingest → chunk → embed → store in Postgres/pgvector → retrieve (dense / lexical / hybrid) → optional cross-encoder rerank → evaluate retrieval speed & quality**.

This repo focuses on **retrieval + benchmarking**, not fancy UI demos.

## Features
- **PGVector-backed vector store** (Postgres + pgvector)
- **Dense retrieval** via sentence embeddings (e.g., BGE)
- **Optional lexical retrieval** (BM25-style) and **hybrid** combinations
- **Optional cross-encoder reranking** for relevance boosts
- **Benchmark script(s)** for raw retrieval speed and reranking latency
- Configurable chunking + metadata flow

## Project Structure
- `loader.py` — loads source documents / dataset into memory
- `chunker.py` — chunking logic
- `embed_chunks_to_pg.py` — embeds chunks and inserts them into PGVector collection
- `test_retrieval.py` — retrieval benchmarking (dense / lexical / hybrid / rerank)
- `rag_qa.py` — example QA flow using retrieved contexts
- `config.py` — central settings (models, DB connection, collection names)
- `artifacts/` — outputs, logs, intermediate results
- `metadata/` — stored metadata for chunks / documents

## Requirements
- Python 3.10+ recommended
- Postgres with **pgvector** enabled
- GPU is optional (CPU works fine; cross-encoders can be slower on CPU)

Install Python deps:
```bash
pip install -r requirements.txt


# News RAG (PGVector + Hybrid Retrieval + Reranking)

A compact RAG prototype for a news-style corpus:
**load → chunk → embed → store in Postgres/pgvector → retrieve (dense / lexical / hybrid) → optional cross-encoder rerank → benchmark speed & relevance**.

This repo is mainly about **retrieval and benchmarking**, not building a flashy app.

---

## What’s in here

- **Postgres + pgvector** as the vector store
- **Dense retrieval** using sentence embeddings (e.g., BGE)
- **Optional lexical retrieval** (BM25-style) and **hybrid** combinations
- **Optional cross-encoder reranking** (SentenceTransformers CrossEncoder)
- Scripts for **embedding into PGVector** and **testing retrieval latency**
- Minimal QA pipeline example (retrieve contexts, then answer with an LLM)

---

## Repo layout

Based on the current project files:

- `loader.py`  
  Loads documents / dataset (source depends on your setup).

- `chunker.py`  
  Splits documents into chunks suitable for embeddings + retrieval.

- `embed_chunks_to_pg.py`  
  Embeds chunks and inserts them into a PGVector collection.

- `test_retrieval.py`  
  Retrieval benchmarking: dense / lexical / hybrid / rerank (depending on what you enable).

- `rag_qa.py`  
  Example QA flow using retrieved chunks as context.

- `config.py`  
  Central configuration: DB connection, collection name, model names, top-k, etc.

- `metadata/`  
  Metadata produced during ingestion/chunking (if used by your pipeline).

- `artifacts/`  
  Outputs/logs/intermediate artifacts.

- `.env`  
  API keys / secrets **(must NOT be committed)**.

---

## Requirements

- Python **3.10+** recommended
- Postgres with **pgvector** enabled
- (Optional) GPU for faster embedding/reranking

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## Environment variables

Create a `.env` file (do not commit it):

```env
OPENROUTER_API_KEY=your_key_here
# Add any other keys used by your scripts
```

**Important:** Ensure `.env` is in `.gitignore`.

---

## Postgres + pgvector setup

1. Install Postgres
2. Enable `pgvector` in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

3. Set your DB connection in `config.py` (or refactor to env vars if you prefer).
   Example connection string pattern:

* `postgresql+psycopg://USER:PASSWORD@localhost:5432/news_rag`

---

## Quickstart

### 1) Configure

Open `config.py` and set:

* `CONNECTION_STRING`
* `COLLECTION_NAME`
* embedding model name (e.g., `BAAI/bge-large-en`)
* any retrieval parameters (top-k, reranker toggle, etc.)

### 2) Load + chunk

Run your ingestion logic (depends on how `loader.py` is written):

```bash
python loader.py
python chunker.py
```

If your pipeline combines these steps elsewhere, skip this and use your actual entrypoint.

### 3) Embed + store in PGVector

```bash
python embed_chunks_to_pg.py
```

This step creates/uses the configured collection and inserts embedded chunks into Postgres.

### 4) Benchmark retrieval

```bash
python test_retrieval.py
```

Typical benchmark modes include:

* dense-only retrieval
* lexical-only retrieval (if implemented/enabled)
* hybrid retrieval (dense + lexical fusion)
* optional cross-encoder reranking latency

### 5) Run QA example (optional)

```bash
python rag_qa.py
```

---

## What “hybrid” means here

Hybrid retrieval generally means combining:

* **Dense** semantic search (embeddings)
* **Lexical** term-based search (BM25-like)

Then either:

* merge results with a simple fusion strategy (e.g., weighted, RRF)
* optionally **rerank** the merged candidates using a cross-encoder

Exact behavior depends on how your `test_retrieval.py` is implemented.

---

## Notes & gotchas

* **Do not commit secrets:** `.env` should be ignored. If you accidentally committed it once, rotate keys.
* Cross-encoders can be slow on CPU. For speed benchmarks:

  * use a smaller reranker model
  * limit rerank candidate count (e.g., rerank top 20)
* If Postgres feels slow, check indexes and connection pooling.

---

## Suggested `.gitignore` (minimum)

At minimum, your repo should ignore:

* `.env`
* `__pycache__/`
* `.ipynb_checkpoints/`
* `artifacts/` (if it contains generated output)
* `metadata/` (if it contains generated output)

(You can expand this depending on how you use those folders.)

---

## License

Pick one (recommended): **MIT** or **Apache-2.0**.
Or leave it unlicensed if you enjoy ambiguity and confusion.

---

## Credits

Built as part of a personal “News RAG” prototype focusing on retrieval quality and speed tradeoffs.

```

