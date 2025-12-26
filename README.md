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
