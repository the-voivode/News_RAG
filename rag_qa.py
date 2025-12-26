# ========== v3.0 HYBRID RETRIEVAL USING BM25 LEXICAL SEARCH AND RRF AND RERANKING WITH CROSS-ENCODER WITH NO NER FILTERING ==========

import os
from typing import List, Tuple

from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import CrossEncoder  # NEW
import json

# ---------- OPENROUTER CLIENT ----------
load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

LLM_MODEL = "x-ai/grok-4.1-fast"

# ---------- CONFIG ----------

CONNECTION_STRING = (
    "postgresql+psycopg://postgres:Amirhosein2002@localhost:5432/news_rag"
)

COLLECTION_NAME = "history_chunks_bge_large"  # must match embed script

MODEL_NAME = "BAAI/bge-large-en"
RERANKER_MODEL = "BAAI/bge-reranker-base"      # NEW: cross-encoder reranker

K_RETRIEVE = 8   # final chunks passed to LLM
K_CANDIDATES = 48  # how many candidates to rerank (after RRF)

engine = create_engine(CONNECTION_STRING, future=True)

# ---------- EMBEDDINGS + VECTOR STORE ----------

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
    pre_delete_collection=False,   # don't drop existing data
)

# ---------- CROSS-ENCODER RERANKER (GLOBAL) ----------

# This is a query-passage cross-encoder: it takes (query, text) pairs
# and outputs a relevance score. Higher = more relevant.
reranker = CrossEncoder(RERANKER_MODEL, device="cpu")


# ---------- LEXICAL SEARCH (BM25-like via Postgres FTS) ----------

def lexical_search(query: str, k: int = 50) -> List[Document]:
    sql = text("""
        SELECT id, document, cmetadata,
               ts_rank_cd(content_fts, plainto_tsquery('english', :q)) AS rank
        FROM langchain_pg_embedding
        WHERE content_fts @@ plainto_tsquery('english', :q)
        ORDER BY rank DESC
        LIMIT :k
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"q": query, "k": k}).fetchall()

    docs: List[Document] = []
    for r in rows:
        meta = r.cmetadata
        if isinstance(meta, str):
            meta = json.loads(meta)
        docs.append(
            Document(
                page_content=r.document,
                metadata=meta,
            )
        )
    return docs


# ---------- RRF FUSION (dense + lexical) ----------

def rrf_fuse(
    dense_docs: List[Document],
    lexical_docs: List[Document],
    k_final: int = 20,
    k_rrf: int = 60,
    w_dense: float = 1.0,
    w_lexical: float = 1.0,
) -> List[Document]:
    scores: dict[str, dict] = {}

    def add_list(docs: List[Document], weight: float):
        for rank, d in enumerate(docs, start=1):
            cid = d.metadata.get("chunk_id")
            if not cid:
                continue
            if cid not in scores:
                scores[cid] = {"doc": d, "score": 0.0}
            scores[cid]["score"] += weight * 1.0 / (k_rrf + rank)

    add_list(dense_docs, w_dense)
    add_list(lexical_docs, w_lexical)

    combined = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [x["doc"] for x in combined[:k_final]]


def retrieve_chunks_hybrid(query: str, k_final: int = 8) -> List[Document]:
    # get candidates
    dense = vector_store.similarity_search(query, k=60)
    lexical = lexical_search(query, k=60)

    fused = rrf_fuse(
        dense_docs=dense,
        lexical_docs=lexical,
        k_final=k_final,
        k_rrf=60,
        w_dense=1.0,
        w_lexical=1.0,
    )
    return fused


# ---------- CROSS-ENCODER RERANKING STEP ----------

def rerank_with_cross_encoder(
    query: str,
    docs: List[Document],
    top_k: int,
) -> List[Document]:
    """
    Take candidate docs, score each with a cross-encoder on (query, chunk_text),
    return top_k by score.
    """
    if not docs:
        return []

    # Prepare (query, text) pairs
    pairs: List[Tuple[str, str]] = [(query, d.page_content) for d in docs]

    # CrossEncoder.predict returns a list/np.array of scores
    scores = reranker.predict(pairs)  # shape: [len(docs)]

    # Attach scores and sort
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)

    top_docs = [d for d, s in scored[:top_k]]
    return top_docs


# ---------- CONTEXT & PROMPT BUILDING ----------

def build_context(docs: List[Document]) -> str:
    """Turn retrieved docs into a readable context block for the LLM."""
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        doc_id = m.get("doc_id", "UNKNOWN")
        p_start = m.get("page_start")
        p_end = m.get("page_end")
        page_str = (
            f"pages {p_start}-{p_end}"
            if p_start is not None and p_end is not None
            else ""
        )
        parts.append(
            f"[{i}] {doc_id} {page_str}\n{d.page_content.strip()}"
        )
    return "\n\n".join(parts)


def build_user_prompt(query: str, context: str) -> str:
    return f"""You are a careful historical analyst.

Use ONLY the information in the CONTEXT below to answer the QUESTION.
If something is not supported by the context, say you don't know.

QUESTION:
{query}

CONTEXT:
{context}

When you answer:
- Cite which source numbers you used like [1], [2–3].
- Be concise but precise.
"""


# ---------- MAIN QA FUNCTION ----------

def answer_question(query: str, k: int = K_RETRIEVE) -> str:
    # 1) Retrieve hybrid candidates, larger than final k
    candidates = retrieve_chunks_hybrid(query, k_final=K_CANDIDATES)

    if not candidates:
        return "I couldn't find any relevant passages in the database."

    # 2) Rerank with cross-encoder, keep top k
    docs = rerank_with_cross_encoder(query, candidates, top_k=k)

    # 3) Build context & prompt
    context = build_context(docs)
    
    prompt = build_user_prompt(query, context)

    # 4) Call OpenRouter LLM
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise historian. Never hallucinate; answer only from the provided context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content


# ---------- CLI LOOP ----------

if __name__ == "__main__":
    while True:
        query = input("\nQuery> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        answer = answer_question(query, k=K_RETRIEVE)
        print("\n=== ANSWER ===")
        print(answer)
        print("\n" + "=" * 80)
