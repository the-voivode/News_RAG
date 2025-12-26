# # test_retrieval.py

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_postgres import PGVector
# from langchain_core.documents import Document

# # adjust these to match embed_chunks_to_pg.py
# CONNECTION_STRING = "postgresql+psycopg://postgres:Amirhosein2002@localhost:5432/news_rag"
# COLLECTION_NAME = "history_chunks_bge_large"  # or whatever you used


# def get_vector_store():
#     """Reconnect to the existing PGVector collection."""
#     embeddings = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-large-en",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True},
#     )

#     vector_store = PGVector(
#         embeddings=embeddings,
#         collection_name=COLLECTION_NAME,
#         connection=CONNECTION_STRING,
#         use_jsonb=True,
#         pre_delete_collection=False,  # IMPORTANT: don't drop your data
#     )
#     return vector_store


# def pretty_print_results(results):
#     """results = list of (Document, score) from similarity_search_with_score."""
#     print("\n=== RESULTS ===")
#     for rank, (doc, score) in enumerate(results, start=1):
#         meta = doc.metadata or {}
#         doc_id = meta.get("doc_id", "UNKNOWN_DOC")
#         page_start = meta.get("page_start")
#         page_end = meta.get("page_end")

#         page_info = ""
#         if page_start is not None and page_end is not None:
#             if page_start == page_end:
#                 page_info = f"p.{page_start}"
#             else:
#                 page_info = f"p.{page_start}-{page_end}"

#         print(f"{rank}. score={score:.4f}")
#         print(f"   doc_id: {doc_id}")
#         if page_info:
#             print(f"   pages:  {page_info}")

#         snippet = doc.page_content[:400].replace("\n", " ")
#         print(f"   text:   {snippet}...")
#         print("-" * 80)


# def main():
#     vs = get_vector_store()
#     print("Connected to PGVector. Type a query (or 'exit').")

#     while True:
#         query = input("\nQuery> ").strip()
#         if not query:
#             continue
#         if query.lower() in {"exit", "quit", "q"}:
#             break

#         results = vs.similarity_search_with_score(query, k=5)
#         pretty_print_results(results)

#     print("Bye.")


# if __name__ == "__main__":
#     main()


import json
import time
from typing import List, Tuple

from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sqlalchemy import create_engine, text
from sentence_transformers import CrossEncoder

# ---------- CONFIG ----------

CONNECTION_STRING = "postgresql+psycopg://postgres:Amirhosein2002@localhost:5432/news_rag"
COLLECTION_NAME = "history_chunks_bge_large"  # must match embed script

MODEL_NAME = "BAAI/bge-large-en"       # must match what you used for embedding
RERANKER_MODEL = "BAAI/bge-reranker-base"

K_RETRIEVE = 8        # final top-k to inspect
K_CANDIDATES = 48     # candidate pool for hybrid + rerank

engine = create_engine(CONNECTION_STRING, future=True)

# ---------- EMBEDDINGS + VECTOR STORE ----------

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"},            # "cuda" if you can
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
    pre_delete_collection=False,
)

# ---------- CROSS-ENCODER RERANKER ----------

reranker = CrossEncoder(RERANKER_MODEL, device="cpu")  # "cuda" if available


# ---------- LEXICAL SEARCH (Postgres FTS, inline tsvector) ----------

def lexical_search(query: str, k: int = 50) -> List[Document]:
    sql = text("""
        SELECT id, document, cmetadata,
               ts_rank_cd(
                   to_tsvector('english', document),
                   plainto_tsquery('english', :q)
               ) AS rank
        FROM langchain_pg_embedding
        WHERE to_tsvector('english', document) @@ plainto_tsquery('english', :q)
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
                metadata=meta or {},
            )
        )
    return docs


# ---------- DENSE RETRIEVAL (PGVector) ----------

def dense_search(query: str, k: int = 20) -> List[Document]:
    return vector_store.similarity_search(query, k=k)


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


# ---------- CROSS-ENCODER RERANKING ----------

def rerank_with_cross_encoder(
    query: str,
    docs: List[Document],
    top_k: int,
) -> List[Document]:
    if not docs:
        return []

    pairs: List[Tuple[str, str]] = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)  # list/np.array of floats

    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)

    return [d for d, s in scored[:top_k]]


# ---------- PRETTY PRINT ----------

def pretty_print_docs(docs: List[Document], header: str):
    print(f"\n=== {header} (k={len(docs)}) ===")
    if not docs:
        print("No docs.")
        return

    for i, d in enumerate(docs, start=1):
        m = d.metadata or {}
        doc_id = m.get("doc_id", "UNKNOWN_DOC")
        p_start = m.get("page_start")
        p_end = m.get("page_end")

        if p_start is not None and p_end is not None:
            if p_start == p_end:
                page_info = f"p.{p_start}"
            else:
                page_info = f"p.{p_start}-{p_end}"
        else:
            page_info = ""

        snippet = d.page_content.replace("\n", " ")[:300]

        print(f"{i}. doc_id={doc_id} {page_info}")
        print(f"   {snippet}...")
        print("-" * 80)


# ---------- MODES ----------

def run_dense_only(query: str, k: int = K_RETRIEVE):
    t0 = time.perf_counter()
    docs = dense_search(query, k=k)
    t1 = time.perf_counter()
    pretty_print_docs(docs, header=f"DENSE ONLY (time={t1 - t0:.3f}s)")


def run_full_pipeline(query: str):
    t0 = time.perf_counter()
    dense = dense_search(query, k=K_CANDIDATES)
    t1 = time.perf_counter()

    lexical = lexical_search(query, k=K_CANDIDATES)
    t2 = time.perf_counter()

    fused = rrf_fuse(
        dense_docs=dense,
        lexical_docs=lexical,
        k_final=K_CANDIDATES,
        k_rrf=60,
        w_dense=1.0,
        w_lexical=1.0,
    )
    t3 = time.perf_counter()

    reranked = rerank_with_cross_encoder(query, fused, top_k=K_RETRIEVE)
    t4 = time.perf_counter()

    print(
        f"\nTIMINGS:"
        f" dense={t1 - t0:.3f}s,"
        f" lexical={t2 - t1:.3f}s,"
        f" rrf={t3 - t2:.3f}s,"
        f" rerank={t4 - t3:.3f}s,"
        f" total={t4 - t0:.3f}s"
    )

    pretty_print_docs(reranked, header="FULL PIPELINE (dense+lexical+RRF+cross-encoder)")


# ---------- CLI LOOP ----------

if __name__ == "__main__":
    print("Connected. Modes:")
    print(" 1 = dense only")
    print(" 2 = full pipeline (dense + lexical + RRF + cross-encoder)")
    print(" q = quit")

    while True:
        mode = input("\nMode [1/2/q]> ").strip().lower()
        if mode in {"q", "quit", "exit"}:
            break
        if mode not in {"1", "2"}:
            print("Pick 1, 2 or q.")
            continue

        query = input("Query> ").strip()
        if not query:
            continue

        if mode == "1":
            run_dense_only(query, k=K_RETRIEVE)
        else:
            run_full_pipeline(query)
