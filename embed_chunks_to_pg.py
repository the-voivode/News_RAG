import json
from pathlib import Path

import re
import unicodedata

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

# ---------- CONFIG ----------
CHUNKS_PATH = Path(r"V:\\ML Projects\\News RAG\src\\artifacts\\chunks.jsonl")  # adjust if needed

CONNECTION_STRING = "postgresql+psycopg://postgres:Amirhosein2002@127.0.0.1:5432/news_rag"
COLLECTION_NAME = "history_chunks_bge_large"

BATCH_SIZE = 256
MIN_WORDS_PER_CHUNK = 40  # filter useless junk


def clean_chunk_text(text: str) -> str:
    # Normalize Unicode (quotes, dashes, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Remove NUL bytes explicitly
    text = text.replace("\x00", "")

    # Optionally: strip *other* control characters (except \n, \t)
    text = re.sub(r"[\x01-\x08\x0B-\x0C\x0E-\x1F]", " ", text)

    # Collapse weird spacing
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def iter_chunk_docs(path: Path):
    """Stream chunks.jsonl and yield LangChain Document objects."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            raw_text = record["content"]
            text = clean_chunk_text(raw_text)

            # Skip if after cleaning it's empty or tiny
            if len(text.split()) < MIN_WORDS_PER_CHUNK:
                continue

            yield Document(
                page_content=text,
                metadata={
                    "chunk_id": record["chunk_id"],
                    "doc_id": record["doc_id"],
                    "page_start": record["page_start"],
                    "page_end": record["page_end"],
                },
            )



def main():
    # 1) Embedding model: BGE-large-en
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en",
        model_kwargs={"device": "cuda"},             # use "cpu" if no GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    # 2) Vector store backed by Postgres + pgvector
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,           # metadata as jsonb
        pre_delete_collection=False,  # set True only if you want to DROP and recreate
    )

    docs_buffer = []
    total = 0

    for doc in iter_chunk_docs(CHUNKS_PATH):
        docs_buffer.append(doc)
        if len(docs_buffer) >= BATCH_SIZE:
            vector_store.add_documents(docs_buffer)
            total += len(docs_buffer)
            print(f"Indexed {total} chunks...")
            docs_buffer = []

    if docs_buffer:
        vector_store.add_documents(docs_buffer)
        total += len(docs_buffer)
        print(f"Indexed {total} chunks (final batch).")

    print("Done embedding & indexing.")


if __name__ == "__main__":
    main()
