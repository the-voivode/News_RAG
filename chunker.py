import json
from pathlib import Path
from collections import defaultdict
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- CONFIG ----
input_json_path = r"X:\\ML Projects\\News RAG\src\\artifacts\\all_docs.jsonl"  # your file
min_words_per_chunk = 50   # filter junk
chunk_size_chars = 800    # ~300–800 tokens depending on text
chunk_overlap_chars = 100

# ---- 1) LOAD JSON ----
with open(input_json_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]   # if you used JSONL, replace this with: data = [json.loads(line) for line in f]

# ---- 2) GROUP PAGES BY DOCUMENT ----
# We'll treat metadata['source'] as doc_id for now.
docs_pages = defaultdict(list)  # doc_id -> list of {page, text}

for row in data:
    text = row["page_content"]
    meta = row["metadata"]
    doc_id = meta.get("doc_id") or meta["source"]
    page = meta.get("page", 1)

    docs_pages[doc_id].append({
        "page": page,
        "text": text
    })

# sort pages per doc
for doc_id in docs_pages:
    docs_pages[doc_id].sort(key=lambda x: x["page"])

# ---- 3) DEFINE SPLITTER ----
# Char-based splitter approximating 300–800 tokens.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size_chars,
    chunk_overlap=chunk_overlap_chars,
    separators=["\n\n", "\n", ". ", " "]
)

# ---- 4) BUILD CHUNKS WITH PAGE MAPPING ----
all_chunks = []  # will hold dicts with chunk_id, doc_id, page_start, page_end, content

for doc_id, pages in docs_pages.items():
    # Concatenate all pages' text but remember char spans for each page
    full_text = ""
    page_spans = []  # list of (page_no, start_idx, end_idx)

    for p in pages:
        start_idx = len(full_text)
        # add two newlines between pages so boundaries aren't smashed
        full_text += p["text"].strip() + "\n\n"
        end_idx = len(full_text)
        page_spans.append((p["page"], start_idx, end_idx))

    # Split concatenated text into overlapping chunks
    splits = splitter.split_text(full_text)

    # For mapping chunks back to page ranges,
    # keep a moving pointer so .find() stays in order and doesn’t hit earlier duplicates
    search_start = 0

    for chunk_text in splits:
        # Filter out short / useless chunks by word count
        word_count = len(chunk_text.split())
        if word_count < min_words_per_chunk:
            continue

        # Find char start & end of this chunk in full_text
        idx = full_text.find(chunk_text, search_start)
        if idx == -1:
            # Fallback: try from beginning (rare, but just in case)
            idx = full_text.find(chunk_text)
            if idx == -1:
                # Give up on mapping pages for this chunk
                # You can skip or continue here; I'll skip mapping but still keep chunk.
                page_start = pages[0]["page"]
                page_end = pages[-1]["page"]
                start_idx = None
                end_idx = None
            else:
                start_idx = idx
                end_idx = idx + len(chunk_text)
        else:
            start_idx = idx
            end_idx = idx + len(chunk_text)
            search_start = end_idx

        # Map char span to pages covered
        pages_covered = []
        if start_idx is not None:
            for page_no, p_start, p_end in page_spans:
                # overlap condition: page span and chunk span intersect
                if p_end > start_idx and p_start < end_idx:
                    pages_covered.append(page_no)

        if pages_covered:
            page_start = min(pages_covered)
            page_end = max(pages_covered)
        else:
            # Fallback if mapping fails completely
            page_start = pages[0]["page"]
            page_end = pages[-1]["page"]

        chunk_id = str(uuid.uuid4())

        all_chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "page_start": page_start,
            "page_end": page_end,
            "content": chunk_text
        })

print(f"Total chunks created: {len(all_chunks)}")


chunks_output_path = r"X:\\ML Projects\\News RAG\src\\artifacts\\chunks.jsonl"
with open(chunks_output_path, "w", encoding="utf-8") as f:
    for ch in all_chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")

print(f"Chunks saved to {chunks_output_path}")