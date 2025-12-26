from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
import unicodedata, re, json, hashlib


pdf_dir = r"X:\\ML Projects\\News RAG\\data\\pdfs"  # your path
output_jsonl_path = r"X:\\ML Projects\\News RAG\src\\artifacts\\all_docs.jsonl"

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def make_doc_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()

pdf_paths = list(Path(pdf_dir).glob("*.pdf"))

with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        doc_id = make_doc_id(path)

        for i, doc in enumerate(pages, start=1):
            cleaned = clean_text(doc.page_content)
            record = {
                "doc_id": doc_id,
                "file_name": path.name,
                "page": i,
                "page_content": cleaned,
                "metadata": {
                    "source": path.stem,
                    "page": i,
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"All documents saved (JSONL) to {output_jsonl_path}")
