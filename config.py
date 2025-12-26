# src/config.py
from pathlib import Path

DATA_DIR = Path("X:\\ML Projects\\News RAG\\data")
BUILD_DIR = Path("X:\\ML Projects\\News RAG\\build")
INDEX_DIR = Path("X:\\ML Projects\\News RAG\\index")

# chunking
TARGET_TOKENS = 800
STRIDE_TOKENS  = 100  

EMBED_MODEL = "BAAI/bge-large-en"
EMBED_BATCH = 64
