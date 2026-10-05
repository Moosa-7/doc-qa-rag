import os
import json
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_DIR = DATA_DIR / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "")
LLM_MODEL = "llama3.2:3b"

def load_chunks():
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]