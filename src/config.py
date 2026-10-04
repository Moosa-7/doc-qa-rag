import json
from pathlib import Path

PDF_DIR = Path("data/pdfs")
CHUNKS_PATH = Path("data/chunks.jsonl")
INDEX_DIR = Path("data/index")
INDEX_PATH = INDEX_DIR / "faiss.index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000     # characters
CHUNK_OVERLAP = 150   # characters shared between neighboring chunks


def load_chunks():
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]