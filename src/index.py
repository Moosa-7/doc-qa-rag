import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL, INDEX_DIR, INDEX_PATH, load_chunks


def build_index():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Indexed {index.ntotal} chunks, dimension {embeddings.shape[1]}")


if __name__ == "__main__":
    build_index()