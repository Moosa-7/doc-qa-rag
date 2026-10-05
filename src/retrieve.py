import os
import sys
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL, INDEX_PATH, load_chunks, QUERY_PREFIX


class Retriever:
    def __init__(self):
        # On Hugging Face Spaces (ZeroGPU) force CPU so CUDA is never touched.
        # Locally, let sentence-transformers pick its default device, as before.
        device = "cpu" if os.getenv("SPACE_ID") else None
        self.model = SentenceTransformer(EMBED_MODEL, device=device)
        self.index = faiss.read_index(str(INDEX_PATH))
        faiss.omp_set_num_threads(1)   # avoids multithreading clashes with torch
        self.chunks = load_chunks()

    def search(self, query, k=4):
        vec = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    for rank, c in enumerate(Retriever().search(query), start=1):
        print(f"\n#{rank} score={c['score']:.3f} {c['source']} p.{c['page']}")
        print(c["text"][:300], "...")