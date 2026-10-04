import sys
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL, INDEX_PATH, load_chunks


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(str(INDEX_PATH))
        self.chunks = load_chunks()

    def search(self, query, k=4):
        vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
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