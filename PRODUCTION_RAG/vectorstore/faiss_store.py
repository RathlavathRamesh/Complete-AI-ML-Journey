from typing import List, Dict
import faiss
import numpy as np
import pickle
import os

class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.documents = []

    def add(self, embedded_chunks: List[Dict]):
        vectors = np.array(
            [c["embedding"] for c in embedded_chunks],
            dtype="float32"
        )

        self.index.add(vectors)
        self.documents.extend(embedded_chunks)

    def search(self, query_vector, top_k: int = 5):
        query_vector = np.array([query_vector], dtype="float32")
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(score)
                results.append(doc)

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/docs.pkl", "rb") as f:
            self.documents = pickle.load(f)


