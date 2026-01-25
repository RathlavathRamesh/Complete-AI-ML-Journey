from sentence_transformers import SentenceTransformer
from vectorstore.faiss_store import FaissVectorStore

# Same embedding model used earlier
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_retriever(path: str):
    store = FaissVectorStore(dim=384)
    store.load(path)
    return store

def retrieve(query: str, store: FaissVectorStore, top_k: int = 5):
    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    results = store.search(query_vector, top_k=top_k)
    return results