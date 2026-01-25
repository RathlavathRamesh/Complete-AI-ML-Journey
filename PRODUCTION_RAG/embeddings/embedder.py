from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Load once (CPU-friendly)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Generates embeddings locally using SentenceTransformers.
    """

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embedded_chunks = []

    for chunk, emb in zip(chunks, embeddings):
        embedded_chunks.append({
            "text": chunk["text"],
            "embedding": emb.tolist(),
            "metadata": chunk["metadata"]
        })

    return embedded_chunks
