from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, chunks, top_n=5):
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)

    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)

    chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return chunks[:top_n]
