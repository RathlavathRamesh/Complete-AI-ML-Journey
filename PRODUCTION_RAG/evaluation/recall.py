def recall_stats(chunks):
    return {
        "avg_score": sum(c.get("score", 0) for c in chunks) / len(chunks)
    }
