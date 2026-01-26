def build_context(chunks):
    return "\n\n".join(f"- {c['text']}" for c in chunks)


def format_sources(chunks):
    sources = []
    for i, c in enumerate(chunks, 1):
        page = c["metadata"].get("page_number", "N/A")
        score = c.get("rerank_score", c.get("score", 0))
        sources.append(f"[{i}] Page {page} (score={round(score,3)})")
    return "\n".join(sources)
