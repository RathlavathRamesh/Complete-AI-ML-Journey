def build_prompt(query, context):
    return f"""
You are an enterprise policy assistant.
Answer ONLY from the context.
If unsure, say "Not found in document".

Context:
{context}

Question:
{query}
Answer:
"""

def build_metrics_insight_prompt(metrics: dict, sources: str) -> str:
    return f"""
You are an AI system analyst evaluating the quality of a Retrieval-Augmented Generation (RAG) response.

Given the system metrics and retrieval evidence below, produce a concise, user-friendly insight.

System Metrics:
- Retrieval Time (ms): {metrics['retrieval_time_ms']}
- Re-rank Time (ms): {metrics['rerank_time_ms']}
- Generation Time (ms): {metrics['generation_time_ms']}
- Faithfulness Score (0–1): {metrics['faithfulness_score']}
- Answerable: {metrics['answerable']}

Retrieved Evidence:
{sources}

Your task:
1. Assess overall answer confidence (High / Medium / Low).
2. Comment on system performance (latency).
3. Comment on answer reliability.
4. Provide a short recommendation to the user.

Rules:
- Do NOT invent facts.
- Base conclusions strictly on the metrics.
- Keep the response professional and concise.
- 4–6 bullet points max.
"""

