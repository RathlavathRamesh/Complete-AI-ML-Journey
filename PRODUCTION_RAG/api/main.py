from fastapi import FastAPI
from pydantic import BaseModel
import time

from retrieval.retriever import load_retriever, retrieve
from reranking.reranker import rerank
from generation.prompts import build_prompt, build_metrics_insight_prompt
from generation.llm import generate, generate_insight
from evaluation.faithfulness import faithfulness
from utils.context import build_context, format_sources

app = FastAPI(title="Policy RAG API")

# Load vector store ONCE at startup
store = load_retriever("vectorstore_data")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: str
    metrics: dict
    insight: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):

    # 1️⃣ RETRIEVAL
    t0 = time.time()
    retrieved_chunks = retrieve(req.question, store, top_k=15)
    retrieval_time = round((time.time() - t0) * 1000, 2)

    # 2️⃣ RE-RANKING
    t1 = time.time()
    reranked_chunks = rerank(req.question, retrieved_chunks, top_n=5)
    rerank_time = round((time.time() - t1) * 1000, 2)

    # 3️⃣ CONTEXT BUILDING
    context = build_context(reranked_chunks)

    # 4️⃣ PROMPT + GENERATION
    t2 = time.time()
    prompt = build_prompt(req.question, context)
    answer = generate(prompt)
    generation_time = round((time.time() - t2) * 1000, 2)

    # 5️⃣ EVALUATION (FAITHFULNESS)
    eval_metrics = faithfulness(answer, context)

    # 6️⃣ METRICS OBJECT (⚠️ MUST COME BEFORE INSIGHT)
    metrics = {
        "retrieval_time_ms": retrieval_time,
        "rerank_time_ms": rerank_time,
        "generation_time_ms": generation_time,
        "faithfulness_score": eval_metrics["faithfulness_score"],
        "answerable": eval_metrics["answerable"]
    }

    # 7️⃣ SOURCES
    sources = format_sources(reranked_chunks)

    # 8️⃣ SYSTEM INSIGHT (LLM ANALYSIS OF METRICS)
    insight_prompt = build_metrics_insight_prompt(metrics, sources)
    system_insight = generate_insight(insight_prompt)

    # 9️⃣ FINAL RESPONSE
    return {
        "answer": answer,
        "sources": sources,
        "metrics": metrics,
        "insight": system_insight
    }
