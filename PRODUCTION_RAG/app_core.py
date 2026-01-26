import os
import time

from retrieval.retriever import load_retriever, retrieve
from reranking.reranker import rerank
from generation.prompts import build_prompt, build_metrics_insight_prompt
from generation.llm import generate, generate_insight
from evaluation.faithfulness import faithfulness
from utils.context import build_context, format_sources

# Load vector store ONCE
base_path = os.path.dirname(__file__)
store_path = os.path.join(base_path, "vectorstore_data")
store = load_retriever(store_path)

def answer_question(question: str):
    # 1️⃣ RETRIEVAL
    t0 = time.time()
    retrieved_chunks = retrieve(question, store, top_k=15)
    retrieval_time = round((time.time() - t0) * 1000, 2)

    # 2️⃣ RE-RANKING
    t1 = time.time()
    reranked_chunks = rerank(question, retrieved_chunks, top_n=5)
    rerank_time = round((time.time() - t1) * 1000, 2)

    # 3️⃣ CONTEXT
    context = build_context(reranked_chunks)

    # 4️⃣ GENERATION
    t2 = time.time()
    prompt = build_prompt(question, context)
    answer = generate(prompt)
    generation_time = round((time.time() - t2) * 1000, 2)

    # 5️⃣ EVALUATION
    eval_metrics = faithfulness(answer, context)

    metrics = {
        "retrieval_time_ms": retrieval_time,
        "rerank_time_ms": rerank_time,
        "generation_time_ms": generation_time,
        "faithfulness_score": eval_metrics["faithfulness_score"],
        "answerable": eval_metrics["answerable"]
    }

    # 6️⃣ SOURCES
    sources = format_sources(reranked_chunks)

    # 7️⃣ INSIGHT LLM
    insight_prompt = build_metrics_insight_prompt(metrics, sources)
    insight = generate_insight(insight_prompt)

    return {
        "answer": answer,
        "sources": sources,
        "metrics": metrics,
        "insight": insight
    }
