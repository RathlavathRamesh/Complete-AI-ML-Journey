from fastapi import FastAPI
from pydantic import BaseModel

from retrieval.retriever import load_retriever, retrieve
from rag.rag_chain import (
    build_context,
    build_prompt,
    generate_answer,
    format_sources
)

app = FastAPI(title="Policy RAG API")

# Load vector store ONCE at startup
store = load_retriever("vectorstore_data")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    chunks = retrieve(req.question, store, top_k=5)

    context = build_context(chunks)
    prompt = build_prompt(req.question, context)

    answer = generate_answer(prompt)
    sources = format_sources(chunks)

    return {
        "answer": answer,
        "sources": sources
    }
