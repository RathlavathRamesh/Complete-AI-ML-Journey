from retrieval.retriever import load_retriever, retrieve
from rag.rag_chain import (
    build_context,
    build_prompt,
    generate_answer,
    format_sources
)

if __name__ == "__main__":
    # 1. Load vector store
    store = load_retriever("vectorstore_data")

    # 2. User query
    query = "What is the responsibility of the HR department regarding this manual?"

    # 3. Retrieve relevant chunks
    chunks = retrieve(query, store, top_k=5)

    # 4. Build RAG prompt
    context = build_context(chunks)
    prompt = build_prompt(query, context)

    # 5. Generate answer
    answer = generate_answer(prompt)

    # 6. PRINT OUTPUT
    print("\nQUESTION:")
    print(query)

    print("\nANSWER:")
    print(answer)

    # ✅ THIS IS WHERE YOUR QUESTION APPLIES
    sources = format_sources(chunks)

    print("\nSOURCES:")
    print(sources)
