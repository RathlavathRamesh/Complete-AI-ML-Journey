from unstructured.partition.pdf import partition_pdf
from ingestion.cleaner import clean_text
from chunking.chunker import chunk_documents
from embeddings.embedder import embed_chunks
from vectorstore.faiss_store import FaissVectorStore
# Only keep meaningful content
ALLOWED_CATEGORIES = {
    "NarrativeText",
    "ListItem"
}

def ingest_pdf(pdf_path: str):
    """
    Ingests a PDF and returns clean narrative text blocks
    suitable for RAG.
    """

    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",              # balanced & safe on MacBook
        infer_table_structure=True
    )

    documents = []

    for el in elements:
        # 1. Structural filtering
        if el.category not in ALLOWED_CATEGORIES:
            continue

        if not el.text:
            continue

        # 2. Minimal cleaning
        cleaned = clean_text(el.text)

        # 3. Drop very small fragments
        if len(cleaned) < 150:
            continue

        documents.append({
            "text": cleaned,
            "metadata": {
                "page_number": el.metadata.page_number,
                "category": el.category
            }
        })

    return documents


if __name__ == "__main__":
    pdf_file = "data/raw/policies_p.pdf"

    # 1. Ingestion
    docs = ingest_pdf(pdf_file)

    # 2. Chunking
    chunks = chunk_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Embeddings
    embedded_chunks = embed_chunks(chunks)
    print(f"Total embedded chunks: {len(embedded_chunks)}")
    print("Embedding vector length:", len(embedded_chunks[0]["embedding"]))

    # 4. Vector store
    vector_store = FaissVectorStore(dim=len(embedded_chunks[0]["embedding"]))
    vector_store.add(embedded_chunks)

    # Save to disk
    vector_store.save("vectorstore_data")

    print("\nFAISS index created and saved successfully.")
