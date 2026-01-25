from typing import List, Dict
import tiktoken

# Tokenizer only for size estimation (model-agnostic)
tokenizer = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 500
MIN_TOKENS = 150


def token_len(text: str) -> int:
    return len(tokenizer.encode(text))


def chunk_documents(docs: List[Dict]) -> List[Dict]:
    """
    Converts ingestion paragraphs into retrieval-ready chunks.
    """
    chunks = []

    buffer_text = ""
    buffer_meta = None

    for doc in docs:
        text = doc["text"]
        meta = doc["metadata"]

        if not buffer_text:
            buffer_text = text
            buffer_meta = meta
            continue

        combined = buffer_text + " " + text

        # If still within size → keep merging
        if token_len(combined) <= MAX_TOKENS:
            buffer_text = combined
            continue

        # Save meaningful chunk
        if token_len(buffer_text) >= MIN_TOKENS:
            chunks.append({
                "text": buffer_text,
                "metadata": buffer_meta
            })

        # Reset buffer
        buffer_text = text
        buffer_meta = meta

    # Flush last chunk
    if buffer_text and token_len(buffer_text) >= MIN_TOKENS:
        chunks.append({
            "text": buffer_text,
            "metadata": buffer_meta
        })

    return chunks
