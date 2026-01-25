import re

def clean_text(text: str) -> str:
    """
    Minimal, safe text cleaning for RAG ingestion.
    We DO NOT try to fix broken PDFs here.
    """
    if not text:
        return ""

    # Normalize whitespace only
    text = re.sub(r"\s+", " ", text).strip()

    return text
