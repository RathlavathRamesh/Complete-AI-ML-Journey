from retrieval.retriever import load_retriever, retrieve
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client=Groq(api_key=GROQ_API_KEY)
def format_sources(chunks):
    sources = []
    for i, c in enumerate(chunks, 1):
        sources.append(
            f"[{i}] Page {c['metadata']['page_number']} (score={c['score']:.3f})"
        )
    return "\n".join(sources)


def generate_answer(prompt):
    response=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def build_context(chunks):
    context = "\n\n".join(
        f"- {c['text']}" for c in chunks
    )
    return context


def build_prompt(query: str, context: str) -> str:
    return f"""
You are a policy assistant.
Answer the question strictly using the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{query}

Answer:
""".strip()
