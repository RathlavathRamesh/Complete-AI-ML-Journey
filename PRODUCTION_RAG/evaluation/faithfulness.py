from sentence_transformers import SentenceTransformer, util

# Lightweight semantic model
_model = SentenceTransformer("all-MiniLM-L6-v2")

def faithfulness(answer: str, context: str):
    """
    Measures semantic alignment between answer and retrieved context.
    Returns a score between 0 and 1.
    """

    if not answer or not context:
        return {
            "faithfulness_score": 0.0,
            "answerable": False
        }

    answer_emb = _model.encode(answer, convert_to_tensor=True)
    context_emb = _model.encode(context, convert_to_tensor=True)

    similarity = util.cos_sim(answer_emb, context_emb).item()

    # Practical enterprise thresholds (NOT academic)
    faithfulness_score = round(float(similarity), 2)

    answerable = faithfulness_score >= 0.35

    return {
        "faithfulness_score": faithfulness_score,
        "answerable": answerable
    }
