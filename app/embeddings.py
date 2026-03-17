import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = "cpu"

# ── BGE models perform better with this query instruction prefix ──────────────
# Applied automatically via encode_kwargs — no manual prefix needed in queries.
ENCODE_KWARGS = {"normalize_embeddings": True}  # cosine similarity ready

MODEL_KWARGS = {"device": DEVICE}
if os.getenv("HF_TOKEN"):
    MODEL_KWARGS["token"] = os.getenv("HF_TOKEN")


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and return the HuggingFace embedding model.

    Uses BAAI/bge-base-en-v1.5 on CPU with normalized embeddings
    for accurate cosine similarity scoring during retrieval.

    Returns:
        HuggingFaceEmbeddings: Ready-to-use embedding model instance.
    """
    print(f"[Embeddings] Loading model: {EMBEDDING_MODEL_NAME} on {DEVICE}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS,
    )
    print("[Embeddings] Model loaded successfully.")
    return embeddings


# Singleton — reused across the app to avoid reloading the model on every call
embedding_model = get_embedding_model()


if __name__ == "__main__":
    # Quick smoke-test — run: python -m app.embeddings
    test_sentences = [
        "What are NovaSoft's refund policies?",
        "How do I contact customer support?",
    ]
    vectors = embedding_model.embed_documents(test_sentences)
    print(f"\nEmbedding dimension: {len(vectors[0])}")
    print(f"Sample vector (first 5 values): {vectors[0][:5]}")
