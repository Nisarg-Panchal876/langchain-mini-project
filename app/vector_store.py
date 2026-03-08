from langchain_community.vectorstores import FAISS
from app.embeddings import embedding_model
from app.document_loader import load_and_split_pdf
import os

# ── Configuration ─────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "vector_store"
)


def build_vector_store(pdf_path: str = None) -> FAISS:
    """
    Load the PDF, generate embeddings for all chunks, and persist the
    FAISS index to disk.

    Args:
        pdf_path: Optional path to the PDF. Uses default from document_loader
                  if not provided.

    Returns:
        FAISS vector store instance.
    """
    kwargs = {"pdf_path": pdf_path} if pdf_path else {}
    chunks = load_and_split_pdf(**kwargs)

    print(f"[VectorStore] Building FAISS index from {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    abs_store_path = os.path.abspath(VECTOR_STORE_PATH)
    os.makedirs(abs_store_path, exist_ok=True)
    vector_store.save_local(abs_store_path)
    print(f"[VectorStore] Index saved to: {abs_store_path}")

    return vector_store


def load_vector_store() -> FAISS:
    """
    Load an existing FAISS index from disk.

    Returns:
        FAISS vector store instance.

    Raises:
        FileNotFoundError: If the index has not been built yet.
    """
    abs_store_path = os.path.abspath(VECTOR_STORE_PATH)

    if not os.path.exists(abs_store_path):
        raise FileNotFoundError(
            f"Vector store not found at: {abs_store_path}\n"
            "Run build_vector_store() first to index your PDF."
        )

    print(f"[VectorStore] Loading FAISS index from: {abs_store_path}")
    vector_store = FAISS.load_local(
        abs_store_path,
        embedding_model,
        allow_dangerous_deserialization=True,  # Safe — we wrote this index ourselves
    )
    print("[VectorStore] Index loaded successfully.")
    return vector_store


def get_or_build_vector_store(pdf_path: str = None) -> FAISS:
    """
    Load the FAISS index from disk if it exists, otherwise build it from the PDF.
    This avoids re-embedding the PDF on every application restart.

    Args:
        pdf_path: Optional path to the PDF (used only when building).

    Returns:
        FAISS vector store instance.
    """
    abs_store_path = os.path.abspath(VECTOR_STORE_PATH)
    index_file = os.path.join(abs_store_path, "index.faiss")

    if os.path.exists(index_file):
        return load_vector_store()

    return build_vector_store(pdf_path)


if __name__ == "__main__":
    # Quick smoke-test — run: python -m app.vector_store
    store = get_or_build_vector_store()
    results = store.similarity_search("What services does NovaSoft offer?", k=3)
    print(f"\nTop 3 retrieved chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] Page {doc.metadata.get('page', '?')}:")
        print(doc.page_content[:300])
