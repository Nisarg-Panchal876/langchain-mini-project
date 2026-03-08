from langchain_community.vectorstores import FAISS
from app.vector_store import get_or_build_vector_store

# ── Configuration ─────────────────────────────────────────────────────────────
TOP_K = 8  # Increased from 5 — ensures dense table/pricing chunks are included

# ── Retriever singleton ───────────────────────────────────────────────────────
_vector_store: FAISS = None


def _get_vector_store() -> FAISS:
    """Lazily initialise the vector store once and reuse it."""
    global _vector_store
    if _vector_store is None:
        _vector_store = get_or_build_vector_store()
    return _vector_store


def retrieve_context(query: str, k: int = TOP_K) -> list:
    """
    Convert a query to embeddings and retrieve the top-k most relevant
    document chunks from the FAISS index.

    Args:
        query: The customer's question / email body.
        k:     Number of chunks to retrieve (default: TOP_K = 5).

    Returns:
        List of LangChain Document objects ordered by relevance (best first).
    """
    store = _get_vector_store()
    print(f"[Retriever] Searching for top-{k} chunks for query: {query!r}")

    # similarity_search_with_score returns (doc, score) pairs — lower L2 = better
    results_with_scores = store.similarity_search_with_score(query, k=k)

    docs = [doc for doc, score in results_with_scores]

    for i, (doc, score) in enumerate(results_with_scores, 1):
        page = doc.metadata.get("page", "?")
        print(f"  [{i}] page={page}  score={score:.4f}  "
              f"preview={doc.page_content[:80].replace(chr(10), ' ')!r}")

    return docs


def format_context(docs: list) -> str:
    """
    Concatenate retrieved chunks into a single context string for the LLM.
    Each chunk is separated by a divider and labelled with its source page.

    Args:
        docs: List of Document objects returned by retrieve_context().

    Returns:
        A formatted multi-chunk context string.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i} — Page {page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def retrieve_and_format(query: str, k: int = TOP_K) -> str:
    """
    Convenience wrapper: retrieve top-k chunks and return them as a
    single formatted context string ready to inject into the LLM prompt.

    Args:
        query: The customer's question / email body.
        k:     Number of chunks to retrieve.

    Returns:
        Formatted context string.
    """
    docs = retrieve_context(query, k=k)
    return format_context(docs)


if __name__ == "__main__":
    # Quick smoke-test — run: python -m app.retriever
    query = "What are NovaSoft's refund and cancellation policies?"
    context = retrieve_and_format(query)
    print(f"\n{'='*60}")
    print("RETRIEVED CONTEXT:")
    print('='*60)
    print(context)
