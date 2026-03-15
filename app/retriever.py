from langchain_community.vectorstores import FAISS
from app.vector_store import get_or_build_vector_store

# ── Configuration ─────────────────────────────────────────────────────────────
TOP_K = 10
MAX_PAGE_EXPANSION_PER_PAGE = 3
MAX_TOTAL_CONTEXT_CHUNKS = 16

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

    # Use MMR first for better diversity, then backfill with scored similarity.
    mmr_docs = store.max_marginal_relevance_search(query, k=k, fetch_k=max(30, k * 3))
    results_with_scores = store.similarity_search_with_score(query, k=max(k, 6))
    scored_docs = [doc for doc, score in results_with_scores]

    docs = _dedupe_docs(mmr_docs + scored_docs)
    docs = _expand_with_related_page_chunks(store, docs)
    docs = docs[:MAX_TOTAL_CONTEXT_CHUNKS]

    for i, (doc, score) in enumerate(results_with_scores, 1):
        page = doc.metadata.get("page", "?")
        print(f"  [{i}] page={page}  score={score:.4f}  "
              f"preview={doc.page_content[:80].replace(chr(10), ' ')!r}")

    return docs


def _dedupe_docs(docs: list) -> list:
    """Remove duplicate chunks while preserving order."""
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content.strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _expand_with_related_page_chunks(store: FAISS, docs: list) -> list:
    """
    Expand retrieved chunks with a few extra chunks from the same pages.

    This helps when structured lists are split across adjacent chunks and only
    one item is initially retrieved.
    """
    page_hits = {}
    for doc in docs:
        page = doc.metadata.get("page")
        if page is not None:
            page_hits.setdefault(page, 0)

    # FAISS uses an in-memory docstore when built from documents.
    all_docs = list(store.docstore._dict.values())

    expanded = list(docs)
    existing_text = {d.page_content.strip() for d in docs}

    for candidate in all_docs:
        page = candidate.metadata.get("page")
        if page not in page_hits:
            continue
        if page_hits[page] >= MAX_PAGE_EXPANSION_PER_PAGE:
            continue

        content_key = candidate.page_content.strip()
        if content_key in existing_text:
            continue

        expanded.append(candidate)
        existing_text.add(content_key)
        page_hits[page] += 1

    return expanded


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
