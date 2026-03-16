from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm_model import llm
from app.retriever import retrieve_context, format_context

# ── Prompt Template ───────────────────────────────────────────────────────────
EMAIL_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an AI assistant working for NovaSoft Technologies.

Your task is to answer customer emails using ONLY the provided company knowledge base context.

Rules:
- Use only the information from the context.
- Keep the full response between 80 and 140 words.
- Use plain, direct language and short sentences.
- Include only the most relevant 2-4 points for the customer query.
- Do NOT paste long checklists, full policy text, or training curriculum details.
- If a long list exists in context, summarize it into one short sentence and offer to share full details on request.
- If specific numbers/prices/dates are explicitly required and present in context, include only the necessary ones.
- If the context does not contain the answer, say you will forward the query to the relevant team.
- Write the response in a professional email format.
- Keep the response concise and helpful.
- Use exactly this structure:
    Subject: <short relevant subject>
    Dear Customer,
    <one short paragraph>
    <optional second short paragraph>
    Sincerely,
    NovaSoft Technologies

Context:
{context}

Customer Query:
{question}

Write a professional email reply.""",
)

# ── Chain ─────────────────────────────────────────────────────────────────────
# PromptTemplate | LLM | StrOutputParser
rag_email_chain = EMAIL_PROMPT_TEMPLATE | llm | StrOutputParser()


def _dedupe_docs(docs: list) -> list:
    """De-duplicate retrieved chunks while preserving order."""
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content.strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _build_context_for_query(query: str) -> str:
    """
    Build context with a targeted fallback for checklist-style internship
    preparation questions to avoid partial requirement answers.
    """
    primary_docs = retrieve_context(query)

    query_lower = query.lower()
    needs_requirement_fallback = any(
        token in query_lower
        for token in [
            "prepare",
            "preparation",
            "application process",
            "what should we prepare",
            "internship",
            "cohort",
        ]
    )

    if not needs_requirement_fallback:
        return format_context(primary_docs)

    # Targeted retrieval query to capture full "prepare the following" sections.
    fallback_query = (
        "Application process prepare the following documents "
        "technical CV github portfolio statement of interest"
    )
    fallback_docs = retrieve_context(fallback_query, k=12)

    merged_docs = _dedupe_docs(primary_docs + fallback_docs)
    return format_context(merged_docs)


def generate_rag_email(query: str) -> dict:
    """
    Full RAG pipeline: retrieve relevant context from the knowledge base,
    then generate a professional email reply using the LLM.

    Args:
        query: The customer's question or email body.

    Returns:
        A dict with:
          - "reply":   The generated email text.
          - "context": The raw retrieved context passed to the LLM
                       (useful for debugging / audit).
    """
    print(f"[RAGChain] Processing query: {query!r}")

    # Step 1 — Retrieve relevant chunks from FAISS
    context = _build_context_for_query(query)

    # Step 2 — Generate email reply
    reply = rag_email_chain.invoke({"context": context, "question": query})
    reply = _enforce_brief_email(reply)

    print("[RAGChain] Email reply generated.")
    return {"reply": reply, "context": context}


def _enforce_brief_email(reply: str) -> str:
    """Normalize and constrain generated email length while keeping it complete."""
    text = "\n".join(line.rstrip() for line in reply.strip().splitlines() if line.strip())

    if "Sincerely," not in text:
        text = f"{text}\n\nSincerely,\nNovaSoft Technologies"
    elif "NovaSoft Technologies" not in text:
        text = text.replace("Sincerely,", "Sincerely,\nNovaSoft Technologies")

    max_words = 150
    words = text.split()
    if len(words) <= max_words:
        return text

    signature = "\n\nSincerely,\nNovaSoft Technologies"
    body_words = words[:120]
    truncated_body = " ".join(body_words).rstrip(" ,;") + "."

    if "Subject:" not in truncated_body:
        truncated_body = "Subject: Response from NovaSoft Technologies\nDear Customer,\n" + truncated_body
    elif "Dear Customer," not in truncated_body:
        truncated_body = truncated_body.replace("\n", "\nDear Customer,\n", 1)

    return truncated_body + signature


if __name__ == "__main__":
    # Quick smoke-test — run: python -m app.rag_chain
    test_query = "What is NovaSoft's refund policy for cancelled subscriptions?"
    result = generate_rag_email(test_query)
    print(f"\n{'='*60}")
    print("GENERATED EMAIL REPLY:")
    print("="*60)
    print(result["reply"])
    print(f"\n{'='*60}")
    print("CONTEXT USED:")
    print("="*60)
    print(result["context"])
