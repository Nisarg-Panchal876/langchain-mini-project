from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm_model import llm
from app.retriever import retrieve_and_format

# ── Prompt Template ───────────────────────────────────────────────────────────
EMAIL_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an AI assistant working for NovaSoft Technologies.

Your task is to answer customer emails using ONLY the provided company knowledge base context.

Rules:
- Use only the information from the context.
- If specific numbers, prices, or figures are present in the context that are relevant to the question, you MUST include them in your response.
- If the context does not contain the answer, say you will forward the query to the relevant team.
- Write the response in a professional email format.
- Keep the response concise and helpful.
- Always end the email with:
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
    context = retrieve_and_format(query)

    # Step 2 — Generate email reply
    reply = rag_email_chain.invoke({"context": context, "question": query})

    print("[RAGChain] Email reply generated.")
    return {"reply": reply, "context": context}


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
