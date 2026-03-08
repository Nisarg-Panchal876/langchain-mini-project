from fastapi import FastAPI, HTTPException
from app.schemas import GetEmail
from langchain_core.prompts import PromptTemplate
from app.llm_model import llm
from app.spam import generate_spam_reply
from app.rag_chain import generate_rag_email

app = FastAPI()

@app.post("/analyze")
def analyze_intent(data : GetEmail):

        template = PromptTemplate(
            template="""
        You are an AI assistant working for a company.
Your ONLY task is to classify an email subject into exactly one of the following categories:

Company Inquiry

Sales / Purchase Intent

Irrelevant / Casual

Customer Support / Issue Resolution

Rules:

Use ONLY the email subject provided.

Choose exactly ONE category.

Do NOT explain your reasoning.

Do NOT add extra text.

Output ONLY the category name.

Examples:
Subject: Are you hiring backend interns?
Output: Company Inquiry

Subject: Need pricing details for your product
Output: Sales / Purchase Intent

Subject: Just wanted to say great work
Output: Irrelevant / Casual

If the email indicates a user complaint or issue regarding a service/product feature, such as:
"Unable to access my order history — app keeps crashing"
Output: Customer Support / Issue Resolution

Now classify the following email subject:
Subject: {subject}"""
,
            input_variables=["subject"],
            validate_template=True
        )

        prompt=template.invoke({"subject" : data.subject})

        result = llm.invoke(prompt)
        category = result.content.strip()
        
        # Check if email is spam and generate reply if needed
        spam_reply = None
        if category.lower() in ["irrelevant / casual", "irrelevant/casual"]:
            spam_reply = generate_spam_reply.invoke({"subject": data.subject, "body": data.body})
        
        return {"category": category, "is_spam": category.lower() in ["irrelevant / casual", "irrelevant/casual"], "spam_reply": spam_reply}


@app.post("/company-inquiry")
def company_inquiry(data: GetEmail):
    """
    RAG endpoint — answers company-related customer queries using the
    internal knowledge base PDF and generates a professional email reply.

    The query is built from the email subject + body for maximum context.
    """
    query = f"{data.subject}\n\n{data.body}".strip()

    try:
        result = generate_rag_email(query)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge base index not found. "
                "Please ensure company_docs.pdf is placed at "
                "data/company_docs.pdf and the vector store has been built. "
                f"Details: {str(e)}"
            ),
        )

    return {
        "sender_email": data.sender_email,
        "subject": data.subject,
        "reply": result["reply"],
        "context_used": result["context"],
    }