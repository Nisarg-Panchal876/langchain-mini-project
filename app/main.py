from fastapi import FastAPI, HTTPException
from app.schemas import (
    ApprovalLinks,
    ApprovalActionResponse,
    AnalyzeResponse,
    GetEmail,
    LatestEmailRecord,
    ReplyGenerationResponse,
)
from langchain_core.prompts import PromptTemplate
from app.llm_model import llm
from app.spam import generate_spam_reply
from app.rag_chain import generate_rag_email

app = FastAPI()

latest_email: dict[str, str] = {}
APPROVAL_BASE_URL = "http://localhost:8000"
DEPARTMENT_APPROVER_EMAIL = "approvals@novasoft.local"


def send_email(to_email: str, subject: str, body: str) -> None:
    """Simulate sending an email reply to the original sender."""
    print(f"[Email] Sending to={to_email} subject={subject!r}")
    print(f"[Email] Body:\n{body}")


def build_approval_links() -> ApprovalLinks:
    """Build the latest-email approval and rejection URLs."""
    return ApprovalLinks(
        approve=f"{APPROVAL_BASE_URL}/approve",
        reject=f"{APPROVAL_BASE_URL}/reject",
    )


def store_latest_email(sender_email: str, subject: str, reply_text: str) -> None:
    """Store the most recently generated reply for approval."""
    latest_email.clear()
    latest_email.update(
        LatestEmailRecord(
            sender_email=sender_email,
            subject=subject,
            reply_text=reply_text,
        ).model_dump()
    )


def send_approval_request(sender_email: str, subject: str, reply_text: str) -> None:
    """Simulate sending a department approval request email."""
    links = build_approval_links()
    body = (
        "A generated reply is waiting for department approval.\n\n"
        f"Original Sender: {sender_email}\n"
        f"Subject: {subject}\n\n"
        "Generated Reply:\n"
        f"{reply_text}\n\n"
        "Approve:\n"
        f"{links.approve}\n\n"
        "Reject:\n"
        f"{links.reject}"
    )
    send_email(
        to_email=DEPARTMENT_APPROVER_EMAIL,
        subject=f"Approval Needed: {subject}",
        body=body,
    )


def classify_email_subject(subject: str) -> str:
    """Classify an email subject into one of the supported intent categories."""
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
Subject: {subject}""",
        input_variables=["subject"],
        validate_template=True,
    )

    prompt = template.invoke({"subject": subject})
    result = llm.invoke(prompt)
    return result.content.strip()

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_intent(data: GetEmail):
    category = classify_email_subject(data.subject)
    return {"category": category}


@app.post("/spam-reply", response_model=ReplyGenerationResponse, status_code=202)
def spam_reply(data: GetEmail):
    """Generate a spam reply and store it for approval before sending."""
    reply = generate_spam_reply.invoke({"subject": data.subject, "body": data.body})
    store_latest_email(data.sender_email, data.subject, reply)
    approval_links = build_approval_links()
    send_approval_request(data.sender_email, data.subject, reply)

    return {
        "sender_email": data.sender_email,
        "subject": data.subject,
        "reply": reply,
        "message": "Reply generated and queued for department approval.",
        "approval_links": approval_links,
    }


@app.post("/company-inquiry", response_model=ReplyGenerationResponse, status_code=202)
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

    store_latest_email(data.sender_email, data.subject, result["reply"])
    approval_links = build_approval_links()
    send_approval_request(data.sender_email, data.subject, result["reply"])

    return {
        "sender_email": data.sender_email,
        "subject": data.subject,
        "reply": result["reply"],
        "message": "Reply generated and stored for department approval.",
        "approval_links": approval_links,
        "context_used": result["context"],
    }


@app.get("/approve", response_model=ApprovalActionResponse)
def approve_reply():
    """Return a simple confirmation when the approve link is clicked."""
    return {"message": "Approve clicked"}


@app.get("/reject", response_model=ApprovalActionResponse)
def reject_reply():
    """Return a simple confirmation when the reject link is clicked."""
    return {"message": "Reject clicked"}
