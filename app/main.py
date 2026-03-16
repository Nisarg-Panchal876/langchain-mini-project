from fastapi import FastAPI, HTTPException
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
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
pending_approvals: dict[str, dict[str, str]] = {}
APPROVAL_BASE_URL = "http://localhost:8000"
DEPARTMENT_APPROVER_EMAIL = "approvals@novasoft.local"
N8N_APPROVAL_WEBHOOK_URL = os.getenv(
    "N8N_APPROVAL_WEBHOOK_URL",
    "http://localhost:5678/webhook/9a712a44-58be-4ba1-b9ab-ea879c957afe",
)


def send_email(to_email: str, subject: str, body: str) -> None:
    """Simulate sending an email reply to the original sender."""
    print(f"[Email] Sending to={to_email} subject={subject!r}")
    print(f"[Email] Body:\n{body}")


def build_approval_links(approval_id: str) -> ApprovalLinks:
    """Build approval and rejection URLs for one approval record."""
    return ApprovalLinks(
        approve=f"{APPROVAL_BASE_URL}/approve?id={approval_id}",
        reject=f"{APPROVAL_BASE_URL}/reject?id={approval_id}",
    )


def create_pending_approval(sender_email: str, subject: str, reply_text: str) -> str:
    """Create a pending approval record and return its ID."""
    approval_id = uuid4().hex
    pending_approvals[approval_id] = {
        "sender_email": sender_email,
        "subject": subject,
        "reply_text": reply_text,
        "status": "pending",
    }
    return approval_id


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


def send_approval_request(
    sender_email: str,
    subject: str,
    reply_text: str,
    approval_id: str,
) -> None:
    """Simulate sending a department approval request email."""
    links = build_approval_links(approval_id)
    body = (
        "A generated reply is waiting for department approval.\n\n"
        f"Approval ID: {approval_id}\n"
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


def notify_n8n_approval_event(approval_id: str, action: str, record: dict[str, str]) -> None:
    """Send an approval/rejection event to n8n webhook."""
    payload = {
        "approval_id": approval_id,
        "action": action,
        "sender_email": record["sender_email"],
        "subject": record["subject"],
        "reply_text": record["reply_text"],
    }
    request = Request(
        N8N_APPROVAL_WEBHOOK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=8) as response:
            status_code = response.getcode()
            if status_code < 200 or status_code >= 300:
                raise HTTPException(
                    status_code=502,
                    detail=f"n8n webhook returned non-success status: {status_code}",
                )
    except HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"n8n webhook HTTP error: {exc.code}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to reach n8n webhook: {exc.reason}",
        ) from exc


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


def is_quota_or_rate_limit_error(exc: Exception) -> bool:
    """Return True when the exception indicates model quota/rate limiting."""
    message = str(exc).lower()
    return (
        "resource_exhausted" in message
        or "quota" in message
        or "rate limit" in message
        or "429" in message
    )


def classify_email_subject_fallback(subject: str) -> str:
    """Simple local fallback when the LLM is unavailable."""
    text = subject.lower()
    if any(word in text for word in ["issue", "error", "unable", "problem", "crash", "support", "help"]):
        return "Customer Support / Issue Resolution"
    if any(word in text for word in ["price", "pricing", "quote", "buy", "purchase", "demo", "plan"]):
        return "Sales / Purchase Intent"
    if any(word in text for word in ["hiring", "career", "company", "about", "partnership", "contact"]):
        return "Company Inquiry"
    return "Irrelevant / Casual"

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_intent(data: GetEmail):
    try:
        category = classify_email_subject(data.subject)
    except Exception as exc:
        if is_quota_or_rate_limit_error(exc):
            category = classify_email_subject_fallback(data.subject)
        else:
            raise HTTPException(status_code=502, detail=f"LLM classification failed: {exc}") from exc
    return {"category": category}


@app.post("/spam-reply", response_model=ReplyGenerationResponse, status_code=202)
def spam_reply(data: GetEmail):
    """Generate a spam reply and store it for approval before sending."""
    try:
        reply = generate_spam_reply.invoke({"subject": data.subject, "body": data.body})
    except Exception as exc:
        if is_quota_or_rate_limit_error(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Reply generation is temporarily unavailable due to model quota/rate limits. "
                    "Please retry after a minute or increase API quota."
                ),
            ) from exc
        raise HTTPException(status_code=502, detail=f"Reply generation failed: {exc}") from exc

    store_latest_email(data.sender_email, data.subject, reply)
    approval_id = create_pending_approval(data.sender_email, data.subject, reply)
    approval_links = build_approval_links(approval_id)
    send_approval_request(data.sender_email, data.subject, reply, approval_id)

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
    except Exception as exc:
        if is_quota_or_rate_limit_error(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    "RAG reply generation is temporarily unavailable due to model quota/rate limits. "
                    "Please retry after a minute or increase API quota."
                ),
            ) from exc
        raise HTTPException(status_code=502, detail=f"RAG generation failed: {exc}") from exc

    store_latest_email(data.sender_email, data.subject, result["reply"])
    approval_id = create_pending_approval(data.sender_email, data.subject, result["reply"])
    approval_links = build_approval_links(approval_id)
    send_approval_request(data.sender_email, data.subject, result["reply"], approval_id)

    return {
        "sender_email": data.sender_email,
        "subject": data.subject,
        "reply": result["reply"],
        "message": "Reply generated and stored for department approval.",
        "approval_links": approval_links,
        "context_used": result["context"],
    }


@app.get("/approve", response_model=ApprovalActionResponse)
def approve_reply(id: str):
    """Mark a pending approval record as approved by ID."""
    record = pending_approvals.get(id)
    if not record:
        raise HTTPException(status_code=404, detail="Approval record not found")
    if record["status"] != "pending":
        return {"message": f"Approval already processed with status: {record['status']}"}

    record["status"] = "processing_approved"
    try:
        notify_n8n_approval_event(id, "approved", record)
    except HTTPException:
        record["status"] = "pending"
        raise

    record["status"] = "approved"
    return {"message": f"Approve clicked for id={id}"}


@app.get("/reject", response_model=ApprovalActionResponse)
def reject_reply(id: str):
    """Mark a pending approval record as rejected by ID."""
    record = pending_approvals.get(id)
    if not record:
        raise HTTPException(status_code=404, detail="Approval record not found")
    if record["status"] != "pending":
        return {"message": f"Approval already processed with status: {record['status']}"}

    record["status"] = "processing_rejected"
    try:
        notify_n8n_approval_event(id, "rejected", record)
    except HTTPException:
        record["status"] = "pending"
        raise

    record["status"] = "rejected"
    return {"message": f"Reject clicked for id={id}"}
