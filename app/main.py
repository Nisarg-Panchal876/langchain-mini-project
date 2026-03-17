from fastapi import FastAPI, HTTPException
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from email.utils import parseaddr
from typing import Any
from uuid import uuid4
import requests
from app.schemas import (
    ApprovalLinks,
    ApprovalActionResponse,
    AnalyzeResponse,
    GetEmail,
    LatestEmailRecord,
    ReplyGenerationResponse,
    LeadScoringData,
    SalesIntentResponse,
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
COMPANY_ENRICHMENT_API_KEY = "Wj56W5puuAIh78b86fLXdWgNyBcCPl2I"
PERSONAL_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "icloud.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
    "gmx.com",
    "mail.com",
    "yandex.com",
}
TARGET_INDUSTRY_KEYWORDS = {
    "software",
    "technology",
    "saas",
    "fintech",
    "it services",
    "cloud",
    "artificial intelligence",
}


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


def has_sales_purchase_signals(subject: str, body: str) -> bool:
    """Heuristic sales-intent detector using subject + body text."""
    text = f"{subject} {body}".lower()
    sales_keywords = [
        "price",
        "pricing",
        "quote",
        "buy",
        "purchase",
        "demo",
        "plan",
        "subscription",
        "enterprise",
        "sales",
        "trial",
        "proposal",
    ]
    return any(keyword in text for keyword in sales_keywords)


def extract_sender_domain(sender_email: str) -> str:
    """Extract and normalize domain from sender email."""
    _, parsed_email = parseaddr(sender_email or "")
    candidate_email = (parsed_email or sender_email or "").strip()

    if "@" not in candidate_email:
        return ""

    domain = candidate_email.rsplit("@", 1)[1].strip().lower()
    return domain.strip("<>\"'()[]{}.,; ")


def parse_numeric_value(value: Any) -> float | None:
    """Parse numeric values from API fields that might be numbers or strings."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    cleaned = value.strip().replace(",", "").lower()
    cleaned = re.sub(
        r"(?i)^(over|under|approx|approximately)\s*-?", "", cleaned)
    cleaned = cleaned.replace("+", "")

    def parse_single_numeric_token(token: str) -> float | None:
        token = token.strip().lower()
        if not token:
            return None

        multiplier = 1.0
        if token.endswith("bn"):
            multiplier = 1_000_000_000.0
            token = token[:-2]
        elif token.endswith("mn"):
            multiplier = 1_000_000.0
            token = token[:-2]
        elif token.endswith("b"):
            multiplier = 1_000_000_000.0
            token = token[:-1]
        elif token.endswith("m"):
            multiplier = 1_000_000.0
            token = token[:-1]
        elif token.endswith("k"):
            multiplier = 1_000.0
            token = token[:-1]

        token = re.sub(r"[^0-9.]", "", token)
        if not token:
            return None

        try:
            return float(token) * multiplier
        except ValueError:
            return None

    # Handle ranges such as 10m-50m or 500m-1b.
    range_match = re.match(
        r"^\s*([0-9.]+\s*[a-z]{0,2})\s*-\s*([0-9.]+\s*[a-z]{0,2})\s*$", cleaned)
    if range_match:
        lower_bound = parse_single_numeric_token(range_match.group(1))
        upper_bound = parse_single_numeric_token(range_match.group(2))
        if upper_bound is not None:
            return upper_bound
        return lower_bound

    return parse_single_numeric_token(cleaned)


def fetch_company_enrichment(domain: str) -> LeadScoringData:
    """Fetch company enrichment data for a company domain."""
    url = f"https://api.thecompaniesapi.com/v2/companies/{domain}"
    try:
        response = requests.get(
            url,
            params={"token": COMPANY_ENRICHMENT_API_KEY},
            timeout=10,
        )
    except requests.RequestException:
        return LeadScoringData()

    # 404 means the company record was not found for this domain.
    # Treat it as missing enrichment, not an API failure for the whole request.
    if response.status_code == 404:
        return LeadScoringData()

    if response.status_code >= 400:
        return LeadScoringData()

    try:
        data = response.json()
    except ValueError:
        return LeadScoringData()

    about_data = data.get("about", {}) if isinstance(data, dict) else {}
    analytics_data = data.get(
        "analytics", {}) if isinstance(data, dict) else {}
    finances_data = data.get("finances", {}) if isinstance(data, dict) else {}
    locations_data = data.get(
        "locations", {}) if isinstance(data, dict) else {}
    headquarters = locations_data.get(
        "headquarters", {}) if isinstance(locations_data, dict) else {}

    city_name = ((headquarters.get("city") or {}).get("name")
                 if isinstance(headquarters, dict) else None)
    state_name = ((headquarters.get("state") or {}).get("name")
                  if isinstance(headquarters, dict) else None)
    country_name = ((headquarters.get("country") or {}).get(
        "name") if isinstance(headquarters, dict) else None)
    location_parts = [part for part in [
        city_name, state_name, country_name] if part]
    normalized_location = ", ".join(location_parts) if location_parts else None

    industries = about_data.get("industries") if isinstance(
        about_data, dict) else None
    if isinstance(industries, list) and industries:
        normalized_industry = ", ".join(str(item)
                                        for item in industries if item)
    else:
        normalized_industry = None

    return LeadScoringData(
        company_name=(
            data.get("company_name")
            or about_data.get("name")
            or about_data.get("nameLegal")
        ),
        industry=(
            data.get("industry")
            or about_data.get("industry")
            or normalized_industry
        ),
        employee_count=parse_numeric_value(
            data.get("employee_count")
            or about_data.get("totalEmployeesExact")
            or about_data.get("totalEmployees")
        ),
        revenue=parse_numeric_value(
            data.get("revenue")
            or finances_data.get("revenue")
        ),
        location=(
            data.get("location")
            or normalized_location
            or about_data.get("yearFoundedPlace")
        ),
        monthly_visitors=parse_numeric_value(
            data.get("monthly_visitors")
            or analytics_data.get("monthlyVisitors")
        ),
    )


def classify_lead_from_score(score: int) -> str:
    if score > 12:
        return "High-value lead"
    if 6 <= score <= 12:
        return "Medium lead"
    return "Low lead"


def compute_lead_score(
    is_company_email: bool,
    enrichment_data: LeadScoringData,
) -> tuple[int, list[str]]:
    """Compute lead score and return scoring reasons."""
    score = 0
    reasons: list[str] = []

    if enrichment_data.employee_count and enrichment_data.employee_count > 1000:
        score += 5
        reasons.append("Employees > 1000 (+5)")

    if enrichment_data.revenue and enrichment_data.revenue > 1_000_000_000:
        score += 5
        reasons.append("Revenue > 1B (+5)")

    industry_value = (enrichment_data.industry or "").lower().replace(
        "-", " ").replace("_", " ")
    if any(keyword in industry_value for keyword in TARGET_INDUSTRY_KEYWORDS):
        score += 3
        reasons.append("Target industry (+3)")

    if enrichment_data.monthly_visitors and enrichment_data.monthly_visitors > 1_000_000:
        score += 2
        reasons.append("High traffic (+2)")

    if is_company_email:
        score += 2
        reasons.append("Company email (+2)")

    return score, reasons


def compute_customer_need_score(subject: str, body: str) -> tuple[int, list[str]]:
    """Estimate customer demand/order size score between 0 and 10 using heuristics."""
    text = f"{subject} {body}".lower()
    score = 1
    signals: list[str] = []

    if any(word in text for word in ["pricing", "price", "quote", "demo", "proposal", "purchase", "buy"]):
        score += 3
        signals.append("Buying intent keywords detected")

    if any(word in text for word in ["enterprise", "contract", "annual", "license", "licenses", "users", "seats"]):
        score += 2
        signals.append("Commercial scope terms detected")

    if any(word in text for word in ["urgent", "asap", "immediately", "this week", "this month", "timeline"]):
        score += 2
        signals.append("Urgency or timeline detected")

    number_matches = re.findall(r"\b\d{2,}\b", text)
    if number_matches:
        largest_number = max(int(item) for item in number_matches)
        if largest_number >= 500:
            score += 2
            signals.append("Large numeric requirement detected")
        elif largest_number >= 50:
            score += 1
            signals.append("Moderate numeric requirement detected")

    if re.search(r"[$€£]\s*\d+|\b\d+\s*(usd|eur|inr|dollars)\b", text):
        score += 1
        signals.append("Budget signal detected")

    score = max(0, min(10, score))
    if not signals:
        signals.append("Limited buying signals found")
    return score, signals


def compute_order_size_score(subject: str, body: str) -> tuple[int, str]:
    """Estimate order size from email and return a score between 1 and 5 using heuristics."""
    text = f"{subject} {body}".lower()
    number_matches = [int(item) for item in re.findall(r"\b\d{2,}\b", text)]
    largest_number = max(number_matches) if number_matches else 0

    if largest_number >= 1000 or "enterprise" in text:
        return 5, "Large enterprise-scale order signal detected"
    if largest_number >= 500 or "annual contract" in text:
        return 4, "Large order/ticket indicators detected"
    if largest_number >= 100 or any(word in text for word in ["team", "department", "multi-site"]):
        return 3, "Medium order indicators detected"
    if any(word in text for word in ["trial", "pilot", "few users", "basic"]):
        return 2, "Small order indicators detected"
    return 1, "No clear order-size signal detected"


def parse_llm_subject_body(content: Any) -> tuple[str | None, str | None]:
    """Extract subject/body from model output that may contain JSON or wrapped JSON."""
    if not isinstance(content, str) or not content.strip():
        return None, None

    text = content.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            subject = parsed.get("subject")
            body = parsed.get("body")
            return subject, body
    except Exception:
        pass

    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed.get("subject"), parsed.get("body")
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            parsed = json.loads(text[start: end + 1])
            if isinstance(parsed, dict):
                return parsed.get("subject"), parsed.get("body")
        except Exception:
            pass

    return None, None


def build_personal_reply_fallback(subject: str, body: str) -> tuple[str, str]:
    """Build a deterministic, useful fallback reply for personal domains."""
    normalized_subject = (subject or "").strip()
    normalized_body = (body or "").strip()

    first_line = re.split(r"[\n.!?]", normalized_body)[
        0].strip() if normalized_body else ""
    topic = first_line if first_line else "your request"
    if len(topic) > 120:
        topic = topic[:117].rstrip() + "..."

    reply_subject = f"Re: {normalized_subject}" if normalized_subject else "Re: Your Inquiry"
    reply_body = (
        "Hello,\n\n"
        f"Thank you for reaching out. We have received your message regarding {topic}.\n"
        "Our team is reviewing your request and will share the next steps shortly.\n\n"
        "Best regards,\n"
        "Support Team"
    )
    return reply_subject, reply_body


def finalize_personal_reply(
    generated_subject: str | None,
    generated_body: str | None,
    original_subject: str,
    sender_body: str,
) -> tuple[str, str]:
    """Ensure personal-domain replies always have a complete subject and body."""
    fallback_subject, fallback_body = build_personal_reply_fallback(
        original_subject, sender_body)

    subject = (generated_subject or "").strip()
    body = (generated_body or "").strip()

    if not subject:
        subject = fallback_subject
    elif not subject.lower().startswith("re:") and original_subject.strip():
        subject = f"Re: {subject}"

    # Keep reply professional and complete even when the model returns very short text.
    if len(body) < 80:
        body = fallback_body

    return subject, body


def generate_sales_intent_response(
    sender_email: str,
    original_subject: str,
    sender_body: str,
    category: str,
    include_lead_details: bool,
    lead_score: int,
    lead_classification: str,
    lead_data: LeadScoringData,
    scoring_reasons: list[str],
) -> tuple[str, str]:
    """Generate AI-driven subject/body response with optional lead details."""
    if not include_lead_details:
        normal_prompt = PromptTemplate(
            template="""
You are a professional email assistant.
Create a professional reply in JSON only with keys: "subject" and "body".

Context:
- Sender Email: {sender_email}
- Original Subject: {original_subject}
- Sender Body: {sender_body}
- Classified Category: {category}

Instructions:
- Write a natural auto-generated reply to the sender.
- Subject must be clear and email-ready.
- Body must be complete and include:
    1) Greeting
    2) Acknowledgement of sender request
    3) Next step or expected follow-up
    4) Professional closing
- Do not include lead score, lead classification, scoring logic, or enrichment data.
- Keep the message polite, practical, and at least 4 lines.
- Output valid JSON only, no markdown.
""",
            input_variables=[
                "sender_email",
                "original_subject",
                "sender_body",
                "category",
            ],
            validate_template=True,
        )

        normal_prompt_value = normal_prompt.invoke(
            {
                "sender_email": sender_email,
                "original_subject": original_subject,
                "sender_body": sender_body,
                "category": category,
            }
        )

        try:
            llm_result = llm.invoke(normal_prompt_value)
            parsed_subject, parsed_body = parse_llm_subject_body(
                llm_result.content)
            return finalize_personal_reply(
                parsed_subject,
                parsed_body,
                original_subject,
                sender_body,
            )
        except Exception:
            return build_personal_reply_fallback(original_subject, sender_body)

    prompt = PromptTemplate(
        template="""
You are an outbound business email assistant.
Create a concise email response output in JSON only with keys: "subject" and "body".

Context:
- Sender Email: {sender_email}
- Original Subject: {original_subject}
- Sender Body: {sender_body}
- Classified Category: {category}
- Lead Score: {lead_score}
- Lead Classification: {lead_classification}
- Scoring Logic Used: {scoring_reasons}
- Main Data used in Lead Scoring:
  - company_name: {company_name}
  - industry: {industry}
  - employee_count: {employee_count}
  - revenue: {revenue}
  - location: {location}
  - monthly_visitors: {monthly_visitors}

Instructions:
- Subject should reflect priority.
- If lead classification is High-value lead, make subject urgent (example style: High Priority Lead Contact Now !!).
- Body must include:
  1) A brief reply to sender's message.
  2) A short section describing lead scoring logic used.
  3) Sender email and any sender contact details seen in sender body.
  4) Main data used in lead scoring.
- Explain that final lead score = company strength score + customer need score.
- Keep body practical and professional.
- Output valid JSON only, no markdown.
""",
        input_variables=[
            "sender_email",
            "original_subject",
            "sender_body",
            "category",
            "lead_score",
            "lead_classification",
            "scoring_reasons",
            "company_name",
            "industry",
            "employee_count",
            "revenue",
            "location",
            "monthly_visitors",
        ],
        validate_template=True,
    )

    prompt_value = prompt.invoke(
        {
            "sender_email": sender_email,
            "original_subject": original_subject,
            "sender_body": sender_body,
            "category": category,
            "lead_score": str(lead_score),
            "lead_classification": lead_classification,
            "scoring_reasons": "; ".join(scoring_reasons) if scoring_reasons else "No score drivers matched",
            "company_name": lead_data.company_name or "N/A",
            "industry": lead_data.industry or "N/A",
            "employee_count": str(lead_data.employee_count) if lead_data.employee_count is not None else "N/A",
            "revenue": str(lead_data.revenue) if lead_data.revenue is not None else "N/A",
            "location": lead_data.location or "N/A",
            "monthly_visitors": str(lead_data.monthly_visitors) if lead_data.monthly_visitors is not None else "N/A",
        }
    )

    try:
        llm_result = llm.invoke(prompt_value)
        parsed_subject, parsed_body = parse_llm_subject_body(
            llm_result.content)
        subject = parsed_subject or "Lead Update"
        body = parsed_body or "Thank you for your message."
        return subject, body
    except Exception:
        # Deterministic fallback when model output is unavailable/non-JSON.
        if lead_classification == "High-value lead":
            subject = "High Priority Lead Contact Now !!"
        elif lead_classification == "Medium lead":
            subject = "Medium Priority Lead Follow-up"
        else:
            subject = "Low Priority Lead Review"

        body = (
            f"Sender's Body:\n{sender_body}\n\n"
            f"Sender Email: {sender_email}\n"
            f"Lead Classification: {lead_classification}\n"
            f"Lead Score: {lead_score}\n"
            f"Lead Scoring Logic Used: {', '.join(scoring_reasons) if scoring_reasons else 'No score drivers matched'}\n\n"
            "Main Data used in Lead Scoring:\n"
            f"- Company Name: {lead_data.company_name or 'N/A'}\n"
            f"- Industry: {lead_data.industry or 'N/A'}\n"
            f"- Employee Count: {lead_data.employee_count if lead_data.employee_count is not None else 'N/A'}\n"
            f"- Revenue: {lead_data.revenue if lead_data.revenue is not None else 'N/A'}\n"
            f"- Location: {lead_data.location or 'N/A'}\n"
            f"- Monthly Visitors: {lead_data.monthly_visitors if lead_data.monthly_visitors is not None else 'N/A'}\n"
        )
        return subject, body


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_intent(data: GetEmail):
    try:
        category = classify_email_subject(data.subject)
    except Exception as exc:
        if is_quota_or_rate_limit_error(exc):
            category = classify_email_subject_fallback(data.subject)
        else:
            raise HTTPException(
                status_code=502, detail=f"LLM classification failed: {exc}") from exc
    return {"category": category}


@app.post("/spam-reply", response_model=ReplyGenerationResponse, status_code=202)
def spam_reply(data: GetEmail):
    """Generate a spam reply and store it for approval before sending."""
    try:
        reply = generate_spam_reply.invoke(
            {"subject": data.subject, "body": data.body})
    except Exception as exc:
        if is_quota_or_rate_limit_error(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Reply generation is temporarily unavailable due to model quota/rate limits. "
                    "Please retry after a minute or increase API quota."
                ),
            ) from exc
        raise HTTPException(
            status_code=502, detail=f"Reply generation failed: {exc}") from exc

    store_latest_email(data.sender_email, data.subject, reply)
    approval_id = create_pending_approval(
        data.sender_email, data.subject, reply)
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
        raise HTTPException(
            status_code=502, detail=f"RAG generation failed: {exc}") from exc

    store_latest_email(data.sender_email, data.subject, result["reply"])
    approval_id = create_pending_approval(
        data.sender_email, data.subject, result["reply"])
    approval_links = build_approval_links(approval_id)
    send_approval_request(data.sender_email, data.subject,
                          result["reply"], approval_id)

    return {
        "sender_email": data.sender_email,
        "subject": data.subject,
        "reply": result["reply"],
        "message": "Reply generated and stored for department approval.",
        "approval_links": approval_links,
        "context_used": result["context"],
    }


@app.post("/sales-purchase-intent", response_model=SalesIntentResponse)
def sales_purchase_intent(data: GetEmail):
    """
    Process Sales / Purchase Intent emails with domain enrichment and lead scoring.
    For company domains, always enrich lead data before scoring.
    """
    category = "Sales / Purchase Intent"

    sender_domain = extract_sender_domain(data.sender_email)
    is_personal_domain = sender_domain in PERSONAL_EMAIL_DOMAINS or not sender_domain
    is_company_domain = bool(sender_domain) and not is_personal_domain

    lead_data = LeadScoringData()
    enrichment_api_called = False

    if is_company_domain:
        lead_data = fetch_company_enrichment(sender_domain)
        enrichment_api_called = True

    base_lead_score, scoring_reasons = compute_lead_score(
        is_company_domain, lead_data)

    email_content_score, email_content_signals = compute_customer_need_score(
        data.subject,
        data.body,
    )

    order_size_score, order_size_reason = compute_order_size_score(
        data.subject,
        data.body,
    )

    # Final score = base lead score + email content score + order size score.
    lead_score = base_lead_score + email_content_score + order_size_score
    scoring_reasons = scoring_reasons + [
        f"Email content score +{email_content_score} based on: {', '.join(email_content_signals)}",
        f"Order size score +{order_size_score} (1-5) based on: {order_size_reason}",
    ]
    lead_classification = classify_lead_from_score(lead_score)

    generated_subject, generated_body = generate_sales_intent_response(
        sender_email=data.sender_email,
        original_subject=data.subject,
        sender_body=data.body,
        category=category,
        include_lead_details=is_company_domain,
        lead_score=lead_score,
        lead_classification=lead_classification,
        lead_data=lead_data,
        scoring_reasons=scoring_reasons,
    )

    return {
        "sender_email": data.sender_email,
        "sender_domain": sender_domain,
        "category": category,
        "is_personal_domain": is_personal_domain,
        "enrichment_api_called": enrichment_api_called,
        "lead_score": lead_score,
        "lead_classification": lead_classification,
        "generated_subject": generated_subject,
        "generated_body": generated_body,
        "lead_data_used": lead_data,
    }


@app.get("/approve", response_model=ApprovalActionResponse)
def approve_reply(id: str):
    """Mark a pending approval record as approved by ID."""
    record = pending_approvals.get(id)
    if not record:
        raise HTTPException(
            status_code=404, detail="Approval record not found")
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
        raise HTTPException(
            status_code=404, detail="Approval record not found")
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
