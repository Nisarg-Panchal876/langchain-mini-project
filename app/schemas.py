from typing import Optional

from pydantic import BaseModel


class GetEmail(BaseModel):
    sender_email: str
    subject: str
    body: str


class AnalyzeResponse(BaseModel):
    category: str


class ApprovalLinks(BaseModel):
    approve: str
    reject: str


class ReplyGenerationResponse(BaseModel):
    sender_email: str
    subject: str
    reply: str
    message: str
    approval_links: ApprovalLinks
    context_used: Optional[str] = None


class LatestEmailRecord(BaseModel):
    sender_email: str
    reply_text: str
    subject: str


class ApprovalActionResponse(BaseModel):
    message: Optional[str] = None
    error: Optional[str] = None


class LeadScoringData(BaseModel):
    company_name: Optional[str] = None
    industry: Optional[str] = None
    employee_count: Optional[float] = None
    revenue: Optional[float] = None
    location: Optional[str] = None
    monthly_visitors: Optional[float] = None


class SalesIntentResponse(BaseModel):
    sender_email: str
    sender_domain: str
    category: str
    is_personal_domain: bool
    enrichment_api_called: bool
    lead_score: int
    lead_classification: str
    generated_subject: str
    generated_body: str
    lead_data_used: LeadScoringData
