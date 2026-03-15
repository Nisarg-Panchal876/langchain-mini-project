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