from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from app.llm_model import llm


@tool
def generate_spam_reply(subject: str, body: str) -> str:
    """
    Generate a polite and professional reply to spam emails.
    
    This tool is called automatically when an email is detected as spam or unwanted.
    It creates a courteous, professional response that discourages further contact 
    without being rude or offensive. The response is formal, concise, and appropriate 
    for business communication.
    
    Args:
        subject (str): The subject line of the spam email
        body (str): The body/content of the spam email
    
    Returns:
        str: A polite, professional reply (2-3 sentences) to the spam email
    """
    
    template = PromptTemplate(
        template="""You are a professional corporate email assistant. Your task is to generate a professional company response to unsolicited emails.

Your company handles spam emails professionally by:
1. Thanking them for contacting the company
2. Briefly mentioning that the content may not be relevant to current business needs
3. Inviting them to reach out if they have queries about the company's services/products
4. Keeping a warm, professional tone without being dismissive

Guidelines:
- Keep reply SHORT (2-3 sentences maximum)
- Start with "Thank you for reaching out" or similar
- Mention a random company domain/service (e.g., "our services", "our solutions", "our products")
- Be FORMAL and PROFESSIONAL
- Maintain a warm, welcoming tone
- Do NOT be rude or dismissive
- Do NOT apologize excessively
- Provide ONLY the email body text, nothing else
- End with an invitation to contact if they have relevant queries

Email Details:
Subject: {subject}
Body: {body}

Generate a professional company reply to this email:""",
        input_variables=["subject", "body"],
        validate_template=True
    )
    
    prompt = template.invoke({"subject": subject, "body": body})
    result = llm.invoke(prompt)
    
    return result.content.strip()
