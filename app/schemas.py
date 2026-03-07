from pydantic import BaseModel
from typing import Optional 

class GetEmail(BaseModel):
    sender_email : str
    subject : str
    body : str