from pydantic import BaseModel

class SendMessageRequest(BaseModel):
    to: str
    message: str
