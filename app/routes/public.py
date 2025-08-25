# app/routes/public.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.agents.public_agent import PublicAgent

router = APIRouter()
public_agent = PublicAgent()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def public_chat(req: ChatRequest):
    response = public_agent.respond(req.message)
    return {"response": response}
