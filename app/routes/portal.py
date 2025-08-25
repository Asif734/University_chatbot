# app/routes/portal.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.agents.orchestration_agent import OrchestrationAgent
from app.agents.public_agent import PublicAgent
from app.agents.private_agent import PrivateAgent
from app.agents.mental_health_agent import MentalHealthAgent
from app.agents.memory_manager import MemoryManager
from app.agents.llm_interface import local_llm
from app.utils.auth import get_current_user
from app.utils.email_alert import send_alert_if_needed

router = APIRouter()
# Single OrchestrationAgent instance
orchestration_agent = OrchestrationAgent(
    llm=local_llm,
    public_agent=PublicAgent(),
    private_agent=PrivateAgent(),
    mental_health_agent=MentalHealthAgent(),
    memory_manager=MemoryManager(max_history=20)
)

class PortalChatRequest(BaseModel):
    message: str
@router.get("/portal/chat")
async def chat_endpoint(user=Depends(get_current_user)):
    return {"message": f"Hello {user.full_name}"}

@router.post("/chat")
async def portal_chat(req: PortalChatRequest, user=Depends(get_current_user)):
    response, category, risk_flag = orchestration_agent.handle_message(req.message, user_id=user.id)
    
    # If mental health risk detected, send email alert
    if risk_flag:
        send_alert_if_needed(user, req.message)
    
    return {"response": response, "category": category}
