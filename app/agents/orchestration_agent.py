# agents/orchestration_agent.py
from .memory_manager import MemoryManager

from app.utils.auth import create_access_token

access_token = create_access_token({"sub": "student123", "email": "student@uni.edu"})
class OrchestrationAgent:
    """
    Routes user messages to the correct agent:
    - PUBLIC (RAG-enabled)
    - PRIVATE (student data)
    - MENTAL_HEALTH (retrieval + LLM fallback)
    Tracks conversation in memory and raises admin alerts if needed.
    """
    def __init__(self, llm, public_agent, private_agent, mental_health_agent, memory_manager: MemoryManager):
        self.llm = llm
        self.public_agent = public_agent
        self.private_agent = private_agent
        self.mental_health_agent = mental_health_agent
        self.memory_manager = memory_manager

    # ------------------ Classification ------------------
    def classify_message(self, message: str) -> str:
        """
        Lightweight classification: PUBLIC, PRIVATE, MENTAL_HEALTH
        - Use keywords to save computation
        """
        keywords_private = ["grades", "result", "cgpa", "due", "password", "id", "login"]
        keywords_mental = ["stress", "depression", "anxiety", "help", "sad", "suicide"]

        msg_lower = message.lower()
        if any(k in msg_lower for k in keywords_private):
            return "PRIVATE"
        elif any(k in msg_lower for k in keywords_mental):
            return "MENTAL_HEALTH"
        else:
            return "PUBLIC"

    # ------------------ Handle Message ------------------
    def handle_message(self, message: str, student_id: str = None) -> str:
        category = self.classify_message(message)
        self.memory_manager.add_message("User", f"[{category}] {message}")

        # Route to correct agent
        if category == "PUBLIC":
            response = self.public_agent.respond(message)
        elif category == "PRIVATE":
            if student_id is None:
                response = "Private information requires student login."
            else:
                response = self.private_agent.respond(message, student_id=student_id)
        elif category == "MENTAL_HEALTH":
            response = self.mental_health_agent.respond(message)
            # Optional: trigger admin alert for high-risk messages
            if self._is_high_risk(message):
                self._notify_admin(student_id, message)
        else:
            response = "Sorry, I couldn’t classify your request."

        self.memory_manager.add_message("Assistant", response)
        return response

    # ------------------ Admin Alert ------------------
    def _is_high_risk(self, message: str) -> bool:
        """
        Detect high-risk messages for mental health.
        Very simple keyword-based for now. Extendable.
        """
        high_risk_keywords = ["suicide", "die", "kill myself", "harm"]
        msg_lower = message.lower()
        return any(k in msg_lower for k in high_risk_keywords)

    def _notify_admin(self, student_id: str, message: str):
        """
        Send email alert to admin. Here we just print for now.
        """
        print(f"[ALERT] High-risk mental health message from {student_id}: {message}")
        # Integrate with SMTP, Twilio, or internal admin notification
