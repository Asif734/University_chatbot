# agents/memory_manager.py
from typing import List, Dict

class MemoryManager:
    """
    Stores conversation history for multiple users or sessions.
    Provides methods to add messages, get summaries, and fetch recent context.
    """

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.buffer: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """
        Add a message to the buffer. Role can be 'User' or 'Assistant'.
        Keeps only the most recent `max_history` messages.
        """
        self.buffer.append({"role": role, "content": content})
        if len(self.buffer) > self.max_history:
            self.buffer.pop(0)  # remove oldest message to maintain max history

    def get_recent(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Get the last `n` messages from buffer.
        """
        return self.buffer[-n:]

    def get_summary(self) -> str:
        """
        Returns a simple text summary of all messages in buffer.
        Can be used as context for RAG or LLM prompts.
        """
        summary_lines = [f"{m['role']}: {m['content']}" for m in self.buffer]
        return "\n".join(summary_lines)

    def clear(self):
        """
        Clear the conversation buffer.
        """
        self.buffer = []
