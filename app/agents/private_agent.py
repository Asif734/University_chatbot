# agents/private_agent.py
import json
from .llm_interface import local_llm

class PrivateAgent:
    def __init__(self, json_file=r"C:\Users\Asif\VSCODE\University Chatbot\app\data\private_student.json"):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def generate_prompt(self, message: str, student_id: str):
        student = next((s for s in self.data if s["student_id"] == student_id), None)
        if not student:
            return "[PrivateAgent] Student not found."

        prompt = (
            f"Use the following student info to answer safely:\n"
            f"Name: {student['name']}\n"
            f"Grades: {student.get('grades', '')}\n"
            f"Schedule: {student.get('schedule', '')}\n"
            f"Question: {message}\nAnswer:"
        )
        return prompt

    def respond(self, message: str, student_id: str):
        prompt = self.generate_prompt(message, student_id)
        return local_llm(prompt)
