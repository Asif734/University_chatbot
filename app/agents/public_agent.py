# # agents/public_agent.py
# from typing import List
# from .llm_interface import local_llm
# from app.vectorstore.rag_store import RAGStore
# import textract  # For PDFs and images
# import pandas as pd

# class PublicAgent:
#     def __init__(self):
#         self.rag = RAGStore()

#     # ------------------ Adding Knowledge ------------------
#     def add_text(self, text: str):
#         self.rag.add_documents([text])

#     def add_file(self, filepath: str):
#         ext = filepath.split(".")[-1].lower()
#         if ext in ["txt"]:
#             with open(filepath, "r", encoding="utf-8") as f:
#                 text = f.read()
#         elif ext in ["pdf", "png", "jpg", "jpeg"]:
#             text = textract.process(filepath).decode("utf-8")
#         elif ext in ["xls", "xlsx"]:
#             df = pd.read_excel(filepath)
#             text = "\n".join([str(x) for x in df.values.flatten()])
#         else:
#             raise ValueError(f"Unsupported file type: {ext}")
#         self.add_text(text)

#     # ------------------ Querying ------------------
#     def generate_prompt(self, query: str, context: List[str]) -> str:
#         prompt = (
#             "You are a helpful university assistant. Use the following context to answer the question concisely.\n\n"
#         )
#         if context:
#             prompt += "\n---\n".join(context) + "\n\n"
#         prompt += f"Question: {query}\nAnswer:"
#         return prompt

#     def respond(self, query: str, top_k=3) -> str:
#         context = self.rag.query(query, top_k=top_k)
#         prompt = self.generate_prompt(query, context)
#         return local_llm(prompt)
    

#     def add_text(self, text: str):
#         self.rag.add_document(text)  # Use singular, your RAGStore has add_document

#public_agent.py
from typing import List
from .llm_interface import local_llm
from app.vectorstore.rag_store import RAGStore
import textract  # For PDFs and images
import pandas as pd


class PublicAgent:
    def __init__(self):
        self.rag = RAGStore()

    # ------------------ Adding Knowledge ------------------
    def add_text(self, text: str):
        """Add raw text into RAG store"""
        self.rag.add_document(text)

    def add_file(self, filepath: str):
        """Extract text from file and add into RAG store"""
        ext = filepath.split(".")[-1].lower()
        if ext in ["txt"]:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext in ["pdf", "png", "jpg", "jpeg"]:
            text = textract.process(filepath).decode("utf-8")
        elif ext in ["xls", "xlsx"]:
            df = pd.read_excel(filepath)
            text = "\n".join([str(x) for x in df.values.flatten()])
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        self.add_text(text)

    # ------------------ Querying ------------------
    def generate_prompt(self, query: str, context_chunks: List[dict]) -> str:
        """Build a prompt for the LLM using retrieved context"""
        prompt = (
            "You are a helpful university assistant. "
            "Use the following context to answer the question concisely.\n\n"
        )
        if context_chunks:
            prompt += "\n---\n".join([chunk["text"] for chunk in context_chunks]) + "\n\n"
        prompt += f"Question: {query}\nAnswer:"
        return prompt
        
    def respond(self, query: str, top_k=3) -> str:
        """Retrieve context from RAG + call LLM only if knowledge exists"""
        context_chunks = self.rag.search(query, top_k=top_k)

        if not context_chunks:
            # No knowledge available → return safe fallback
            return "Sorry, I don't have information on that."

        prompt = self.generate_prompt(query, context_chunks)
        return local_llm(prompt)