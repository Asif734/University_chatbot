from fastapi import FastAPI, UploadFile, Form
from typing import List
from app.agents.public_agent_rag import PublicAgentRAG
import shutil
import os

app = FastAPI()
agent = PublicAgentRAG()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------- Upload Endpoint -----------------
@app.post("/upload/")
async def upload_files(files: List[UploadFile]):
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        agent.add_document(file_path)
    return {"status": "Files ingested successfully", "files": [file.filename for file in files]}


# ----------------- Add Raw Text Endpoint (textarea) -----------------
@app.post("/add_text/")
async def add_text(
    text: str = Form(..., description="Paste your full article or long text here."),
    source: str = Form("manual_input", description="Optional source label for the text")
):
    agent.add_text(text, source=source)
    return {"status": "Text ingested successfully", "source": source}


# ----------------- Query Endpoint -----------------
@app.post("/query/")
async def query_agent(query: str = Form(..., description="Enter your question here")):
    response = agent.query(query)
    return {"answer": response}
