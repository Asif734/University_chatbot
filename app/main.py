# app/main.py
from fastapi import FastAPI
from app.routes import public, portal, docs_update

app = FastAPI(title="University Chatbot")

# Include routes
app.include_router(public.router, prefix="/public", tags=["Public Chat"])
app.include_router(portal.router, prefix="/portal", tags=["Student Portal Chat"])
app.include_router(docs_update.router, prefix="/docs", tags=["RAG Update"])
