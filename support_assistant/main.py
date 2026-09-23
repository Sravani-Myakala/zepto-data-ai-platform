from fastapi import FastAPI
from pydantic import BaseModel, Field

from support_assistant.assistant import ask_assistant


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto policy support assistant",
    version="1.0.0"
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="User's question"
    )


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant is running"
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    response = ask_assistant(request.query)

    return AskResponse(
        answer=response.answer,
        sources=response.sources,
        confidence=response.confidence
    )