"""
RAG Document Q&A App
====================
A production-style Retrieval-Augmented Generation pipeline that lets users
upload PDFs/text files and ask questions answered using their content.

Stack: FastAPI · LangChain · FAISS · OpenAI · sentence-transformers
"""

import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag.ingestion import ingest_document
from rag.retriever import Retriever
from rag.generator import generate_answer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document Q&A",
    description="Upload documents, ask questions — answers grounded in your content.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory session store  {session_id: Retriever}
sessions: dict[str, Retriever] = {}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Schemas ────────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 4           # how many chunks to retrieve


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]       # the retrieved chunks used as context
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks_indexed: int
    message: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse("static/index.html")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Ingest a document into a FAISS vector store.
    Returns a session_id used for subsequent Q&A.
    """
    allowed = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use: {allowed}")

    # Save upload
    session_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{session_id}{suffix}"
    content = await file.read()
    save_path.write_bytes(content)
    log.info("Saved upload → %s", save_path)

    # Ingest: load → chunk → embed → index
    chunks = ingest_document(save_path)
    retriever = Retriever(chunks)
    sessions[session_id] = retriever

    log.info("Indexed %d chunks for session %s", len(chunks), session_id)
    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        chunks_indexed=len(chunks),
        message="Document indexed successfully. You can now ask questions.",
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """
    Retrieve relevant chunks and generate a grounded answer.
    """
    retriever = sessions.get(req.session_id)
    if not retriever:
        raise HTTPException(404, "Session not found. Please upload a document first.")

    # Retrieve top-k semantically similar chunks
    relevant_chunks = retriever.retrieve(req.question, top_k=req.top_k)

    # Generate answer with LLM, grounded in retrieved context
    answer = generate_answer(question=req.question, context_chunks=relevant_chunks)

    return AnswerResponse(
        answer=answer,
        sources=relevant_chunks,
        session_id=req.session_id,
    )


@app.get("/sessions/{session_id}")
async def session_info(session_id: str):
    """Check if a session exists and how many chunks are indexed."""
    retriever = sessions.get(session_id)
    if not retriever:
        raise HTTPException(404, "Session not found.")
    return {"session_id": session_id, "chunks": retriever.count()}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clear a session from memory."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found.")
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted."}


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(sessions)}
