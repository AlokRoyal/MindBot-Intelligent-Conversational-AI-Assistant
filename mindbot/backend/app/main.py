import base64
import os
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings, UPLOAD_DIR
from app.db.session import Base, engine, get_db
from app.db.models import Conversation, Message, Document
from app.utils.file_utils import (
    save_upload_file,
    is_allowed_document,
    is_allowed_audio,
    is_allowed_image,
    extract_text_from_pdf,
    extract_text_from_plain_file,
    file_extension,
)
from app.utils.chunking import chunk_text
from app.services.moderation_service import moderate_text
from app.services.llm_service import chat_completion, SYSTEM_PROMPT, vision_completion
from app.services.speech_service import transcribe_audio
from app.services.rag_service import add_document_chunks, retrieve_context


Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None


class ChatResponse(BaseModel):
    conversation_id: int
    reply: str
    context_used: bool = False


class UploadResponse(BaseModel):
    success: bool
    filename: str
    detail: str


def get_or_create_conversation(db: Session, conversation_id: Optional[int]) -> Conversation:
    if conversation_id:
        convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if convo:
            return convo

    convo = Conversation(title="MindBot Chat")
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo


def get_recent_messages(db: Session, conversation_id: int, limit: int = 12):
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(msgs))


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}


@app.get("/conversations/{conversation_id}")
def conversation_history(conversation_id: int, db: Session = Depends(get_db)):
    convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )

    return {
        "conversation_id": convo.id,
        "title": convo.title,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": str(m.created_at),
            }
            for m in messages
        ],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    moderation = moderate_text(user_text)
    if moderation.get("flagged"):
        raise HTTPException(status_code=400, detail="Input flagged by moderation")

    convo = get_or_create_conversation(db, request.conversation_id)

    db.add(Message(conversation_id=convo.id, role="user", content=user_text))
    db.commit()

    context = retrieve_context(user_text, top_k=settings.top_k_docs)
    context_used = bool(context.strip())

    recent_messages = get_recent_messages(db, convo.id, limit=settings.max_history_turns)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_used:
        messages.append(
            {
                "role": "system",
                "content": "Use the following retrieved document context when relevant:\n\n" + context,
            }
        )

    for msg in recent_messages:
        messages.append({"role": msg.role, "content": msg.content})

    reply = chat_completion(messages)

    db.add(Message(conversation_id=convo.id, role="assistant", content=reply))
    db.commit()

    return ChatResponse(conversation_id=convo.id, reply=reply, context_used=context_used)


@app.post("/upload/document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not is_allowed_document(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported document type")

    saved_path = await save_upload_file(file, UPLOAD_DIR)
    ext = file_extension(file.filename)

    if ext == ".pdf":
        text = extract_text_from_pdf(saved_path)
    else:
        text = extract_text_from_plain_file(saved_path)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from this file")

    chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    add_document_chunks(chunks, source=file.filename)

    doc = Document(
        filename=file.filename,
        filepath=str(saved_path),
        filetype=ext,
        status="indexed",
    )
    db.add(doc)
    db.commit()

    return UploadResponse(success=True, filename=file.filename, detail=f"Indexed {len(chunks)} chunks")


@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    if not is_allowed_audio(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported audio type")

    saved_path = await save_upload_file(file, UPLOAD_DIR)
    transcript = transcribe_audio(str(saved_path))
    return {"success": True, "filename": file.filename, "transcript": transcript}


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...), prompt: str = Form("Describe this image in detail.")):
    if not is_allowed_image(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported image type")

    saved_path = await save_upload_file(file, UPLOAD_DIR)
    with open(saved_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    mime_type = file.content_type or "image/png"
    result = vision_completion(prompt, image_base64=image_base64, mime_type=mime_type)

    return {"success": True, "filename": file.filename, "result": result}