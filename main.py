import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from ingest import ingest_pdf
from search import search_and_answer

load_dotenv()

app = FastAPI(title="DocuMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of uploaded documents
# { doc_id: original_filename }
documents = {}


class AskRequest(BaseModel):
    doc_id: str
    question: str


@app.get("/")
def root():
    return {"message": "Welcome to DocuMind API", "status": "running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, ingests it into ChromaDB,
    and returns a doc_id for future questions.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    save_path = f"uploads/{doc_id}.pdf"

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunk_count = ingest_pdf(save_path, doc_id)

        documents[doc_id] = file.filename

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks_stored": chunk_count,
            "message": "Document uploaded and indexed successfully."
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Accepts a doc_id and a question,
    returns an AI-generated answer from the document.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = search_and_answer(request.doc_id, request.question)

        return {
            "doc_id": request.doc_id,
            "question": result["question"],
            "answer": result["answer"],
            "sources_used": len(result["source_chunks"])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/documents")
def list_documents():
    """
    Returns all uploaded documents in this session.
    """
    return {"documents": documents}

class GeneralAskRequest(BaseModel):
    question: str


@app.post("/general-ask")
async def general_ask_endpoint(request: GeneralAskRequest):
    """
    Answers any general question without document context.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        from search import general_ask
        answer = general_ask(request.question)
        return {
            "question": request.question,
            "answer": answer,
            "mode": "general"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get answer: {str(e)}")