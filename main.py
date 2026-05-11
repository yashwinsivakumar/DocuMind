import os
import json
import uuid
import shutil
import sqlite3
import secrets
from datetime import datetime
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import bcrypt

from ingest import ingest_pdf
from search import search_and_answer, search_and_answer_multi

load_dotenv()

app = FastAPI(title="DocuMind API")
BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "history_store.json"
AUTH_DB_FILE = BASE_DIR / "auth.db"
HISTORY_LIMIT = 200
history_lock = Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of uploaded documents
# { doc_id: original_filename }
documents = {}
qa_history = []


def get_auth_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(AUTH_DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def init_auth_db() -> None:
    with get_auth_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.commit()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_history() -> None:
    global qa_history

    if not HISTORY_FILE.exists():
        qa_history = []
        return

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            loaded_history = json.load(file)
            qa_history = loaded_history if isinstance(loaded_history, list) else []
    except Exception:
        qa_history = []


def persist_history() -> None:
    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(qa_history, file, ensure_ascii=False, indent=2)


def build_history_entry(question: str, answer: str, mode: str, doc_names: list[str]) -> dict:
    return {
        "q": question,
        "a": answer,
        "mode": mode,
        "timestamp": now_iso(),
        "docNames": doc_names,
    }


def add_history_entry(entry: dict) -> None:
    with history_lock:
        qa_history.insert(0, entry)
        del qa_history[HISTORY_LIMIT:]
        persist_history()


load_history()
init_auth_db()


class AskRequest(BaseModel):
    doc_id: str
    question: str


class AskMultiRequest(BaseModel):
    doc_ids: list[str]
    question: str


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


def get_current_user_id(authorization: str | None = Header(default=None)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header.")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token.")

    with get_auth_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT user_id FROM sessions WHERE token = ?",
            (token,),
        )
        session_row = cursor.fetchone()

    if not session_row:
        raise HTTPException(status_code=401, detail="Session expired or invalid token.")

    return int(session_row["user_id"])


@app.get("/")
def root():
    return {"message": "Welcome to DocuMind API", "status": "running"}


@app.post("/auth/signup")
def signup(request: SignupRequest):
    email = request.email.strip().lower()
    password = request.password.strip()

    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    with get_auth_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        existing_user = cursor.fetchone()
        if existing_user:
            raise HTTPException(status_code=409, detail="Email is already registered.")

        cursor.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, hash_password(password), now_iso()),
        )
        connection.commit()

    return {"message": "User created successfully."}


@app.post("/auth/login")
def login(request: LoginRequest):
    email = request.email.strip().lower()
    password = request.password

    with get_auth_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, password_hash FROM users WHERE email = ?",
            (email,),
        )
        user_row = cursor.fetchone()

        if not user_row or not verify_password(password, user_row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        token = secrets.token_urlsafe(32)
        cursor.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, int(user_row["id"]), now_iso()),
        )
        connection.commit()

    return {"token": token, "token_type": "Bearer"}


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header.")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token.")

    with get_auth_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
        connection.commit()

    return {"message": "Logged out successfully."}


@app.get("/auth/me")
def me(user_id: int = Depends(get_current_user_id)):
    with get_auth_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id, email, created_at FROM users WHERE id = ?", (user_id,))
        user_row = cursor.fetchone()

    if not user_row:
        raise HTTPException(status_code=404, detail="User not found.")

    return {
        "id": int(user_row["id"]),
        "email": user_row["email"],
        "created_at": user_row["created_at"],
    }


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
        doc_name = documents.get(request.doc_id, request.doc_id)
        history_entry = build_history_entry(request.question, result["answer"], "single", [doc_name])
        add_history_entry(history_entry)

        return {
            "doc_id": request.doc_id,
            "question": result["question"],
            "answer": result["answer"],
            "sources_used": len(result["source_chunks"]),
            "history_entry": history_entry,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/ask-multi")
async def ask_question_multi(request: AskMultiRequest):
    """
    Accepts multiple doc_ids and a question,
    searches across all documents and returns an aggregated answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if not request.doc_ids:
        raise HTTPException(status_code=400, detail="At least one document ID is required.")

    try:
        result = search_and_answer_multi(request.doc_ids, request.question)
        doc_names = [documents.get(doc_id, doc_id) for doc_id in request.doc_ids]
        history_entry = build_history_entry(request.question, result["answer"], "multi", doc_names)
        add_history_entry(history_entry)

        return {
            "doc_ids": result["doc_ids"],
            "question": result["question"],
            "answer": result["answer"],
            "total_chunks_searched": result["chunk_count"],
            "source_chunks": result["source_chunks"],
            "history_entry": history_entry,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-document search failed: {str(e)}")


@app.get("/documents")
def list_documents():
    """
    Returns all uploaded documents in this session.
    """
    return {"documents": documents}


@app.get("/history")
def get_history():
    """Returns persisted Q&A history."""
    return {"history": qa_history}

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
        history_entry = build_history_entry(request.question, answer, "general", [])
        add_history_entry(history_entry)
        return {
            "question": request.question,
            "answer": answer,
            "mode": "general",
            "history_entry": history_entry,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get answer: {str(e)}")