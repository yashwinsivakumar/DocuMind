import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import json

from ingest import ingest_pdf
from search import search_and_answer, general_ask
from database import init_db, get_db

load_dotenv()

app = FastAPI(title="DocuMind API")
security = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "documind-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
documents = {}

init_db()


# ─── Auth Helpers ────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(email: str, user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": email, "user_id": user_id, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        return None
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return {"email": payload.get("sub"), "user_id": payload.get("user_id")}
    except JWTError:
        return None


# ─── Pydantic Models ─────────────────────────────────────────

class AuthRequest(BaseModel):
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str

class AskRequest(BaseModel):
    doc_id: str
    question: str

class GeneralAskRequest(BaseModel):
    question: str

class SaveHistoryRequest(BaseModel):
    question: str
    answer: str
    mode: str
    doc_names: Optional[list[str]] = []


# ─── Core Routes ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Welcome to DocuMind API", "status": "running"}


@app.get("/auth/config")
def auth_config():
    return {"google_client_id": GOOGLE_CLIENT_ID}


# ─── Auth Routes ─────────────────────────────────────────────

@app.post("/auth/signup")
def signup(req: AuthRequest):
    db = get_db()
    existing = db.execute(
        "SELECT id FROM users WHERE email = ?", (req.email,)
    ).fetchone()

    if existing:
        db.close()
        raise HTTPException(status_code=400, detail="Email already registered.")

    password_hash = hash_password(req.password)
    db.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        (req.email, password_hash)
    )
    db.commit()
    db.close()
    return {"message": "Account created. Please log in."}


@app.post("/auth/login")
def login(req: AuthRequest):
    db = get_db()
    user = db.execute(
        "SELECT id, password_hash FROM users WHERE email = ?", (req.email,)
    ).fetchone()
    db.close()

    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_token(req.email, user["id"])
    return {"token": token, "email": req.email}


@app.post("/auth/google")
def google_signin(req: GoogleAuthRequest):
    try:
        info = id_token.verify_oauth2_token(
            req.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )
        email = info.get("email")
        google_id = info.get("sub")

        db = get_db()
        user = db.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()

        if not user:
            db.execute(
                "INSERT INTO users (email, google_id) VALUES (?, ?)",
                (email, google_id)
            )
            db.commit()
            user = db.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()

        token = create_token(email, user["id"])
        db.close()
        return {"token": token, "email": email}

    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Google sign-in failed: {str(e)}")


@app.post("/auth/logout")
def logout():
    return {"message": "Logged out."}


@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {"email": user["email"]}


# ─── History Routes ───────────────────────────────────────────

@app.post("/history")
def save_history(req: SaveHistoryRequest, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Login to save history.")

    db = get_db()
    db.execute(
        """INSERT INTO history (user_id, question, answer, mode, doc_names)
           VALUES (?, ?, ?, ?, ?)""",
        (user["user_id"], req.question, req.answer,
         req.mode, json.dumps(req.doc_names))
    )
    db.commit()
    db.close()
    return {"message": "Saved."}


@app.get("/history")
def get_history(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Login to view history.")

    db = get_db()
    rows = db.execute(
        """SELECT question, answer, mode, doc_names, created_at
           FROM history WHERE user_id = ?
           ORDER BY created_at DESC LIMIT 50""",
        (user["user_id"],)
    ).fetchall()
    db.close()

    history = []
    for row in rows:
        history.append({
            "q": row["question"],
            "a": row["answer"],
            "mode": row["mode"],
            "docNames": json.loads(row["doc_names"] or "[]"),
            "displayTime": row["created_at"]
        })

    return {"history": history}


# ─── Document Routes ──────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
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
async def ask_question(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = search_and_answer(req.doc_id, req.question)
        return {
            "doc_id": req.doc_id,
            "question": result["question"],
            "answer": result["answer"],
            "sources_used": len(result["source_chunks"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/general-ask")
async def general_ask_endpoint(req: GeneralAskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = general_ask(req.question)
        return {"question": req.question, "answer": answer, "mode": "general"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")