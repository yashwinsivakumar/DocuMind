import fitz
import os
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    all_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            all_text.append(f"[Page {page_num + 1}]\n{text}")

    doc.close()

    full_text = "\n\n".join(all_text)

    if not full_text.strip():
        raise ValueError("No text could be extracted. PDF may be scanned/image-based.")

    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    embeddings = []

    for i, chunk in enumerate(chunks):
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        embeddings.append(response.embeddings[0].values)
        print(f"  Embedded chunk {i + 1}/{len(chunks)}")

    return embeddings


def store_in_chromadb(doc_id: str, chunks: list[str], embeddings: list[list[float]]):
    """
    Stores chunks and their embeddings in ChromaDB.
    Each document gets its own collection.
    """
    collection = chroma_client.get_or_create_collection(name=doc_id)

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    print(f"  Stored {len(chunks)} chunks in ChromaDB collection: {doc_id}")


def ingest_pdf(pdf_path: str, doc_id: str) -> int:
    """
    Full pipeline: extract → chunk → embed → store.
    Returns the number of chunks stored.
    """
    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)

    print(f"Chunking text...")
    chunks = chunk_text(text)
    print(f"  Created {len(chunks)} chunks")

    print(f"Embedding chunks...")
    embeddings = embed_chunks(chunks)

    print(f"Storing in ChromaDB...")
    store_in_chromadb(doc_id, chunks, embeddings)

    return len(chunks)


