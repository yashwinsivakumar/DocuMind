import fitz
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


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


