import os
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def embed_query(question: str) -> list[float]:
    """
    Embeds the user's question using the same Gemini model
    used during ingestion — this is critical for matching.
    """
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )
    return response.embeddings[0].values


def search_chunks(doc_id: str, question: str, top_k: int = 5) -> list[str]:
    """
    Searches ChromaDB for the most relevant chunks
    to the user's question.
    """
    collection = chroma_client.get_collection(name=doc_id)

    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    chunks = results["documents"][0]
    return chunks


def ask_gemini(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant for DocuMind, a smart document search app.
Answer the user's question using ONLY the document excerpts provided below.
If the answer is not found in the excerpts, say "I couldn't find that information in the document."
Never make up information.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""

    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt
    )

    return response.text


def search_and_answer(doc_id: str, question: str) -> dict:
    """
    Full query pipeline: embed question → search → answer.
    Returns both the answer and the source chunks used.
    """
    print(f"Searching for: {question}")

    chunks = search_chunks(doc_id, question)
    print(f"  Found {len(chunks)} relevant chunks")

    print("  Asking Gemini...")
    answer = ask_gemini(question, chunks)

    return {
        "question": question,
        "answer": answer,
        "source_chunks": chunks
    }

def general_ask(question: str) -> str:
    """
    Answers any question freely without document context.
    Used for follow-up or general questions.
    """
    prompt = f"""You are a helpful AI assistant called DocuMind.
Answer the following question clearly and helpfully.

QUESTION: {question}

ANSWER:"""

    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text