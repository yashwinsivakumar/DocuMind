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
    Searches ChromaDB for the most relevant chunks in a single document.
    """
    collection = chroma_client.get_collection(name=doc_id)

    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    chunks = results["documents"][0]
    return chunks


def search_chunks_multi(doc_ids: list[str], question: str, top_k: int = 5) -> list[dict]:
    """
    Searches across multiple documents and returns chunks with source info.
    Returns list of dicts: {"doc_id": str, "chunk": str}
    """
    query_embedding = embed_query(question)
    all_results = []

    for doc_id in doc_ids:
        try:
            collection = chroma_client.get_collection(name=doc_id)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k // len(doc_ids), collection.count())
            )
            
            for chunk in results["documents"][0]:
                all_results.append({"doc_id": doc_id, "chunk": chunk})
        except Exception as e:
            print(f"  Warning: Could not search document {doc_id}: {e}")

    # Sort by relevance (simple approach: results are already ranked per doc)
    return all_results[:top_k]


def ask_gemini(question: str, context_chunks: list[str], multi_doc: bool = False) -> str:
    """
    Generates answer from context chunks.
    If multi_doc=True, assumes chunks already include source info.
    """
    context = "\n\n---\n\n".join(context_chunks)

    if multi_doc:
        prompt = f"""You are a helpful assistant for DocuMind, a smart document search app.
Answer the user's question using ONLY the document excerpts provided below.
If the answer is not found in the excerpts, say "I couldn't find that information in the documents."
Never make up information. You are searching across multiple documents.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""
    else:
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
    Single document search.
    Returns both the answer and the source chunks used.
    """
    print(f"Searching for: {question}")

    chunks = search_chunks(doc_id, question)
    print(f"  Found {len(chunks)} relevant chunks")

    print("  Asking Gemini...")
    answer = ask_gemini(question, chunks, multi_doc=False)

    return {
        "question": question,
        "answer": answer,
        "source_chunks": chunks
    }


def search_and_answer_multi(doc_ids: list[str], question: str) -> dict:
    """
    Full query pipeline for multiple documents.
    Returns answer and source chunks with document IDs.
    """
    print(f"Searching across {len(doc_ids)} document(s) for: {question}")

    results = search_chunks_multi(doc_ids, question)
    print(f"  Found {len(results)} relevant chunks")

    # Format context with source info
    context_with_sources = []
    for result in results:
        source_text = f"[From {result['doc_id']}]\n{result['chunk']}"
        context_with_sources.append(source_text)

    print("  Asking Gemini...")
    answer = ask_gemini(question, context_with_sources, multi_doc=True)

    return {
        "question": question,
        "answer": answer,
        "doc_ids": doc_ids,
        "source_chunks": results,
        "chunk_count": len(results)
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