import os
import time
from typing import List, Optional

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

KB_INDEX_DIR = "faiss_kb_index"
KB_HASH_FILE = os.path.join(KB_INDEX_DIR, "kb_hash.txt")
L2_THRESHOLD = 0.85


def _embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


def _read_stored_hash() -> Optional[str]:
    if os.path.exists(KB_HASH_FILE):
        with open(KB_HASH_FILE, "r") as f:
            return f.read().strip()
    return None


def _write_hash(file_hash: str) -> None:
    os.makedirs(KB_INDEX_DIR, exist_ok=True)
    with open(KB_HASH_FILE, "w") as f:
        f.write(file_hash)


@st.cache_resource(show_spinner=False)
def get_kb_vectorstore(file_hash: str, documents_repr: str) -> FAISS:
    emb = _embeddings()
    stored_hash = _read_stored_hash()

    if stored_hash == file_hash and os.path.isdir(KB_INDEX_DIR):
        return FAISS.load_local(
            KB_INDEX_DIR,
            emb,
            allow_dangerous_deserialization=True,
        )

    from src.document_processing import load_knowledge_base
    docs, _, _, _ = load_knowledge_base("data/knowledge_base.jsonl")
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(KB_INDEX_DIR)
    _write_hash(file_hash)
    return vs


def build_resume_vectorstore(chunks: List[Document]) -> Optional[FAISS]:
    for attempt in range(3):
        try:
            return FAISS.from_documents(chunks, _embeddings())
        except Exception:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None
    return None


def retrieve_resume_context(vectorstore: Optional[FAISS], query: str, k: int = 3) -> str:
    if not vectorstore:
        return ""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception:
        return ""


def get_reference_if_relevant(
    vectorstore: FAISS,
    query: str,
    threshold: float = L2_THRESHOLD,
) -> Optional[str]:
    results = vectorstore.similarity_search_with_score(query, k=1)
    if not results:
        return None

    doc, distance = results[0]
    if distance >= threshold:
        return None

    for line in doc.page_content.split("\n"):
        if line.startswith("Ideal Answer:"):
            return line.replace("Ideal Answer:", "").strip()
    return None
