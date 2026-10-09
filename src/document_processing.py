import hashlib
import json
import os
import tempfile
from typing import Dict, List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def hash_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_resume_from_upload(uploaded_file) -> Tuple[List[Document], str]:
    file_bytes = uploaded_file.getvalue()
    file_hash = hash_bytes(file_bytes)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks, file_hash


def load_knowledge_base(
    jsonl_path: str,
) -> Tuple[List[Document], List[Dict], Dict[str, Dict], str]:
    file_hash = hash_file(jsonl_path)
    raw_entries: List[Dict] = []
    documents: List[Document] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            raw_entries.append(entry)

            rubric_text = "\n".join(f"  - {r}" for r in entry["grading_rubric"])
            content = (
                f"Topic: {entry['topic']}\n"
                f"Difficulty: {entry['difficulty']}\n"
                f"Question: {entry['question']}\n"
                f"Ideal Answer: {entry['ideal_answer']}\n"
                f"Grading Criteria:\n{rubric_text}"
            )
            doc = Document(
                page_content=content,
                metadata={
                    "id": entry["id"],
                    "topic": entry["topic"],
                    "difficulty": entry["difficulty"],
                },
            )
            documents.append(doc)

    kb_dict = {entry["id"]: entry for entry in raw_entries}
    return documents, raw_entries, kb_dict, file_hash
