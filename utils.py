"""FAISS persistence helpers."""

from langchain_community.vectorstores import FAISS
from typing import Any


def save_faiss(vectorstore: FAISS, folder_path: str = "faiss_index") -> None:
    """Persist FAISS vectorstore to disk."""
    vectorstore.save_local(folder_path)


def load_faiss(folder_path: str = "faiss_index", embeddings: Any = None) -> FAISS:
    """Load a FAISS vectorstore from disk.
    
    allow_dangerous_deserialization=True is required by langchain-community >= 0.3.
    This is safe when you control the index files (not loading from untrusted sources).
    """
    if embeddings is None:
        raise ValueError("Embeddings instance is required to load a FAISS index.")
    return FAISS.load_local(
        folder_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
