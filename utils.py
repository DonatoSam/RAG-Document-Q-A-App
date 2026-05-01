"""Utility helpers for FAISS persistence and small helpers."""

from langchain_community.vectorstores import FAISS
from typing import Any


def save_faiss(vectorstore: FAISS, folder_path: str = "faiss_index") -> None:
    """Persist FAISS vectorstore to disk (folder)."""
    vectorstore.save_local(folder_path)


def load_faiss(folder_path: str = "faiss_index", embeddings: Any = None) -> FAISS:
    """Load a FAISS vectorstore from disk. Caller must supply the embeddings object."""
    if embeddings is None:
        raise ValueError("Embeddings instance required to load FAISS index")
    return FAISS.load_local(folder_path, embeddings)
