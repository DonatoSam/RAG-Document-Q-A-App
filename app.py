"""
RAG PDF Chatbot - Streamlit app

Uses:
- Groq for LLM inference (ChatGroq wrapper)
- Hugging Face sentence-transformers (local, no API key needed)
- FAISS for vector storage

Environment variables required:
- GROQ_API_KEY: Groq API key for LLM
"""

import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

from utils import save_faiss, load_faiss
from groq_client import ChatGroq, test_groq_connection

load_dotenv()

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")

st.title("RAG PDF Chatbot - Groq + Sentence Transformers")

# Session state helpers
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: keys & options
st.sidebar.header("Configuration")
st.sidebar.write("Provide API keys as environment variables or in a .env file.")
st.sidebar.text_input("Groq API key (GROQ_API_KEY)", value=os.getenv("GROQ_API_KEY", ""), key="_groq_key")

current_key = st.session_state.get("_groq_key") or os.getenv("GROQ_API_KEY")
if current_key:
    if not test_groq_connection(current_key):
        st.sidebar.warning("⚠️ Warning: Could not connect to Groq API. Check your internet connection or API key validity.", icon="⚠️")
    else:
        st.sidebar.success("✅ Groq API connection successful!")

st.sidebar.markdown("---")
st.sidebar.markdown("Deployment: Hugging Face Spaces (Streamlit). Add `GROQ_API_KEY` as a secret.")

def process_pdf_and_build_index(pdf_path: str, embeddings) -> FAISS:
    """Load PDF, split into chunks, create FAISS index using provided embeddings."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Build FAISS vectorstore from documents
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Saving PDF to a temporary file..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded_file.read())
        tmp.flush()
        pdf_path = tmp.name
    st.success("PDF saved")

    if st.button("Process document and build index"):
        groq_key = current_key
        if not groq_key:
            st.error("Missing GROQ_API_KEY. Set it in your environment or Space secrets.")
        else:
            with st.spinner("Loading local embeddings and indexing document..."):
                try:
                    # HuggingFace embeddings via LangChain Community
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore = process_pdf_and_build_index(pdf_path, embeddings)
                    st.session_state.vectorstore = vectorstore
                    # Persist index to disk for reuse
                    save_faiss(vectorstore, folder_path="faiss_index")
                    st.success("Index built and saved (faiss_index/)")
                except Exception as e:
                    st.exception(e)

# Chat UI - only shown when index exists
if st.session_state.vectorstore is not None:
    st.subheader("Chat with your document")
    query = st.text_input("Your question", key="input_query")
    top_k = st.slider("Retriever top K", min_value=1, max_value=8, value=4)

    if st.button("Send") and query:
        try:
            with st.spinner("Retrieving relevant passages and generating answer..."):
                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

                groq_key = current_key
                if not groq_key:
                    st.error("GROQ_API_KEY is not set. Cannot call LLM.")
                else:
                    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama3-8b-8192"), api_key=groq_key, temperature=0.0)

                    # Build RetrievalQA chain
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                    answer = qa.run(query)

                    # Update chat history
                    st.session_state.chat_history.append({"q": query, "a": answer})

        except Exception as e:
            st.exception(e)

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        for item in reversed(st.session_state.chat_history[-20:]):
            st.markdown(f"**User:** {item['q']}")
            st.markdown(f"**Assistant:** {item['a']}")

    if st.button("Clear index and history"):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.success("Cleared index and chat history")

else:
    # Try to load persisted index if present
    if os.path.exists("faiss_index") and st.button("Load existing index from disk"):
        try:
            with st.spinner("Loading model and index..."):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = load_faiss(folder_path="faiss_index", embeddings=embeddings)
                st.session_state.vectorstore = vectorstore
                st.success("Loaded FAISS index from faiss_index/")
        except Exception as e:
            st.exception(e)
