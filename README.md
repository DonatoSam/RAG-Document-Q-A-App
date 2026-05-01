# RAG PDF Chatbot — Groq (LLM) + Gemini embeddings

This project is a production-ready PDF chatbot built with:

- Python
- Streamlit (UI)
- LangChain (chains, retriever)
- FAISS (vector store)
- Groq API (LLM inference — `ChatGroq` wrapper)
- Google Generative AI / Gemini embeddings (`GoogleGenerativeAIEmbeddings`)

Features:

- Upload a PDF and build a FAISS index using Gemini embeddings
- Retrieve passages and answer questions using Groq LLM
- Clean modular structure ready for Hugging Face Spaces
- Persistence for FAISS index (`faiss_index/`) and a modern Streamlit chat UI

Environment variables (required):

- `GROQ_API_KEY` — Groq API key for LLM
- `GOOGLE_API_KEY` — Google API key for Gemini embeddings

Quick start (local)

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set environment variables (PowerShell example):

```powershell
$env:GROQ_API_KEY = "<your_groq_api_key_here>"
$env:GOOGLE_API_KEY = "<your_google_api_key_here>"
streamlit run app.py
```

Alternatively, create a `.env` file from the provided `.env.example` and fill it with your keys (the app loads `.env` via `python-dotenv`):

```text
cp .env.example .env
# then edit .env and add your keys
```

Security reminder: never commit `.env` or real API keys to version control. Use the repository/space secret feature when deploying to Hugging Face Spaces.

Hugging Face Spaces

- Create a new Space (Streamlit). Add this repository's files.
- Add `GROQ_API_KEY` and `GOOGLE_API_KEY` as Secrets in the Space settings.

Notes & tips

- The Groq client `groq_client.py` implements a minimal LangChain-compatible LLM wrapper. If Groq changes their API endpoints or request schema, update the client accordingly.
- For larger projects, store FAISS indexes in durable storage and reuse them instead of rebuilding on every run.
- The app uses `ChatGroq` as the single LLM provider and Gemini embeddings for vectorization.
