# 📄 RAG PDF Chatbot

Chat with any PDF using **Groq** (free LLM API) + **Sentence Transformers** (local embeddings) + **FAISS** — all free, no GPU needed.

---

## 🚀 Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (get one free at https://console.groq.com)
cp .env.example .env
# Edit .env and paste your key

# 3. Run
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

> You can also enter the API key directly in the sidebar — no `.env` file required.

---

## ☁️ Free Cloud Deployment

### Hugging Face Spaces (Streamlit)
1. Push this repo to HF
2. Settings → Repository secrets → add `GROQ_API_KEY`

### Streamlit Community Cloud
1. Connect your GitHub repo
2. App Settings → Secrets → add:
   ```
   GROQ_API_KEY = "gsk_..."
   ```

### Render
1. New Web Service → connect repo
2. Environment → add `GROQ_API_KEY`

---

## 🏗️ Stack

| Component | Package | Purpose |
|-----------|---------|---------|
| LLM | `langchain-groq` + Groq API | Free, fast text generation |
| Embeddings | `sentence-transformers` | Local vector embeddings (no API key) |
| Vector DB | `faiss-cpu` | In-memory similarity search |
| PDF parsing | `pypdf` | Extract text from PDFs |
| UI | `streamlit` | Web interface |
| Orchestration | `langchain` (LCEL) | RAG pipeline |

---

## 🤖 Supported Models (all free on Groq)

- `llama3-8b-8192` — fastest, great for most tasks
- `llama3-70b-8192` — smarter, slower
- `llama-3.1-8b-instant` — instant responses
- `llama-3.3-70b-versatile` — best quality
- `gemma2-9b-it` — Google's Gemma 2
- `mixtral-8x7b-32768` — longest context window
