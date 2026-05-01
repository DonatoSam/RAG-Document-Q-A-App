# Technical Project Review: RAG PDF Chatbot

## 1. Project Overview

This project is a **Retrieval-Augmented Generation (RAG) PDF Chatbot**. It allows a user to upload a PDF document and subsequently ask natural language questions about the document's contents. The application scans the document, extracts snippets relevant to the user's question, and uses a Large Language Model (LLM) to synthesize a direct answer.

### Technologies & Frameworks

- **Python**: Core programming language.
- **Streamlit**: Framework for the web-based interactive UI.
- **LangChain**: Orchestration framework that connects the document loaders, splitters, vector stores, and LLM chains.
- **FAISS**: (Facebook AI Similarity Search) An in-memory vector database used for fast similarity search.

### APIs & Models

- **LLM (Inference)**: **Groq API** (specifically the `llama3-8b-8192` model by default) is used to generate the final chat responses. Groq is chosen for its extremely fast generation speeds.
- **Embeddings**: **Google Generative AI API** (Gemini embeddings, specifically `models/embedding-001`) is used to convert the PDF text chunks into vector representations.

---

## 2. Architecture & File Breakdown

The project follows a standard modular RAG architecture with a decoupled LLM client and persistent vector storage.

### Important Files

- **`app.py`**: The main entry point. It contains the Streamlit UI logic, handles file uploads, manages Streamlit's session state, orchestrates the LangChain pipeline, and displays the chat interface.
- **`groq_client.py`**: A custom wrapper class (`ChatGroq`) that inherits from LangChain's base LLM class. It directly formats user prompts and handles the REST API call to Groq's chat completion endpoint.
- **`utils.py`**: Contains helper functions (`save_faiss` and `load_faiss`) to persist the generated FAISS indexes to local disk (inside a `faiss_index/` folder) and retrieve them later, saving API calls and processing time on restarts.
- **`requirements.txt`**: Lists all third-party Python packages required to run the project.
- **`.env.example` & `.env`**: Configuration files storing sensitive API keys (`GROQ_API_KEY`, `GOOGLE_API_KEY`).

---

## 3. Execution Flow (Step-by-Step)

Here is exactly what happens when a user uses the app:

1.  **PDF Upload**: The user uploads a PDF via the Streamlit interface. Streamlit catches this file and saves it temporarily to the local file system.
2.  **Text Chunking**:
    - LangChain's `PyPDFLoader` reads the text from the temporary PDF.
    - The `RecursiveCharacterTextSplitter` breaks the text into smaller chunks of 1000 characters, with a 200-character overlap to ensure context isn't lost between paragraphs.
3.  **Embeddings**: Each text chunk is sent to the Google Generative AI API via the `GoogleGenerativeAIEmbeddings` class, which converts the words into numerical arrays (vectors) that capture semantic meaning.
4.  **Vector Database**: The vectors, along with their original text, are inserted into a **FAISS** index. This index is kept in Streamlit's memory so it can be queried, and also saved to disk (`faiss_index/`) via `utils.py`.
5.  **Retrieval**: When the user enters a question in the chat, the question is also converted to an embedding. FAISS compares the question's vector against all the document vectors and retrieves the top `K` (default 4) most similar text chunks.
6.  **Final AI Response**: The retrieved chunks (context) and the user's question are packed into a prompt template by LangChain's `RetrievalQA` chain. This massive prompt is sent to the **Groq API** via our custom `ChatGroq` client. The LLM reads the context, drafts an answer, and Streamlit displays it in the chat UI.

---

## 4. Current Issues & Cleanups Needed

While the logic is structurally sound, the project has a few environmental and dependency issues.

### Dependency & Execution Issues (Broken Parts)

- **Google Generative AI on Python 3.10**: Currently, `google-generative-ai` failed to traverse pip resolution on your specific Python 3.10 environment. This prevents the embeddings from working.
- **LangChain Import Errors**: During testing, an error (`ModuleNotFoundError: No module named 'langchain.document_loaders'`) occurred. This indicates a broken or partially installed LangChain environment.

### Outdated Imports

- The current `app.py` uses legacy LangChain imports (e.g., `from langchain.document_loaders import PyPDFLoader`, `from langchain.chat_models...`).
- _Modern standard (LangChain v0.1.0+)_ dictates importing these from `langchain_community` (e.g., `langchain_community.document_loaders`).

### Duplicate/Unnecessary Files

- There are no explicit duplicate files currently; older integrations (like the old `groq_embeddings.py` and old OpenAI references) were cleanly removed in the previous refactoring step.

---

## 5. Summary: What works vs. What needs fixing

**What is working:**

- Streamlit UI and logic.
- The modular project structure.
- The custom Groq client (`groq_client.py`).
- FAISS Persistence mechanisms (`utils.py`).

**What needs fixing:**

- **The Python Environment**: The gap between Python 3.10 and the `google-generative-ai` package must be resolved (either by upgrading to Python 3.11 or swapping to a different embedding provider like `sentence-transformers`).
- **LangChain Version & Imports**: `requirements.txt` should be updated to target `langchain>=0.1.0` and `langchain_community`, and `app.py` should be updated to use the new community import paths to avoid the `ModuleNotFoundError`.

**Next Logical Step:**
Fix the environment constraints by either migrating imports to `langchain_community` and swapping the embedding model to bypass the broken Google SDK on Python 3.10, or setting up a clean Python 3.11 environment.
