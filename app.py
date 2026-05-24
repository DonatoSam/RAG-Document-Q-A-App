"""
DocMind — RAG PDF Chatbot
Stack: Groq · Sentence Transformers · FAISS · LangChain v1.x
"""

import os
import tempfile
import requests

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

st.set_page_config(
    page_title="DocMind — PDF Chat",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400,0,0');

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    color: #e6edf3;
}
.material-symbols-outlined, .material-icons, .material-symbols-rounded, .material-symbols-sharp {
    font-family: 'Material Symbols Outlined' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}

/* ── Base ─────────────────────────────────────────────────────── */
.stApp { background-color: #0d1117; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem 2rem !important; max-width: 960px !important; }

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #010409 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem 1rem 1rem !important; }
[data-testid="stSidebar"] label { color: #8b949e !important; font-size: 0.8rem !important; font-weight: 500 !important; }

/* ── Widgets ──────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background-color: #0d1117 !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 6px !important; font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus { border-color: #388bfd !important; box-shadow: 0 0 0 3px rgba(56,139,253,0.12) !important; }
.stTextInput > div > div > input::placeholder { color: #484f58 !important; }

.stSelectbox [data-baseweb="select"] > div {
    background-color: #0d1117 !important; border: 1px solid #30363d !important;
    border-radius: 6px !important; color: #e6edf3 !important; font-size: 0.84rem !important;
}

.stSlider [data-baseweb="slider"] { margin-top: 0.3rem !important; }

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    background-color: #21262d !important; color: #cdd9e5 !important;
    border: 1px solid #363b42 !important; border-radius: 6px !important;
    font-size: 0.84rem !important; font-weight: 500 !important;
    padding: 0.42rem 1rem !important; transition: all 0.15s !important;
}
.stButton > button:hover {
    background-color: #30363d !important; border-color: #484f58 !important; color: #e6edf3 !important;
}
.stButton > button[kind="primary"] {
    background-color: #238636 !important; border-color: #2ea043 !important; color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #2ea043 !important; border-color: #3fb950 !important;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
    background: transparent !important; border-bottom: 1px solid #21262d !important;
    padding: 0 !important; gap: 0.6rem !important; border-radius: 0 !important;
}
div[data-baseweb="tab"] {
    background: transparent !important; border: none !important; border-radius: 0 !important;
    color: #8b949e !important; font-size: 0.84rem !important; font-weight: 500 !important;
    padding: 0.6rem 1rem !important; border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
}
div[data-baseweb="tab"]:hover { color: #cdd9e5 !important; }
div[data-baseweb="tab"][aria-selected="true"] {
    color: #e6edf3 !important; border-bottom-color: #f78166 !important;
}
div[data-baseweb="tab-panel"] { padding: 1.4rem 0 0 0 !important; }

/* ── File uploader ────────────────────────────────────────────── */
[data-testid="stFileUploader"] section {
    background-color: #0d1117 !important;
    border: 1px dashed #30363d !important; border-radius: 8px !important;
    padding: 1.5rem !important; transition: border-color 0.15s !important;
}
[data-testid="stFileUploader"] section:hover { border-color: #58a6ff !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: #8b949e !important; font-size: 0.84rem !important; }
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploader"] button {
    background-color: #21262d !important; color: #cdd9e5 !important;
    border: 1px solid #363b42 !important; border-radius: 6px !important;
    font-size: 0.84rem !important; font-weight: 500 !important;
    padding: 0.42rem 1rem !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #30363d !important; border-color: #484f58 !important; color: #e6edf3 !important;
}
[data-testid="stFileUploader"] button span {
    display: none !important;
}
[data-testid="stFileUploader"] button::before {
    content: "Upload";
}

/* ── Progress ─────────────────────────────────────────────────── */
div[data-testid="stProgressBar"] > div { background-color: #21262d !important; border-radius: 4px !important; }
div[data-testid="stProgressBar"] > div > div { background-color: #238636 !important; border-radius: 4px !important; }

/* ── Chat input ───────────────────────────────────────────────── */
div[data-testid="stChatInput"] > div {
    background-color: #0d1117 !important; border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
div[data-testid="stChatInput"] > div:focus-within {
    border-color: #388bfd !important; box-shadow: 0 0 0 3px rgba(56,139,253,0.12) !important;
}

/* ── Alerts ───────────────────────────────────────────────────── */
div[data-testid="stAlert"] { border-radius: 6px !important; font-size: 0.84rem !important; }

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
if "vectorstore"  not in st.session_state: st.session_state.vectorstore  = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "doc_name"     not in st.session_state: st.session_state.doc_name     = ""
if "doc_chunks"   not in st.session_state: st.session_state.doc_chunks   = 0

# ─── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"

MODELS = {
    "llama-3.1-8b-instant":    "llama-3.1-8b-instant  (fastest)",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile  (smartest)",
    "llama3-8b-8192":          "llama3-8b-8192",
    "gemma2-9b-it":            "gemma2-9b-it  (Google)",
    "mixtral-8x7b-32768":      "mixtral-8x7b-32768  (long ctx)",
}

RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer using ONLY the context below.\n"
    "If not found, say: I couldn't find that in the document.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try: key = st.secrets.get("GROQ_API_KEY", "")
        except Exception: pass
    return key or ""


def test_connection(api_key):
    if not api_key: return False, "No API key."
    try:
        r = requests.get("https://api.groq.com/openai/v1/models",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=8)
        if r.status_code == 200:
            return True, f"Connected — {len(r.json().get('data', []))} models available"
        msg = r.json().get("error", {}).get("message", r.text)[:80]
        return False, f"HTTP {r.status_code}: {msg}"
    except requests.exceptions.ConnectionError: return False, "Network error"
    except requests.exceptions.Timeout:         return False, "Timed out"
    except Exception as e:                      return False, str(e)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"batch_size": 16, "normalize_embeddings": True},
    )


def _fmt_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_chain(vs, api_key, model, k):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    llm = ChatGroq(model=model, api_key=api_key, temperature=0.0, max_tokens=1024)
    return (
        {"context": retriever | RunnableLambda(_fmt_docs), "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )


def save_index(vs, path="faiss_index"):   vs.save_local(path)
def load_index(emb, path="faiss_index"):
    return FAISS.load_local(path, emb, allow_dangerous_deserialization=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h2 style='color:#e6edf3;font-size:1.1rem;font-weight:700;margin:0 0 4px 0;'>DocMind</h2>"
        "<p style='color:#484f58;font-size:0.73rem;margin:0 0 1.2rem 0;'>RAG PDF Chatbot · Groq + FAISS</p>"
        "<hr style='border:none;border-top:1px solid #21262d;margin:0 0 1.2rem 0;'>",
        unsafe_allow_html=True,
    )

    groq_key = st.text_input(
        "Groq API Key",
        value=_resolve_key(),
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )

    groq_model = st.selectbox(
        "Model",
        options=list(MODELS.keys()),
        format_func=lambda x: MODELS[x],
    )

    top_k = st.slider("Context chunks (k)", min_value=1, max_value=10, value=4)

    st.markdown("<hr style='border:none;border-top:1px solid #21262d;margin:1rem 0;'>", unsafe_allow_html=True)

    if groq_key:
        if st.button("Test Connection", use_container_width=True):
            with st.spinner("Pinging Groq API…"):
                ok, msg = test_connection(groq_key)
            if ok: st.success(f"Connected: {msg}")
            else:  st.error(f"Connection failed: {msg}")
    else:
        st.info("Add your Groq API key above.\n\nFree key → [console.groq.com](https://console.groq.com)")

    st.markdown("<hr style='border:none;border-top:1px solid #21262d;margin:1rem 0;'>", unsafe_allow_html=True)

    if st.session_state.vectorstore is not None:
        st.markdown(
            f"<p style='color:#3fb950;font-size:0.78rem;font-weight:600;margin:0 0 2px 0;'>● Document loaded</p>"
            f"<p style='color:#8b949e;font-size:0.78rem;margin:0;word-break:break-all;'>{st.session_state.doc_name}</p>"
            + (f"<p style='color:#484f58;font-size:0.73rem;margin:2px 0 8px 0;'>{st.session_state.doc_chunks} chunks indexed</p>" if st.session_state.doc_chunks else ""),
            unsafe_allow_html=True,
        )
        if st.button("Unload Document", use_container_width=True):
            st.session_state.vectorstore  = None
            st.session_state.chat_history = []
            st.session_state.doc_name     = ""
            st.session_state.doc_chunks   = 0
            st.rerun()
    else:
        st.markdown("<p style='color:#484f58;font-size:0.78rem;'>No document loaded</p>", unsafe_allow_html=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid #21262d;margin:1rem 0;'>"
        "<p style='color:#484f58;font-size:0.73rem;line-height:1.6;'>"
        "Deploy free on:<br>"
        "<a href='https://huggingface.co/spaces' style='color:#58a6ff;'>Hugging Face Spaces</a><br>"
        "<a href='https://streamlit.io/cloud' style='color:#58a6ff;'>Streamlit Community Cloud</a><br>"
        "<a href='https://render.com' style='color:#58a6ff;'>Render</a><br>"
        "<span style='color:#30363d;'>Add GROQ_API_KEY as a secret.</span>"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — UPLOAD STATE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.vectorstore is None:

    st.markdown(
        "<h1 style='color:#e6edf3;font-size:1.6rem;font-weight:700;margin:0 0 0.3rem 0;'>Upload a PDF</h1>"
        "<p style='color:#8b949e;font-size:0.88rem;margin:0 0 1.6rem 0;'>"
        "Index any PDF to start chatting with it using Groq's LLMs.</p>",
        unsafe_allow_html=True,
    )

    tab_up, tab_saved = st.tabs(["Upload new PDF", "Load saved index"])

    # ── Upload ─────────────────────────────────────────────────────────────────
    with tab_up:
        uploaded_file = st.file_uploader(
            "pdf_drop",
            type=["pdf"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            st.markdown(
                f"<div style='background:#161b22;border:1px solid #21262d;border-radius:8px;"
                f"padding:0.75rem 1rem;display:flex;align-items:center;gap:10px;margin:0.8rem 0;'>"
                f"<div>"
                f"<p style='color:#e6edf3;font-size:0.85rem;font-weight:500;margin:0;'>{uploaded_file.name}</p>"
                f"<p style='color:#484f58;font-size:0.75rem;margin:2px 0 0 0;'>{uploaded_file.size / 1024:.1f} KB · PDF</p>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                go = st.button("Build Index", type="primary", use_container_width=True)

            if go:
                if not groq_key:
                    st.error("Set your Groq API key in the sidebar first.")
                else:
                    bar = st.progress(0, text="Starting…")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        pdf_path = tmp.name
                    bar.progress(10, text="File saved…")

                    try:
                        bar.progress(20, text="Loading embedding model (all-MiniLM-L6-v2)…")
                        emb = get_embeddings()
                        bar.progress(45, text="Embedding model ready")
                    except Exception as e:
                        st.error(f"Embedding error: {e}")
                        st.stop()

                    try:
                        bar.progress(55, text="Parsing PDF…")
                        raw_docs = PyPDFLoader(pdf_path).load()
                        chunks = RecursiveCharacterTextSplitter(
                            chunk_size=800, chunk_overlap=120
                        ).split_documents(raw_docs)
                        bar.progress(72, text=f"Created {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"PDF parse error: {e}")
                        st.stop()

                    try:
                        bar.progress(82, text="Building FAISS index…")
                        vs = FAISS.from_documents(chunks, emb)
                        save_index(vs)
                        bar.progress(100, text="Done!")
                        st.session_state.vectorstore = vs
                        st.session_state.doc_name    = uploaded_file.name
                        st.session_state.doc_chunks  = len(chunks)
                    except Exception as e:
                        st.error(f"Index error: {e}")
                        st.stop()

                    st.rerun()

    # ── Load saved ─────────────────────────────────────────────────────────────
    with tab_saved:
        if os.path.exists("faiss_index"):
            st.markdown(
                "<div style='background:#161b22;border:1px solid #21262d;border-radius:8px;"
                "padding:0.9rem 1rem;margin-bottom:1rem;'>"
                "<p style='color:#e6edf3;font-size:0.88rem;font-weight:500;margin:0 0 4px 0;'>"
                "Saved index found</p>"
                "<p style='color:#8b949e;font-size:0.8rem;margin:0;'>"
                "A FAISS index from a previous session is saved on disk.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Load Index", type="primary"):
                with st.spinner("Loading embedding model and index…"):
                    try:
                        emb = get_embeddings()
                        vs  = load_index(emb)
                        st.session_state.vectorstore = vs
                        st.session_state.doc_name    = "Loaded from disk"
                        st.session_state.doc_chunks  = 0
                        st.rerun()
                    except Exception as e:
                        st.error(f"Load failed: {e}")
        else:
            st.markdown(
                "<div style='background:#161b22;border:1px solid #21262d;border-radius:8px;"
                "padding:0.9rem 1rem;'>"
                "<p style='color:#8b949e;font-size:0.88rem;margin:0;'>"
                "No saved index found. Upload a PDF first — the index saves automatically.</p>"
                "</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — CHAT STATE
# ══════════════════════════════════════════════════════════════════════════════
else:

    # ── Document + controls header ──────────────────────────────────────────────
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        chunks_txt = f" · {st.session_state.doc_chunks} chunks" if st.session_state.doc_chunks else ""
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #21262d;border-radius:8px;"
            f"padding:0.65rem 1rem;display:flex;align-items:center;gap:8px;'>"
            f"<span style='color:#3fb950;font-size:0.7rem;'>●</span>"
            f"<span style='color:#e6edf3;font-size:0.85rem;font-weight:500;'>{st.session_state.doc_name}</span>"
            f"<span style='color:#30363d;'>·</span>"
            f"<span style='color:#484f58;font-size:0.78rem;'>{groq_model}{chunks_txt}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("New document", use_container_width=True):
            st.session_state.vectorstore  = None
            st.session_state.chat_history = []
            st.session_state.doc_name     = ""
            st.session_state.doc_chunks   = 0
            st.rerun()

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Chat history ────────────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='text-align:center;padding:3rem 1rem;background:#161b22;"
            "border:1px solid #21262d;border-radius:10px;margin-bottom:1rem;'>"
            "<p style='color:#8b949e;font-size:0.9rem;font-weight:500;margin:0 0 4px 0;'>"
            "Document indexed and ready</p>"
            "<p style='color:#484f58;font-size:0.8rem;margin:0;'>"
            "Ask anything — summaries, key points, specific details</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.chat_history[-30:]:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(msg["q"])
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(msg["a"])

    # ── Chat input ──────────────────────────────────────────────────────────────
    query = st.chat_input("Ask a question about the document…")

    if query:
        if not groq_key:
            st.error("Groq API key missing — add it in the sidebar.")
        else:
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                with st.spinner("Generating answer…"):
                    try:
                        chain  = build_chain(st.session_state.vectorstore, groq_key, groq_model, top_k)
                        answer = chain.invoke(query)
                        st.markdown(answer)
                        st.session_state.chat_history.append({"q": query, "a": answer})
                    except Exception as e:
                        err = str(e)
                        if "401" in err or "api_key" in err.lower():
                            st.error("Invalid API key — update it in the sidebar.")
                        elif "429" in err:
                            st.warning("Rate limit reached — wait a moment and retry.")
                        elif "decommission" in err.lower() or "model_not_active" in err.lower():
                            st.error("This model was decommissioned. Select another in the sidebar.")
                        else:
                            st.exception(e)

    # ── Footer ──────────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("<hr style='border:none;border-top:1px solid #21262d;margin:1.5rem 0 0.8rem 0;'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🧹 Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            st.markdown(
                f"<p style='color:#484f58;font-size:0.76rem;padding-top:0.45rem;'>"
                f"{len(st.session_state.chat_history)} messages</p>",
                unsafe_allow_html=True,
            )
