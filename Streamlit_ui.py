"""
===============================================================================
RAGStack
===============================================================================
Author      : Mohit Srivastava
Email       : mosriva@gmail.com
Version     : v1.0.0
License     : MIT
Description :
    A fully local Retrieval-Augmented Generation (RAG) assistant
    for secure, context-aware document intelligence and question answering over private enterprise PDFs.

Key Technologies:
  - Streamlit (UI)
  - FAISS (vector retrieval)
  - Sentence-Transformers (embeddings)
  - Ollama (local LLM inference)
  - PyMuPDF (PDF parsing)

Features:
  • Auto-index PDFs on upload (no extra clicks)
  • De-duplicate via persistent manifest (SHA-1)
  • Manual index rebuild from existing PDFs
  • Context-based Q&A using local LLMs
  • Persistent FAISS index and chat history
  • Health check indicators (Ollama API, model status, index readiness)

Citation (add DOI when available):
  Srivastava, M. (2025). RAGStack:
  Reference implementation of RAGStack — a privacy-first, local RAG architecture for secure enterprise document intelligence.
  Zenodo. DOI: <will be added once issued>
===============================================================================
"""

import os
import json
import hashlib
import pickle
from datetime import datetime
from functools import lru_cache

import faiss
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException
from sentence_transformers import SentenceTransformer
import streamlit as st

st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")

# -------- CONFIG --------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

INDEX_DIR = "index"
EMBEDDINGS_FILE = os.path.join(INDEX_DIR, "embeddings.pkl")
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
MANIFEST_FILE = os.path.join(INDEX_DIR, "manifest.json")

LOGS_DIR = "logs"
CHAT_HISTORY_FILE = os.path.join(LOGS_DIR, "chat_history.pkl")
LOG_FILE = os.path.join(LOGS_DIR, "session_log.txt")
EXPORT_CSV_FILE = os.path.join(LOGS_DIR, "chat_export.csv")

PDF_SAVE_DIR = "uploaded_pdfs"

OLLAMA_ENDPOINT = "http://localhost:11434"
OLLAMA_GENERATE = f"{OLLAMA_ENDPOINT}/api/generate"
OLLAMA_TAGS = f"{OLLAMA_ENDPOINT}/api/tags"

MAX_CONTEXT_CHUNKS = 5

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(PDF_SAVE_DIR, exist_ok=True)

# -------- HELPERS: Index status --------
def index_ready(ix) -> bool:
    try:
        return ix is not None and getattr(ix, "ntotal", 0) > 0
    except Exception:
        return False


def index_size(ix) -> int:
    try:
        return int(getattr(ix, "ntotal", 0))
    except Exception:
        return 0


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load SentenceTransformer on CPU, handle Torch meta-tensor issues."""
    try:
        return SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    except NotImplementedError:
        return SentenceTransformer(FALLBACK_EMBED_MODEL, device="cpu")
    except Exception as e:
        st.error(f"Embedding model failed to load: {e}")
        st.stop()


# -------- HELPERS: Ollama health --------
def check_ollama_alive() -> bool:
    try:
        r = requests.get(OLLAMA_TAGS, timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False


def list_ollama_models():
    try:
        r = requests.get(OLLAMA_TAGS, timeout=5)
        r.raise_for_status()
        data = r.json()
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def model_available_in_ollama(model_name: str) -> bool:
    models = list_ollama_models()
    return any(model_name == m or model_name == m.split(":")[0] for m in models)


# -------- HELPERS: Manifest & hashing --------
def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"files": []}
    return {"files": []}


def save_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def file_sha1(file_bytes: bytes) -> str:
    h = hashlib.sha1()
    h.update(file_bytes)
    return h.hexdigest()


# -------- LOAD COMPONENTS --------
def load_components():
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
        ix = faiss.read_index(FAISS_INDEX_FILE)
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
        ch = data.get("chunks", [])
    else:
        ix, ch = None, []
    return ix, ch


index, chunks = load_components()
manifest = load_manifest()
known_hashes = {entry["sha1"] for entry in manifest.get("files", [])}


# -------- INGESTION --------
def extract_text_chunks_from_pdf(file_path, file_name, chunk_size=300):
    """
    Extract text from a PDF and split into ~chunk_size-word chunks.
    Skips empty pages and empty chunks.
    """
    doc = fitz.open(file_path)
    text_chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text() or ""
        words = text.split()
        if not words:
            continue
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if not chunk:
                continue
            text_chunks.append(
                {
                    "chunk": chunk,
                    "source": {"filename": file_name, "page": page_num + 1},
                }
            )
    return text_chunks


def update_index_with_uploaded_file(file):
    """Add a single PDF (already in memory) into the FAISS index."""
    global index, chunks

    file_path = os.path.join(PDF_SAVE_DIR, file.name)

    # Save only if not present (sometimes already saved by uploader flow)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f_out:
            f_out.write(file.getbuffer())

    # Extract chunks
    new_chunks = extract_text_chunks_from_pdf(file_path, file.name)
    if not new_chunks:
        st.warning(f"⚠️ No text extracted from {file.name}")
        return

    # Compute embeddings (lazy load embedder here)
    model = get_embedder()
    texts = [c["chunk"] for c in new_chunks]
    new_embeddings = model.encode(texts)
    embedding_dim = new_embeddings.shape[1]

    # Build or extend index
    if index is None:
        index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.asarray(new_embeddings, dtype=np.float32))

    # Extend and persist metadata
    chunks.extend(new_chunks)
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"chunks": chunks}, f)


# -------- SELF-HEAL (Manual Rebuild Only) --------
def rebuild_index_from_pdfs():
    """
    Rebuild FAISS index and embeddings from all PDFs on disk.
    Updates manifest as needed.
    """
    global index, chunks, manifest, known_hashes

    index, chunks = None, []
    if "files" not in manifest:
        manifest = {"files": []}
        known_hashes = set()

    rebuilt = 0
    for fname in sorted(os.listdir(PDF_SAVE_DIR)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(PDF_SAVE_DIR, fname)
        try:
            with open(fpath, "rb") as f:
                data = f.read()

            class _Tmp:
                name = fname

                def getbuffer(self):
                    return data

            update_index_with_uploaded_file(_Tmp())

            sha1 = file_sha1(data)
            if sha1 not in known_hashes:
                manifest["files"].append(
                    {
                        "name": fname,
                        "sha1": sha1,
                        "size": len(data),
                        "indexed_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                known_hashes.add(sha1)

            rebuilt += 1
        except Exception:
            # Skip any problematic file, do not crash rebuild
            pass

    save_manifest(manifest)
    return rebuilt


# NOTE: We intentionally remove auto-heal at import-time, as it can
# trigger segfaults with heavy C-extensions on some macOS setups.
# Rebuild is now *manual* via a sidebar button.


# -------- RAG FUNCTIONS --------
def search_similar_chunks(
    query: str, top_k: int = MAX_CONTEXT_CHUNKS, min_similarity: float | None = None
):
    """
    Retrieve the top_k most relevant chunks for a query and return
    (chunk_dict, similarity) pairs, where similarity ∈ (0, 1] and higher is better.

    FAISS IndexFlatL2 returns L2 distances (smaller is better).
    We convert distance d to normalized similarity: sim = 1 / (1 + d).
    """

    if not index_ready(index):
        raise ValueError("No documents indexed yet. Please upload PDFs first.")

    # Embed query (on CPU)
    model = get_embedder()
    query_embedding = model.encode([query])

    # FAISS expects float32
    distances, indices = index.search(
        np.asarray(query_embedding, dtype=np.float32), top_k
    )

    results = []
    for rank, i in enumerate(indices[0]):
        if i == -1:
            continue
        d = float(distances[0][rank])  # L2 distance from FAISS
        sim = 1.0 / (1.0 + d)  # normalized similarity in (0,1]
        if (min_similarity is None) or (sim >= min_similarity):
            results.append((chunks[i], sim))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def build_prompt(context_chunks, user_query):
    if not context_chunks:
        context = "(no matching context found)"
    else:
        context = "\n\n".join(
            [
                f"[Source: {c['source']['filename']}, page {c['source']['page']}]\n{c['chunk']}"
                for c, _ in context_chunks
            ]
        )
    return f"""You are a helpful enterprise assistant. Use the following context.

Context:
{context}

Question:
{user_query}

Answer:"""


def query_ollama(prompt, model_name):
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_GENERATE, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except RequestException as e:
        raise RuntimeError(
            "❌ Could not reach the Ollama API at http://localhost:11434.\n"
            "Ensure Ollama is running and the selected model is installed."
        ) from e


def log_interaction(query, context_chunks, answer, model_used):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] MODEL: {model_used}\n")
        f.write(f"Q: {query}\n")
        for c, similarity in context_chunks:
            src = (
                f"[Source: {c['source']['filename']}, page {c['source']['page']}]"
            )
            f.write(f"{src} (Similarity: {similarity:.2f})\n")
        f.write(f"Answer: {answer.strip()}\n")
        f.write("-" * 40 + "\n")


# -------- CHAT HISTORY --------
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "rb") as f:
            return pickle.load(f)
    return []


def save_chat_history(chat_log):
    with open(CHAT_HISTORY_FILE, "wb") as f:
        pickle.dump(chat_log, f)


def export_chat_to_csv(chat_log):
    pd.DataFrame(chat_log).to_csv(EXPORT_CSV_FILE, index=False)


# -------- UI --------
st.title("📄🔍 RAGStack: Enterprise RAG Assistant")
st.caption("Reference implementation of RAGStack — a privacy-first, local RAG architecture for secure enterprise document intelligence.")

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Select LLM model:", ["mistral", "llama2", "phi"])
    st.caption("Make sure the model is installed in Ollama.")

    # Precompute visible PDFs (ignore .DS_Store and non-PDFs)
    pdf_list = [
        f
        for f in sorted(os.listdir(PDF_SAVE_DIR))
        if not f.startswith(".") and f.lower().endswith(".pdf")
    ]

    st.header("🩺 Health Check")
    ok_ollama = check_ollama_alive()
    st.markdown(f"- Ollama API: {'🟢 Running' if ok_ollama else '🔴 Not reachable'}")
    if ok_ollama:
        avail = model_available_in_ollama(model_choice)
        st.markdown(
            f"- Model `{model_choice}`: {'🟢 Available' if avail else '🟠 Missing'}"
        )
    else:
        st.markdown(f"- Model `{model_choice}`: 🟠 Unknown (Ollama down)")
    st.markdown(
        f"- Indexed Chunks: **{index_size(index)}** "
        f"{'🟢' if index_ready(index) else '🟠'}"
    )
    st.markdown(f"- Uploaded PDFs: **{len(pdf_list)}**")

    # Upload (Auto-index with de-duplication)
    st.header("🗂️ Upload PDF(s)")
    uploaded_files = st.file_uploader(
        "Drop PDFs here (auto-index on upload)",
        type="pdf",
        accept_multiple_files=True,
        key="uploader_autoinDEX",
    )

    if uploaded_files:
        new_count = 0
        for uf in uploaded_files:
            data = uf.read()
            sha1 = file_sha1(data)

            if sha1 in known_hashes:
                st.info(f"Skipped (already indexed): {uf.name}")
                continue

            # Save to disk with collision-safe name
            dst = os.path.join(PDF_SAVE_DIR, uf.name)
            base, ext = os.path.splitext(uf.name)
            i = 1
            while os.path.exists(dst):
                dst = os.path.join(PDF_SAVE_DIR, f"{base}_{i}{ext}")
                i += 1
            with open(dst, "wb") as f:
                f.write(data)

            class _Tmp:
                name = os.path.basename(dst)

                def getbuffer(self):
                    return data

            update_index_with_uploaded_file(_Tmp())

            manifest["files"].append(
                {
                    "name": os.path.basename(dst),
                    "sha1": sha1,
                    "size": len(data),
                    "indexed_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            known_hashes.add(sha1)
            new_count += 1

        save_manifest(manifest)
        if new_count:
            st.success(f"✅ Indexed {new_count} new file(s).")
        else:
            st.info("No new files to index.")

        # refresh pdf_list after uploads
        pdf_list = [
            f
            for f in sorted(os.listdir(PDF_SAVE_DIR))
            if not f.startswith(".") and f.lower().endswith(".pdf")
        ]

    if pdf_list:
        st.subheader("📁 Uploaded Documents")
        for f in pdf_list:
            st.markdown(f"- {f}")

        # Manual rebuild button (replaces auto-heal at import time)
        if st.button("🔁 Rebuild index from existing PDFs"):
            with st.spinner("Rebuilding index from PDFs on disk..."):
                rebuilt = rebuild_index_from_pdfs()
                index, chunks = load_components()
            st.success(f"Rebuilt index from {rebuilt} PDF file(s).")
            st.rerun()

    # History
    st.header("🕘 Chat History")
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = load_chat_history()
    if not st.session_state.chat_log:
        st.info("No chat history yet.")
    else:
        for entry in reversed(st.session_state.chat_log):
            with st.expander(entry["question"]):
                st.markdown(f"**Model:** {entry['model']}")
                st.write(entry["answer"])
        cols = st.columns(2)
        with cols[0]:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_log = []
                save_chat_history([])
                st.success("History cleared.")
                st.rerun()
        with cols[1]:
            if st.button("📤 Export Q&A to CSV"):
                export_chat_to_csv(st.session_state.chat_log)
                st.success(f"Exported to `{EXPORT_CSV_FILE}`")


# -------- MAIN (Q&A with explicit submit) --------
ready = index_ready(index)

with st.form("qa_form", clear_on_submit=False):
    user_query = st.text_input(
        "💬 Ask a question:",
        "",
        placeholder="Upload PDFs first",
        disabled=not ready,
        key="qa_input",
    )
    ask = st.form_submit_button("🔎 Ask", disabled=not ready)

if not ready:
    st.info("📎 Upload at least one PDF before asking questions.")

if ask:
    with st.spinner("Searching and generating answer..."):
        try:
            context_chunks = search_similar_chunks(user_query)
            full_prompt = build_prompt(context_chunks, user_query)
            answer = query_ollama(full_prompt, model_choice)

            st.subheader("🔎 Answer")
            st.write(answer or "No answer returned.")

            with st.expander("📚 Sources & Context", expanded=bool(context_chunks)):
                if not context_chunks:
                    st.write("No relevant context found.")
                else:
                    for c, similarity in context_chunks:
                        st.markdown(
                            f"**{c['source']['filename']}**, page {c['source']['page']} "
                            f"*(Similarity: {similarity:.2f})*"
                        )
                        st.caption(c["chunk"])

            st.session_state.chat_log.append(
                {"question": user_query, "answer": answer, "model": model_choice}
            )
            save_chat_history(st.session_state.chat_log)
            log_interaction(user_query, context_chunks, answer, model_choice)

        except ValueError as ve:
            st.warning(str(ve))
        except RuntimeError as re:
            st.error(str(re))
        except Exception as e:
            st.error(f"Unexpected error: {e}")

