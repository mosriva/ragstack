# 🚀 Release Notes – RAGStack

## 📌 Version v1.0.0  
**Release Date:** 2025-12-09  
**Status:** Initial Public Release  
**License:** MIT  
**Repository:** https://github.com/mosriva/ragstack
**DOI (Zenodo):** 10.5281/zenodo.17878948 

---

## ✨ Overview
RAGStack v1.0.0 delivers the **first public reference implementation** of a fully local Retrieval-Augmented Generation framework, optimized for **offline enterprise environments** with **zero cloud/API dependency** and **full data sovereignty**.

This release includes:
- Fully local RAG pipeline
- Real-time Q&A on enterprise PDFs
- Similarity scoring & audit traceability  
- Streamlit-based chat UI  
- Self-healing FAISS indexing mechanism

---

## 🆕 What's New in v1.0.0

| Feature Category | Highlights |
|------------------|------------|
| 🔒 **Security & Privacy** | 100% local execution, no external data transfer |
| 📚 **RAG Pipeline** | PyMuPDF → SentenceTransformers → FAISS → Ollama |
| 💬 **UI & Interaction** | Streamlit app with PDF upload, model selection, session export |
| 🔁 **Audit & Persistence** | SHA-1 deduplication, manifest tracking, index auto-rebuild |
| 📈 **Retrieval Ranking** | Normalized similarity scoring (0–1 scale) |
| 🧪 **Performance** | Internal tests show Avg. query time ~5s with Mistral 7B on CPU |
| 📁 **Data Handling** | Local folders for uploads, index, and logs |

---

## 🔍 Known Limitations

| Limitation | Notes |
|-----------|-------|
| CPU-only by default | GPU acceleration supported but not auto-detected |
| Basic chunking | Advanced semantic chunking optional |
| Parsing quality | Depends on PDF formatting (OCR currently unsupported) |

---

## 🧭 Roadmap (Upcoming Enhancements)
📌 OCR support for scanned PDF ingestion  
📌 Benchmark integration (e.g., HotpotQA, BioASQ)  
📌 Fine-tuned retrieval for enterprise domain datasets  
📌 Support for GPU auto-detection + performance scaling  
📌 Expand multi-format support (DOCX, TXT)

---

## 🛠 Installation (Quick Start)

```
git clone https://github.com/mosriva/ragstack.git
cd ragstack
python3 -m venv rag_venv
source rag_venv/bin/activate   # macOS / Linux
# .\rag_venv\Scripts\activate   # Windows PowerShell
pip install --upgrade pip 
pip install -r requirements.txt

ollama pull mistral
streamlit run streamlit_ui.py --server.port 8501
```
📎 Citation

If you use this in academic or enterprise research:

Srivastava, M. (2025). RAGStack: A Privacy-First GenAI Retrieval-Augmented Generation Architecture for Secure Enterprise Document Intelligence. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

🤝 Contributors
| Name                 | Role                       |
| -------------------- | -------------------------- |
| **Mohit Srivastava** | Lead Architect & Developer |

🎯 Suggestions and contributions are welcome! See CONTRIBUTING.md for details.


🚀 Thank you for using Enterprise RAGStack!
For feedback, open an issue or email mosriva@gmail.com
