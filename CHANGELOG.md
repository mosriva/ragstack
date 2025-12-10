# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.0.0] – 2025-12-06
### Added
- Initial public release of **RAGStack**.
- Reference implementation of the RAGStack architecture for privacy-first enterprise RAG.
- End-to-end local RAG pipeline:
  - PDF ingestion via PyMuPDF  
  - SentenceTransformers-based embeddings  
  - FAISS vector similarity search  
  - Locally hosted LLM inference using Ollama  
- Streamlit UI with Q&A chat interface, source traceability, and CSV export.
- Auto-indexing upon PDF upload with SHA-1 deduplication (`index/manifest.json`).
- Self-healing index mechanism: automatic FAISS rebuild from `uploaded_pdfs/` if index is missing or corrupt.
- Normalized similarity scoring (`1 / (1 + distance)`, range 0–1) for intuitive relevance display.
- Health monitoring panel (LLM status, model availability, index/document counters).
- Persistent chat session logging stored in `logs/`.

### Known Limitations
- Parsing quality may vary based on PDF formatting or OCR/scanning quality.
- No built-in GPU acceleration toggle (operates in CPU-mode by default; accelerator support available via Ollama).
- Basic chunking logic; advanced semantic chunking (e.g., LangChain splitters) available via manual extension.

[v1.0.0]: https://github.com/mosriva/ragstack/releases/tag/v1.0.0
