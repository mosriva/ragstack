# Contributing to RAGStack

We appreciate your interest in improving the **RAGStack**.
Contributions of all kinds—code, documentation, performance improvements, and feature suggestions—are welcome.

## 🧭 Contribution Workflow

1. Fork & Clone the repository  
```
   git clone https://github.com/mosriva/ragstack.git
   cd ragstack
```
2. Create a feature branch
```
   git checkout -b feature/my-improvement
```
3. Make your changes
   
   Follow PEP8 conventions (black formatting recommended)
   
   Use meaningful commit messages
   
4. Test your changes
   
   Run the Streamlit app locally
```
   ollama run mistral
   streamlit run streamlit_ui.py
```
   Upload a few PDFs to verify indexing, retrieval, and responses
   
5. Submit a Pull Request
   
   Provide a concise title and summary
   
   Describe your motivation and test coverage
   
   Link any related issues (if applicable)
   
🧪 Development Setup

   Recommended environment:
   
      python3.10 -m venv rag_venv
      
      source rag_venv/bin/activate  # or rag_venv\Scripts\activate on Windows
      
      pip install -r requirements.txt
      
   Run app:
```      
      ollama run mistral
      streamlit run streamlit_ui.py
```
🧱 Code Style
| Area    | Standard                                              |
| ------- | ----------------------------------------------------- |
| Python  | [PEP 8](https://peps.python.org/pep-0008/)            |
| Commits | Conventional commits (`feat:`, `fix:`, `docs:`, etc.) |
| Docs    | Markdown (README.md, CHANGELOG.md)                    |
| Testing | Manual and unit tests (optional pytest integration)   |

🧩 Contribution Ideas

Add support for additional local models (Gemma, Phi-2, etc.)

Improve UI with multi-doc comparison

Integrate LangChain or LlamaIndex retrieval pipelines

Extend to support DOCX and TXT ingestion

Add RAG evaluation metrics (retrieval precision, answer faithfulness)

🏁 License and Attribution

By contributing, you agree that your submissions will be licensed under the same MIT License used by this project.

🙌 Thanks!

Thank you for helping advance secure, privacy-first RAG architectures for enterprise AI applications 🚀
If you have questions or want to propose larger enhancements, feel free to open a GitHub Issue or start a Discussion.
