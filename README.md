# 🏢 RAGStack 
*Reference implementation of **RAGStack: A Privacy-First GenAI Retrieval-Augmented Generation Architecture for Secure Enterprise Document Intelligence***  

<p align="center">
  <a href="https://github.com/mosriva/ragstack/stargazers"><img src="https://img.shields.io/github/stars/mosriva/ragstack?style=flat-square&color=yellow" alt="GitHub Stars"></a>
  <a href="https://github.com/mosriva/ragstack/releases"><img src="https://img.shields.io/github/v/release/mosriva/ragstack?style=flat-square&color=blue" alt="Latest Release"></a>
  <a href="https://zenodo.org/badge/latestdoi/zenodo.XXXXXXX"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg" alt="DOI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License: MIT"></a>
  <a href="https://techrxiv.org"><img src="https://img.shields.io/badge/Preprint-TechRxiv-orange?style=flat-square" alt="TechRxiv"></a>
</p>

---

## 📘 Overview
The **RAGStack (v1.0.0)** is the reference implementation of the **RAGStack** architecture described in the paper  
“*RAGStack: A Privacy-First GenAI Retrieval-Augmented Generation Architecture for Secure Enterprise Document Intelligence*”.
It combines **Ollama**, **FAISS**, and **Streamlit** to deliver an air-gapped GenAI workflow — ensuring **data privacy**, **cloud-free execution**, and **reproducibility**.
This project demonstrates how modular, open-source GenAI architectures can power internal enterprise search, compliance review, and knowledge-retrieval systems.

---

## 🚀 Key Features
- 📁 Automatic PDF ingestion and text chunking  
- 🔎 Contextual retrieval using FAISS vector search  
- 🧠 Local LLMs via **Ollama** (Mistral, LLaMA2, Phi)  
- 💬 Streamlit-based conversational interface  
- 🧾 Persistent chat history + CSV export  
- 🗂️ Self-healing FAISS index (auto-rebuilt if missing)  
- 🔄 SHA-1 deduplication and manifest tracking  
- 📈 Normalized **similarity scoring** (0 – 1) for intuitive relevance display  

---

## 🧱 System Architecture

The high-level architecture of RAGStack is shown below, illustrating ingestion, vectorization, retrieval, and local reasoning flow.


───────────────────────── RAGStack: Privacy-First Enterprise GenAI ──────────────────────────

                  (Air-Gapped / On-Prem / No External APIs / Full Data Sovereignty)
         
─────────────────────────────────────────────────────────────────────────────────────────

<pre>

DOCUMENT INGESTION & VECTORIZATION                   QUERY, RETRIEVAL & LOCAL REASONING

</pre>


──────────────────────────────────                    ──────────────────────────────────

    ┌──────────────────────────┐                       ┌──────────────────────────┐
    │      PDF Documents       │                       │        User Query        │
    │  Uploaded via Streamlit  │                       │   Streamlit Q&A Form     │
    │  Rebuild Index if Missing│                       └─────────────┬────────────┘
    └─────────────┬────────────┘                                     │
                  │                                                  ▼
                  ▼                                  ┌──────────────────────────┐
    ┌──────────────────────────┐                     │     Query Embeddings     │
    │    PyMuPDF Parsing       │                     │   SentenceTransformers   │
    │    300-word Chunking     │                     └─────────────┬────────────┘
    └─────────────┬────────────┘                                   │
                  │                                                ▼
                  ▼                                  ┌──────────────────────────┐
    ┌──────────────────────────┐                     │   FAISS Top-K Retrieval  │
    │  SentenceTransformers    │◄────────────────────┤  Normalized Similarity   │
    │  all-MiniLM-L6-v2        │   Nearest Neighbors └─────────────┬────────────┘
    └─────────────┬────────────┘                                   │
                  │                                                ▼
                  ▼                                  ┌──────────────────────────┐
    ┌──────────────────────────┐                     │    Prompt Construction   │
    │     FAISS Vector Index   │                     │ Context + Filename + Pg  │
    │     FlatL2 (Persistent)  │                     └─────────────┬────────────┘
    └─────────────┬────────────┘                                   │
                  │                                                ▼
                  ▼                                  ┌──────────────────────────┐
    ┌──────────────────────────────────────────┐     │  Answer + Attribution    │
    │      Persistent Storage & Audit Logs     │◄────┤ Streamlit UI + Export    │
    │ uploaded_pdfs / index / logs directories │     └──────────────────────────┘
    └──────────────────────────────────────────┘

                                   ┌──────────────────────────────────┐
                                   │        Local LLM Runtime         │
                                   │  Ollama (Mistral / LLaMA2 / Phi) │
                                   └──────────────────────────────────┘

                                   ┌──────────────────────────────────┐
                                   │   Health & Governance Panel      │
                                   │ • Ollama API & Model Status      │
                                   │ • Index & Document Counters      │
                                   │ • De-duplication Manifest        │
                                   │ • Exportable Audit Logs (CSV)    │
                                   └──────────────────────────────────┘


---
## ⚙️ Components and Workflow

| Component | Description | Library / Tool |
|------------|--------------|----------------|
| **PDF Parser** | Extracts text and metadata | PyMuPDF |
| **Chunking Module** | Splits text into coherent blocks | Native loop (optional LangChain splitter) |
| **Embedding Generator** | Converts text to vector representation | `sentence-transformers` |
| **Vector Store** | Stores / retrieves embeddings | FAISS |
| **Retriever** | Retrieves top-k chunks using **normalized similarity (1 / (1 + distance))** | FAISS |
| **Local LLM** | Generates grounded answers | Ollama (Mistral / LLaMA2 / Phi) |
| **UI Layer** | Uploads, chat, export | Streamlit |

**Workflow Summary**
1. Upload one or more PDFs via the Streamlit UI.  
2. Text is parsed, chunked, and embedded locally.  
3. A query retrieves the top-k most similar chunks.  
4. Context is appended to the LLM prompt for grounding.  
5. Answers + sources are displayed and logged.

---

## 🧩 Technical Stack
- **Language:** Python 3.10 +  
- **Frameworks:** Streamlit, FAISS, SentenceTransformers  
- **LLM Runtime:** Ollama (Mistral, LLaMA2, Phi)  
- **Persistence:** Local files for PDFs, FAISS index, and chat logs  
- **Platform:** Offline / air-gapped enterprise environments  

---

## 🧰 Installation & Setup
1. **Clone the Repository**
```
   git clone https://github.com/mosriva/ragstack.git
   cd ragstack
```
2. Create & Activate Virtual Environment (Recommended)
```
python3 -m venv rag_venv
source rag_venv/bin/activate    # macOS / Linux
# .\rag_venv\Scripts\activate # Windows PowerShell
```
3. Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
4. Start Ollama and Load Model
```
ollama pull mistral
ollama run mistral
```
5. Launch Streamlit App
```
streamlit run streamlit_ui.py --server.port 8501
```
📌 Optional (GPU Acceleration & Alternative Models)
```
ollama pull llama2   # or phi, mistral:latest, etc. To switch models, use the dropdown in the Streamlit UI.
#⚠ If switching models (e.g., phi, llama2), ensure that model name matches exactly in Streamlit dropdown and Ollama supports the model locally.
```
🔄 Uninstall / Environment Reset
```
pip uninstall -y torch sentence-transformers streamlit faiss-cpu pymupdf pandas
rm -rf rag_venv index uploaded_pdfs logs

# Optional: Remove cached Ollama models if needed
# ollama rm mistral; ollama rm llama2; ollama rm phi
```
After uninstall/reset, recreate the environment starting from Step 2.


🖥️ macOS Users — Important Runtime Fix (Avoid Segmentation Fault)

Some macOS systems (especially M1/M2/M3 chips) may experience segmentation faults when running Streamlit with PyTorch, FAISS, and HuggingFace tokenizers together.
This is a known issue related to macOS fork safety + parallel tokenizers.

To ensure stable execution, set the following environment variables before launching the app:
```
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

Then start the application:
```
streamlit run streamlit_ui.py --server.port 8501
```
Why this is needed?

TOKENIZERS_PARALLELISM=false
Prevents HuggingFace tokenizers from spawning threads too early (avoids deadlocks/segfaults).

OMP_NUM_THREADS=1
Reduces thread contention inside PyTorch and FAISS on macOS.

OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
Allows PyTorch + tokenizers to run safely after macOS performs a fork (Streamlit internally forks processes).

## 🧪 Example Interaction

**📝 Query:**  
> What are the key risks mentioned in the data governance policy?

**💡 Response (sample):**  
The policy highlights risks related to unauthorized access, data-retention violations, and compliance failures.

**📎 Sources:**  
- *data-governance.pdf*, Page 4  
- *data-policy.pdf*, Page 1  

## 📊 Evaluation Summary
| Metric                 | Description                                 | Result (Example) |
| ---------------------- | ------------------------------------------- | ---------------- |
| Response Latency       | Avg. query time (Mistral 7B local)          | ~5-15 s           |
| Retrieval Precision    | Relevance of retrieved chunks (manual eval) | 0.89             |
| Context Token Coverage | % of retrieved text used in final prompt    | ~78 %            |
| Memory Usage           | Peak FAISS index memory (100 PDFs)          | ~180 MB          |


## 📂 Repository Structure
```
ragstack/
├── README.md                   # Project overview
├── streamlit_ui.py            # Main application
├── requirements.txt           # Python dependencies
├── CHANGELOG.md               # Version history
├── CONTRIBUTING.md            # Contribution guidelines
├── release_notes.md           # Release-specific highlights
├── citation.cff               # Citation metadata
├── LICENSE                    # MIT open-source license
├── .gitignore
├── uploaded_pdfs/
│   └── .gitkeep               # (ignored in git)
├── index/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
└── docs/
    └──Architecture/  
        └── ragstack_architecture_figure1.png

```
## ⚖️ License

This project is released under the MIT License.
You are free to use, modify, and distribute it with attribution.

## 🧠 Citation

If you use this work in your research, please cite:

Srivastava, M. (2025). RAGStack: A Privacy-First GenAI Retrieval-Augmented Generation Architecture for Secure Enterprise Document Intelligence. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

## 🙌 Acknowledgments & Inspiration

Ollama – Local LLM runtime for private inference

FAISS – Efficient vector similarity search

LangChain / LlamaIndex – RAG design patterns

Streamlit – Rapid UI prototyping for GenAI

## 🔗 Related Links

TechRxiv Preprint: [Link to be added once published]

Zenodo Implementation Archive: [Link to be added DOI once issued]

GitHub Repository: https://github.com/mosriva/ragstack
