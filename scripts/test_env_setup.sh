#!/usr/bin/env bash
# ========================================
# test_env_setup.sh
# Environment setup & validation script
# For Enterprise RAG Assistant (Linux/macOS)
# Usage: chmod +x test_env_setup.sh && ./test_env_setup.sh
# ========================================
set -e  # Exit on error
echo "🧪 Starting Enterprise RAG Assistant test environment setup..."

# ================================
# 1️⃣ Detect Python executable
# ================================
if command -v python3 &> /dev/null; then
  PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
  PYTHON_CMD="python"
else
  echo "❌ Python not found. Please install Python 3.10+ and retry."
  exit 1
fi
echo "✔ Using Python command: $PYTHON_CMD"

# ================================
# 2️⃣ OPTIONAL: Cleanup old venv
# ================================
echo "🧹 Cleaning old virtual environment (if exists)..."
rm -rf rag_venv || true

# ================================
# 3️⃣ Create fresh virtual env
# ================================
echo "🆕 Creating virtual environment..."
$PYTHON_CMD -m venv rag_venv

# Activate virtual environment
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
  source rag_venv/bin/activate
elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "win32"* ]]; then
  echo "⚠ On Windows PowerShell, activate manually:"
  echo ".\\rag_venv\\Scripts\\activate"
else
  echo "⚠ Unknown OS. Please activate virtual environment manually."
fi
echo "✔ Virtual environment activated"

# ================================
# 4️⃣ Upgrade pip
# ================================
echo "⬆ Upgrading pip..."
pip install --upgrade pip

# ================================
# 5️⃣ Remove conflicting dependencies
# ================================
echo "🧹 Cleaning potential conflicting installations..."
for pkg in torch faiss faiss-cpu sentence-transformers streamlit pymupdf pandas; do
  if pip show "$pkg" > /dev/null 2>&1; then
    pip uninstall -y "$pkg"
  fi
done

# ================================
# 6️⃣ Install dependencies
# ================================
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# ================================
# 7️⃣ Check Ollama installation
# ================================
echo "🔍 Checking Ollama..."
if ! command -v ollama &> /dev/null; then
  echo "⚠ Ollama is not installed."
  echo "   Visit https://ollama.ai for installation instructions."
else
  echo "✔ Ollama is installed."
fi

# ================================
# 8️⃣ Quick import validation
# ================================
echo "🔍 Verifying imports..."
$PYTHON_CMD - <<EOF
import torch, faiss, pandas, fitz, streamlit
from sentence_transformers import SentenceTransformer
print("🚀 All Python libraries imported successfully!")
print(f"Torch version: {torch.__version__}")
print(f"FAISS version: {faiss.__version__}")
EOF

echo -e "\n🎯 Environment setup complete!"
echo "👉 Start the app using:"
echo "   streamlit run streamlit_ui.py --server.port 8501"
