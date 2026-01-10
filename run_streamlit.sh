#!/bin/bash
# Run the Citizen Support Assistant - Streamlit Frontend
# 100% FREE - Uses Ollama (local LLM)

set -e

echo "🏛️ Citizen Support Assistant - Streamlit UI"
echo "============================================"
echo "100% FREE - No API keys required!"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Ollama is running
echo "🔍 Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running!"
    echo ""
    echo "Please start Ollama first:"
    echo "  1. Install Ollama: https://ollama.ai/download"
    echo "  2. Pull a model: ollama pull llama3.2"
    echo "  3. Start Ollama: ollama serve"
    echo ""
    exit 1
fi

# Check if llama3.2 model is available
if ! ollama list | grep -q "llama3.2"; then
    echo "⚠️  llama3.2 model not found. Pulling it now..."
    ollama pull llama3.2
fi

echo "✅ Ollama is running"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Create data directories
mkdir -p data/chroma data/tts_output data models/whisper

# Run Streamlit
echo ""
echo "🚀 Starting Streamlit..."
echo "   URL: http://localhost:8501"
echo ""
echo "   Using Ollama with local models (FREE)"
echo "   Press Ctrl+C to stop"
echo ""

cd streamlit_app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
