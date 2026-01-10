#!/bin/bash
# Run the Citizen Support Assistant locally
# 100% FREE - Uses Ollama (local LLM)

set -e

echo "🏛️ Citizen Support Assistant - Local Development"
echo "================================================"
echo "100% FREE - No API keys required!"
echo ""

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

echo "✅ Ollama is running with llama3.2"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
fi

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
mkdir -p data/chroma data/tts_output models/whisper

# Run the application
echo ""
echo "🚀 Starting server..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "   Using Ollama with llama3.2 (FREE, LOCAL)"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
