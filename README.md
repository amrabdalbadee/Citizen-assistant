# 🏛️ Citizen Support Assistant

An AI-powered voice-first service assistant for government services. This prototype demonstrates how citizens can interact with government information through both text and voice, receiving accurate, context-aware guidance.

## ✨ 100% FREE - No API Keys Required!

This project runs entirely on your local machine using:
- **Ollama** - Local LLM (Llama 3.2)
- **Faster-Whisper** - Local speech-to-text
- **Edge TTS** - Free Microsoft neural voices
- **ChromaDB** - Local vector database
- **Sentence Transformers** - Local embeddings

## 🎯 Features

- **Multi-Modal Input**: Accept both text and audio (voice) queries
- **Speech-to-Text**: Automatic transcription using Whisper (optimized with faster-whisper)
- **Intelligent Retrieval**: RAG-based answers strictly from official knowledge documents
- **Text-to-Speech**: Neural voice synthesis for audio responses (accessibility)
- **Session Management**: Context-aware follow-up questions within a session
- **Hallucination Prevention**: Strict grounding in source documents with confidence scoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CITIZEN SUPPORT ASSISTANT                          │
│                            (100% Local & Free)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐     ┌──────────────────────────────────────────────────────┐  │
│  │  Client  │────▶│                   FastAPI Server                      │  │
│  │(Web/App) │     │                                                       │  │
│  └──────────┘     │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│                   │  │ /query      │  │ /query/audio│  │ /session    │   │  │
│                   │  │ (Text)      │  │ (Voice)     │  │ (History)   │   │  │
│                   │  └──────┬──────┘  └──────┬──────┘  └─────────────┘   │  │
│                   └─────────┼────────────────┼───────────────────────────┘  │
│                             │                │                              │
│                             │         ┌──────▼──────┐                       │
│                             │         │ STT Service │  ◄── FREE (Local)     │
│                             │         │ (Whisper)   │                       │
│                             │         └──────┬──────┘                       │
│                             │                │                              │
│                             ▼                ▼                              │
│                   ┌─────────────────────────────────────┐                   │
│                   │          RAG Pipeline               │                   │
│                   │  ┌─────────────┐  ┌─────────────┐  │                   │
│                   │  │  Embeddings │  │   ChromaDB  │  │  ◄── FREE (Local) │
│                   │  │(MiniLM-L6)  │◀▶│(Vector Store)│  │                   │
│                   │  └─────────────┘  └─────────────┘  │                   │
│                   │         │                          │                   │
│                   │  ┌──────▼──────┐  ┌─────────────┐  │                   │
│                   │  │  Retriever  │──▶│   Ollama    │  │  ◄── FREE (Local) │
│                   │  │   (k=4)     │  │ (Llama 3.2) │  │                   │
│                   │  └─────────────┘  └──────┬──────┘  │                   │
│                   └──────────────────────────┼─────────┘                   │
│                                              │                              │
│                                       ┌──────▼──────┐                       │
│                                       │ TTS Service │  ◄── FREE (Edge TTS)  │
│                                       │ (Edge TTS)  │                       │
│                                       └─────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start (M1 Mac)

### Prerequisites

- **Ollama** installed and running (see Step 1)
- **Python 3.11+** or **Docker**

### Step 1: Install Ollama

```bash
# Install Ollama (if not already installed)
# Download from: https://ollama.ai/download

# Pull the Llama 3.2 model (recommended for M1)
ollama pull llama3.2

# Verify Ollama is running
ollama list
```

### Step 2: Clone and Configure

```bash
# Clone the repository
cd citizen-assistant

# Copy environment template (no API keys needed!)
cp .env.example .env
```

### Step 3: Run the Application

#### Option A: Streamlit UI (Recommended) 🎨

The Streamlit frontend provides a beautiful chat interface with:
- 💬 Conversation history stored in SQLite database
- 🎤 Voice input (upload audio files)
- 🔊 Text-to-speech responses
- 🤖 LLM model selector
- 📚 Source citations

```bash
# Make script executable
chmod +x run_streamlit.sh

# Run the Streamlit app
./run_streamlit.sh

# Or manually:
source venv/bin/activate
pip install -r requirements.txt
cd streamlit_app
streamlit run app.py
```

**Open:** http://localhost:8501

#### Option B: FastAPI Backend (For API access)

```bash
# Run the backend API
./run.sh

# Or manually:
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Open:** http://localhost:8000/docs

#### Option C: Run with Docker

```bash
# Streamlit UI only (recommended)
docker compose up streamlit --build

# Or both UI and API
docker compose --profile api up --build
```

### Step 4: Test the Application

**Streamlit UI:** Open http://localhost:8501 and start chatting!

**API (if running):**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I apply for a passport?"}'
```

---

## 🖥️ Streamlit UI Features

### Chat Interface
- Clean, modern chat UI with message bubbles
- Real-time streaming responses
- Confidence indicators for each response
- Expandable source citations

### Conversation Management
- 📁 Persistent storage in SQLite database
- 🔍 Search through past conversations
- 📝 Auto-generated conversation titles
- 🗑️ Delete conversations

### Multimodal Support
- 📝 Text input
- 🎤 Audio file upload (WAV, MP3, OGG, M4A)
- 🔊 Text-to-speech responses (multiple voices)

### Model Selection
- Switch between Ollama models on-the-fly
- Shows installed vs. available models
- Recommended models for M1 Mac:
  - `llama3.2` - Fast, good quality
  - `mistral` - Excellent quality
  - `llama3.1:8b` - Best quality

## 📁 Project Structure

```
citizen-assistant/
├── app/                        # FastAPI Backend
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── core/
│   │   └── config.py          # Configuration management
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── services/
│   │   ├── rag_service.py     # RAG pipeline (LangChain + ChromaDB)
│   │   ├── stt_service.py     # Speech-to-Text (Whisper)
│   │   ├── tts_service.py     # Text-to-Speech (Edge TTS)
│   │   └── session_service.py # Session management
│   └── main.py                # FastAPI application
│
├── streamlit_app/              # Streamlit Frontend
│   ├── app.py                 # Main Streamlit application
│   ├── database/
│   │   └── models.py          # SQLite database for conversations
│   ├── services/
│   │   └── chat_service.py    # Chat service with RAG
│   └── components/            # Reusable UI components
│
├── knowledge/                  # Knowledge base documents
│   ├── passport_application.txt
│   └── birth_certificate.txt
│
├── tests/                      # Test suite
├── docs/                       # Documentation
│
├── Dockerfile                  # Backend container
├── Dockerfile.streamlit        # Frontend container
├── docker-compose.yml          # Docker orchestration
├── requirements.txt            # Python dependencies
├── run.sh                      # Run backend script
├── run_streamlit.sh            # Run frontend script
├── Makefile                    # Convenience commands
├── .env.example               # Environment template
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | LLM provider: `groq`, `openai`, or `ollama` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model identifier |
| `GROQ_API_KEY` | - | Groq API key (required if using Groq) |
| `STT_MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `STT_DEVICE` | `cpu` | Compute device: `cpu` or `cuda` |
| `TTS_ENABLED` | `true` | Enable text-to-speech responses |
| `CHUNK_SIZE` | `500` | Document chunk size for embedding |
| `RETRIEVAL_K` | `4` | Number of chunks to retrieve |

### Adding Knowledge Documents

Place `.txt` files in the `knowledge/` directory. The system will automatically ingest them on startup.

Or use the API:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{"filename": "new_service.txt", "content": "Document content here..."}'
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/query` | Process text query |
| POST | `/api/v1/query/audio` | Process audio query |
| GET | `/api/v1/session/{id}` | Get session information |
| DELETE | `/api/v1/session/{id}` | Delete session |
| POST | `/api/v1/ingest` | Ingest new knowledge document |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/stats` | System statistics |

Full API documentation available at `/docs` (Swagger UI) or `/redoc`.

## 🎯 Design Decisions

### Why Ollama (Default LLM)?

- **Cost**: 100% FREE - runs locally on your machine
- **Privacy**: Data never leaves your computer
- **M1 Optimized**: Excellent performance on Apple Silicon
- **Easy Setup**: Just `ollama pull llama3.2`
- **Offline**: Works without internet connection
- **Alternative**: Can easily switch to Groq (free cloud) or OpenAI

### Recommended Models for M1 Mac

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `llama3.2` | 2B | ★★★★★ | ★★★☆☆ | Fast responses, simple Q&A |
| `llama3.2:3b` | 3B | ★★★★☆ | ★★★★☆ | Better quality, still fast |
| `mistral` | 7B | ★★★☆☆ | ★★★★★ | Excellent quality |
| `llama3.1:8b` | 8B | ★★☆☆☆ | ★★★★★ | Best quality, slower |

### Why Faster-Whisper (STT)?

- **Performance**: 4x faster than original Whisper via CTranslate2
- **Memory**: INT8 quantization reduces memory by ~50%
- **Accuracy**: Comparable to original Whisper models
- **Free**: No API costs, runs completely locally
- **M1 Compatible**: Works great on Apple Silicon

### Why ChromaDB (Vector Store)?

- **Simplicity**: Embedded database, no external setup
- **Performance**: Fast similarity search with HNSW
- **Persistence**: Data survives restarts
- **Free**: Open source, no costs
- **Migration Path**: Easy to swap for Pinecone/Weaviate in production

### Why Edge TTS?

- **Quality**: Microsoft's neural TTS voices
- **Cost**: 100% FREE, no API key required
- **Variety**: Multiple languages and voices
- **Async**: Non-blocking synthesis

## ⚡ Latency Optimization

1. **STT Optimization**:
   - `beam_size=1` for faster decoding
   - VAD filter to skip silence
   - INT8 quantization

2. **LLM Optimization**:
   - Groq's LPU for sub-second inference
   - Limited conversation history
   - Streaming support (configurable)

3. **Retrieval Optimization**:
   - Pre-computed embeddings
   - HNSW index for approximate NN search
   - Optimal chunk size (500 chars)

4. **Architecture**:
   - Async throughout (FastAPI + async services)
   - Singleton pattern for model reuse
   - Connection pooling

## 📈 Scalability Considerations

For handling 1,000+ concurrent requests:

### Current Architecture (Prototype)
- Single instance, in-memory sessions
- Embedded ChromaDB
- Suitable for ~50-100 concurrent users

### Production Scaling Strategy

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │    (nginx)      │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
    │ API #1  │        │ API #2  │        │ API #3  │
    │(FastAPI)│        │(FastAPI)│        │(FastAPI)│
    └────┬────┘        └────┬────┘        └────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
         │  Redis  │   │Pinecone │   │  LLM    │
         │(Sessions)│  │(Vectors)│   │ (Groq)  │
         └─────────┘   └─────────┘   └─────────┘
```

**Key Changes for Scale:**
1. **Sessions**: Replace in-memory with Redis Cluster
2. **Vector Store**: Migrate to Pinecone or Weaviate (managed)
3. **API**: Deploy multiple instances behind load balancer
4. **STT**: Use cloud APIs (Deepgram, AssemblyAI) for parallel processing
5. **Caching**: Add response caching for common queries

## 🔒 Security & Accuracy

### Hallucination Prevention
- System prompt explicitly restricts answers to source documents
- Confidence scoring based on retrieval relevance
- "I don't have information" fallback for low-confidence queries

### Data Privacy (Government Context)
- No data sent to external services except LLM inference
- Session data automatically expires
- Audio files cleaned up after processing
- No PII logging

### Input Validation
- Pydantic models for all inputs
- File type validation for audio
- Query length limits

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Text query with session
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What documents do I need for a new passport?"}'

# Follow-up question (use session_id from previous response)
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How much does it cost?", "session_id": "<SESSION_ID>"}'

# Get session history
curl http://localhost:8000/api/v1/session/<SESSION_ID>
```

## 🛠️ Development

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=your_key_here

# Run the application
uvicorn app.main:app --reload --port 8000
```

### Code Quality

```bash
# Format code
black app/ tests/
isort app/ tests/

# Type checking
mypy app/

# Linting
ruff check app/
```

## 📋 Handling Edge Cases

### Ambiguous Queries
The system handles vague queries by:
1. Retrieving most relevant documents
2. Providing available information with lower confidence
3. Suggesting clarification when needed

### Unrelated Queries
For questions outside the knowledge base:
- Returns: "I don't have information about that in my knowledge base"
- Suggests contacting the relevant government office
- Does NOT hallucinate or make up information

### Poor Audio Quality
- Confidence score reflects transcription quality
- Returns error message if transcription fails
- Suggests re-recording with clearer speech

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

---

**Built for Government Service Excellence** 🏛️
