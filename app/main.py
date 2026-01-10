"""
Citizen Support Assistant - Main Application

An AI-powered voice-first service assistant for government services.
Supports both text and audio input, with context-aware responses
based on official government knowledge documents.

Architecture:
- FastAPI for high-performance async API
- LangChain + ChromaDB for RAG pipeline
- Faster Whisper for STT
- Edge TTS for voice synthesis
- In-memory session management (Redis-ready)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.core.config import get_settings
from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup:
    - Initialize and preload models for faster first request
    - Validate configuration
    - Create necessary directories
    
    Shutdown:
    - Cleanup temporary files
    - Close connections
    """
    logger.info("Starting Citizen Support Assistant...")
    
    # Create data directories
    Path("./data/chroma").mkdir(parents=True, exist_ok=True)
    Path("./data/tts_output").mkdir(parents=True, exist_ok=True)
    Path("./models/whisper").mkdir(parents=True, exist_ok=True)
    
    # Preload models if configured
    if settings.preload_models:
        logger.info("Preloading models...")
        try:
            # Import services to trigger model loading
            from app.services.rag_service import rag_service
            from app.services.stt_service import stt_service
            from app.services.tts_service import tts_service
            
            logger.info(f"RAG Service: {'healthy' if rag_service.is_healthy() else 'unhealthy'}")
            logger.info(f"STT Service: {'healthy' if stt_service.is_healthy() else 'unhealthy'}")
            logger.info(f"TTS Service: {'healthy' if tts_service.is_healthy() else 'unhealthy'}")
            
        except Exception as e:
            logger.error(f"Error preloading models: {e}")
            # Continue startup - services will initialize on first request
    
    logger.info(f"Application started - Version {settings.app_version}")
    logger.info(f"LLM Provider: {settings.llm_provider} / Model: {settings.llm_model}")
    logger.info(f"STT Model: {settings.stt_model}")
    logger.info(f"TTS Enabled: {settings.tts_enabled}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Citizen Support Assistant...")
    
    # Cleanup TTS files
    try:
        from app.services.tts_service import tts_service
        tts_service.cleanup_old_files(max_age_seconds=0)
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
    
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
    ## AI-Powered Citizen Support Assistant
    
    A voice-first service assistant that helps citizens navigate government services.
    
    ### Features
    - **Multi-Modal Input**: Accept both text and audio queries
    - **Speech-to-Text**: Automatic transcription of voice input using Whisper
    - **Intelligent Retrieval**: RAG-based answers from official knowledge documents
    - **Text-to-Speech**: Optional audio responses for accessibility
    - **Session Management**: Context-aware follow-up questions
    
    ### Usage
    1. Submit a question via `/api/v1/query` (text) or `/api/v1/query/audio` (voice)
    2. Use the returned `session_id` for follow-up questions
    3. Optionally request audio responses for voice output
    
    ### Architecture
    - LangChain + ChromaDB for RAG pipeline
    - Faster Whisper for optimized STT
    - Edge TTS for neural voice synthesis
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for TTS output
tts_output_dir = Path("./data/tts_output")
tts_output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/audio", StaticFiles(directory=str(tts_output_dir)), name="audio")

# Include API routes
app.include_router(api_router, prefix=settings.api_prefix, tags=["Assistant"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI-powered citizen support assistant for government services",
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health"
    }


# For running directly with Python
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1  # Single worker for prototype
    )
