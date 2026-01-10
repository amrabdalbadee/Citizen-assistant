"""
API Routes for the Citizen Support Assistant.

Endpoints:
- POST /query - Text-based query
- POST /query/audio - Audio-based query (voice input)
- GET /session/{session_id} - Get session information
- POST /ingest - Ingest new knowledge document
- GET /health - Health check
- GET /audio/{filename} - Serve TTS audio files
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    AudioQueryRequest,
    SessionInfo,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    ErrorResponse,
    SourceDocument
)
from app.services.rag_service import rag_service
from app.services.stt_service import stt_service
from app.services.tts_service import tts_service
from app.services.session_service import session_manager
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Process a text query",
    description="Submit a text question about government services and receive an AI-generated response.",
    responses={
        200: {"description": "Successful response"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def process_text_query(request: QueryRequest) -> QueryResponse:
    """
    Process a text-based query.
    
    The query is processed through the RAG pipeline:
    1. Retrieve relevant documents from the knowledge base
    2. Generate response using LLM with retrieved context
    3. Optionally synthesize audio response
    4. Store interaction in session history
    """
    start_time = time.time()
    
    try:
        # Get or create session
        session_id = request.session_id
        if not session_id or session_manager.get_session(session_id) is None:
            session_id = session_manager.create_session()
        
        # Get conversation history for context
        history = session_manager.get_history(session_id)
        
        # Process query through RAG
        result = await rag_service.query(
            question=request.query,
            conversation_history=history
        )
        
        # Store conversation turn
        session_manager.add_turn(session_id, "user", request.query)
        session_manager.add_turn(session_id, "assistant", result["response"])
        
        # Generate audio if requested
        audio_url = None
        if request.include_audio_response:
            tts_result = await tts_service.synthesize(result["response"])
            audio_url = tts_result.get("audio_url")
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            session_id=session_id,
            query=request.query,
            response=result["response"],
            sources=[
                SourceDocument(**source) for source in result["sources"]
            ],
            audio_url=audio_url,
            transcription=None,
            confidence=result["confidence"],
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/query/audio",
    response_model=QueryResponse,
    summary="Process an audio query",
    description="Submit an audio file containing a spoken question and receive an AI-generated response.",
    responses={
        200: {"description": "Successful response"},
        400: {"model": ErrorResponse, "description": "Invalid audio format"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def process_audio_query(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A)"),
    session_id: Optional[str] = Form(default=None, description="Session ID for conversation continuity"),
    include_audio_response: bool = Form(default=True, description="Include synthesized audio in response"),
    language: str = Form(default="en", description="Expected language for transcription")
) -> QueryResponse:
    """
    Process an audio-based query (voice input).
    
    Pipeline:
    1. Validate audio format
    2. Transcribe audio to text using Whisper
    3. Process transcribed query through RAG
    4. Generate audio response (TTS)
    5. Store interaction in session history
    
    Latency Optimization:
    - Whisper uses VAD to skip silence
    - Parallel processing of TTS while returning response
    """
    start_time = time.time()
    
    try:
        # Validate content type
        content_type = audio.content_type or "audio/wav"
        if content_type not in stt_service.get_supported_formats():
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {content_type}. Supported: {stt_service.get_supported_formats()}"
            )
        
        # Get or create session
        if not session_id or session_manager.get_session(session_id) is None:
            session_id = session_manager.create_session()
        
        # Transcribe audio
        transcription_result = await stt_service.transcribe(
            audio_file=audio.file,
            language=language,
            filename=audio.filename or "audio.wav"
        )
        
        transcribed_text = transcription_result["text"]
        
        if not transcribed_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio. Please ensure clear speech and try again."
            )
        
        # Get conversation history
        history = session_manager.get_history(session_id)
        
        # Process query through RAG
        result = await rag_service.query(
            question=transcribed_text,
            conversation_history=history
        )
        
        # Store conversation turn
        session_manager.add_turn(session_id, "user", transcribed_text)
        session_manager.add_turn(session_id, "assistant", result["response"])
        
        # Generate audio response
        audio_url = None
        if include_audio_response:
            tts_result = await tts_service.synthesize(result["response"])
            audio_url = tts_result.get("audio_url")
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            session_id=session_id,
            query=transcribed_text,
            response=result["response"],
            sources=[
                SourceDocument(**source) for source in result["sources"]
            ],
            audio_url=audio_url,
            transcription=transcribed_text,
            confidence=result["confidence"],
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/session/{session_id}",
    response_model=SessionInfo,
    summary="Get session information",
    description="Retrieve session details including conversation history.",
    responses={
        200: {"description": "Session information"},
        404: {"model": ErrorResponse, "description": "Session not found"}
    }
)
async def get_session(session_id: str) -> SessionInfo:
    """Get session information and conversation history."""
    session_info = session_manager.get_session_info(session_id)
    
    if session_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found or expired: {session_id}"
        )
    
    return session_info


@router.delete(
    "/session/{session_id}",
    summary="Delete a session",
    description="End a session and clear its conversation history.",
    responses={
        200: {"description": "Session deleted"},
        404: {"model": ErrorResponse, "description": "Session not found"}
    }
)
async def delete_session(session_id: str) -> dict:
    """Delete a session and its history."""
    if not session_manager.delete_session(session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    return {"message": "Session deleted successfully", "session_id": session_id}


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a knowledge document",
    description="Add a new document to the knowledge base for retrieval.",
    responses={
        200: {"description": "Document ingested successfully"},
        400: {"model": ErrorResponse, "description": "Invalid document"}
    }
)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Ingest a new knowledge document into the vector store.
    
    The document is:
    1. Split into chunks
    2. Embedded using the sentence transformer
    3. Stored in ChromaDB for retrieval
    """
    try:
        chunks_created = rag_service.ingest_document(
            content=request.content,
            filename=request.filename
        )
        
        return IngestResponse(
            success=True,
            filename=request.filename,
            chunks_created=chunks_created,
            message=f"Document '{request.filename}' ingested successfully with {chunks_created} chunks"
        )
        
    except Exception as e:
        logger.error(f"Error ingesting document: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/audio/{filename}",
    summary="Get audio file",
    description="Retrieve a synthesized audio response file.",
    responses={
        200: {"description": "Audio file"},
        404: {"description": "Audio file not found"}
    }
)
async def get_audio_file(filename: str) -> FileResponse:
    """Serve TTS audio files."""
    file_path = Path("./data/tts_output") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=filename
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of all system components."
)
async def health_check() -> HealthResponse:
    """
    Check health of all components:
    - RAG Service (embeddings, vector store, LLM)
    - STT Service (Whisper model)
    - TTS Service (Edge TTS)
    - Session Manager
    """
    components = {
        "rag_service": rag_service.is_healthy(),
        "stt_service": stt_service.is_healthy(),
        "tts_service": tts_service.is_healthy(),
        "session_manager": True  # In-memory, always healthy
    }
    
    all_healthy = all(components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=settings.app_version,
        components=components
    )


@router.get(
    "/stats",
    summary="Get system statistics",
    description="Get statistics about the system state."
)
async def get_stats() -> dict:
    """Get system statistics."""
    return {
        "active_sessions": session_manager.get_active_session_count(),
        "documents_in_store": rag_service.get_document_count(),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "stt_model": settings.stt_model
    }
