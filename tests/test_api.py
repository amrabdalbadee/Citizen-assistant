"""
Tests for the Citizen Support Assistant API.

These tests verify:
- API endpoint functionality
- Request/response validation
- Error handling
- Session management
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import json


# We'll patch the services to avoid loading heavy models during testing
@pytest.fixture
def mock_services():
    """Mock all heavy services for testing."""
    with patch("app.services.rag_service.rag_service") as mock_rag, \
         patch("app.services.stt_service.stt_service") as mock_stt, \
         patch("app.services.tts_service.tts_service") as mock_tts:
        
        # Configure RAG service mock
        mock_rag.is_healthy.return_value = True
        mock_rag.query = AsyncMock(return_value={
            "response": "To apply for a passport, you need to gather the required documents...",
            "sources": [
                {
                    "content": "Required Documents: Original birth certificate...",
                    "source": "passport_application.txt",
                    "relevance_score": 0.92
                }
            ],
            "confidence": 0.92
        })
        mock_rag.get_document_count.return_value = 10
        
        # Configure STT service mock
        mock_stt.is_healthy.return_value = True
        mock_stt.transcribe = AsyncMock(return_value={
            "text": "How do I apply for a passport?",
            "language": "en",
            "confidence": 0.95,
            "duration": 2.5
        })
        mock_stt.get_supported_formats.return_value = [
            "audio/wav", "audio/mp3", "audio/mpeg"
        ]
        
        # Configure TTS service mock
        mock_tts.is_healthy.return_value = True
        mock_tts.synthesize = AsyncMock(return_value={
            "audio_path": "/tmp/response.mp3",
            "audio_url": "/api/v1/audio/response.mp3",
            "duration_estimate": 5.0
        })
        
        yield {
            "rag": mock_rag,
            "stt": mock_stt,
            "tts": mock_tts
        }


@pytest.fixture
def client(mock_services):
    """Create test client with mocked services."""
    from app.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check_healthy(self, client, mock_services):
        """Test health check returns healthy status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert data["components"]["rag_service"] is True
        assert data["components"]["stt_service"] is True
        assert data["components"]["tts_service"] is True
    
    def test_health_check_degraded(self, client, mock_services):
        """Test health check returns degraded when a service is down."""
        mock_services["rag"].is_healthy.return_value = False
        
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"


class TestTextQueryEndpoint:
    """Tests for the text query endpoint."""
    
    def test_text_query_success(self, client, mock_services):
        """Test successful text query processing."""
        response = client.post(
            "/api/v1/query",
            json={"query": "How do I apply for a passport?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "response" in data
        assert "sources" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
        assert data["confidence"] > 0
    
    def test_text_query_with_session(self, client, mock_services):
        """Test text query with existing session."""
        # First query to create session
        response1 = client.post(
            "/api/v1/query",
            json={"query": "What documents do I need?"}
        )
        session_id = response1.json()["session_id"]
        
        # Follow-up query with session
        response2 = client.post(
            "/api/v1/query",
            json={
                "query": "How much does it cost?",
                "session_id": session_id
            }
        )
        
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
    
    def test_text_query_with_audio_response(self, client, mock_services):
        """Test text query requesting audio response."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "How do I renew my passport?",
                "include_audio_response": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["audio_url"] is not None
    
    def test_text_query_empty_query(self, client, mock_services):
        """Test text query with empty string."""
        response = client.post(
            "/api/v1/query",
            json={"query": ""}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_text_query_too_long(self, client, mock_services):
        """Test text query exceeding max length."""
        response = client.post(
            "/api/v1/query",
            json={"query": "x" * 3000}  # Max is 2000
        )
        
        assert response.status_code == 422  # Validation error


class TestAudioQueryEndpoint:
    """Tests for the audio query endpoint."""
    
    def test_audio_query_success(self, client, mock_services):
        """Test successful audio query processing."""
        # Create a simple WAV file content (minimal valid header)
        audio_content = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00\x00\x00\x04\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00"
        
        response = client.post(
            "/api/v1/query/audio",
            files={"audio": ("test.wav", audio_content, "audio/wav")},
            data={"include_audio_response": "true"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "transcription" in data
        assert data["transcription"] == "How do I apply for a passport?"
        assert "response" in data
    
    def test_audio_query_unsupported_format(self, client, mock_services):
        """Test audio query with unsupported format."""
        mock_services["stt"].get_supported_formats.return_value = ["audio/wav"]
        
        response = client.post(
            "/api/v1/query/audio",
            files={"audio": ("test.xyz", b"fake content", "audio/xyz")},
            data={}
        )
        
        assert response.status_code == 400
        assert "Unsupported audio format" in response.json()["detail"]


class TestSessionEndpoint:
    """Tests for session management endpoints."""
    
    def test_get_session(self, client, mock_services):
        """Test retrieving session information."""
        # Create session via query
        response1 = client.post(
            "/api/v1/query",
            json={"query": "Test query"}
        )
        session_id = response1.json()["session_id"]
        
        # Get session info
        response2 = client.get(f"/api/v1/session/{session_id}")
        
        assert response2.status_code == 200
        data = response2.json()
        assert data["session_id"] == session_id
        assert "conversation_history" in data
        assert len(data["conversation_history"]) > 0
    
    def test_get_nonexistent_session(self, client, mock_services):
        """Test retrieving non-existent session."""
        response = client.get("/api/v1/session/nonexistent-session-id")
        
        assert response.status_code == 404
    
    def test_delete_session(self, client, mock_services):
        """Test deleting a session."""
        # Create session
        response1 = client.post(
            "/api/v1/query",
            json={"query": "Test query"}
        )
        session_id = response1.json()["session_id"]
        
        # Delete session
        response2 = client.delete(f"/api/v1/session/{session_id}")
        assert response2.status_code == 200
        
        # Verify session is gone
        response3 = client.get(f"/api/v1/session/{session_id}")
        assert response3.status_code == 404


class TestIngestEndpoint:
    """Tests for the document ingestion endpoint."""
    
    def test_ingest_document(self, client, mock_services):
        """Test ingesting a new document."""
        mock_services["rag"].ingest_document = MagicMock(return_value=5)
        
        response = client.post(
            "/api/v1/ingest",
            json={
                "filename": "test_document.txt",
                "content": "This is test content for the knowledge base."
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test_document.txt"
        assert data["chunks_created"] == 5


class TestStatsEndpoint:
    """Tests for the statistics endpoint."""
    
    def test_get_stats(self, client, mock_services):
        """Test retrieving system statistics."""
        response = client.get("/api/v1/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "active_sessions" in data
        assert "documents_in_store" in data
        assert "llm_provider" in data
        assert "stt_model" in data


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root(self, client, mock_services):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
