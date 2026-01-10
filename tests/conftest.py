"""
Pytest configuration and fixtures.
"""

import pytest
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing."""
    os.environ["DEBUG"] = "true"
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["GROQ_API_KEY"] = "test_key"
    os.environ["PRELOAD_MODELS"] = "false"
    os.environ["TTS_ENABLED"] = "false"
    
    from app.core.config import Settings
    return Settings()


@pytest.fixture
def sample_audio_wav():
    """Generate a minimal valid WAV file for testing."""
    # Minimal WAV header + empty audio data
    return (
        b"RIFF"
        b"\x24\x00\x00\x00"  # File size
        b"WAVE"
        b"fmt "
        b"\x10\x00\x00\x00"  # Chunk size
        b"\x01\x00"  # Audio format (PCM)
        b"\x01\x00"  # Num channels
        b"\x00\x04\x00\x00"  # Sample rate
        b"\x00\x04\x00\x00"  # Byte rate
        b"\x01\x00"  # Block align
        b"\x08\x00"  # Bits per sample
        b"data"
        b"\x00\x00\x00\x00"  # Data size
    )


@pytest.fixture
def sample_knowledge_document():
    """Sample knowledge document for testing."""
    return """
    # Test Government Service
    
    ## Requirements
    - Document A
    - Document B
    - Photo ID
    
    ## Fees
    - Standard: $50
    - Expedited: $100
    
    ## Processing Time
    5-7 business days for standard, 2-3 days for expedited.
    """
