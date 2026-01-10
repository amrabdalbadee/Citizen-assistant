"""
Configuration settings for the Citizen Support Assistant.
Uses Pydantic Settings for environment variable management.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Settings
    app_name: str = "Citizen Support Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API Settings
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]
    
    # LLM Settings
    llm_provider: Literal["groq", "openai", "ollama"] = "ollama"
    llm_model: str = "llama3.2"  # Default for Ollama (good for M1)
    llm_temperature: float = 0.1  # Low for factual responses
    llm_max_tokens: int = 1024
    
    # API Keys (optional - only needed if using cloud providers)
    groq_api_key: str = ""
    openai_api_key: str = ""
    
    # Ollama Settings (default - runs locally)
    ollama_base_url: str = "http://host.docker.internal:11434"  # Docker to host
    ollama_base_url_local: str = "http://localhost:11434"  # Direct local
    
    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence Transformers model
    embedding_dimension: int = 384
    
    # Vector Store Settings
    chroma_persist_directory: str = "./data/chroma"
    collection_name: str = "government_services"
    
    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 4  # Number of chunks to retrieve
    
    # Speech-to-Text Settings
    stt_model: str = "base"  # Whisper model size: tiny, base, small, medium, large
    stt_device: str = "cpu"  # or "cuda" for GPU
    stt_compute_type: str = "int8"  # Quantization for faster-whisper
    
    # Text-to-Speech Settings (Bonus)
    tts_enabled: bool = True
    tts_voice: str = "en-US-AriaNeural"  # Microsoft Edge TTS voice
    tts_rate: str = "+0%"  # Speech rate adjustment
    
    # Session Settings
    session_timeout_minutes: int = 30
    max_history_length: int = 10  # Max conversation turns to keep
    
    # Latency Optimization
    use_streaming: bool = True
    preload_models: bool = True
    
    # Knowledge Base
    knowledge_directory: str = "./knowledge"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
