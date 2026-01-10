"""
Speech-to-Text Service using Faster Whisper.

Faster Whisper uses CTranslate2 for optimized inference, providing:
- 4x faster transcription than original Whisper
- Lower memory usage through INT8 quantization
- Comparable accuracy to original models
"""

import logging
from pathlib import Path
from typing import Optional, BinaryIO
import tempfile
import os

from faster_whisper import WhisperModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class STTService:
    """
    Speech-to-Text service using Faster Whisper.
    
    Architecture Decision:
    - Using faster-whisper instead of OpenAI's whisper for 4x speed improvement
    - CTranslate2 backend enables INT8 quantization for reduced memory
    - Model is loaded once at startup and reused for all requests
    - Supports streaming audio processing for lower latency
    """
    
    _instance: Optional["STTService"] = None
    _model: Optional[WhisperModel] = None
    
    def __new__(cls):
        """Singleton pattern ensures model is loaded only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the STT service."""
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """
        Load the Whisper model with optimized settings.
        
        Model Sizes and Trade-offs:
        - tiny: Fastest, lowest accuracy (~39M parameters)
        - base: Good balance for simple queries (~74M parameters)
        - small: Better accuracy, still fast (~244M parameters)
        - medium: High accuracy, moderate speed (~769M parameters)
        - large: Best accuracy, slowest (~1.55B parameters)
        
        For citizen queries (typically clear speech), 'base' offers
        the best latency/accuracy trade-off.
        """
        logger.info(f"Loading Whisper model: {settings.stt_model}")
        
        try:
            self._model = WhisperModel(
                settings.stt_model,
                device=settings.stt_device,
                compute_type=settings.stt_compute_type,
                download_root="./models/whisper"
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    async def transcribe(
        self,
        audio_file: BinaryIO,
        language: str = "en",
        filename: str = "audio.wav"
    ) -> dict:
        """
        Transcribe audio to text.
        
        Args:
            audio_file: Binary audio file object
            language: Expected language code (e.g., 'en', 'ar')
            filename: Original filename for format detection
            
        Returns:
            dict with transcription results:
            - text: Full transcribed text
            - language: Detected/used language
            - confidence: Average confidence score
            - duration: Audio duration in seconds
            
        Latency Optimization:
        - Uses beam_size=1 for faster decoding (slight accuracy trade-off)
        - VAD filter removes silence for shorter processing
        - Temperature=0 for deterministic, faster results
        """
        if self._model is None:
            raise RuntimeError("STT model not initialized")
        
        # Save uploaded file to temp location for processing
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Transcribe with optimized settings
            segments, info = self._model.transcribe(
                tmp_path,
                language=language if language != "auto" else None,
                beam_size=1,  # Faster decoding
                best_of=1,
                temperature=0,  # Deterministic
                vad_filter=True,  # Filter out silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            )
            
            # Collect all segments
            text_parts = []
            total_confidence = 0
            segment_count = 0
            
            for segment in segments:
                text_parts.append(segment.text.strip())
                total_confidence += segment.avg_logprob
                segment_count += 1
            
            full_text = " ".join(text_parts)
            
            # Calculate average confidence (convert log prob to 0-1 scale)
            avg_confidence = 0.0
            if segment_count > 0:
                avg_log_prob = total_confidence / segment_count
                # Log prob is typically -0 to -1, convert to confidence
                avg_confidence = min(1.0, max(0.0, 1.0 + avg_log_prob))
            
            result = {
                "text": full_text,
                "language": info.language,
                "confidence": round(avg_confidence, 3),
                "duration": round(info.duration, 2)
            }
            
            logger.info(
                f"Transcription complete: {len(full_text)} chars, "
                f"language={info.language}, confidence={avg_confidence:.2f}"
            )
            
            return result
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    def get_supported_formats(self) -> list[str]:
        """Return list of supported audio formats."""
        return [
            "audio/wav",
            "audio/wave", 
            "audio/x-wav",
            "audio/mp3",
            "audio/mpeg",
            "audio/ogg",
            "audio/flac",
            "audio/m4a",
            "audio/mp4",
            "audio/webm"
        ]
    
    def is_healthy(self) -> bool:
        """Check if the STT service is operational."""
        return self._model is not None


# Global service instance
stt_service = STTService()
