"""
Text-to-Speech Service using Edge TTS.

Edge TTS provides access to Microsoft's neural TTS voices:
- High quality, natural-sounding speech
- Free to use (no API key required)
- Multiple languages and voices available
- Async API for non-blocking synthesis
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Optional
import asyncio
import uuid

import edge_tts

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TTSService:
    """
    Text-to-Speech service using Microsoft Edge TTS.
    
    Architecture Decision:
    - Edge TTS chosen for high-quality neural voices without API costs
    - Async synthesis prevents blocking during audio generation
    - Output stored temporarily and served via static file endpoint
    - Voice selection optimized for government service context (clear, professional)
    """
    
    # Available high-quality voices for different use cases
    VOICES = {
        "en-US": {
            "female": "en-US-AriaNeural",      # Natural, conversational
            "male": "en-US-GuyNeural",          # Professional, clear
            "news": "en-US-JennyNeural"         # Newsreader style
        },
        "en-GB": {
            "female": "en-GB-SoniaNeural",
            "male": "en-GB-RyanNeural"
        },
        "ar-SA": {
            "female": "ar-SA-ZariyahNeural",
            "male": "ar-SA-HamedNeural"
        },
        "es-ES": {
            "female": "es-ES-ElviraNeural",
            "male": "es-ES-AlvaroNeural"
        }
    }
    
    def __init__(self):
        """Initialize TTS service."""
        self.output_dir = Path("./data/tts_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_voice = settings.tts_voice
        self.rate = settings.tts_rate
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None
    ) -> dict:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier (e.g., 'en-US-AriaNeural')
            rate: Speech rate adjustment (e.g., '+10%', '-10%')
            
        Returns:
            dict with:
            - audio_path: Path to generated audio file
            - audio_url: URL endpoint for accessing the audio
            - duration_estimate: Estimated duration in seconds
            
        Latency Optimization:
        - Async synthesis allows concurrent processing
        - Output files cleaned up after configurable timeout
        - Streaming support possible for longer responses
        """
        if not settings.tts_enabled:
            return {"audio_path": None, "audio_url": None, "duration_estimate": 0}
        
        voice = voice or self.default_voice
        rate = rate or self.rate
        
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        output_path = self.output_dir / f"response_{file_id}.mp3"
        
        try:
            # Create TTS communication instance
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate
            )
            
            # Save to file
            await communicate.save(str(output_path))
            
            # Estimate duration (rough: ~150 words per minute)
            word_count = len(text.split())
            duration_estimate = word_count / 150 * 60
            
            logger.info(
                f"TTS synthesis complete: {len(text)} chars -> {output_path.name}"
            )
            
            return {
                "audio_path": str(output_path),
                "audio_url": f"/api/v1/audio/{output_path.name}",
                "duration_estimate": round(duration_estimate, 1)
            }
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return {"audio_path": None, "audio_url": None, "duration_estimate": 0}
    
    async def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """
        List available voices, optionally filtered by language.
        
        Args:
            language: Language code to filter by (e.g., 'en')
            
        Returns:
            List of voice dictionaries with name, language, gender
        """
        voices = await edge_tts.list_voices()
        
        if language:
            voices = [v for v in voices if v["Locale"].startswith(language)]
        
        return [
            {
                "name": v["ShortName"],
                "language": v["Locale"],
                "gender": v["Gender"]
            }
            for v in voices
        ]
    
    def cleanup_old_files(self, max_age_seconds: int = 3600):
        """Remove audio files older than max_age_seconds."""
        import time
        
        current_time = time.time()
        for file_path in self.output_dir.glob("response_*.mp3"):
            if current_time - file_path.stat().st_mtime > max_age_seconds:
                file_path.unlink()
                logger.debug(f"Cleaned up old TTS file: {file_path.name}")
    
    def is_healthy(self) -> bool:
        """Check if TTS service is operational."""
        return settings.tts_enabled and self.output_dir.exists()


# Global service instance
tts_service = TTSService()
