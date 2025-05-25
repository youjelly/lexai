"""
Text-to-speech endpoints for synthesis and streaming
"""

import io
import time
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...tts import tts_service, multilingual_tts, voice_manager, TTSConfig
from ...utils.logging import get_logger
from ..middleware import validate_text_length, validate_language_code
from config import settings

logger = get_logger(__name__)

router = APIRouter()


class TTSSynthesizeRequest(BaseModel):
    text: str
    language: Optional[str] = None
    voice_id: Optional[str] = None
    speed: float = 1.0
    model: Optional[str] = None


class TTSStreamRequest(BaseModel):
    text: str
    language: Optional[str] = None
    voice_id: Optional[str] = None
    chunk_size_ms: int = 100


class TTSVoiceInfo(BaseModel):
    model: str
    name: str
    language: str
    description: str
    multi_speaker: bool
    speakers: Optional[List[str]] = None


@router.post("/synthesize")
async def synthesize_speech(request: TTSSynthesizeRequest):
    """Convert text to speech"""
    
    # Validate input
    await validate_text_length(request.text, max_length=5000)
    
    if request.language:
        await validate_language_code(request.language)
    
    try:
        start_time = time.time()
        
        # Set voice if specified
        if request.voice_id:
            voice_manager.set_active_voice(request.voice_id)
        
        # Synthesize using multilingual TTS
        audio = await multilingual_tts.synthesize(
            text=request.text,
            language=request.language,
            voice_id=request.voice_id
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert to WAV format
        import soundfile as sf
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, 22050, format='WAV')
        wav_buffer.seek(0)
        
        # Add headers with metadata
        headers = {
            "X-Processing-Time-Ms": str(int(processing_time)),
            "X-Audio-Duration-Ms": str(int(len(audio) / 22.05)),  # 22050 Hz sample rate
            "X-Language": request.language or "auto"
        }
        
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_tts(request: TTSStreamRequest):
    """Stream TTS synthesis for long text"""
    
    # Validate input
    await validate_text_length(request.text, max_length=50000)  # Allow longer text for streaming
    
    if request.language:
        await validate_language_code(request.language)
    
    try:
        # Set voice if specified
        if request.voice_id:
            voice_manager.set_active_voice(request.voice_id)
        
        # Create TTS config
        config = TTSConfig(
            language=request.language or "en",
            speaker_wav=None,
            stream=True
        )
        
        # Split text into sentences for better streaming
        import re
        sentences = re.split(r'(?<=[.!?])\s+', request.text)
        
        async def audio_generator():
            """Generate audio chunks"""
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Synthesize sentence
                audio = await tts_service.synthesize(sentence, config)
                
                # Convert to bytes
                import soundfile as sf
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, audio, 22050, format='WAV')
                wav_data = wav_buffer.getvalue()
                
                # Stream in chunks
                chunk_size = request.chunk_size_ms * 22050 // 1000 * 2  # bytes
                for i in range(0, len(wav_data), chunk_size):
                    yield wav_data[i:i + chunk_size]
        
        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={
                "X-Streaming": "true",
                "X-Language": request.language or "auto"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS streaming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices")
async def list_tts_voices(language: Optional[str] = None):
    """List available TTS voices by language"""
    
    if language:
        await validate_language_code(language)
    
    try:
        # Get available models
        available_models = tts_service.get_available_models()
        
        voices = []
        
        for model in available_models:
            # Filter by language if specified
            if language and language not in model.get("languages", []):
                continue
            
            # Get model info
            model_info = tts_service.get_model_info(model["name"])
            
            if model_info:
                voice_info = TTSVoiceInfo(
                    model=model["name"],
                    name=model.get("description", model["name"]),
                    language=language or "multilingual",
                    description=model.get("description", ""),
                    multi_speaker="speakers" in model_info,
                    speakers=model_info.get("speakers", [])[:10]  # Limit speakers list
                )
                voices.append(voice_info)
        
        # Also add custom voices
        custom_voices = voice_manager.list_voices(language=language)
        
        for voice in custom_voices:
            voice_info = TTSVoiceInfo(
                model="custom",
                name=voice.profile.name,
                language=voice.profile.language,
                description=voice.profile.description,
                multi_speaker=False,
                speakers=None
            )
            voices.append(voice_info)
        
        return {
            "voices": voices,
            "total": len(voices),
            "language": language
        }
        
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/languages/detect")
async def detect_text_language(text: str):
    """Detect language from text for TTS"""
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        language, confidence = multilingual_tts.detect_language(text)
        
        # Get language info
        supported_languages = multilingual_tts.get_supported_languages()
        language_info = next(
            (lang for lang in supported_languages if lang["code"] == language),
            None
        )
        
        return {
            "detected_language": language,
            "confidence": confidence,
            "language_name": language_info["name"] if language_info else language.upper(),
            "supported": language_info is not None,
            "recommended_model": multilingual_tts.get_best_model_for_language(language)
        }
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def optimize_tts_settings(
    language: str,
    text_sample: Optional[str] = None
):
    """Get optimized TTS settings for a language"""
    
    await validate_language_code(language)
    
    try:
        optimization = await multilingual_tts.optimize_for_language(
            language,
            text_sample
        )
        
        return optimization
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/info")
async def get_tts_model_info(model_name: str):
    """Get detailed information about a TTS model"""
    
    try:
        # Load model if not already loaded
        await tts_service.load_model(model_name)
        
        # Get model info
        info = tts_service.get_model_info(model_name)
        
        if not info:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))