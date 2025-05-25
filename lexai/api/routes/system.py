"""
System endpoints for health, status, and language detection
"""

import os
import psutil
import torch
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...models import model_manager
from ...tts import multilingual_tts, LANGUAGE_CODES
from ...websocket import session_manager, connection_manager
from ...utils.logging import get_logger
from config import settings

logger = get_logger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str = "0.1.0"


class SystemStatus(BaseModel):
    models_loaded: Dict[str, bool]
    storage_usage: Dict[str, Dict[str, float]]
    active_sessions: int
    active_connections: int
    gpu_available: bool
    gpu_memory: Optional[Dict[str, float]]


class LanguageDetectionRequest(BaseModel):
    text: Optional[str] = None
    audio_data: Optional[str] = None  # Base64 encoded audio


class LanguageDetectionResponse(BaseModel):
    detected_language: str
    confidence: float
    supported: bool


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import time
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time()
    )


@router.get("/status", response_model=SystemStatus)
async def system_status():
    """Get system status including models and storage"""
    
    # Check loaded models
    loaded_models = model_manager.get_loaded_models()
    models_status = {
        "ultravox": any(m["name"] == "ultravox" for m in loaded_models),
        "tts": len(loaded_models) > 0
    }
    
    # Get storage usage
    storage_usage = {}
    
    # Persistent storage
    persistent_stat = os.statvfs("/mnt/storage")
    persistent_total = persistent_stat.f_blocks * persistent_stat.f_frsize
    persistent_free = persistent_stat.f_available * persistent_stat.f_frsize
    persistent_used = persistent_total - persistent_free
    
    storage_usage["persistent"] = {
        "total_gb": persistent_total / (1024**3),
        "used_gb": persistent_used / (1024**3),
        "free_gb": persistent_free / (1024**3),
        "usage_percent": (persistent_used / persistent_total) * 100
    }
    
    # Ephemeral storage
    try:
        ephemeral_stat = os.statvfs("/opt/dlami/nvme")
        ephemeral_total = ephemeral_stat.f_blocks * ephemeral_stat.f_frsize
        ephemeral_free = ephemeral_stat.f_available * ephemeral_stat.f_frsize
        ephemeral_used = ephemeral_total - ephemeral_free
        
        storage_usage["ephemeral"] = {
            "total_gb": ephemeral_total / (1024**3),
            "used_gb": ephemeral_used / (1024**3),
            "free_gb": ephemeral_free / (1024**3),
            "usage_percent": (ephemeral_used / ephemeral_total) * 100
        }
    except:
        storage_usage["ephemeral"] = {
            "total_gb": 0,
            "used_gb": 0,
            "free_gb": 0,
            "usage_percent": 0
        }
    
    # Get active sessions
    active_sessions = len(await session_manager.get_active_sessions())
    active_connections = len(connection_manager.connections)
    
    # GPU status
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    
    if gpu_available:
        gpu_memory = {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    
    return SystemStatus(
        models_loaded=models_status,
        storage_usage=storage_usage,
        active_sessions=active_sessions,
        active_connections=active_connections,
        gpu_available=gpu_available,
        gpu_memory=gpu_memory
    )


@router.get("/languages")
async def list_languages():
    """List all supported languages"""
    
    supported_languages = multilingual_tts.get_supported_languages()
    
    # Add language codes mapping
    languages_with_codes = []
    for lang in supported_languages:
        lang_info = lang.copy()
        # Add alternative codes
        lang_info["alternative_codes"] = [
            code for code, mapped in LANGUAGE_CODES.items()
            if mapped == lang["code"]
        ]
        languages_with_codes.append(lang_info)
    
    return {
        "languages": languages_with_codes,
        "total": len(languages_with_codes),
        "default": settings.DEFAULT_LANGUAGE
    }


@router.post("/language/detect", response_model=LanguageDetectionResponse)
async def detect_language(request: LanguageDetectionRequest):
    """Detect language from text or audio"""
    
    if not request.text and not request.audio_data:
        raise HTTPException(
            status_code=400,
            detail="Either text or audio_data must be provided"
        )
    
    if request.text:
        # Detect from text
        language, confidence = multilingual_tts.detect_language(request.text)
        
        # Check if supported
        supported_langs = [lang["code"] for lang in multilingual_tts.get_supported_languages()]
        supported = language in supported_langs
        
        return LanguageDetectionResponse(
            detected_language=language,
            confidence=confidence,
            supported=supported
        )
    
    else:
        # Audio detection would require decoding and processing
        # For MVP, return not implemented
        raise HTTPException(
            status_code=501,
            detail="Audio language detection not implemented in MVP"
        )


@router.get("/info")
async def api_info():
    """Get API information"""
    
    return {
        "name": "LexAI API",
        "version": "0.1.0",
        "description": "Real-time multimodal AI voice assistant",
        "capabilities": {
            "real_time_streaming": True,
            "voice_cloning": True,
            "multilingual": True,
            "languages": len(multilingual_tts.get_supported_languages()),
            "max_audio_length_seconds": settings.MAX_AUDIO_LENGTH,
            "sample_rate": settings.SAMPLE_RATE
        },
        "endpoints": {
            "websocket": f"ws://{settings.EXTERNAL_HOST or '3.129.5.177'}:{settings.PORT}/ws/audio/{{session_id}}",
            "api_docs": "/api/docs",
            "health": "/api/health"
        }
    }


@router.get("/models")
async def list_models():
    """List available and loaded models"""
    
    loaded_models = model_manager.get_loaded_models()
    
    # Get available TTS models
    from ...tts import tts_service
    available_tts = tts_service.get_available_models()
    
    return {
        "loaded": loaded_models,
        "available": {
            "ultravox": ["fixie-ai/ultravox-v0_5-llama-3_1-8b"],
            "tts": available_tts
        }
    }