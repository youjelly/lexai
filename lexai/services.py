"""
Singleton services for model management
Ensures only one instance of each model is loaded in memory
"""

from typing import Optional
import asyncio
from .models.ultravox_service import UltravoxService
from .tts.tts_service import TTSService
from .utils.logging import get_logger

logger = get_logger(__name__)


class ModelServices:
    """Singleton container for AI model services"""
    
    _instance = None
    _initialized = False
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ModelServices._initialized:
            self.ultravox_service: Optional[UltravoxService] = None
            self.tts_service: Optional[TTSService] = None
            ModelServices._initialized = True
    
    async def get_ultravox(self) -> UltravoxService:
        """Get or create Ultravox service instance"""
        async with self._lock:
            if self.ultravox_service is None:
                logger.info("Creating Ultravox service instance")
                # Force garbage collection before loading
                import torch
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                self.ultravox_service = UltravoxService()
                await self.ultravox_service.initialize()
            return self.ultravox_service
    
    def get_tts(self) -> TTSService:
        """Get or create TTS service instance"""
        if self.tts_service is None:
            logger.info("Creating TTS service instance")
            self.tts_service = TTSService()
        return self.tts_service


# Global singleton instance
model_services = ModelServices()