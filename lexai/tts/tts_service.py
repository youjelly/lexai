"""
Main TTS service with Coqui TTS integration
Provides multilingual synthesis with streaming and voice cloning support
"""

import os
import json
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Union, AsyncIterator
import asyncio
from pathlib import Path
import tempfile
import threading
from dataclasses import dataclass
import soundfile as sf
import time
from contextlib import asynccontextmanager

from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.audio import AudioProcessor

from ..utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass
class TTSConfig:
    """Configuration for TTS synthesis"""
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    language: str = "en"
    speaker: Optional[str] = None
    speaker_wav: Optional[Union[str, List[str]]] = None
    speed: float = 1.0
    emotion: Optional[str] = None
    stream: bool = True
    use_gpu: bool = True
    sample_rate: int = 22050  # Most TTS models use 22.05kHz


class TTSService:
    """Main TTS service for text-to-speech synthesis"""
    
    def __init__(self):
        self.models_path = Path(settings.TTS_MODEL_PATH)
        self.cache_path = Path(settings.TTS_CACHE_PATH)
        self.processing_path = Path(settings.AUDIO_PROCESSING_PATH)
        
        # Model registry and loaded models
        self.model_registry = self._load_model_registry()
        self.loaded_models: Dict[str, TTS] = {}
        self.model_lock = threading.Lock()
        
        # Default model - Use VITS for English since it's faster and doesn't need voice samples
        self.default_model = "tts_models/en/vctk/vits"
        
        # Initialize paths
        self._init_paths()
    
    def _init_paths(self):
        """Initialize required directories"""
        for path in [self.models_path, self.cache_path, self.processing_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load the model registry created during download"""
        registry_path = self.models_path / "model_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning("Model registry not found. Run download_models.py first.")
            return {"models": {}}
    
    def _is_known_model(self, model_name: str) -> bool:
        """Check if model is a known Coqui TTS model"""
        known_patterns = [
            "tts_models/en/vctk/vits",
            "tts_models/en/jenny/jenny",
            "tts_models/multilingual/",
            "tts_models/"
        ]
        return any(model_name.startswith(pattern) for pattern in known_patterns)
    
    async def initialize(self, model_name: Optional[str] = None):
        """Initialize TTS service with a specific model"""
        model_name = model_name or self.default_model
        
        if model_name not in self.loaded_models:
            await self.load_model(model_name)
    
    async def load_model(self, model_name: str) -> TTS:
        """Load a TTS model"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Check if model is in registry - skip check for known models
        if model_name not in self.model_registry.get("models", {}) and not self._is_known_model(model_name):
            raise ValueError(f"Model {model_name} not found in registry. Available models: {list(self.model_registry.get('models', {}).keys())}")
        
        logger.debug(f"Loading TTS model: {model_name}")
        
        # Load model in thread pool
        model = await asyncio.get_event_loop().run_in_executor(
            None,
            self._load_model_sync,
            model_name
        )
        
        return model
    
    def _load_model_sync(self, model_name: str) -> TTS:
        """Synchronously load a TTS model"""
        with self.model_lock:
            # Double-check after acquiring lock
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]
            
            # Unload other models to free memory (only keep one TTS model)
            if len(self.loaded_models) > 0:
                for loaded_model_name in list(self.loaded_models.keys()):
                    if loaded_model_name != model_name:
                        logger.info(f"Unloading model {loaded_model_name} to free memory")
                        del self.loaded_models[loaded_model_name]
                        # Force garbage collection
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
            
            # Set TTS home directory
            os.environ['TTS_HOME'] = str(self.models_path)
            
            # Initialize TTS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                tts = TTS(
                    model_name=model_name,
                    progress_bar=False,
                    gpu=torch.cuda.is_available()
                )
                
                self.loaded_models[model_name] = tts
                logger.info(f"Model loaded successfully: {model_name} on {device}")
                
                return tts
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
    
    async def synthesize(
        self,
        text: str,
        config: Optional[TTSConfig] = None
    ) -> np.ndarray:
        """Synthesize speech from text"""
        if config is None:
            config = TTSConfig()
        
        # Ensure model is loaded
        await self.initialize(config.model_name)
        
        # Get the model
        tts = self.loaded_models[config.model_name]
        
        # Synthesize in thread pool
        audio = await asyncio.get_event_loop().run_in_executor(
            None,
            self._synthesize_sync,
            tts,
            text,
            config
        )
        
        return audio
    
    def _synthesize_sync(
        self,
        tts: TTS,
        text: str,
        config: TTSConfig
    ) -> np.ndarray:
        """Synchronously synthesize speech"""
        try:
            # Create temp file for output
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                dir=self.processing_path,
                delete=False
            ) as temp_file:
                temp_path = temp_file.name
            
            # Synthesize based on model capabilities
            if config.speaker_wav and hasattr(tts, 'tts_with_vc'):
                # Voice cloning synthesis
                tts.tts_to_file(
                    text=text,
                    speaker_wav=config.speaker_wav,
                    language=config.language,
                    file_path=temp_path,
                    split_sentences=True
                )
            elif hasattr(tts, 'tts_to_file'):
                # Check if this is XTTS v2 without voice cloning
                is_xtts_v2 = config.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
                
                if is_xtts_v2 and not config.speaker_wav:
                    # XTTS v2 requires voice samples, can't work without them
                    # Fall back to a different model for default voice
                    logger.warning("XTTS v2 requires voice samples, falling back to VITS for English")
                    
                    # For English, use VITS instead
                    if config.language == "en":
                        # Load VITS model synchronously
                        vits_model_name = "tts_models/en/vctk/vits"
                        if vits_model_name not in self.loaded_models:
                            self._load_model_sync(vits_model_name)
                        vits_tts = self.loaded_models[vits_model_name]
                        
                        # Use VITS with a default speaker
                        vits_tts.tts_to_file(
                            text=text,
                            file_path=temp_path,
                            speaker="p225",  # Default VCTK speaker
                            split_sentences=True
                        )
                    else:
                        # For non-English, we need voice samples for XTTS v2
                        raise ValueError(f"XTTS v2 requires voice samples for synthesis in {config.language}")
                elif is_xtts_v2 and config.speaker_wav:
                    # Use XTTS v2 with provided voice sample
                    tts.tts_to_file(
                        text=text,
                        speaker_wav=config.speaker_wav,
                        language=config.language,
                        file_path=temp_path,
                        split_sentences=True
                    )
                else:
                    # Standard synthesis for other models
                    synthesis_params = {
                        "text": text,
                        "file_path": temp_path
                    }
                    
                    # Add language if multilingual
                    if self._is_multilingual(config.model_name):
                        synthesis_params["language"] = config.language
                    
                    # Add speaker if multi-speaker
                    if config.speaker and self._is_multispeaker(tts):
                        synthesis_params["speaker"] = config.speaker
                    
                    # Add speed if supported
                    if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts'):
                        synthesis_params["speed"] = config.speed
                    
                    tts.tts_to_file(**synthesis_params)
            else:
                raise ValueError(f"Model {config.model_name} does not support synthesis")
            
            # Load the synthesized audio
            audio, sample_rate = sf.read(temp_path)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            return audio
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise
    
    async def synthesize_stream(
        self,
        text: str,
        config: Optional[TTSConfig] = None,
        chunk_size_ms: int = 100
    ) -> AsyncIterator[np.ndarray]:
        """Stream synthesized speech in chunks"""
        if config is None:
            config = TTSConfig()
        
        # For streaming, we'll synthesize the full text first
        # then stream it in chunks (true streaming requires special models)
        audio = await self.synthesize(text, config)
        
        # Calculate chunk size in samples
        sample_rate = config.sample_rate
        chunk_samples = int(sample_rate * chunk_size_ms / 1000)
        
        # Stream chunks
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            yield chunk
            
            # Small delay to simulate real-time streaming
            await asyncio.sleep(chunk_size_ms / 1000)
    
    def _is_multilingual(self, model_name: str) -> bool:
        """Check if model supports multiple languages"""
        model_info = self.model_registry.get("models", {}).get(model_name, {})
        languages = model_info.get("languages", [])
        return len(languages) > 1
    
    def _is_multispeaker(self, tts: TTS) -> bool:
        """Check if model supports multiple speakers"""
        try:
            return hasattr(tts, 'speakers') and tts.speakers is not None and len(tts.speakers) > 0
        except:
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available TTS models"""
        models = []
        
        for model_name, info in self.model_registry.get("models", {}).items():
            models.append({
                "name": model_name,
                "description": info.get("description", ""),
                "languages": info.get("languages", []),
                "type": info.get("type", "tts"),
                "loaded": model_name in self.loaded_models
            })
        
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        if model_name not in self.model_registry.get("models", {}):
            return None
        
        info = self.model_registry["models"][model_name].copy()
        info["loaded"] = model_name in self.loaded_models
        
        # Add runtime info if model is loaded
        if model_name in self.loaded_models:
            tts = self.loaded_models[model_name]
            
            # Get speakers if available
            if self._is_multispeaker(tts):
                info["speakers"] = tts.speakers
            
            # Get supported languages from model
            if hasattr(tts, 'languages'):
                info["supported_languages"] = tts.languages
        
        return info
    
    def get_speakers(self, model_name: str) -> Optional[List[str]]:
        """Get available speakers for a model"""
        if model_name not in self.loaded_models:
            return None
        
        tts = self.loaded_models[model_name]
        
        if self._is_multispeaker(tts):
            return tts.speakers
        
        return None
    
    async def save_audio(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: int = 22050
    ):
        """Save audio to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            sf.write,
            str(output_path),
            audio,
            sample_rate
        )
        
        logger.info(f"Audio saved to {output_path}")
    
    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        with self.model_lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                
                # Force garbage collection
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Model unloaded: {model_name}")
    
    def cleanup(self):
        """Clean up all loaded models"""
        model_names = list(self.loaded_models.keys())
        
        for model_name in model_names:
            self.unload_model(model_name)
        
        logger.info("TTS service cleanup complete")
    
    @asynccontextmanager
    async def session(self, model_name: Optional[str] = None):
        """Context manager for TTS session"""
        model_name = model_name or self.default_model
        
        try:
            await self.initialize(model_name)
            yield self
        finally:
            # Could implement session-specific cleanup
            pass


# Global TTS service instance
tts_service = TTSService()