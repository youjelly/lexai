"""
TTS Model Configuration and Registry
Central configuration for all TTS models and their capabilities
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class ModelType(Enum):
    TTS = "tts"
    VOCODER = "vocoder"
    ENCODER = "encoder"


class ModelQuality(Enum):
    LOW = "low"       # Fast but lower quality
    MEDIUM = "medium" # Balanced
    HIGH = "high"     # Best quality but slower


@dataclass
class ModelConfig:
    """Configuration for a TTS model"""
    name: str
    type: ModelType
    languages: List[str]
    description: str
    quality: ModelQuality
    supports_voice_cloning: bool = False
    supports_streaming: bool = False
    multi_speaker: bool = False
    sample_rate: int = 22050
    max_text_length: int = 1000
    recommended_chunk_size: int = 100  # For streaming
    gpu_memory_mb: int = 1000  # Estimated GPU memory usage
    cpu_compatible: bool = True
    speed_factor: float = 1.0  # Relative speed (1.0 = realtime)
    
    # Model-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)


# Model Registry
TTS_MODEL_REGISTRY = {
    # Multilingual models
    "tts_models/multilingual/multi-dataset/xtts_v2": ModelConfig(
        name="XTTS v2",
        type=ModelType.TTS,
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"],
        description="Best quality multilingual TTS with voice cloning support",
        quality=ModelQuality.HIGH,
        supports_voice_cloning=True,
        supports_streaming=True,
        multi_speaker=False,
        sample_rate=24000,
        max_text_length=500,  # Per chunk for streaming
        gpu_memory_mb=2500,
        speed_factor=0.8,  # Slightly slower than realtime
        parameters={
            "temperature": 0.65,
            "length_penalty": 1.0,
            "repetition_penalty": 10.0,
            "top_k": 50,
            "top_p": 0.85
        }
    ),
    
    "tts_models/multilingual/multi-dataset/your_tts": ModelConfig(
        name="YourTTS",
        type=ModelType.TTS,
        languages=["en", "pt", "fr", "it", "pl", "es", "de", "nl", "cs", "ar", "tr", "ru", "zh-cn"],
        description="Multilingual voice cloning model with good quality",
        quality=ModelQuality.MEDIUM,
        supports_voice_cloning=True,
        supports_streaming=False,
        multi_speaker=True,
        sample_rate=16000,
        max_text_length=300,
        gpu_memory_mb=1500,
        speed_factor=1.2
    ),
    
    # English models
    "tts_models/en/vctk/vits": ModelConfig(
        name="VITS VCTK",
        type=ModelType.TTS,
        languages=["en"],
        description="Fast English multi-speaker model",
        quality=ModelQuality.MEDIUM,
        supports_voice_cloning=False,
        supports_streaming=False,
        multi_speaker=True,
        sample_rate=22050,
        max_text_length=500,
        gpu_memory_mb=800,
        speed_factor=2.0,  # Very fast
        parameters={
            "speaker_id": 0  # Default speaker
        }
    ),
    
    "tts_models/en/jenny/jenny": ModelConfig(
        name="Jenny",
        type=ModelType.TTS,
        languages=["en"],
        description="High quality English female voice",
        quality=ModelQuality.HIGH,
        supports_voice_cloning=False,
        supports_streaming=False,
        multi_speaker=False,
        sample_rate=22050,
        max_text_length=1000,
        gpu_memory_mb=600,
        speed_factor=1.5
    ),
    
    # Vocoder models
    "vocoder_models/universal/libri-tts/wavegrad": ModelConfig(
        name="WaveGrad",
        type=ModelType.VOCODER,
        languages=[],  # Language agnostic
        description="High quality neural vocoder",
        quality=ModelQuality.HIGH,
        sample_rate=22050,
        gpu_memory_mb=1000,
        speed_factor=0.5  # Slower but high quality
    ),
    
    "vocoder_models/en/ljspeech/hifigan_v2": ModelConfig(
        name="HiFiGAN v2",
        type=ModelType.VOCODER,
        languages=["en"],
        description="Fast neural vocoder",
        quality=ModelQuality.MEDIUM,
        sample_rate=22050,
        gpu_memory_mb=500,
        speed_factor=10.0  # Very fast
    ),
    
    # Encoder models
    "encoder_models/universal/libri-tts/wavegrad": ModelConfig(
        name="Universal Speaker Encoder",
        type=ModelType.ENCODER,
        languages=[],  # Language agnostic
        description="Speaker encoder for voice cloning",
        quality=ModelQuality.HIGH,
        sample_rate=16000,
        gpu_memory_mb=400,
        parameters={
            "embedding_dim": 256
        }
    )
}


# Model selection rules
MODEL_SELECTION_RULES = {
    "quality_priority": {
        # Best model for each language prioritizing quality
        "en": "tts_models/en/jenny/jenny",
        "multilingual": "tts_models/multilingual/multi-dataset/xtts_v2",
        "default": "tts_models/multilingual/multi-dataset/xtts_v2"
    },
    "speed_priority": {
        # Fastest model for each language
        "en": "tts_models/en/vctk/vits",
        "multilingual": "tts_models/multilingual/multi-dataset/your_tts",
        "default": "tts_models/en/vctk/vits"
    },
    "voice_cloning": {
        # Best voice cloning model for each language
        "en": "tts_models/multilingual/multi-dataset/xtts_v2",
        "multilingual": "tts_models/multilingual/multi-dataset/xtts_v2",
        "default": "tts_models/multilingual/multi-dataset/your_tts"
    }
}


# Language capabilities
LANGUAGE_CAPABILITIES = {
    "en": {
        "name": "English",
        "native_models": ["tts_models/en/vctk/vits", "tts_models/en/jenny/jenny"],
        "recommended_model": "tts_models/en/jenny/jenny",
        "voice_cloning_model": "tts_models/multilingual/multi-dataset/xtts_v2"
    },
    "es": {
        "name": "Spanish",
        "native_models": [],
        "recommended_model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_cloning_model": "tts_models/multilingual/multi-dataset/xtts_v2"
    },
    "fr": {
        "name": "French",
        "native_models": [],
        "recommended_model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_cloning_model": "tts_models/multilingual/multi-dataset/xtts_v2"
    },
    "de": {
        "name": "German",
        "native_models": [],
        "recommended_model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_cloning_model": "tts_models/multilingual/multi-dataset/xtts_v2"
    },
    "zh": {
        "name": "Chinese",
        "native_models": [],
        "recommended_model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_cloning_model": "tts_models/multilingual/multi-dataset/xtts_v2"
    },
    "ja": {
        "name": "Japanese",
        "native_models": [],
        "recommended_model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "voice_cloning_model": "tts_models/multilingual/multi-dataset/xtts_v2"
    }
}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model"""
    return TTS_MODEL_REGISTRY.get(model_name)


def get_models_for_language(language: str, model_type: ModelType = ModelType.TTS) -> List[str]:
    """Get all models that support a specific language"""
    models = []
    
    for model_name, config in TTS_MODEL_REGISTRY.items():
        if config.type == model_type:
            if not config.languages or language in config.languages:
                models.append(model_name)
    
    return models


def select_model(
    language: str,
    priority: str = "quality",
    require_voice_cloning: bool = False,
    require_streaming: bool = False
) -> Optional[str]:
    """Select best model based on requirements"""
    
    # Get models for language
    models = get_models_for_language(language)
    
    if not models:
        return None
    
    # Filter by requirements
    if require_voice_cloning:
        models = [
            m for m in models
            if TTS_MODEL_REGISTRY[m].supports_voice_cloning
        ]
    
    if require_streaming:
        models = [
            m for m in models
            if TTS_MODEL_REGISTRY[m].supports_streaming
        ]
    
    if not models:
        return None
    
    # Select based on priority
    if priority == "quality":
        # Sort by quality
        models.sort(
            key=lambda m: (
                TTS_MODEL_REGISTRY[m].quality.value,
                -TTS_MODEL_REGISTRY[m].speed_factor
            ),
            reverse=True
        )
    elif priority == "speed":
        # Sort by speed
        models.sort(
            key=lambda m: TTS_MODEL_REGISTRY[m].speed_factor,
            reverse=True
        )
    
    return models[0] if models else None


def estimate_synthesis_time(
    text_length: int,
    model_name: str,
    include_loading: bool = False
) -> float:
    """Estimate synthesis time in seconds"""
    config = get_model_config(model_name)
    
    if not config:
        return 0.0
    
    # Base synthesis time
    words = text_length / 5  # Rough estimate
    speaking_time = words / 150 * 60  # 150 words per minute
    
    synthesis_time = speaking_time / config.speed_factor
    
    # Add loading time if requested
    if include_loading:
        # Rough estimate based on model size
        loading_time = config.gpu_memory_mb / 1000 * 2  # 2 seconds per GB
        synthesis_time += loading_time
    
    return synthesis_time


def get_model_requirements(model_name: str) -> Dict[str, Any]:
    """Get system requirements for a model"""
    config = get_model_config(model_name)
    
    if not config:
        return {}
    
    return {
        "gpu_memory_mb": config.gpu_memory_mb,
        "cpu_compatible": config.cpu_compatible,
        "sample_rate": config.sample_rate,
        "supported_languages": config.languages,
        "features": {
            "voice_cloning": config.supports_voice_cloning,
            "streaming": config.supports_streaming,
            "multi_speaker": config.multi_speaker
        }
    }