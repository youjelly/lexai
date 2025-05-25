"""
LexAI TTS module
Provides text-to-speech synthesis with voice cloning and multilingual support
"""

from .tts_service import (
    TTSService,
    TTSConfig,
    tts_service
)

from .voice_cloning import (
    VoiceCloning,
    VoiceProfile,
    voice_cloning
)

from .voice_manager import (
    VoiceManager,
    VoiceMetadata,
    voice_manager
)

from .multilingual_tts import (
    MultilingualTTS,
    MultilingualConfig,
    multilingual_tts,
    LANGUAGE_CODES,
    LANGUAGE_MODEL_MAP
)

from .model_config import (
    ModelType,
    ModelQuality,
    ModelConfig,
    TTS_MODEL_REGISTRY,
    MODEL_SELECTION_RULES,
    LANGUAGE_CAPABILITIES,
    get_model_config,
    get_models_for_language,
    select_model,
    estimate_synthesis_time,
    get_model_requirements
)

__all__ = [
    # Services
    'TTSService',
    'VoiceCloning',
    'VoiceManager',
    'MultilingualTTS',
    
    # Configurations
    'TTSConfig',
    'VoiceProfile',
    'VoiceMetadata',
    'MultilingualConfig',
    'ModelType',
    'ModelQuality',
    'ModelConfig',
    
    # Global instances
    'tts_service',
    'voice_cloning',
    'voice_manager',
    'multilingual_tts',
    
    # Constants and registries
    'LANGUAGE_CODES',
    'LANGUAGE_MODEL_MAP',
    'TTS_MODEL_REGISTRY',
    'MODEL_SELECTION_RULES',
    'LANGUAGE_CAPABILITIES',
    
    # Utility functions
    'get_model_config',
    'get_models_for_language',
    'select_model',
    'estimate_synthesis_time',
    'get_model_requirements'
]