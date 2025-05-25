"""
Multilingual TTS handling for LexAI
Automatic language detection, model switching, and cross-lingual synthesis
"""

import os
import json
from typing import Optional, Dict, Any, List, Union, Tuple
import asyncio
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import langdetect
from langdetect import detect_langs, LangDetectException
import re

from .tts_service import TTSService, TTSConfig
from .voice_cloning import VoiceCloning
from ..utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


# Language to model mapping
LANGUAGE_MODEL_MAP = {
    # Best multilingual models
    "multilingual": [
        "tts_models/multilingual/multi-dataset/xtts_v2",
        "tts_models/multilingual/multi-dataset/your_tts"
    ],
    
    # Language-specific models
    "en": [
        "tts_models/en/vctk/vits",
        "tts_models/en/jenny/jenny",
        "tts_models/multilingual/multi-dataset/xtts_v2"
    ],
    "es": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "fr": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "de": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "it": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "pt": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "pl": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "tr": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "ru": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "nl": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "cs": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "ar": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "zh": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "ja": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "hu": ["tts_models/multilingual/multi-dataset/xtts_v2"],
    "ko": ["tts_models/multilingual/multi-dataset/xtts_v2"]
}

# Language code normalization
LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "polish": "pl",
    "turkish": "tr",
    "russian": "ru",
    "dutch": "nl",
    "czech": "cs",
    "arabic": "ar",
    "chinese": "zh",
    "mandarin": "zh",
    "japanese": "ja",
    "hungarian": "hu",
    "korean": "ko"
}


@dataclass
class MultilingualConfig:
    """Configuration for multilingual TTS"""
    auto_detect_language: bool = True
    fallback_language: str = "en"
    cross_lingual_voice: bool = True  # Use voice from one language for another
    language_mixing: bool = False  # Allow mixing languages in same synthesis
    preferred_models: Dict[str, str] = None
    
    def __post_init__(self):
        if self.preferred_models is None:
            self.preferred_models = {}


class MultilingualTTS:
    """Handles multilingual text-to-speech with automatic language detection"""
    
    def __init__(
        self,
        tts_service: Optional[TTSService] = None,
        voice_cloning: Optional[VoiceCloning] = None
    ):
        self.tts_service = tts_service or TTSService()
        self.voice_cloning = voice_cloning or VoiceCloning()
        
        # Model registry
        self.model_registry = self._load_model_registry()
        
        # Language detection cache
        self.detection_cache: Dict[str, str] = {}
        
        # Default configuration
        self.default_config = MultilingualConfig()
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry to know available languages"""
        registry_path = Path(settings.TTS_MODEL_PATH) / "model_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        
        return {"models": {}}
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language from text with confidence score"""
        
        # Check cache
        cache_key = text[:100]  # Use first 100 chars as key
        if cache_key in self.detection_cache:
            return self.detection_cache[cache_key], 1.0
        
        try:
            # Detect languages
            langs = detect_langs(text)
            
            if langs:
                # Get top detection
                lang = langs[0]
                
                # Normalize language code
                lang_code = self._normalize_language_code(lang.lang)
                confidence = lang.prob
                
                # Cache result
                self.detection_cache[cache_key] = lang_code
                
                logger.debug(f"Detected language: {lang_code} (confidence: {confidence:.2f})")
                
                return lang_code, confidence
                
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
        
        # Return fallback
        return self.default_config.fallback_language, 0.0
    
    def _normalize_language_code(self, code: str) -> str:
        """Normalize language code to standard format"""
        code = code.lower()
        
        # Check if it's already a valid code
        if code in LANGUAGE_MODEL_MAP:
            return code
        
        # Check mappings
        if code in LANGUAGE_CODES:
            return LANGUAGE_CODES[code]
        
        # Handle special cases
        if code.startswith("zh"):
            return "zh"
        
        # Default to first two characters
        return code[:2]
    
    def get_best_model_for_language(self, language: str) -> Optional[str]:
        """Get the best available model for a language"""
        language = self._normalize_language_code(language)
        
        # Check preferred models first
        if language in self.default_config.preferred_models:
            preferred = self.default_config.preferred_models[language]
            if preferred in self.model_registry.get("models", {}):
                return preferred
        
        # Get models for language
        models = LANGUAGE_MODEL_MAP.get(language, LANGUAGE_MODEL_MAP["multilingual"])
        
        # Find first available model
        for model in models:
            if model in self.model_registry.get("models", {}):
                return model
        
        # Fallback to default multilingual
        return "tts_models/multilingual/multi-dataset/xtts_v2"
    
    async def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice_id: Optional[str] = None,
        config: Optional[MultilingualConfig] = None,
        **kwargs
    ) -> np.ndarray:
        """Synthesize text with automatic language handling"""
        
        if config is None:
            config = self.default_config
        
        # Detect language if not provided
        if language is None and config.auto_detect_language:
            language, confidence = self.detect_language(text)
            
            # Use fallback if confidence is low
            if confidence < 0.5:
                logger.warning(f"Low confidence language detection ({confidence:.2f}), using fallback")
                language = config.fallback_language
        elif language is None:
            language = config.fallback_language
        else:
            language = self._normalize_language_code(language)
        
        # Don't log the text content, just the action
        logger.debug(f"Synthesizing in {language}")
        
        # Handle voice cloning
        if voice_id:
            return await self._synthesize_with_voice(text, language, voice_id, **kwargs)
        
        # Get best model for language
        model_name = self.get_best_model_for_language(language)
        
        # If we got XTTS v2 but no voice, use VITS for English
        if model_name == "tts_models/multilingual/multi-dataset/xtts_v2" and language == "en":
            # XTTS v2 needs voice samples, use VITS instead for default voice
            model_name = "tts_models/en/vctk/vits"
            logger.info(f"Using VITS instead of XTTS v2 for default English voice")
        
        # Create TTS config
        tts_config = TTSConfig(
            model_name=model_name,
            language=language,
            speaker="p225" if model_name == "tts_models/en/vctk/vits" else None,  # VITS needs speaker ID
            **kwargs
        )
        
        # Synthesize
        return await self.tts_service.synthesize(text, tts_config)
    
    async def _synthesize_with_voice(
        self,
        text: str,
        language: str,
        voice_id: str,
        **kwargs
    ) -> np.ndarray:
        """Synthesize using voice cloning"""
        
        # Load voice profile
        profile = self.voice_cloning.load_profile(voice_id)
        
        if not profile:
            raise ValueError(f"Voice profile {voice_id} not found")
        
        # Check if cross-lingual synthesis is needed
        if profile.language != language and not self.default_config.cross_lingual_voice:
            logger.warning(
                f"Voice language ({profile.language}) doesn't match text language ({language}). "
                "Enable cross_lingual_voice for better results."
            )
        
        # Synthesize with voice cloning
        return await self.voice_cloning.synthesize_with_voice(
            text,
            profile,
            language=language,
            **kwargs
        )
    
    async def synthesize_mixed_language(
        self,
        segments: List[Tuple[str, str]],
        voice_id: Optional[str] = None,
        transition_duration_ms: int = 100
    ) -> np.ndarray:
        """Synthesize text with multiple languages"""
        
        if not self.default_config.language_mixing:
            raise ValueError("Language mixing is not enabled")
        
        audio_segments = []
        
        for text, language in segments:
            # Synthesize segment
            audio = await self.synthesize(
                text,
                language=language,
                voice_id=voice_id
            )
            
            audio_segments.append(audio)
        
        # Concatenate with smooth transitions
        return self._concatenate_audio_smooth(audio_segments, transition_duration_ms)
    
    def _concatenate_audio_smooth(
        self,
        segments: List[np.ndarray],
        transition_ms: int
    ) -> np.ndarray:
        """Concatenate audio segments with smooth transitions"""
        
        if not segments:
            return np.array([])
        
        if len(segments) == 1:
            return segments[0]
        
        # Calculate transition samples
        sample_rate = 22050  # Standard TTS sample rate
        transition_samples = int(sample_rate * transition_ms / 1000)
        
        result = []
        
        for i, segment in enumerate(segments):
            if i == 0:
                # First segment
                result.append(segment)
            else:
                # Create smooth transition
                prev_end = result[-1][-transition_samples:] if len(result[-1]) > transition_samples else result[-1]
                curr_start = segment[:transition_samples] if len(segment) > transition_samples else segment
                
                # Linear crossfade
                fade_out = np.linspace(1, 0, len(prev_end))
                fade_in = np.linspace(0, 1, len(curr_start))
                
                # Apply crossfade
                crossfade = prev_end * fade_out + curr_start * fade_in
                
                # Replace end of previous segment
                if len(result[-1]) > transition_samples:
                    result[-1] = result[-1][:-transition_samples]
                
                # Add crossfade and rest of current segment
                result.append(crossfade)
                if len(segment) > transition_samples:
                    result.append(segment[transition_samples:])
        
        return np.concatenate(result)
    
    async def detect_and_split_languages(self, text: str) -> List[Tuple[str, str]]:
        """Detect and split text by language changes"""
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        
        segments = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Detect language for sentence
            language, _ = self.detect_language(sentence)
            segments.append((sentence, language))
        
        # Merge consecutive segments with same language
        merged = []
        current_text = []
        current_lang = None
        
        for text, lang in segments:
            if lang == current_lang:
                current_text.append(text)
            else:
                if current_text:
                    merged.append((". ".join(current_text) + ".", current_lang))
                current_text = [text]
                current_lang = lang
        
        if current_text:
            merged.append((". ".join(current_text) + ".", current_lang))
        
        return merged
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Get list of supported languages with available models"""
        
        supported = []
        
        for lang_code, models in LANGUAGE_MODEL_MAP.items():
            if lang_code == "multilingual":
                continue
            
            # Find available models
            available_models = [
                m for m in models
                if m in self.model_registry.get("models", {})
            ]
            
            if available_models:
                # Get language name
                lang_name = next(
                    (k for k, v in LANGUAGE_CODES.items() if v == lang_code),
                    lang_code.upper()
                )
                
                supported.append({
                    "code": lang_code,
                    "name": lang_name.title(),
                    "models": available_models,
                    "model_count": len(available_models)
                })
        
        # Sort by name
        supported.sort(key=lambda x: x["name"])
        
        return supported
    
    async def optimize_for_language(
        self,
        language: str,
        text_sample: Optional[str] = None
    ) -> Dict[str, Any]:
        """Optimize TTS settings for a specific language"""
        
        language = self._normalize_language_code(language)
        
        # Get best model
        model = self.get_best_model_for_language(language)
        
        optimization = {
            "language": language,
            "recommended_model": model,
            "settings": {}
        }
        
        # Language-specific optimizations
        if language == "zh":
            optimization["settings"]["speed"] = 0.9  # Slower for tonal language
        elif language == "en":
            optimization["settings"]["speed"] = 1.0
        elif language in ["es", "it"]:
            optimization["settings"]["speed"] = 1.1  # Slightly faster for Romance languages
        
        # Test with sample if provided
        if text_sample:
            try:
                start_time = asyncio.get_event_loop().time()
                
                config = TTSConfig(
                    model_name=model,
                    language=language,
                    **optimization["settings"]
                )
                
                await self.tts_service.synthesize(text_sample, config)
                
                synthesis_time = asyncio.get_event_loop().time() - start_time
                optimization["synthesis_time"] = synthesis_time
                optimization["realtime_factor"] = len(text_sample) / synthesis_time
                
            except Exception as e:
                logger.error(f"Optimization test failed: {e}")
                optimization["test_error"] = str(e)
        
        return optimization
    
    def set_language_preference(self, language: str, model: str):
        """Set preferred model for a language"""
        language = self._normalize_language_code(language)
        self.default_config.preferred_models[language] = model
        logger.info(f"Set preferred model for {language}: {model}")
    
    def clear_cache(self):
        """Clear language detection cache"""
        self.detection_cache.clear()
        logger.info("Cleared language detection cache")


# Global multilingual TTS instance
multilingual_tts = MultilingualTTS()