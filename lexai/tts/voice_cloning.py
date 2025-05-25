"""
Voice cloning pipeline for Coqui TTS
Enables voice cloning with minimal samples (10-30 seconds)
"""

import os
import torch
import numpy as np
import soundfile as sf
import librosa
from typing import Optional, Dict, Any, List, Union, Tuple
import asyncio
from pathlib import Path
import tempfile
import shutil
import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime

from TTS.api import TTS
from TTS.utils.audio import AudioProcessor
from TTS.encoder.utils.generic_utils import setup_encoder_model
from TTS.encoder.utils.prepare_voxceleb import extract_embedding

from ..utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass
class VoiceProfile:
    """Voice profile for cloning"""
    id: str
    name: str
    description: str = ""
    source_files: List[str] = None
    embedding_path: Optional[str] = None
    sample_rate: int = 22050
    total_duration: float = 0.0
    language: str = "en"
    created_at: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.source_files is None:
            self.source_files = []
        if self.metadata is None:
            self.metadata = {}
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class VoiceCloning:
    """Voice cloning pipeline for creating and using custom voices"""
    
    def __init__(self):
        self.voices_path = Path(settings.VOICE_FILES_PATH)
        self.samples_path = Path(settings.VOICE_SAMPLES_PATH)
        self.processing_path = Path(settings.AUDIO_PROCESSING_PATH) / "voice_cloning"
        self.models_path = Path(settings.TTS_MODEL_PATH)
        
        # Voice cloning models
        self.cloning_model = "tts_models/multilingual/multi-dataset/your_tts"
        self.encoder_model = "encoder_models/universal/libri-tts/wavegrad"
        
        # Speaker encoder
        self.speaker_encoder = None
        self.tts_model = None
        
        # Initialize paths
        self._init_paths()
    
    def _init_paths(self):
        """Initialize required directories"""
        for path in [self.voices_path, self.samples_path, self.processing_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize voice cloning models"""
        if self.speaker_encoder is None:
            await self._load_encoder()
        
        if self.tts_model is None:
            await self._load_tts_model()
    
    async def _load_encoder(self):
        """Load speaker encoder model"""
        logger.info("Loading speaker encoder model...")
        
        # Set TTS home
        os.environ['TTS_HOME'] = str(self.models_path)
        
        # Load encoder in thread pool
        self.speaker_encoder = await asyncio.get_event_loop().run_in_executor(
            None,
            self._load_encoder_sync
        )
        
        logger.info("Speaker encoder loaded successfully")
    
    def _load_encoder_sync(self):
        """Synchronously load speaker encoder"""
        try:
            # Try to use TTS API first
            tts = TTS(model_name=self.encoder_model, gpu=torch.cuda.is_available())
            return tts
        except:
            # Fallback to manual loading if needed
            logger.warning("Failed to load encoder via TTS API, trying manual load")
            return None
    
    async def _load_tts_model(self):
        """Load TTS model for voice cloning"""
        logger.info(f"Loading voice cloning model: {self.cloning_model}")
        
        self.tts_model = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: TTS(
                model_name=self.cloning_model,
                gpu=torch.cuda.is_available()
            )
        )
        
        logger.info("Voice cloning model loaded successfully")
    
    async def create_voice_profile(
        self,
        name: str,
        audio_files: List[Union[str, Path]],
        description: str = "",
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None
    ) -> VoiceProfile:
        """Create a voice profile from audio samples"""
        
        # Ensure models are loaded
        await self.initialize()
        
        # Generate unique ID
        profile_id = self._generate_profile_id(name)
        
        # Create profile directory
        profile_dir = self.voices_path / profile_id
        profile_dir.mkdir(exist_ok=True)
        
        # Process audio files
        processed_files = []
        total_duration = 0.0
        
        for audio_file in audio_files:
            processed_file, duration = await self._process_voice_sample(
                audio_file,
                profile_dir
            )
            processed_files.append(str(processed_file))
            total_duration += duration
        
        # Check minimum duration
        if total_duration < 10.0:
            logger.warning(f"Total audio duration ({total_duration:.1f}s) is less than recommended 10s")
        
        # Create voice embedding
        embedding_path = await self._create_voice_embedding(
            processed_files,
            profile_dir
        )
        
        # Create profile
        profile = VoiceProfile(
            id=profile_id,
            name=name,
            description=description,
            source_files=processed_files,
            embedding_path=str(embedding_path),
            total_duration=total_duration,
            language=language,
            metadata=metadata or {}
        )
        
        # Save profile metadata
        self._save_profile(profile)
        
        logger.info(f"Voice profile created: {name} (ID: {profile_id}, Duration: {total_duration:.1f}s)")
        
        return profile
    
    def _generate_profile_id(self, name: str) -> str:
        """Generate unique profile ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_clean = "".join(c for c in name.lower() if c.isalnum() or c in "_-")
        return f"{name_clean}_{timestamp}"
    
    async def _process_voice_sample(
        self,
        audio_file: Union[str, Path],
        output_dir: Path
    ) -> Tuple[Path, float]:
        """Process and validate voice sample"""
        audio_file = Path(audio_file)
        
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Load audio
        audio, sr = await asyncio.get_event_loop().run_in_executor(
            None,
            librosa.load,
            str(audio_file),
            None  # Preserve original sample rate
        )
        
        # Resample if needed
        target_sr = 22050
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Normalize audio
        audio = audio / np.abs(audio).max() * 0.95
        
        # Calculate duration
        duration = len(audio) / target_sr
        
        # Save processed file
        output_file = output_dir / f"sample_{audio_file.stem}.wav"
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            sf.write,
            str(output_file),
            audio,
            target_sr
        )
        
        return output_file, duration
    
    async def _create_voice_embedding(
        self,
        audio_files: List[str],
        output_dir: Path
    ) -> Path:
        """Create voice embedding from audio samples"""
        
        # Concatenate all audio samples
        all_audio = []
        
        for audio_file in audio_files:
            audio, sr = await asyncio.get_event_loop().run_in_executor(
                None,
                sf.read,
                audio_file
            )
            all_audio.append(audio)
        
        # Concatenate with small silence between samples
        silence = np.zeros(int(0.5 * 22050))  # 0.5s silence
        concatenated = []
        
        for i, audio in enumerate(all_audio):
            concatenated.append(audio)
            if i < len(all_audio) - 1:
                concatenated.append(silence)
        
        concatenated_audio = np.concatenate(concatenated)
        
        # Save concatenated audio
        concat_path = output_dir / "concatenated.wav"
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            sf.write,
            str(concat_path),
            concatenated_audio,
            22050
        )
        
        # Create embedding
        embedding_path = output_dir / "speaker_embedding.npy"
        
        if self.speaker_encoder and hasattr(self.speaker_encoder, 'compute_embedding'):
            # Use TTS encoder
            embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                self.speaker_encoder.compute_embedding,
                str(concat_path)
            )
            
            np.save(embedding_path, embedding)
        else:
            # For models that don't need pre-computed embeddings
            # just use the concatenated audio file
            shutil.copy(concat_path, embedding_path.with_suffix('.wav'))
        
        return embedding_path
    
    def _save_profile(self, profile: VoiceProfile):
        """Save voice profile metadata"""
        profile_file = self.voices_path / profile.id / "profile.json"
        
        with open(profile_file, 'w') as f:
            json.dump(asdict(profile), f, indent=2)
    
    async def synthesize_with_voice(
        self,
        text: str,
        voice_profile: Union[str, VoiceProfile],
        language: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Synthesize text using a cloned voice"""
        
        # Ensure models are loaded
        await self.initialize()
        
        # Load profile if ID provided
        if isinstance(voice_profile, str):
            voice_profile = self.load_profile(voice_profile)
        
        if not voice_profile:
            raise ValueError("Voice profile not found")
        
        # Use profile language if not specified
        if language is None:
            language = voice_profile.language
        
        # Get reference audio
        reference_wav = voice_profile.embedding_path
        
        # If embedding is .npy, use concatenated audio instead
        if reference_wav.endswith('.npy'):
            concat_wav = Path(reference_wav).parent / "concatenated.wav"
            if concat_wav.exists():
                reference_wav = str(concat_wav)
            else:
                # Use first source file as fallback
                reference_wav = voice_profile.source_files[0]
        
        # Synthesize using voice cloning model
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            dir=self.processing_path,
            delete=False
        ) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Synthesize
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.tts_model.tts_to_file,
                text,
                reference_wav,
                language,
                temp_path
            )
            
            # Load synthesized audio
            audio, sr = await asyncio.get_event_loop().run_in_executor(
                None,
                sf.read,
                temp_path
            )
            
            return audio
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def load_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """Load a voice profile by ID"""
        profile_file = self.voices_path / profile_id / "profile.json"
        
        if not profile_file.exists():
            return None
        
        with open(profile_file, 'r') as f:
            data = json.load(f)
        
        return VoiceProfile(**data)
    
    def list_profiles(self) -> List[VoiceProfile]:
        """List all available voice profiles"""
        profiles = []
        
        for profile_dir in self.voices_path.iterdir():
            if profile_dir.is_dir():
                profile = self.load_profile(profile_dir.name)
                if profile:
                    profiles.append(profile)
        
        return sorted(profiles, key=lambda p: p.created_at, reverse=True)
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete a voice profile"""
        profile_dir = self.voices_path / profile_id
        
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
            logger.info(f"Deleted voice profile: {profile_id}")
            return True
        
        return False
    
    async def validate_voice_quality(
        self,
        voice_profile: Union[str, VoiceProfile],
        test_text: str = "Hello, this is a test of the voice cloning system."
    ) -> Dict[str, Any]:
        """Validate voice quality by analyzing synthesis output"""
        
        # Synthesize test audio
        audio = await self.synthesize_with_voice(test_text, voice_profile)
        
        # Analyze audio quality
        quality_metrics = await asyncio.get_event_loop().run_in_executor(
            None,
            self._analyze_audio_quality,
            audio
        )
        
        return quality_metrics
    
    def _analyze_audio_quality(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio quality metrics"""
        
        # Basic metrics
        metrics = {
            "duration": len(audio) / 22050,
            "peak_amplitude": float(np.abs(audio).max()),
            "rms_energy": float(np.sqrt(np.mean(audio**2))),
            "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))
        }
        
        # Spectral features
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        metrics["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(S=magnitude)))
        metrics["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(S=magnitude)))
        
        # Quality assessment
        metrics["quality_score"] = self._calculate_quality_score(metrics)
        
        return metrics
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-1)"""
        score = 1.0
        
        # Penalize clipping
        if metrics["peak_amplitude"] > 0.99:
            score *= 0.9
        
        # Check for silence
        if metrics["rms_energy"] < 0.01:
            score *= 0.5
        
        # Good spectral characteristics
        if 2000 < metrics["spectral_centroid"] < 4000:
            score *= 1.1
        
        return min(1.0, score)
    
    def cleanup(self):
        """Clean up resources"""
        self.speaker_encoder = None
        self.tts_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Voice cloning cleanup complete")


# Global voice cloning instance
voice_cloning = VoiceCloning()