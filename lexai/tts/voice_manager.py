"""
Voice management system for LexAI
Handles voice library, switching, previews, and versioning
"""

import os
import json
import shutil
import asyncio
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from datetime import datetime
import tempfile
import zipfile
from dataclasses import dataclass, asdict
import hashlib

import numpy as np
import soundfile as sf

from .voice_cloning import VoiceProfile, VoiceCloning
from .tts_service import TTSService, TTSConfig
from ..utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass
class VoiceMetadata:
    """Extended metadata for voice management"""
    profile: VoiceProfile
    tags: List[str] = None
    category: str = "custom"
    is_default: bool = False
    usage_count: int = 0
    last_used: Optional[str] = None
    version: int = 1
    parent_id: Optional[str] = None
    preview_text: str = "Hello, this is a preview of the voice."
    preview_audio_path: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class VoiceManager:
    """Manages voice library with CRUD operations, versioning, and caching"""
    
    def __init__(
        self,
        voice_cloning: Optional[VoiceCloning] = None,
        tts_service: Optional[TTSService] = None
    ):
        self.voices_path = Path(settings.VOICE_FILES_PATH)
        self.cache_path = Path(settings.TTS_CACHE_PATH) / "voices"
        self.backup_path = self.voices_path / "backups"
        
        # Services
        self.voice_cloning = voice_cloning or VoiceCloning()
        self.tts_service = tts_service or TTSService()
        
        # Voice cache
        self.voice_cache: Dict[str, VoiceMetadata] = {}
        self.active_voice_id: Optional[str] = None
        
        # Initialize paths
        self._init_paths()
        
        # Load voice library
        self._load_voice_library()
    
    def _init_paths(self):
        """Initialize required directories"""
        for path in [self.voices_path, self.cache_path, self.backup_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _load_voice_library(self):
        """Load all voices into cache"""
        self.voice_cache.clear()
        
        # Load custom voices
        for profile in self.voice_cloning.list_profiles():
            metadata = self._load_voice_metadata(profile.id)
            if metadata:
                self.voice_cache[profile.id] = metadata
            else:
                # Create default metadata
                self.voice_cache[profile.id] = VoiceMetadata(profile=profile)
        
        # Load system voices
        self._load_system_voices()
        
        logger.info(f"Loaded {len(self.voice_cache)} voices")
    
    def _load_voice_metadata(self, voice_id: str) -> Optional[VoiceMetadata]:
        """Load extended metadata for a voice"""
        metadata_file = self.voices_path / voice_id / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            # Load profile
            profile = self.voice_cloning.load_profile(voice_id)
            if not profile:
                return None
            
            # Create metadata with profile
            data['profile'] = profile
            return VoiceMetadata(**data)
            
        except Exception as e:
            logger.error(f"Failed to load metadata for {voice_id}: {e}")
            return None
    
    def _save_voice_metadata(self, metadata: VoiceMetadata):
        """Save extended metadata for a voice"""
        metadata_file = self.voices_path / metadata.profile.id / "metadata.json"
        
        # Convert to dict without profile (profile is saved separately)
        data = asdict(metadata)
        del data['profile']
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_system_voices(self):
        """Load built-in system voices"""
        # This would load any pre-configured system voices
        # For now, we'll add a placeholder
        pass
    
    async def create_voice(
        self,
        name: str,
        audio_files: List[Union[str, Path]],
        description: str = "",
        language: str = "en",
        tags: Optional[List[str]] = None,
        category: str = "custom",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new voice with full metadata"""
        
        # Create voice profile
        profile = await self.voice_cloning.create_voice_profile(
            name=name,
            audio_files=audio_files,
            description=description,
            language=language,
            metadata=metadata
        )
        
        # Create extended metadata
        voice_metadata = VoiceMetadata(
            profile=profile,
            tags=tags or [],
            category=category
        )
        
        # Generate preview
        await self._generate_voice_preview(voice_metadata)
        
        # Save metadata
        self._save_voice_metadata(voice_metadata)
        
        # Add to cache
        self.voice_cache[profile.id] = voice_metadata
        
        logger.info(f"Created voice: {name} (ID: {profile.id})")
        
        return profile.id
    
    async def _generate_voice_preview(self, metadata: VoiceMetadata):
        """Generate preview audio for a voice"""
        try:
            # Synthesize preview
            audio = await self.voice_cloning.synthesize_with_voice(
                metadata.preview_text,
                metadata.profile
            )
            
            # Save preview
            preview_path = self.voices_path / metadata.profile.id / "preview.wav"
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                sf.write,
                str(preview_path),
                audio,
                22050
            )
            
            metadata.preview_audio_path = str(preview_path)
            
        except Exception as e:
            logger.error(f"Failed to generate preview for {metadata.profile.name}: {e}")
    
    def get_voice(self, voice_id: str) -> Optional[VoiceMetadata]:
        """Get voice metadata by ID"""
        return self.voice_cache.get(voice_id)
    
    def list_voices(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> List[VoiceMetadata]:
        """List voices with optional filtering"""
        voices = list(self.voice_cache.values())
        
        # Filter by category
        if category:
            voices = [v for v in voices if v.category == category]
        
        # Filter by tags
        if tags:
            voices = [v for v in voices if any(tag in v.tags for tag in tags)]
        
        # Filter by language
        if language:
            voices = [v for v in voices if v.profile.language == language]
        
        # Sort by usage and name
        voices.sort(key=lambda v: (-v.usage_count, v.profile.name))
        
        return voices
    
    async def update_voice(
        self,
        voice_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        preview_text: Optional[str] = None
    ) -> bool:
        """Update voice metadata"""
        metadata = self.get_voice(voice_id)
        if not metadata:
            return False
        
        # Update fields
        if name:
            metadata.profile.name = name
        if description:
            metadata.profile.description = description
        if tags is not None:
            metadata.tags = tags
        if category:
            metadata.category = category
        if preview_text:
            metadata.preview_text = preview_text
            # Regenerate preview
            await self._generate_voice_preview(metadata)
        
        # Save changes
        self._save_voice_metadata(metadata)
        self.voice_cloning._save_profile(metadata.profile)
        
        return True
    
    def delete_voice(self, voice_id: str, backup: bool = True) -> bool:
        """Delete a voice with optional backup"""
        metadata = self.get_voice(voice_id)
        if not metadata:
            return False
        
        # Create backup if requested
        if backup:
            self._backup_voice(voice_id)
        
        # Delete from voice cloning
        self.voice_cloning.delete_profile(voice_id)
        
        # Remove from cache
        del self.voice_cache[voice_id]
        
        # Clear active voice if it was deleted
        if self.active_voice_id == voice_id:
            self.active_voice_id = None
        
        logger.info(f"Deleted voice: {voice_id}")
        
        return True
    
    def _backup_voice(self, voice_id: str):
        """Create a backup of a voice"""
        metadata = self.get_voice(voice_id)
        if not metadata:
            return
        
        # Create backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{voice_id}_{timestamp}.zip"
        backup_file = self.backup_path / backup_name
        
        # Create zip archive
        voice_dir = self.voices_path / voice_id
        
        with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in voice_dir.rglob('*'):
                if file_path.is_file():
                    arc_name = file_path.relative_to(voice_dir)
                    zf.write(file_path, arc_name)
        
        logger.info(f"Created backup: {backup_file}")
    
    async def clone_voice(
        self,
        voice_id: str,
        new_name: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> str:
        """Clone an existing voice with optional modifications"""
        metadata = self.get_voice(voice_id)
        if not metadata:
            raise ValueError(f"Voice {voice_id} not found")
        
        # Create new voice from existing samples
        new_id = await self.create_voice(
            name=new_name,
            audio_files=metadata.profile.source_files,
            description=f"Cloned from {metadata.profile.name}",
            language=metadata.profile.language,
            tags=metadata.tags + ["cloned"],
            category=metadata.category
        )
        
        # Update version info
        new_metadata = self.get_voice(new_id)
        new_metadata.parent_id = voice_id
        new_metadata.version = metadata.version + 1
        
        # Apply modifications if any
        if modifications:
            await self.update_voice(new_id, **modifications)
        
        self._save_voice_metadata(new_metadata)
        
        return new_id
    
    def set_active_voice(self, voice_id: str) -> bool:
        """Set the active voice for synthesis"""
        if voice_id not in self.voice_cache:
            return False
        
        self.active_voice_id = voice_id
        
        # Update usage statistics
        metadata = self.voice_cache[voice_id]
        metadata.usage_count += 1
        metadata.last_used = datetime.now().isoformat()
        self._save_voice_metadata(metadata)
        
        logger.info(f"Set active voice: {metadata.profile.name}")
        
        return True
    
    def get_active_voice(self) -> Optional[VoiceMetadata]:
        """Get the currently active voice"""
        if self.active_voice_id:
            return self.get_voice(self.active_voice_id)
        
        # Return default voice if set
        for metadata in self.voice_cache.values():
            if metadata.is_default:
                return metadata
        
        return None
    
    async def synthesize_with_active_voice(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Synthesize text using the active voice"""
        voice = self.get_active_voice()
        
        if not voice:
            # Fallback to standard TTS
            config = TTSConfig(language=language or "en", **kwargs)
            return await self.tts_service.synthesize(text, config)
        
        # Use voice cloning
        return await self.voice_cloning.synthesize_with_voice(
            text,
            voice.profile,
            language=language,
            **kwargs
        )
    
    async def compare_voices(
        self,
        text: str,
        voice_ids: List[str],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Union[np.ndarray, str]]:
        """Compare multiple voices by synthesizing the same text"""
        results = {}
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for voice_id in voice_ids:
            metadata = self.get_voice(voice_id)
            if not metadata:
                logger.warning(f"Voice {voice_id} not found")
                continue
            
            try:
                # Synthesize
                audio = await self.voice_cloning.synthesize_with_voice(
                    text,
                    metadata.profile
                )
                
                results[voice_id] = audio
                
                # Save if output directory provided
                if output_dir:
                    output_file = output_dir / f"{voice_id}_{metadata.profile.name}.wav"
                    
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        sf.write,
                        str(output_file),
                        audio,
                        22050
                    )
                    
                    results[f"{voice_id}_path"] = str(output_file)
                    
            except Exception as e:
                logger.error(f"Failed to synthesize with voice {voice_id}: {e}")
                results[voice_id] = None
        
        return results
    
    async def export_voice(self, voice_id: str, output_path: Union[str, Path]) -> bool:
        """Export a voice for sharing or backup"""
        metadata = self.get_voice(voice_id)
        if not metadata:
            return False
        
        output_path = Path(output_path)
        
        # Create export package
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy voice files
            voice_dir = self.voices_path / voice_id
            export_dir = temp_path / "voice"
            shutil.copytree(voice_dir, export_dir)
            
            # Add export metadata
            export_info = {
                "exported_at": datetime.now().isoformat(),
                "lexai_version": "0.1.0",
                "voice_metadata": asdict(metadata)
            }
            
            with open(temp_path / "export_info.json", 'w') as f:
                json.dump(export_info, f, indent=2)
            
            # Create zip
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        arc_name = file_path.relative_to(temp_path)
                        zf.write(file_path, arc_name)
        
        logger.info(f"Exported voice to {output_path}")
        return True
    
    async def import_voice(self, import_path: Union[str, Path]) -> str:
        """Import a voice from export package"""
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract package
            with zipfile.ZipFile(import_path, 'r') as zf:
                zf.extractall(temp_path)
            
            # Load export info
            with open(temp_path / "export_info.json", 'r') as f:
                export_info = json.load(f)
            
            # Get voice data
            voice_data = export_info["voice_metadata"]
            profile_data = voice_data["profile"]
            
            # Generate new ID for imported voice
            new_name = f"{profile_data['name']}_imported"
            
            # Get audio files from the export
            voice_dir = temp_path / "voice"
            audio_files = []
            
            for source_file in profile_data["source_files"]:
                # Find the file in the export
                source_name = Path(source_file).name
                for file_path in voice_dir.glob(f"*/{source_name}"):
                    if file_path.exists():
                        audio_files.append(file_path)
                        break
            
            if not audio_files:
                raise ValueError("No audio files found in import package")
            
            # Create new voice
            new_id = await self.create_voice(
                name=new_name,
                audio_files=audio_files,
                description=profile_data.get("description", ""),
                language=profile_data.get("language", "en"),
                tags=voice_data.get("tags", []) + ["imported"],
                category=voice_data.get("category", "custom")
            )
            
            logger.info(f"Imported voice: {new_name} (ID: {new_id})")
            
            return new_id
    
    def get_voice_statistics(self) -> Dict[str, Any]:
        """Get statistics about the voice library"""
        total_voices = len(self.voice_cache)
        
        # Count by category
        categories = {}
        for metadata in self.voice_cache.values():
            categories[metadata.category] = categories.get(metadata.category, 0) + 1
        
        # Count by language
        languages = {}
        for metadata in self.voice_cache.values():
            lang = metadata.profile.language
            languages[lang] = languages.get(lang, 0) + 1
        
        # Most used voices
        most_used = sorted(
            self.voice_cache.values(),
            key=lambda m: m.usage_count,
            reverse=True
        )[:5]
        
        return {
            "total_voices": total_voices,
            "categories": categories,
            "languages": languages,
            "most_used": [
                {
                    "id": m.profile.id,
                    "name": m.profile.name,
                    "usage_count": m.usage_count
                }
                for m in most_used
            ],
            "active_voice": self.active_voice_id
        }
    
    def cleanup(self):
        """Clean up voice manager resources"""
        self.voice_cache.clear()
        self.active_voice_id = None
        logger.info("Voice manager cleanup complete")


# Global voice manager instance
voice_manager = VoiceManager()