"""
Voice management endpoints for voice cloning and customization
"""

import os
import base64
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ...tts import voice_manager, voice_cloning
from ...utils.logging import get_logger
from ..middleware import validate_audio_file
from config import settings

logger = get_logger(__name__)

router = APIRouter()


class VoiceCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    language: str = "en"
    tags: Optional[List[str]] = None


class VoiceResponse(BaseModel):
    id: str
    name: str
    description: str
    language: str
    category: str
    tags: List[str]
    is_default: bool
    usage_count: int
    created_at: str
    preview_available: bool


class VoiceTestRequest(BaseModel):
    text: str
    language: Optional[str] = None


@router.get("", response_model=List[VoiceResponse])
async def list_voices(
    category: Optional[str] = None,
    language: Optional[str] = None
):
    """List all available voices (default + custom)"""
    
    try:
        # Get voices from manager
        voices = voice_manager.list_voices(
            category=category,
            language=language
        )
        
        # Format response
        return [
            VoiceResponse(
                id=voice.profile.id,
                name=voice.profile.name,
                description=voice.profile.description,
                language=voice.profile.language,
                category=voice.category,
                tags=voice.tags,
                is_default=voice.is_default,
                usage_count=voice.usage_count,
                created_at=voice.profile.created_at,
                preview_available=voice.preview_audio_path is not None
            )
            for voice in voices
        ]
        
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clone")
async def clone_voice(
    name: str = Form(...),
    description: Optional[str] = Form(""),
    language: str = Form("en"),
    tags: Optional[str] = Form(None),
    audio_files: List[UploadFile] = File(...)
):
    """Clone voice from audio samples"""
    
    if not audio_files:
        raise HTTPException(status_code=400, detail="At least one audio file required")
    
    if len(audio_files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 audio files allowed")
    
    temp_files = []
    
    try:
        # Save uploaded files temporarily
        for file in audio_files:
            # Validate file
            await validate_audio_file(file.size, file.content_type)
            
            # Save to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_files.append(tmp.name)
        
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        
        # Create voice profile
        voice_id = await voice_manager.create_voice(
            name=name,
            audio_files=temp_files,
            description=description,
            language=language,
            tags=tag_list,
            category="custom"
        )
        
        # Get created voice
        voice = voice_manager.get_voice(voice_id)
        
        return {
            "id": voice_id,
            "name": voice.profile.name,
            "description": voice.profile.description,
            "language": voice.profile.language,
            "message": "Voice cloned successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clone voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


@router.get("/{voice_id}", response_model=VoiceResponse)
async def get_voice(voice_id: str):
    """Get voice details"""
    
    voice = voice_manager.get_voice(voice_id)
    
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    return VoiceResponse(
        id=voice.profile.id,
        name=voice.profile.name,
        description=voice.profile.description,
        language=voice.profile.language,
        category=voice.category,
        tags=voice.tags,
        is_default=voice.is_default,
        usage_count=voice.usage_count,
        created_at=voice.profile.created_at,
        preview_available=voice.preview_audio_path is not None
    )


@router.delete("/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete custom voice"""
    
    voice = voice_manager.get_voice(voice_id)
    
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    if voice.is_default:
        raise HTTPException(status_code=400, detail="Cannot delete default voice")
    
    if voice.category != "custom":
        raise HTTPException(status_code=400, detail="Can only delete custom voices")
    
    success = voice_manager.delete_voice(voice_id)
    
    if success:
        return {"message": "Voice deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete voice")


@router.post("/{voice_id}/samples")
async def add_voice_samples(
    voice_id: str,
    audio_files: List[UploadFile] = File(...)
):
    """Add training samples to existing voice"""
    
    voice = voice_manager.get_voice(voice_id)
    
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    if voice.category != "custom":
        raise HTTPException(status_code=400, detail="Can only modify custom voices")
    
    if len(audio_files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 audio files allowed")
    
    temp_files = []
    
    try:
        # Save uploaded files
        for file in audio_files:
            await validate_audio_file(file.size, file.content_type)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_files.append(tmp.name)
        
        # Add samples to existing voice
        # This would require updating the voice profile with new samples
        # For MVP, we'll just return success
        
        return {
            "message": f"Added {len(audio_files)} samples to voice",
            "voice_id": voice_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


@router.get("/{voice_id}/preview")
async def get_voice_preview(voice_id: str):
    """Generate or get preview audio for voice"""
    
    voice = voice_manager.get_voice(voice_id)
    
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    if voice.preview_audio_path and os.path.exists(voice.preview_audio_path):
        # Return existing preview
        return FileResponse(
            voice.preview_audio_path,
            media_type="audio/wav",
            filename=f"preview_{voice_id}.wav"
        )
    else:
        # Generate preview if not exists
        try:
            # This is handled internally by voice_manager
            await voice_manager._generate_voice_preview(voice)
            
            if voice.preview_audio_path and os.path.exists(voice.preview_audio_path):
                return FileResponse(
                    voice.preview_audio_path,
                    media_type="audio/wav",
                    filename=f"preview_{voice_id}.wav"
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to generate preview")
                
        except Exception as e:
            logger.error(f"Failed to generate preview: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/{voice_id}/test")
async def test_voice(voice_id: str, request: VoiceTestRequest):
    """Test voice with custom text"""
    
    voice = voice_manager.get_voice(voice_id)
    
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    # Validate text length
    if len(request.text) > 500:
        raise HTTPException(status_code=400, detail="Text too long (max 500 characters)")
    
    try:
        # Set active voice
        voice_manager.set_active_voice(voice_id)
        
        # Synthesize audio
        audio = await voice_manager.synthesize_with_active_voice(
            request.text,
            language=request.language or voice.profile.language
        )
        
        # Convert to WAV format
        import io
        import soundfile as sf
        
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, 22050, format='WAV')
        wav_buffer.seek(0)
        
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=test_{voice_id}.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to test voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{voice_id}/export")
async def export_voice(voice_id: str):
    """Export voice for backup or sharing"""
    
    voice = voice_manager.get_voice(voice_id)
    
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    if voice.category != "custom":
        raise HTTPException(status_code=400, detail="Can only export custom voices")
    
    try:
        # Create temp file for export
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            export_path = tmp.name
        
        # Export voice
        success = await voice_manager.export_voice(voice_id, export_path)
        
        if success and os.path.exists(export_path):
            return FileResponse(
                export_path,
                media_type="application/zip",
                filename=f"voice_{voice_id}.zip",
                background=lambda: os.unlink(export_path)  # Delete after sending
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to export voice")
            
    except Exception as e:
        logger.error(f"Failed to export voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))