"""
Audio processing endpoints for file upload, download, and transcription
"""

import os
import uuid
import tempfile
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...models import audio_processor, streaming_inference
from ...utils.logging import get_logger
from ..middleware import validate_audio_file
from config import settings

logger = get_logger(__name__)

router = APIRouter()


class AudioUploadResponse(BaseModel):
    audio_id: str
    filename: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int
    format: str


class TranscriptionRequest(BaseModel):
    audio_id: Optional[str] = None
    language: Optional[str] = None
    prompt: Optional[str] = None


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration_seconds: float
    processing_time_ms: float


# Storage path for uploaded audio
AUDIO_UPLOAD_PATH = Path(settings.TEMP_FILES_PATH) / "uploads"
AUDIO_UPLOAD_PATH.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for processing"""
    
    # Validate file
    await validate_audio_file(file.size, file.content_type)
    
    # Generate audio ID
    audio_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".wav"
    saved_filename = f"{audio_id}{file_ext}"
    saved_path = AUDIO_UPLOAD_PATH / saved_filename
    
    try:
        # Save uploaded file
        content = await file.read()
        with open(saved_path, "wb") as f:
            f.write(content)
        
        # Get audio info
        audio_info = audio_processor.get_audio_info(saved_path)
        
        return AudioUploadResponse(
            audio_id=audio_id,
            filename=file.filename,
            size_bytes=len(content),
            duration_seconds=audio_info.get("duration", 0),
            sample_rate=audio_info.get("sample_rate", 16000),
            format=audio_info.get("format", "unknown")
        )
        
    except Exception as e:
        # Clean up on error
        if saved_path.exists():
            saved_path.unlink()
        
        logger.error(f"Failed to upload audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{audio_id}")
async def download_audio(audio_id: str):
    """Download audio file"""
    
    # Find audio file
    audio_file = None
    for file in AUDIO_UPLOAD_PATH.iterdir():
        if file.stem == audio_id:
            audio_file = file
            break
    
    if not audio_file or not audio_file.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_file,
        media_type="audio/wav",
        filename=audio_file.name
    )


@router.delete("/{audio_id}")
async def delete_audio(audio_id: str):
    """Delete audio file"""
    
    # Find and delete audio file
    deleted = False
    for file in AUDIO_UPLOAD_PATH.iterdir():
        if file.stem == audio_id:
            file.unlink()
            deleted = True
            break
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return {"message": "Audio file deleted successfully"}


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    request: TranscriptionRequest = None,
    file: Optional[UploadFile] = File(None)
):
    """Transcribe audio to text using Ultravox"""
    
    if not request and not file:
        raise HTTPException(
            status_code=400,
            detail="Either audio_id or file must be provided"
        )
    
    audio_path = None
    temp_file = None
    
    try:
        # Get audio file path
        if file:
            # Validate uploaded file
            await validate_audio_file(file.size, file.content_type)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_file = tmp.name
                audio_path = Path(tmp.name)
        
        elif request and request.audio_id:
            # Find uploaded audio
            for f in AUDIO_UPLOAD_PATH.iterdir():
                if f.stem == request.audio_id:
                    audio_path = f
                    break
            
            if not audio_path:
                raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Process audio file
        audio_data, sample_rate = audio_processor.process_file(audio_path)
        
        # Get duration
        duration = len(audio_data) / sample_rate
        
        # Transcribe using streaming inference
        import time
        start_time = time.time()
        
        # Create inference config
        from ...models import InferenceConfig
        config = InferenceConfig(
            language=request.language if request else None,
            stream=False
        )
        
        # Process audio
        full_text = ""
        detected_language = request.language if request else "en"
        
        async for text in streaming_inference.ultravox_service.process_audio(
            audio_data,
            sample_rate,
            prompt=request.prompt if request else None,
            config=config
        ):
            full_text += text
        
        # Detect language from response if not specified
        if not request or not request.language:
            detected_language = streaming_inference.ultravox_service.detect_language(full_text) or "en"
        
        processing_time = (time.time() - start_time) * 1000
        
        return TranscriptionResponse(
            text=full_text.strip(),
            language=detected_language,
            duration_seconds=duration,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@router.get("/{audio_id}/info")
async def get_audio_info(audio_id: str):
    """Get detailed information about audio file"""
    
    # Find audio file
    audio_file = None
    for file in AUDIO_UPLOAD_PATH.iterdir():
        if file.stem == audio_id:
            audio_file = file
            break
    
    if not audio_file or not audio_file.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Get audio info
    info = audio_processor.get_audio_info(audio_file)
    
    # Add file info
    info["audio_id"] = audio_id
    info["filename"] = audio_file.name
    info["size_bytes"] = audio_file.stat().st_size
    info["created_at"] = audio_file.stat().st_ctime
    
    return info


@router.post("/convert")
async def convert_audio(
    audio_id: str,
    target_format: str = "wav",
    target_sample_rate: int = 16000
):
    """Convert audio to different format or sample rate"""
    
    # Find audio file
    audio_file = None
    for file in AUDIO_UPLOAD_PATH.iterdir():
        if file.stem == audio_id:
            audio_file = file
            break
    
    if not audio_file or not audio_file.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    if target_format not in ["wav", "mp3", "flac"]:
        raise HTTPException(status_code=400, detail="Unsupported target format")
    
    try:
        # Load and process audio
        audio_data, original_sr = audio_processor.process_file(audio_file)
        
        # Resample if needed
        if target_sample_rate != original_sr:
            audio_data = audio_processor._resample(
                audio_data,
                original_sr,
                target_sample_rate
            )
        
        # Save converted file
        converted_id = str(uuid.uuid4())
        converted_filename = f"{converted_id}.{target_format}"
        converted_path = AUDIO_UPLOAD_PATH / converted_filename
        
        audio_processor.save_processed_audio(
            audio_data,
            converted_path,
            target_sample_rate
        )
        
        return {
            "audio_id": converted_id,
            "filename": converted_filename,
            "format": target_format,
            "sample_rate": target_sample_rate,
            "message": "Audio converted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to convert audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))