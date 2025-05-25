from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import uuid
from datetime import datetime
import logging

from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be audio format")
        
        file_extension = file.filename.split(".")[-1]
        file_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        file_path = os.path.join(settings.AUDIO_PROCESSING_PATH, f"{file_id}.{file_extension}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Audio file uploaded: {file_path}")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "path": file_path,
            "size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Error uploading audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{file_id}")
async def download_audio(file_id: str):
    try:
        # Check multiple locations for the audio file
        search_paths = [
            settings.AUDIO_PROCESSING_PATH,
            settings.AUDIO_CONVERSATIONS_PATH,
            settings.AUDIO_SESSIONS_PATH,
            settings.TEMP_FILES_PATH
        ]
        
        for path in search_paths:
            for ext in ["wav", "mp3", "ogg", "m4a", "flac"]:
                file_path = os.path.join(path, f"{file_id}.{ext}")
                if os.path.exists(file_path):
                    return FileResponse(
                        file_path,
                        media_type=f"audio/{ext}",
                        filename=f"{file_id}.{ext}"
                    )
        
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))