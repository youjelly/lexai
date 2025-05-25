"""
Simple session management endpoints
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...websocket import session_manager
from ...utils.logging import get_logger
from ..middleware import validate_language_code

logger = get_logger(__name__)

router = APIRouter()


class SessionCreate(BaseModel):
    language: str = "en"
    voice_id: Optional[str] = None
    client_info: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    session_id: str
    created_at: float
    updated_at: float
    is_active: bool
    language: Optional[str]
    voice_id: Optional[str]
    duration_seconds: float
    message_count: int
    total_audio_seconds: float


class SessionUpdate(BaseModel):
    language: Optional[str] = None
    voice_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """Create new streaming session"""
    
    # Validate language
    if request.language:
        await validate_language_code(request.language)
    
    try:
        # Create session
        session = await session_manager.create_session(
            user_id=None,  # No auth in MVP
            client_info=request.client_info
        )
        
        # Update with initial settings
        await session_manager.update_session(
            session.session_id,
            language=request.language,
            voice_id=request.voice_id
        )
        
        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            is_active=session.is_active,
            language=session.language,
            voice_id=session.voice_id,
            duration_seconds=session.duration_seconds,
            message_count=session.message_count,
            total_audio_seconds=session.total_audio_seconds
        )
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session details"""
    
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        is_active=session.is_active,
        language=session.language,
        voice_id=session.voice_id,
        duration_seconds=session.duration_seconds,
        message_count=session.message_count,
        total_audio_seconds=session.total_audio_seconds
    )


@router.delete("/{session_id}")
async def end_session(session_id: str):
    """End streaming session"""
    
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    success = await session_manager.end_session(session_id)
    
    if success:
        return {"message": "Session ended successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to end session")


@router.put("/{session_id}/language")
async def update_session_language(session_id: str, language: str):
    """Change session language"""
    
    # Validate language
    await validate_language_code(language)
    
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    success = await session_manager.update_session(
        session_id,
        language=language
    )
    
    if success:
        return {
            "message": "Language updated successfully",
            "language": language
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to update language")


@router.patch("/{session_id}")
async def update_session(session_id: str, update: SessionUpdate):
    """Update session settings"""
    
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate language if provided
    if update.language:
        await validate_language_code(update.language)
    
    # Update session
    update_data = {}
    if update.language:
        update_data["language"] = update.language
    if update.voice_id:
        update_data["voice_id"] = update.voice_id
    if update.metadata:
        update_data.update(update.metadata)
    
    success = await session_manager.update_session(session_id, **update_data)
    
    if success:
        updated_session = await session_manager.get_session(session_id)
        return SessionResponse(
            session_id=updated_session.session_id,
            created_at=updated_session.created_at,
            updated_at=updated_session.updated_at,
            is_active=updated_session.is_active,
            language=updated_session.language,
            voice_id=updated_session.voice_id,
            duration_seconds=updated_session.duration_seconds,
            message_count=updated_session.message_count,
            total_audio_seconds=updated_session.total_audio_seconds
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to update session")


@router.get("")
async def list_sessions(
    limit: int = 20,
    active_only: bool = False
):
    """List all sessions"""
    
    try:
        # Get all sessions
        sessions = await session_manager.get_active_sessions()
        
        # Filter if needed
        if active_only:
            sessions = [s for s in sessions if s.is_active]
        
        # Sort by most recent
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        
        # Limit results
        sessions = sessions[:limit]
        
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "created_at": datetime.fromtimestamp(s.created_at).isoformat(),
                    "updated_at": datetime.fromtimestamp(s.updated_at).isoformat(),
                    "is_active": s.is_active,
                    "language": s.language,
                    "duration_seconds": s.duration_seconds,
                    "message_count": s.message_count
                }
                for s in sessions
            ],
            "total": len(sessions),
            "active_count": sum(1 for s in sessions if s.is_active)
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_session_history(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100
):
    """Get historical session data"""
    
    try:
        history = await session_manager.get_session_history(
            user_id=None,  # No auth in MVP
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        return {
            "sessions": history,
            "total": len(history),
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_session_statistics():
    """Get session statistics"""
    
    try:
        stats = session_manager.get_statistics()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/transcript")
async def add_transcript_to_session(
    session_id: str,
    text: str,
    duration_ms: float,
    language: Optional[str] = None
):
    """Add transcript segment to session (for testing)"""
    
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Add transcript segment
        await session_manager.add_transcript_segment(
            session_id,
            {
                "text": text,
                "duration_ms": duration_ms,
                "language": language or session.language
            }
        )
        
        return {"message": "Transcript added successfully"}
        
    except Exception as e:
        logger.error(f"Failed to add transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))