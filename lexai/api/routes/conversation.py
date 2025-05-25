"""
Conversation endpoints for managing chat sessions
"""

import os
import json
import zipfile
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ...database import operations as db_ops
from ...websocket import connection_manager, session_manager
from ...utils.logging import get_logger
from config import settings

logger = get_logger(__name__)

router = APIRouter()
ws_router = APIRouter()


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    language: str = "en"
    voice_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int
    language: str
    metadata: Dict[str, Any]


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]


class ConversationExportRequest(BaseModel):
    format: str = "json"  # json, txt, or zip
    include_audio: bool = False


# WebSocket endpoint
@ws_router.websocket("/ws/audio/{session_id}")
async def websocket_audio_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time audio streaming"""
    await websocket.accept()
    
    try:
        # Handle connection through connection manager
        await connection_manager.handle_connection(websocket, f"/ws/audio/{session_id}")
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


@router.post("", response_model=ConversationResponse)
async def create_conversation(request: ConversationCreate):
    """Start a new conversation"""
    
    try:
        # Create conversation in database
        conversation = await db_ops.create_conversation(
            user_id=None,  # No auth in MVP
            metadata={
                "title": request.title,
                "language": request.language,
                "voice_id": request.voice_id,
                **(request.metadata or {})
            }
        )
        
        return ConversationResponse(
            id=str(conversation.id),
            title=conversation.metadata.get("title"),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=0,
            language=conversation.metadata.get("language", "en"),
            metadata=conversation.metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Get conversation details and history"""
    
    try:
        # Get conversation
        conversation = await db_ops.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get message count
        messages = await db_ops.get_conversation_messages(conversation_id, limit=1000)
        
        return ConversationResponse(
            id=str(conversation.id),
            title=conversation.metadata.get("title"),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=len(messages),
            language=conversation.metadata.get("language", "en"),
            metadata=conversation.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 100,
    offset: int = 0
):
    """Get conversation messages"""
    
    try:
        # Check if conversation exists
        conversation = await db_ops.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages
        messages = await db_ops.get_conversation_messages(
            conversation_id,
            limit=limit,
            skip=offset
        )
        
        return [
            MessageResponse(
                id=str(msg.id),
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                metadata=msg.metadata or {}
            )
            for msg in messages
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    
    try:
        # Check if conversation exists
        conversation = await db_ops.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete conversation (cascades to messages)
        await db_ops.delete_conversation(conversation_id)
        
        # Delete any associated audio files
        archive_path = Path(settings.AUDIO_CONVERSATIONS_PATH)
        for date_dir in archive_path.iterdir():
            if date_dir.is_dir():
                conv_dir = date_dir / conversation_id
                if conv_dir.exists():
                    import shutil
                    shutil.rmtree(conv_dir)
                    break
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_conversations(
    limit: int = 20,
    offset: int = 0,
    language: Optional[str] = None
):
    """List recent conversations"""
    
    try:
        # Get all conversations (no user filter in MVP)
        conversations = await db_ops.get_user_conversations(
            user_id=None,
            limit=limit,
            skip=offset
        )
        
        # Filter by language if specified
        if language:
            conversations = [
                c for c in conversations
                if c.metadata.get("language") == language
            ]
        
        # Format response
        return {
            "conversations": [
                {
                    "id": str(conv.id),
                    "title": conv.metadata.get("title"),
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at,
                    "language": conv.metadata.get("language", "en"),
                    "preview": conv.metadata.get("preview", "")
                }
                for conv in conversations
            ],
            "total": len(conversations),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    request: ConversationExportRequest
):
    """Export conversation in various formats"""
    
    try:
        # Get conversation and messages
        conversation = await db_ops.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = await db_ops.get_conversation_messages(conversation_id, limit=10000)
        
        # Create export based on format
        if request.format == "json":
            # JSON export
            export_data = {
                "conversation": {
                    "id": str(conversation.id),
                    "title": conversation.metadata.get("title"),
                    "created_at": conversation.created_at.isoformat(),
                    "language": conversation.metadata.get("language", "en"),
                    "metadata": conversation.metadata
                },
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata or {}
                    }
                    for msg in messages
                ]
            }
            
            return StreamingResponse(
                content=json.dumps(export_data, indent=2),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=conversation_{conversation_id}.json"
                }
            )
            
        elif request.format == "txt":
            # Text export
            text_lines = [
                f"Conversation: {conversation.metadata.get('title', 'Untitled')}",
                f"Date: {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Language: {conversation.metadata.get('language', 'en')}",
                "",
                "=" * 50,
                ""
            ]
            
            for msg in messages:
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                role = msg.role.upper()
                text_lines.append(f"[{timestamp}] {role}: {msg.content}")
                text_lines.append("")
            
            return StreamingResponse(
                content="\n".join(text_lines),
                media_type="text/plain",
                headers={
                    "Content-Disposition": f"attachment; filename=conversation_{conversation_id}.txt"
                }
            )
            
        elif request.format == "zip":
            # ZIP export with optional audio
            import tempfile
            import io
            
            # Create zip in memory
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add JSON data
                export_data = {
                    "conversation": {
                        "id": str(conversation.id),
                        "title": conversation.metadata.get("title"),
                        "created_at": conversation.created_at.isoformat(),
                        "language": conversation.metadata.get("language", "en"),
                        "metadata": conversation.metadata
                    },
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat(),
                            "metadata": msg.metadata or {}
                        }
                        for msg in messages
                    ]
                }
                
                zf.writestr("conversation.json", json.dumps(export_data, indent=2))
                
                # Add text transcript
                text_lines = []
                for msg in messages:
                    timestamp = msg.timestamp.strftime("%H:%M:%S")
                    role = msg.role.upper()
                    text_lines.append(f"[{timestamp}] {role}: {msg.content}")
                
                zf.writestr("transcript.txt", "\n".join(text_lines))
                
                # Add audio files if requested
                if request.include_audio:
                    # Look for archived audio
                    archive_path = Path(settings.AUDIO_CONVERSATIONS_PATH)
                    for date_dir in archive_path.iterdir():
                        if date_dir.is_dir():
                            conv_dir = date_dir / conversation_id
                            if conv_dir.exists():
                                for audio_file in conv_dir.glob("*.wav"):
                                    zf.write(audio_file, f"audio/{audio_file.name}")
                                break
            
            zip_buffer.seek(0)
            
            return StreamingResponse(
                content=zip_buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename=conversation_{conversation_id}.zip"
                }
            )
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid export format: {request.format}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))