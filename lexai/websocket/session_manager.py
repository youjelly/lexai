"""
Session management for WebSocket audio streaming
Handles session lifecycle, persistence, and conversation archiving
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
import shutil

from ..database import operations as db_ops
from ..utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass
class SessionInfo:
    """Information about a streaming session"""
    session_id: str
    user_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Session metadata
    language: Optional[str] = None
    voice_id: Optional[str] = None
    client_info: Dict[str, Any] = field(default_factory=dict)
    
    # Session state
    is_active: bool = True
    duration_seconds: float = 0.0
    message_count: int = 0
    
    # Audio info
    total_audio_seconds: float = 0.0
    input_audio_seconds: float = 0.0
    output_audio_seconds: float = 0.0
    
    # Conversation
    conversation_id: Optional[str] = None
    transcript_segments: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            **asdict(self),
            "created_at_iso": datetime.fromtimestamp(self.created_at).isoformat(),
            "updated_at_iso": datetime.fromtimestamp(self.updated_at).isoformat()
        }


class SessionManager:
    """Manages WebSocket streaming sessions"""
    
    def __init__(self):
        # Paths
        self.sessions_path = Path(settings.AUDIO_SESSIONS_PATH)
        self.conversations_path = Path(settings.AUDIO_CONVERSATIONS_PATH)
        self.temp_path = Path(settings.TEMP_FILES_PATH) / "sessions"
        
        # Active sessions
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.session_handlers: Dict[str, Any] = {}  # Maps to AudioStreamer instances
        
        # Session cleanup
        self.cleanup_interval = 300  # 5 minutes
        self.session_timeout = 3600  # 1 hour
        self.cleanup_task = None
        
        # Initialize paths
        self._init_paths()
    
    def _init_paths(self):
        """Initialize required directories"""
        for path in [self.sessions_path, self.conversations_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize session manager"""
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Load any persisted sessions
        await self._load_persisted_sessions()
        
        logger.info("Session manager initialized")
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        client_info: Optional[Dict[str, Any]] = None
    ) -> SessionInfo:
        """Create a new streaming session"""
        session_id = str(uuid.uuid4())
        
        # Create session info
        session = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            client_info=client_info or {}
        )
        
        # Create conversation in database
        try:
            conversation = await db_ops.create_conversation(
                user_id=user_id,
                metadata={
                    "session_id": session_id,
                    "client_info": client_info,
                    "type": "websocket_stream"
                }
            )
            session.conversation_id = str(conversation.id)
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
        
        # Store session
        self.active_sessions[session_id] = session
        
        # Create session directory
        session_dir = self.sessions_path / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Save initial session info
        await self._save_session(session)
        
        logger.info(f"Created session {session_id} for user {user_id}")
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID"""
        return self.active_sessions.get(session_id)
    
    async def update_session(
        self,
        session_id: str,
        language: Optional[str] = None,
        voice_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Update session information"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        # Update fields
        if language:
            session.language = language
        if voice_id:
            session.voice_id = voice_id
        
        # Update metadata
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        session.updated_at = time.time()
        
        # Save updates
        await self._save_session(session)
        
        return True
    
    async def add_transcript_segment(
        self,
        session_id: str,
        segment: Dict[str, Any]
    ):
        """Add transcript segment to session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        # Add segment
        segment["index"] = len(session.transcript_segments)
        segment["timestamp"] = time.time()
        session.transcript_segments.append(segment)
        
        # Update stats
        session.message_count += 1
        if "duration_ms" in segment:
            session.input_audio_seconds += segment["duration_ms"] / 1000
            session.total_audio_seconds += segment["duration_ms"] / 1000
        
        session.updated_at = time.time()
        
        # Save to database if we have conversation ID
        if session.conversation_id:
            try:
                await db_ops.add_message(
                    conversation_id=session.conversation_id,
                    role="user",
                    content=segment.get("text", ""),
                    metadata=segment
                )
            except Exception as e:
                logger.error(f"Failed to save message: {e}")
    
    async def add_response_segment(
        self,
        session_id: str,
        response: Dict[str, Any]
    ):
        """Add response segment to session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        # Update stats
        if "duration_ms" in response:
            session.output_audio_seconds += response["duration_ms"] / 1000
            session.total_audio_seconds += response["duration_ms"] / 1000
        
        session.updated_at = time.time()
        
        # Save to database
        if session.conversation_id:
            try:
                await db_ops.add_message(
                    conversation_id=session.conversation_id,
                    role="assistant",
                    content=response.get("text", ""),
                    metadata=response
                )
            except Exception as e:
                logger.error(f"Failed to save response: {e}")
    
    async def end_session(self, session_id: str) -> bool:
        """End a streaming session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        # Calculate duration
        session.duration_seconds = time.time() - session.created_at
        session.is_active = False
        session.updated_at = time.time()
        
        # Archive session
        await self._archive_session(session)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Remove handler if exists
        if session_id in self.session_handlers:
            del self.session_handlers[session_id]
        
        logger.info(f"Ended session {session_id}, duration: {session.duration_seconds:.1f}s")
        
        return True
    
    async def _save_session(self, session: SessionInfo):
        """Save session to disk"""
        session_file = self.sessions_path / session.session_id / "session_info.json"
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
    
    async def _archive_session(self, session: SessionInfo):
        """Archive completed session"""
        try:
            # Create archive directory
            archive_date = datetime.fromtimestamp(session.created_at).strftime("%Y-%m-%d")
            archive_dir = self.conversations_path / archive_date / session.session_id
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Save full session data
            archive_file = archive_dir / "session_data.json"
            with open(archive_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            
            # Copy any audio files
            session_dir = self.sessions_path / session.session_id
            if session_dir.exists():
                for audio_file in session_dir.glob("*.wav"):
                    shutil.copy2(audio_file, archive_dir)
            
            # Create summary
            summary = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "duration_seconds": session.duration_seconds,
                "message_count": session.message_count,
                "total_audio_seconds": session.total_audio_seconds,
                "language": session.language,
                "created_at": session.created_at,
                "transcript_preview": session.transcript_segments[:5] if session.transcript_segments else []
            }
            
            summary_file = archive_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Clean up session directory
            if session_dir.exists():
                shutil.rmtree(session_dir)
            
            logger.info(f"Archived session {session.session_id} to {archive_dir}")
            
        except Exception as e:
            logger.error(f"Failed to archive session {session.session_id}: {e}")
    
    async def _load_persisted_sessions(self):
        """Load any persisted sessions from disk"""
        try:
            for session_dir in self.sessions_path.iterdir():
                if session_dir.is_dir():
                    session_file = session_dir / "session_info.json"
                    if session_file.exists():
                        with open(session_file, 'r') as f:
                            data = json.load(f)
                        
                        # Check if session is recent
                        if time.time() - data.get("updated_at", 0) < self.session_timeout:
                            # Recreate session
                            session = SessionInfo(**{
                                k: v for k, v in data.items()
                                if k in SessionInfo.__dataclass_fields__
                            })
                            
                            # Mark as inactive (needs to be reactivated)
                            session.is_active = False
                            
                            self.active_sessions[session.session_id] = session
                            logger.info(f"Loaded persisted session {session.session_id}")
                        else:
                            # Archive old session
                            session = SessionInfo(**{
                                k: v for k, v in data.items()
                                if k in SessionInfo.__dataclass_fields__
                            })
                            await self._archive_session(session)
                            
        except Exception as e:
            logger.error(f"Failed to load persisted sessions: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of inactive sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Find inactive sessions
                now = time.time()
                sessions_to_remove = []
                
                for session_id, session in self.active_sessions.items():
                    # Check timeout
                    if now - session.updated_at > self.session_timeout:
                        sessions_to_remove.append(session_id)
                    # Check inactive sessions
                    elif not session.is_active and now - session.updated_at > 600:  # 10 minutes
                        sessions_to_remove.append(session_id)
                
                # Remove inactive sessions
                for session_id in sessions_to_remove:
                    logger.info(f"Cleaning up inactive session {session_id}")
                    await self.end_session(session_id)
                
                # Clean up old temp files
                for temp_file in self.temp_path.iterdir():
                    if temp_file.is_file():
                        file_age = now - temp_file.stat().st_mtime
                        if file_age > 3600:  # 1 hour
                            temp_file.unlink()
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def register_handler(self, session_id: str, handler: Any):
        """Register a session handler (AudioStreamer)"""
        self.session_handlers[session_id] = handler
    
    def get_handler(self, session_id: str) -> Optional[Any]:
        """Get session handler"""
        return self.session_handlers.get(session_id)
    
    async def get_active_sessions(self, user_id: Optional[str] = None) -> List[SessionInfo]:
        """Get active sessions, optionally filtered by user"""
        sessions = list(self.active_sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        
        # Sort by most recent
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        
        return sessions
    
    async def get_session_history(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical sessions"""
        sessions = []
        
        try:
            # Search archive directories
            for date_dir in sorted(self.conversations_path.iterdir(), reverse=True):
                if not date_dir.is_dir():
                    continue
                
                # Check date range
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if start_date and dir_date < start_date:
                    break
                if end_date and dir_date > end_date:
                    continue
                
                # Check sessions in this date
                for session_dir in date_dir.iterdir():
                    if not session_dir.is_dir():
                        continue
                    
                    summary_file = session_dir / "summary.json"
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                        
                        # Filter by user if specified
                        if user_id and summary.get("user_id") != user_id:
                            continue
                        
                        sessions.append(summary)
                        
                        if len(sessions) >= limit:
                            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
        
        return sessions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        active_count = len(self.active_sessions)
        active_users = len(set(s.user_id for s in self.active_sessions.values() if s.user_id))
        
        total_duration = sum(s.duration_seconds for s in self.active_sessions.values())
        total_audio = sum(s.total_audio_seconds for s in self.active_sessions.values())
        
        return {
            "active_sessions": active_count,
            "active_users": active_users,
            "total_duration_seconds": total_duration,
            "total_audio_seconds": total_audio,
            "handlers_registered": len(self.session_handlers),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "user_id": s.user_id,
                    "duration": s.duration_seconds,
                    "messages": s.message_count,
                    "is_active": s.is_active
                }
                for s in list(self.active_sessions.values())[:10]  # Top 10
            ]
        }
    
    async def cleanup(self):
        """Clean up session manager"""
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # End all active sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.end_session(session_id)
        
        logger.info("Session manager cleanup complete")


# Global session manager instance
session_manager = SessionManager()