from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from bson import ObjectId
import logging

from .connection import get_database
from .config import db_config
from .models import (
    Session, Conversation, Message, AudioFile, 
    VoiceProfile, UserSettings, UsageAnalytics,
    ProcessingStatus, MessageRole
)

logger = logging.getLogger(__name__)


class DatabaseOperations:
    def __init__(self):
        self.db = None
    
    async def initialize(self):
        self.db = get_database()
    
    # Session Operations
    async def create_session(self, session: Session) -> str:
        collection = self.db[db_config.SESSIONS_COLLECTION]
        result = await collection.insert_one(session.dict(by_alias=True))
        return str(result.inserted_id)
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        collection = self.db[db_config.SESSIONS_COLLECTION]
        doc = await collection.find_one({"_id": ObjectId(session_id)})
        return Session(**doc) if doc else None
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        collection = self.db[db_config.SESSIONS_COLLECTION]
        updates["updated_at"] = datetime.utcnow()
        result = await collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    async def end_session(self, session_id: str) -> bool:
        session = await self.get_session(session_id)
        if not session:
            return False
        
        ended_at = datetime.utcnow()
        duration = (ended_at - session.started_at).total_seconds()
        
        return await self.update_session(session_id, {
            "ended_at": ended_at,
            "duration_seconds": duration,
            "is_active": False
        })
    
    async def get_active_sessions(self, user_id: Optional[str] = None) -> List[Session]:
        collection = self.db[db_config.SESSIONS_COLLECTION]
        query = {"is_active": True}
        if user_id:
            query["user_id"] = user_id
        
        cursor = collection.find(query).sort("started_at", -1)
        sessions = []
        async for doc in cursor:
            sessions.append(Session(**doc))
        return sessions
    
    # Conversation Operations
    async def create_conversation(self, conversation: Conversation) -> str:
        collection = self.db[db_config.CONVERSATIONS_COLLECTION]
        result = await collection.insert_one(conversation.dict(by_alias=True))
        return str(result.inserted_id)
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        collection = self.db[db_config.CONVERSATIONS_COLLECTION]
        doc = await collection.find_one({"_id": ObjectId(conversation_id)})
        return Conversation(**doc) if doc else None
    
    async def update_conversation(self, conversation_id: str, updates: Dict[str, Any]) -> bool:
        collection = self.db[db_config.CONVERSATIONS_COLLECTION]
        updates["updated_at"] = datetime.utcnow()
        result = await collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    async def get_user_conversations(
        self, 
        user_id: str, 
        skip: int = 0, 
        limit: int = None,
        include_archived: bool = False
    ) -> Tuple[List[Conversation], int]:
        collection = self.db[db_config.CONVERSATIONS_COLLECTION]
        query = {"user_id": user_id}
        if not include_archived:
            query["is_archived"] = False
        
        limit = limit or db_config.DEFAULT_PAGE_SIZE
        limit = min(limit, db_config.MAX_PAGE_SIZE)
        
        # Get total count
        total = await collection.count_documents(query)
        
        # Get paginated results
        cursor = collection.find(query).sort("updated_at", -1).skip(skip).limit(limit)
        conversations = []
        async for doc in cursor:
            conversations.append(Conversation(**doc))
        
        return conversations, total
    
    # Message Operations
    async def create_message(self, message: Message) -> str:
        collection = self.db[db_config.MESSAGES_COLLECTION]
        result = await collection.insert_one(message.dict(by_alias=True))
        
        # Update conversation stats
        await self.db[db_config.CONVERSATIONS_COLLECTION].update_one(
            {"_id": ObjectId(message.conversation_id)},
            {
                "$inc": {"message_count": 1},
                "$set": {"last_message_at": message.created_at}
            }
        )
        
        return str(result.inserted_id)
    
    async def get_conversation_messages(
        self, 
        conversation_id: str,
        skip: int = 0,
        limit: int = None
    ) -> List[Message]:
        collection = self.db[db_config.MESSAGES_COLLECTION]
        limit = limit or db_config.DEFAULT_PAGE_SIZE
        limit = min(limit, db_config.MAX_PAGE_SIZE)
        
        cursor = collection.find(
            {"conversation_id": conversation_id}
        ).sort("created_at", 1).skip(skip).limit(limit)
        
        messages = []
        async for doc in cursor:
            messages.append(Message(**doc))
        return messages
    
    # Audio File Operations
    async def create_audio_file(self, audio_file: AudioFile) -> str:
        collection = self.db[db_config.AUDIO_FILES_COLLECTION]
        result = await collection.insert_one(audio_file.dict(by_alias=True))
        return str(result.inserted_id)
    
    async def get_audio_file(self, audio_id: str) -> Optional[AudioFile]:
        collection = self.db[db_config.AUDIO_FILES_COLLECTION]
        
        # Try by ObjectId first
        try:
            doc = await collection.find_one({"_id": ObjectId(audio_id)})
            if doc:
                return AudioFile(**doc)
        except:
            pass
        
        # Try by file_id
        doc = await collection.find_one({"file_id": audio_id})
        return AudioFile(**doc) if doc else None
    
    async def update_audio_file(self, audio_id: str, updates: Dict[str, Any]) -> bool:
        collection = self.db[db_config.AUDIO_FILES_COLLECTION]
        result = await collection.update_one(
            {"_id": ObjectId(audio_id)},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    async def get_pending_audio_files(self, limit: int = 10) -> List[AudioFile]:
        collection = self.db[db_config.AUDIO_FILES_COLLECTION]
        cursor = collection.find(
            {"status": ProcessingStatus.PENDING.value}
        ).sort("created_at", 1).limit(limit)
        
        files = []
        async for doc in cursor:
            files.append(AudioFile(**doc))
        return files
    
    # Voice Profile Operations
    async def create_voice_profile(self, profile: VoiceProfile) -> str:
        collection = self.db[db_config.VOICE_PROFILES_COLLECTION]
        result = await collection.insert_one(profile.dict(by_alias=True))
        return str(result.inserted_id)
    
    async def get_voice_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        collection = self.db[db_config.VOICE_PROFILES_COLLECTION]
        doc = await collection.find_one({"_id": ObjectId(profile_id)})
        return VoiceProfile(**doc) if doc else None
    
    async def get_user_voice_profiles(self, user_id: str) -> List[VoiceProfile]:
        collection = self.db[db_config.VOICE_PROFILES_COLLECTION]
        cursor = collection.find(
            {"user_id": user_id, "is_active": True}
        ).sort("created_at", -1)
        
        profiles = []
        async for doc in cursor:
            profiles.append(VoiceProfile(**doc))
        return profiles
    
    async def update_voice_profile(self, profile_id: str, updates: Dict[str, Any]) -> bool:
        collection = self.db[db_config.VOICE_PROFILES_COLLECTION]
        updates["updated_at"] = datetime.utcnow()
        result = await collection.update_one(
            {"_id": ObjectId(profile_id)},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    # User Settings Operations
    async def get_user_settings(self, user_id: str) -> Optional[UserSettings]:
        collection = self.db[db_config.USER_SETTINGS_COLLECTION]
        doc = await collection.find_one({"user_id": user_id})
        
        if not doc:
            # Create default settings
            settings = UserSettings(user_id=user_id)
            await collection.insert_one(settings.dict(by_alias=True))
            return settings
        
        return UserSettings(**doc)
    
    async def update_user_settings(self, user_id: str, updates: Dict[str, Any]) -> bool:
        collection = self.db[db_config.USER_SETTINGS_COLLECTION]
        updates["updated_at"] = datetime.utcnow()
        result = await collection.update_one(
            {"user_id": user_id},
            {"$set": updates},
            upsert=True
        )
        return result.modified_count > 0 or result.upserted_id is not None
    
    # Analytics Operations
    async def record_usage(self, analytics: UsageAnalytics) -> str:
        collection = self.db[db_config.USAGE_ANALYTICS_COLLECTION]
        
        # Aggregate for the day
        start_of_day = analytics.date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        result = await collection.update_one(
            {
                "user_id": analytics.user_id,
                "date": start_of_day
            },
            {
                "$inc": {
                    "total_sessions": analytics.total_sessions,
                    "total_messages": analytics.total_messages,
                    "total_audio_minutes": analytics.total_audio_minutes,
                    "total_tokens_used": analytics.total_tokens_used,
                    "error_count": analytics.error_count
                },
                "$addToSet": {
                    "voice_profiles_used": {"$each": analytics.voice_profiles_used},
                    "languages_used": {"$each": analytics.languages_used},
                    "models_used": {"$each": analytics.models_used}
                }
            },
            upsert=True
        )
        
        return str(result.upserted_id) if result.upserted_id else "updated"
    
    async def get_user_usage(
        self, 
        user_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[UsageAnalytics]:
        collection = self.db[db_config.USAGE_ANALYTICS_COLLECTION]
        cursor = collection.find({
            "user_id": user_id,
            "date": {
                "$gte": start_date,
                "$lte": end_date
            }
        }).sort("date", 1)
        
        usage = []
        async for doc in cursor:
            usage.append(UsageAnalytics(**doc))
        return usage
    
    # Search Operations
    async def search_conversations(
        self, 
        user_id: str, 
        query: str,
        skip: int = 0,
        limit: int = None
    ) -> Tuple[List[Conversation], int]:
        collection = self.db[db_config.CONVERSATIONS_COLLECTION]
        limit = limit or db_config.DEFAULT_PAGE_SIZE
        limit = min(limit, db_config.MAX_PAGE_SIZE)
        
        search_query = {
            "user_id": user_id,
            "$text": {"$search": query}
        }
        
        # Get total count
        total = await collection.count_documents(search_query)
        
        # Get paginated results with text score
        cursor = collection.find(
            search_query,
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).skip(skip).limit(limit)
        
        conversations = []
        async for doc in cursor:
            conversations.append(Conversation(**doc))
        
        return conversations, total
    
    async def search_audio_files(
        self,
        user_id: str,
        transcription_query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = None
    ) -> Tuple[List[AudioFile], int]:
        collection = self.db[db_config.AUDIO_FILES_COLLECTION]
        limit = limit or db_config.DEFAULT_PAGE_SIZE
        limit = min(limit, db_config.MAX_PAGE_SIZE)
        
        query = {"user_id": user_id}
        
        if transcription_query:
            query["$text"] = {"$search": transcription_query}
        
        if tags:
            query["tags"] = {"$in": tags}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = start_date
            if end_date:
                date_query["$lte"] = end_date
            query["created_at"] = date_query
        
        # Get total count
        total = await collection.count_documents(query)
        
        # Get paginated results
        cursor = collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
        
        files = []
        async for doc in cursor:
            files.append(AudioFile(**doc))
        
        return files, total
    
    # Cleanup Operations
    async def cleanup_old_sessions(self, older_than_hours: int = 24) -> int:
        collection = self.db[db_config.SESSIONS_COLLECTION]
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        result = await collection.delete_many({
            "is_active": False,
            "ended_at": {"$lt": cutoff_time}
        })
        
        return result.deleted_count
    
    async def cleanup_temp_audio(self, older_than_hours: int = 1) -> int:
        collection = self.db[db_config.AUDIO_FILES_COLLECTION]
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        # Find temp audio files
        cursor = collection.find({
            "file_path": {"$regex": "^temp/"},
            "created_at": {"$lt": cutoff_time}
        })
        
        deleted_count = 0
        async for doc in cursor:
            # Delete from database
            await collection.delete_one({"_id": doc["_id"]})
            deleted_count += 1
            
            # TODO: Also delete the actual file from storage
        
        return deleted_count


# Singleton instance
db_ops = DatabaseOperations()