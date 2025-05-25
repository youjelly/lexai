import logging
from typing import Dict, List, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT

from .config import db_config

logger = logging.getLogger(__name__)


async def create_indexes(db: AsyncIOMotorDatabase):
    """Create all necessary indexes for optimal performance"""
    
    # Sessions indexes
    await create_session_indexes(db)
    
    # Conversations indexes
    await create_conversation_indexes(db)
    
    # Messages indexes
    await create_message_indexes(db)
    
    # Audio files indexes
    await create_audio_file_indexes(db)
    
    # Voice profiles indexes
    await create_voice_profile_indexes(db)
    
    # User settings indexes
    await create_user_settings_indexes(db)
    
    # Usage analytics indexes
    await create_usage_analytics_indexes(db)
    
    logger.info("All database indexes created successfully")


async def create_session_indexes(db: AsyncIOMotorDatabase):
    collection = db[db_config.SESSIONS_COLLECTION]
    
    indexes = [
        # Primary lookups
        IndexModel([("user_id", ASCENDING), ("is_active", ASCENDING)]),
        IndexModel([("conversation_id", ASCENDING)]),
        
        # Time-based queries
        IndexModel([("started_at", DESCENDING)]),
        IndexModel([("ended_at", DESCENDING)]),
        
        # Active sessions
        IndexModel([
            ("is_active", ASCENDING),
            ("started_at", DESCENDING)
        ]),
        
        # TTL index for automatic cleanup
        IndexModel(
            [("ended_at", ASCENDING)],
            expireAfterSeconds=db_config.SESSION_TTL
        )
    ]
    
    await collection.create_indexes(indexes)
    logger.info("Session indexes created")


async def create_conversation_indexes(db: AsyncIOMotorDatabase):
    collection = db[db_config.CONVERSATIONS_COLLECTION]
    
    indexes = [
        # Primary lookups
        IndexModel([("user_id", ASCENDING), ("is_archived", ASCENDING)]),
        IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)]),
        
        # Text search on title and summary
        IndexModel([("title", TEXT), ("summary", TEXT)]),
        
        # Tag searches
        IndexModel([("tags", ASCENDING)]),
        
        # Time-based queries
        IndexModel([("created_at", DESCENDING)]),
        IndexModel([("last_message_at", DESCENDING)]),
        
        # Voice profile lookups
        IndexModel([("default_voice_profile_id", ASCENDING)])
    ]
    
    await collection.create_indexes(indexes)
    logger.info("Conversation indexes created")


async def create_message_indexes(db: AsyncIOMotorDatabase):
    collection = db[db_config.MESSAGES_COLLECTION]
    
    indexes = [
        # Primary lookups
        IndexModel([("conversation_id", ASCENDING), ("created_at", ASCENDING)]),
        IndexModel([("session_id", ASCENDING)]),
        
        # User messages
        IndexModel([("conversation_id", ASCENDING), ("role", ASCENDING)]),
        
        # Audio file references
        IndexModel([("audio_file_id", ASCENDING)]),
        
        # Time-based queries
        IndexModel([("created_at", DESCENDING)]),
        
        # Model analytics
        IndexModel([("model_used", ASCENDING), ("created_at", DESCENDING)])
    ]
    
    await collection.create_indexes(indexes)
    logger.info("Message indexes created")


async def create_audio_file_indexes(db: AsyncIOMotorDatabase):
    collection = db[db_config.AUDIO_FILES_COLLECTION]
    
    indexes = [
        # Primary lookups
        IndexModel([("file_id", ASCENDING)], unique=True),
        IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)]),
        
        # Session/conversation lookups
        IndexModel([("session_id", ASCENDING)]),
        IndexModel([("conversation_id", ASCENDING)]),
        IndexModel([("message_id", ASCENDING)]),
        
        # Processing queue
        IndexModel([("status", ASCENDING), ("created_at", ASCENDING)]),
        
        # Text search on transcriptions
        IndexModel([("transcription", TEXT)]),
        
        # Tag searches
        IndexModel([("tags", ASCENDING)]),
        
        # Language detection
        IndexModel([("language_detected", ASCENDING)]),
        
        # Time-based queries
        IndexModel([("created_at", DESCENDING)]),
        IndexModel([("processed_at", DESCENDING)]),
        
        # Compound index for search queries
        IndexModel([
            ("user_id", ASCENDING),
            ("status", ASCENDING),
            ("created_at", DESCENDING)
        ])
    ]
    
    await collection.create_indexes(indexes)
    logger.info("Audio file indexes created")


async def create_voice_profile_indexes(db: AsyncIOMotorDatabase):
    collection = db[db_config.VOICE_PROFILES_COLLECTION]
    
    indexes = [
        # Primary lookups
        IndexModel([("user_id", ASCENDING), ("is_active", ASCENDING)]),
        IndexModel([("user_id", ASCENDING), ("name", ASCENDING)]),
        
        # Language lookups
        IndexModel([("language", ASCENDING)]),
        
        # TTS model lookups
        IndexModel([("tts_model", ASCENDING)]),
        
        # Time-based queries
        IndexModel([("created_at", DESCENDING)]),
        IndexModel([("updated_at", DESCENDING)]),
        
        # Sample audio references
        IndexModel([("sample_audio_ids", ASCENDING)])
    ]
    
    await collection.create_indexes(indexes)
    logger.info("Voice profile indexes created")


async def create_user_settings_indexes(db: AsyncIOMotorDatabase):
    collection = db[db_config.USER_SETTINGS_COLLECTION]
    
    indexes = [
        # Primary lookup - unique user_id
        IndexModel([("user_id", ASCENDING)], unique=True),
        
        # Language preferences
        IndexModel([("preferred_language", ASCENDING)]),
        
        # Voice profile references
        IndexModel([("default_voice_profile_id", ASCENDING)])
    ]
    
    await collection.create_indexes(indexes)
    logger.info("User settings indexes created")


async def create_usage_analytics_indexes(db: AsyncIOMotorDatabase):
    collection = db[db_config.USAGE_ANALYTICS_COLLECTION]
    
    indexes = [
        # Primary lookup - unique per user per day
        IndexModel(
            [("user_id", ASCENDING), ("date", ASCENDING)],
            unique=True
        ),
        
        # Time-based queries
        IndexModel([("date", DESCENDING)]),
        
        # Usage tracking
        IndexModel([("total_audio_minutes", DESCENDING)]),
        IndexModel([("total_tokens_used", DESCENDING)]),
        
        # Feature usage
        IndexModel([("models_used", ASCENDING)]),
        IndexModel([("languages_used", ASCENDING)]),
        
        # Compound index for date range queries
        IndexModel([
            ("user_id", ASCENDING),
            ("date", DESCENDING)
        ])
    ]
    
    await collection.create_indexes(indexes)
    logger.info("Usage analytics indexes created")


async def get_index_stats(db: AsyncIOMotorDatabase) -> Dict[str, List[Dict[str, Any]]]:
    """Get statistics about all indexes in the database"""
    stats = {}
    
    collections = [
        db_config.SESSIONS_COLLECTION,
        db_config.CONVERSATIONS_COLLECTION,
        db_config.MESSAGES_COLLECTION,
        db_config.AUDIO_FILES_COLLECTION,
        db_config.VOICE_PROFILES_COLLECTION,
        db_config.USER_SETTINGS_COLLECTION,
        db_config.USAGE_ANALYTICS_COLLECTION
    ]
    
    for collection_name in collections:
        collection = db[collection_name]
        
        # Get index information
        indexes = await collection.index_information()
        
        # Get index stats
        index_stats = []
        for index_name, index_info in indexes.items():
            stats_cmd = await db.command("collStats", collection_name, indexDetails=True)
            
            if "indexSizes" in stats_cmd:
                size = stats_cmd["indexSizes"].get(index_name, 0)
                index_stats.append({
                    "name": index_name,
                    "keys": index_info.get("key"),
                    "size_bytes": size,
                    "unique": index_info.get("unique", False),
                    "sparse": index_info.get("sparse", False)
                })
        
        stats[collection_name] = index_stats
    
    return stats


async def analyze_query_performance(
    db: AsyncIOMotorDatabase,
    collection_name: str,
    query: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze query performance using explain()"""
    collection = db[collection_name]
    
    # Run explain
    explain_result = await collection.find(query).explain()
    
    # Extract key metrics
    execution_stats = explain_result.get("executionStats", {})
    
    return {
        "execution_time_ms": execution_stats.get("executionTimeMillis", 0),
        "total_docs_examined": execution_stats.get("totalDocsExamined", 0),
        "total_keys_examined": execution_stats.get("totalKeysExamined", 0),
        "docs_returned": execution_stats.get("nReturned", 0),
        "index_used": explain_result.get("winningPlan", {}).get("inputStage", {}).get("indexName"),
        "stage": explain_result.get("winningPlan", {}).get("stage"),
        "is_multi_key": explain_result.get("winningPlan", {}).get("isMultiKey", False)
    }