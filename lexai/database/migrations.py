import logging
from typing import List, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import OperationFailure

from .config import db_config
from .connection import get_database, get_sync_client

logger = logging.getLogger(__name__)


class Migration:
    """Base class for database migrations"""
    
    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
        self.applied_at = None
    
    async def up(self, db: AsyncIOMotorDatabase):
        """Apply the migration"""
        raise NotImplementedError
    
    async def down(self, db: AsyncIOMotorDatabase):
        """Rollback the migration"""
        raise NotImplementedError


class InitialSchemaMigration(Migration):
    """Create initial database schema"""
    
    def __init__(self):
        super().__init__("001", "Initial schema creation")
    
    async def up(self, db: AsyncIOMotorDatabase):
        # Create collections with validation
        await self._create_collections(db)
        
        # Set up capped collections for logs if needed
        await self._create_capped_collections(db)
        
        logger.info("Initial schema created")
    
    async def _create_collections(self, db: AsyncIOMotorDatabase):
        """Create all collections with schema validation"""
        
        # Sessions collection
        await db.create_collection(
            db_config.SESSIONS_COLLECTION,
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["user_id", "started_at"],
                    "properties": {
                        "user_id": {"bsonType": "string"},
                        "started_at": {"bsonType": "date"},
                        "is_active": {"bsonType": "bool"}
                    }
                }
            }
        )
        
        # Conversations collection
        await db.create_collection(
            db_config.CONVERSATIONS_COLLECTION,
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["user_id", "created_at"],
                    "properties": {
                        "user_id": {"bsonType": "string"},
                        "created_at": {"bsonType": "date"},
                        "is_archived": {"bsonType": "bool"}
                    }
                }
            }
        )
        
        # Messages collection
        await db.create_collection(
            db_config.MESSAGES_COLLECTION,
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["session_id", "conversation_id", "role", "created_at"],
                    "properties": {
                        "session_id": {"bsonType": "string"},
                        "conversation_id": {"bsonType": "string"},
                        "role": {"enum": ["user", "assistant", "system"]},
                        "created_at": {"bsonType": "date"}
                    }
                }
            }
        )
        
        # Audio files collection
        await db.create_collection(
            db_config.AUDIO_FILES_COLLECTION,
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["user_id", "file_path", "file_id", "created_at"],
                    "properties": {
                        "user_id": {"bsonType": "string"},
                        "file_path": {"bsonType": "string"},
                        "file_id": {"bsonType": "string"},
                        "status": {"enum": ["pending", "processing", "completed", "failed"]},
                        "created_at": {"bsonType": "date"}
                    }
                }
            }
        )
        
        # Voice profiles collection
        await db.create_collection(
            db_config.VOICE_PROFILES_COLLECTION,
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["user_id", "name", "created_at"],
                    "properties": {
                        "user_id": {"bsonType": "string"},
                        "name": {"bsonType": "string"},
                        "is_active": {"bsonType": "bool"},
                        "created_at": {"bsonType": "date"}
                    }
                }
            }
        )
        
        # User settings collection
        await db.create_collection(
            db_config.USER_SETTINGS_COLLECTION,
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["user_id"],
                    "properties": {
                        "user_id": {"bsonType": "string"},
                        "preferred_language": {"bsonType": "string"}
                    }
                }
            }
        )
        
        # Usage analytics collection
        await db.create_collection(
            db_config.USAGE_ANALYTICS_COLLECTION,
            validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["user_id", "date"],
                    "properties": {
                        "user_id": {"bsonType": "string"},
                        "date": {"bsonType": "date"},
                        "total_sessions": {"bsonType": "int", "minimum": 0},
                        "total_messages": {"bsonType": "int", "minimum": 0},
                        "total_audio_minutes": {"bsonType": "double", "minimum": 0}
                    }
                }
            }
        )
    
    async def _create_capped_collections(self, db: AsyncIOMotorDatabase):
        """Create capped collections for logs/events if needed"""
        try:
            # Create a capped collection for real-time events (optional)
            await db.create_collection(
                "events",
                capped=True,
                size=100 * 1024 * 1024,  # 100MB
                max=100000  # Max 100k documents
            )
        except OperationFailure:
            # Collection might already exist
            pass
    
    async def down(self, db: AsyncIOMotorDatabase):
        """Drop all collections"""
        collections = [
            db_config.SESSIONS_COLLECTION,
            db_config.CONVERSATIONS_COLLECTION,
            db_config.MESSAGES_COLLECTION,
            db_config.AUDIO_FILES_COLLECTION,
            db_config.VOICE_PROFILES_COLLECTION,
            db_config.USER_SETTINGS_COLLECTION,
            db_config.USAGE_ANALYTICS_COLLECTION,
            "events"
        ]
        
        for collection in collections:
            try:
                await db.drop_collection(collection)
            except:
                pass


class MigrationRunner:
    """Manages database migrations"""
    
    MIGRATIONS_COLLECTION = "migrations"
    
    def __init__(self):
        self.migrations: List[Migration] = [
            InitialSchemaMigration(),
            # Add new migrations here
        ]
    
    async def run(self):
        """Run all pending migrations"""
        db = get_database()
        
        # Ensure migrations collection exists
        await self._ensure_migrations_collection(db)
        
        # Get applied migrations
        applied = await self._get_applied_migrations(db)
        
        # Run pending migrations
        for migration in self.migrations:
            if migration.version not in applied:
                logger.info(f"Running migration {migration.version}: {migration.description}")
                
                try:
                    await migration.up(db)
                    await self._mark_migration_applied(db, migration)
                    logger.info(f"Migration {migration.version} completed")
                except Exception as e:
                    logger.error(f"Migration {migration.version} failed: {e}")
                    raise
    
    async def rollback(self, target_version: str = None):
        """Rollback migrations to target version"""
        db = get_database()
        
        # Get applied migrations in reverse order
        applied = await self._get_applied_migrations(db)
        
        for migration in reversed(self.migrations):
            if migration.version in applied:
                if target_version and migration.version == target_version:
                    break
                
                logger.info(f"Rolling back migration {migration.version}")
                
                try:
                    await migration.down(db)
                    await self._mark_migration_rolled_back(db, migration)
                    logger.info(f"Migration {migration.version} rolled back")
                except Exception as e:
                    logger.error(f"Rollback of migration {migration.version} failed: {e}")
                    raise
    
    async def _ensure_migrations_collection(self, db: AsyncIOMotorDatabase):
        """Ensure migrations tracking collection exists"""
        collections = await db.list_collection_names()
        
        if self.MIGRATIONS_COLLECTION not in collections:
            await db.create_collection(
                self.MIGRATIONS_COLLECTION,
                validator={
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["version", "description", "applied_at"],
                        "properties": {
                            "version": {"bsonType": "string"},
                            "description": {"bsonType": "string"},
                            "applied_at": {"bsonType": "date"}
                        }
                    }
                }
            )
            
            # Create unique index on version
            collection = db[self.MIGRATIONS_COLLECTION]
            await collection.create_index("version", unique=True)
    
    async def _get_applied_migrations(self, db: AsyncIOMotorDatabase) -> List[str]:
        """Get list of applied migration versions"""
        collection = db[self.MIGRATIONS_COLLECTION]
        
        applied = []
        async for doc in collection.find({}, {"version": 1}):
            applied.append(doc["version"])
        
        return applied
    
    async def _mark_migration_applied(self, db: AsyncIOMotorDatabase, migration: Migration):
        """Mark a migration as applied"""
        collection = db[self.MIGRATIONS_COLLECTION]
        
        await collection.insert_one({
            "version": migration.version,
            "description": migration.description,
            "applied_at": datetime.utcnow()
        })
    
    async def _mark_migration_rolled_back(self, db: AsyncIOMotorDatabase, migration: Migration):
        """Remove a migration from applied list"""
        collection = db[self.MIGRATIONS_COLLECTION]
        
        await collection.delete_one({"version": migration.version})


async def initialize_database():
    """Initialize database with migrations and indexes"""
    logger.info("Initializing database...")
    
    # Run migrations
    runner = MigrationRunner()
    await runner.run()
    
    # Create indexes (from indexing.py)
    from .indexing import create_indexes
    db = get_database()
    await create_indexes(db)
    
    logger.info("Database initialization complete")


async def reset_database(confirm: bool = False):
    """Reset database - WARNING: This will delete all data!"""
    if not confirm:
        raise ValueError("Must confirm database reset by passing confirm=True")
    
    logger.warning("Resetting database - all data will be lost!")
    
    db = get_database()
    
    # Drop all collections
    collections = await db.list_collection_names()
    for collection in collections:
        await db.drop_collection(collection)
        logger.info(f"Dropped collection: {collection}")
    
    # Re-initialize
    await initialize_database()
    
    logger.info("Database reset complete")