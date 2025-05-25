import motor.motor_asyncio
from typing import Optional, Dict, Any
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from config import settings
from .config import db_config

logger = logging.getLogger(__name__)

client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
database: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None
sync_client: Optional[MongoClient] = None


async def init_db():
    global client, database
    try:
        # Build connection URI with authentication if configured
        if db_config.MONGODB_USERNAME and db_config.MONGODB_PASSWORD:
            uri = f"mongodb://{db_config.MONGODB_USERNAME}:{db_config.MONGODB_PASSWORD}@{db_config.MONGODB_HOST}:{db_config.MONGODB_PORT}/?authSource={db_config.MONGODB_AUTH_SOURCE}"
        else:
            uri = f"mongodb://{db_config.MONGODB_HOST}:{db_config.MONGODB_PORT}/"
        
        # Connection options
        options = {
            'serverSelectionTimeoutMS': db_config.SERVER_SELECTION_TIMEOUT,
            'connectTimeoutMS': db_config.CONNECT_TIMEOUT,
            'maxPoolSize': db_config.MAX_POOL_SIZE,
            'minPoolSize': db_config.MIN_POOL_SIZE,
            'maxIdleTimeMS': db_config.MAX_IDLE_TIME,
            'retryWrites': db_config.RETRY_WRITES,
            'retryReads': db_config.RETRY_READS,
            'w': db_config.WRITE_CONCERN,
            'journal': True,
            'readPreference': 'primaryPreferred'
        }
        
        client = motor.motor_asyncio.AsyncIOMotorClient(uri, **options)
        database = client[db_config.MONGODB_DATABASE]
        
        # Test connection
        await client.admin.command('ping')
        logger.info(f"Connected to MongoDB at {db_config.MONGODB_HOST}:{db_config.MONGODB_PORT}")
        
        # Get server info
        server_info = await client.server_info()
        logger.info(f"MongoDB version: {server_info.get('version', 'unknown')}")
        
        # Initialize indexes
        from .indexing import create_indexes
        await create_indexes(database)
        
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error initializing MongoDB: {e}")
        raise


async def close_db():
    global client
    if client:
        client.close()
        logger.info("Disconnected from MongoDB")


def get_database() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    if database is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return database


def get_client() -> motor.motor_asyncio.AsyncIOMotorClient:
    if client is None:
        raise RuntimeError("Database client not initialized. Call init_db() first.")
    return client


def get_sync_client() -> MongoClient:
    """Get synchronous MongoDB client for operations that require sync access"""
    global sync_client
    if sync_client is None:
        if db_config.MONGODB_USERNAME and db_config.MONGODB_PASSWORD:
            uri = f"mongodb://{db_config.MONGODB_USERNAME}:{db_config.MONGODB_PASSWORD}@{db_config.MONGODB_HOST}:{db_config.MONGODB_PORT}/?authSource={db_config.MONGODB_AUTH_SOURCE}"
        else:
            uri = f"mongodb://{db_config.MONGODB_HOST}:{db_config.MONGODB_PORT}/"
        
        sync_client = MongoClient(uri)
    return sync_client


async def check_connection() -> Dict[str, Any]:
    """Check MongoDB connection status and return diagnostics"""
    try:
        db = get_database()
        result = await client.admin.command('ping')
        
        # Get database stats
        stats = await db.command('dbStats')
        
        # Get collection info
        collections = await db.list_collection_names()
        
        return {
            'connected': True,
            'database': db_config.MONGODB_DATABASE,
            'collections': collections,
            'storage_size': stats.get('storageSize', 0),
            'data_size': stats.get('dataSize', 0),
            'index_size': stats.get('indexSize', 0),
            'collections_count': len(collections)
        }
    except Exception as e:
        return {
            'connected': False,
            'error': str(e)
        }