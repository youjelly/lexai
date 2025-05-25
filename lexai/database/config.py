from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class DatabaseConfig(BaseSettings):
    # Connection settings
    MONGODB_HOST: str = Field(default="localhost")
    MONGODB_PORT: int = Field(default=27017)
    MONGODB_DATABASE: str = Field(default="lexai")
    
    # Authentication
    MONGODB_USERNAME: Optional[str] = Field(default=None)
    MONGODB_PASSWORD: Optional[str] = Field(default=None)
    MONGODB_AUTH_SOURCE: str = Field(default="admin")
    
    # Connection pool settings
    MAX_POOL_SIZE: int = Field(default=100)
    MIN_POOL_SIZE: int = Field(default=10)
    MAX_IDLE_TIME: int = Field(default=60000)  # 60 seconds
    
    # Timeouts (in milliseconds)
    SERVER_SELECTION_TIMEOUT: int = Field(default=30000)  # 30 seconds
    CONNECT_TIMEOUT: int = Field(default=10000)  # 10 seconds
    SOCKET_TIMEOUT: int = Field(default=60000)  # 60 seconds
    
    # Write concern
    WRITE_CONCERN: str = Field(default="majority")
    RETRY_WRITES: bool = Field(default=True)
    RETRY_READS: bool = Field(default=True)
    
    # Collection names
    SESSIONS_COLLECTION: str = Field(default="sessions")
    CONVERSATIONS_COLLECTION: str = Field(default="conversations")
    MESSAGES_COLLECTION: str = Field(default="messages")
    AUDIO_FILES_COLLECTION: str = Field(default="audio_files")
    VOICE_PROFILES_COLLECTION: str = Field(default="voice_profiles")
    USER_SETTINGS_COLLECTION: str = Field(default="user_settings")
    USAGE_ANALYTICS_COLLECTION: str = Field(default="usage_analytics")
    
    # TTL settings (in seconds)
    SESSION_TTL: int = Field(default=86400)  # 24 hours
    TEMP_AUDIO_TTL: int = Field(default=3600)  # 1 hour
    
    # Query settings
    DEFAULT_PAGE_SIZE: int = Field(default=20)
    MAX_PAGE_SIZE: int = Field(default=100)
    
    # Full-text search settings
    TEXT_SEARCH_ENABLED: bool = Field(default=True)
    TEXT_SEARCH_LANGUAGE: str = Field(default="english")
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False


db_config = DatabaseConfig()