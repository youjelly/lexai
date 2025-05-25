from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from enum import Enum


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class MongoBaseModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        use_enum_values = True


# Enums
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    M4A = "m4a"
    FLAC = "flac"
    WEBM = "webm"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VoiceGender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


# Audio Metadata
class AudioMetadata(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    format: AudioFormat
    size_bytes: int
    bitrate: Optional[int] = None
    
    # Audio features for search/analysis
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    speaking_rate: Optional[float] = None  # words per minute


# Voice Profile
class VoiceProfile(MongoBaseModel):
    user_id: str
    name: str
    description: Optional[str] = None
    gender: Optional[VoiceGender] = None
    language: str = "en"
    
    # Voice characteristics
    sample_audio_ids: List[str] = Field(default_factory=list)  # Reference to audio files
    voice_embedding: Optional[List[float]] = None  # Vector representation for voice matching
    
    # TTS settings
    tts_model: Optional[str] = None
    tts_voice_id: Optional[str] = None
    pitch_shift: float = 0.0
    speed_scale: float = 1.0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Audio File
class AudioFile(MongoBaseModel):
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    user_id: str
    
    # File information
    file_path: str  # Relative path from audio base directory
    original_filename: Optional[str] = None
    file_id: str  # Unique identifier for API access
    
    # Audio metadata
    audio_metadata: AudioMetadata
    
    # Transcription
    transcription: Optional[str] = None
    transcription_confidence: Optional[float] = None
    language_detected: Optional[str] = None
    
    # Processing status
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    
    # Tags for organization
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Message with audio support
class Message(MongoBaseModel):
    session_id: str
    conversation_id: str
    role: MessageRole
    
    # Content
    text_content: Optional[str] = None
    audio_file_id: Optional[str] = None  # Reference to AudioFile
    
    # For assistant messages
    model_used: Optional[str] = None
    inference_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Session (real-time interaction)
class Session(MongoBaseModel):
    user_id: str
    conversation_id: Optional[str] = None
    
    # Session info
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Voice settings for this session
    voice_profile_id: Optional[str] = None
    input_language: str = "en"
    output_language: str = "en"
    
    # Performance metrics
    total_messages: int = 0
    total_audio_duration_seconds: float = 0.0
    average_response_time_ms: Optional[float] = None
    
    # Connection info
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Conversation (collection of messages)
class Conversation(MongoBaseModel):
    user_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    
    # Conversation settings
    default_voice_profile_id: Optional[str] = None
    language: str = "en"
    
    # Statistics
    message_count: int = 0
    total_duration_seconds: float = 0.0
    last_message_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Organization
    tags: List[str] = Field(default_factory=list)
    is_archived: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# User settings and preferences
class UserSettings(MongoBaseModel):
    user_id: str
    
    # Default preferences
    default_voice_profile_id: Optional[str] = None
    preferred_language: str = "en"
    preferred_tts_model: Optional[str] = None
    
    # Audio preferences
    auto_play_responses: bool = True
    echo_cancellation: bool = True
    noise_suppression: bool = True
    
    # Privacy settings
    store_audio_files: bool = True
    store_transcriptions: bool = True
    
    # Usage limits
    monthly_audio_minutes_limit: Optional[int] = None
    daily_message_limit: Optional[int] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Analytics and usage tracking
class UsageAnalytics(MongoBaseModel):
    user_id: str
    date: datetime  # Date for aggregation (day level)
    
    # Usage metrics
    total_sessions: int = 0
    total_messages: int = 0
    total_audio_minutes: float = 0.0
    total_tokens_used: int = 0
    
    # Performance metrics
    average_response_time_ms: float = 0.0
    error_count: int = 0
    
    # Feature usage
    voice_profiles_used: List[str] = Field(default_factory=list)
    languages_used: List[str] = Field(default_factory=list)
    models_used: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)