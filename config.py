import os
import torch
import psutil
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pathlib import Path
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: Environment = Field(default=Environment.PRODUCTION)
    
    # MongoDB Configuration
    MONGODB_URI: str = Field(default="mongodb://localhost:27017/")
    MONGODB_DATABASE: str = Field(default="lexai")
    MONGODB_DATA_PATH: str = Field(default="/mnt/storage/mongodb")
    
    # Persistent Storage Paths
    MODEL_BASE_PATH: str = Field(default="/mnt/storage/models")
    ULTRAVOX_MODEL_PATH: str = Field(default="/mnt/storage/models/ultravox")
    TTS_MODEL_PATH: str = Field(default="/mnt/storage/models/tts")
    VOICE_FILES_PATH: str = Field(default="/mnt/storage/voices")
    AUDIO_CONVERSATIONS_PATH: str = Field(default="/mnt/storage/audio/conversations")
    AUDIO_SESSIONS_PATH: str = Field(default="/mnt/storage/audio/sessions")
    VOICE_SAMPLES_PATH: str = Field(default="/mnt/storage/audio/voice_samples")
    
    # Fast Ephemeral Storage Paths (NVMe)
    AUDIO_PROCESSING_PATH: str = Field(default="/opt/dlami/nvme/audio_processing")
    CACHE_BASE_PATH: str = Field(default="/opt/dlami/nvme/cache")
    ULTRAVOX_CACHE_PATH: str = Field(default="/opt/dlami/nvme/cache/ultravox")
    TTS_CACHE_PATH: str = Field(default="/opt/dlami/nvme/cache/tts")
    MONGODB_TEMP_PATH: str = Field(default="/opt/dlami/nvme/cache/mongodb_temp")
    TEMP_FILES_PATH: str = Field(default="/opt/dlami/nvme/temp")
    
    # Logging Configuration
    LOG_BASE_PATH: str = Field(default="/mnt/storage/logs")
    MONGODB_LOG_PATH: str = Field(default="/mnt/storage/logs/mongodb")
    API_LOG_PATH: str = Field(default="/mnt/storage/logs/api")
    APP_LOG_PATH: str = Field(default="/mnt/storage/logs/app")
    LOG_LEVEL: str = Field(default="INFO")
    LOG_MAX_SIZE_MB: int = Field(default=100)
    LOG_BACKUP_COUNT: int = Field(default=10)
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # API Configuration - External Access
    HOST: str = Field(default="0.0.0.0")  # Bind to all interfaces for external access
    PORT: int = Field(default=8000)
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    WEBSOCKET_PORT: int = Field(default=8001)
    API_WORKERS: int = Field(default=1)  # Single worker for GPU model loading
    API_RELOAD: bool = Field(default=False)
    API_ACCESS_LOG: bool = Field(default=True)
    
    # External Access Configuration
    ALLOWED_HOSTS: List[str] = Field(default=["*"])  # Allow all hosts for external access
    EXTERNAL_HOST: Optional[str] = Field(default=None)  # EC2 public IP/domain
    
    # CORS Settings - External Access
    CORS_ORIGINS: List[str] = Field(default=["*"])  # Allow all origins for external access
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])
    
    # Security Headers for External Access
    SECURITY_HEADERS: bool = Field(default=True)
    
    # Static Files Configuration
    STATIC_FILES_PATH: str = Field(default="./static")
    SERVE_STATIC_FILES: bool = Field(default=True)
    
    # GPU Configuration
    CUDA_VISIBLE_DEVICES: str = Field(default="0")
    TORCH_CUDA_ARCH_LIST: str = Field(default="8.9")  # Ada Lovelace
    GPU_MEMORY_FRACTION: float = Field(default=0.95)
    MIXED_PRECISION: bool = Field(default=True)
    
    # Model Configuration
    MAX_AUDIO_LENGTH: int = Field(default=30)
    SAMPLE_RATE: int = Field(default=16000)
    DEFAULT_LANGUAGE: str = Field(default="en")
    ULTRAVOX_MODEL_NAME: str = Field(default="fixie-ai/ultravox-v0_5-llama-3_1-8b")
    TTS_MODEL_NAME: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2")
    PRELOAD_MODELS: bool = Field(default=True)
    
    # Inference Settings
    MAX_TOKENS: int = Field(default=512)
    TEMPERATURE: float = Field(default=0.7)
    TOP_P: float = Field(default=0.9)
    REPETITION_PENALTY: float = Field(default=1.1)
    
    # Authentication
    HF_TOKEN: str = Field(default="")
    API_KEY: Optional[str] = Field(default=None)  # Optional API key for production
    
    # Performance Settings
    REQUEST_TIMEOUT: int = Field(default=300)  # 5 minutes
    MAX_REQUEST_SIZE_MB: int = Field(default=50)
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    
    # System Health
    MIN_FREE_DISK_GB: int = Field(default=10)
    MIN_FREE_MEMORY_GB: int = Field(default=4)
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = 'allow'
    
    @validator("ENVIRONMENT", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            v = v.lower()
        return v
    
    @validator("API_WORKERS", pre=True)
    def validate_workers(cls, v, values):
        # For GPU models, limit workers to avoid memory issues
        if v > 1 and values.get("ENVIRONMENT") == Environment.PRODUCTION:
            return 1
        return v
    
    def validate_storage_paths(self) -> dict:
        """Validate all storage paths exist and are writable"""
        issues = []
        
        # Check persistent storage
        persistent_paths = [
            self.MODEL_BASE_PATH,
            self.VOICE_FILES_PATH,
            self.AUDIO_CONVERSATIONS_PATH,
            self.AUDIO_SESSIONS_PATH,
            self.LOG_BASE_PATH
        ]
        
        for path_str in persistent_paths:
            path = Path(path_str)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create {path}: {e}")
            elif not os.access(path, os.W_OK):
                issues.append(f"No write permission for {path}")
        
        # Check ephemeral storage
        ephemeral_paths = [
            self.AUDIO_PROCESSING_PATH,
            self.CACHE_BASE_PATH,
            self.TEMP_FILES_PATH
        ]
        
        for path_str in ephemeral_paths:
            path = Path(path_str)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create {path}: {e}")
        
        return {"valid": len(issues) == 0, "issues": issues}
    
    def get_gpu_info(self) -> dict:
        """Get GPU information and availability"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": 0,
            "gpus": []
        }
        
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            
            for i in range(info["gpu_count"]):
                gpu = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    "memory_allocated": torch.cuda.memory_allocated(i) / (1024**3),
                    "memory_reserved": torch.cuda.memory_reserved(i) / (1024**3)
                }
                info["gpus"].append(gpu)
        
        return info
    
    def get_system_info(self) -> dict:
        """Get system resource information"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage for key paths
        disk_usage = {}
        for name, path in [
            ("persistent", "/mnt/storage"),
            ("ephemeral", "/opt/dlami/nvme")
        ]:
            try:
                usage = psutil.disk_usage(path)
                disk_usage[name] = {
                    "total_gb": usage.total / (1024**3),
                    "used_gb": usage.used / (1024**3),
                    "free_gb": usage.free / (1024**3),
                    "percent": usage.percent
                }
            except:
                disk_usage[name] = None
        
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": cpu_percent,
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "disk_usage": disk_usage,
            "gpu_info": self.get_gpu_info()
        }
    
    def optimize_for_g6e(self):
        """Apply G6e instance optimizations"""
        # Set CUDA environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = self.CUDA_VISIBLE_DEVICES
        os.environ["TORCH_CUDA_ARCH_LIST"] = self.TORCH_CUDA_ARCH_LIST
        
        # Enable TF32 for better performance on Ada Lovelace
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Set memory fraction
        if self.GPU_MEMORY_FRACTION < 1.0:
            torch.cuda.set_per_process_memory_fraction(self.GPU_MEMORY_FRACTION)
        
        # Set mixed precision
        if self.MIXED_PRECISION:
            torch.set_float32_matmul_precision('medium')
    
    def validate_system(self) -> List[str]:
        """Validate system configuration and return any errors"""
        errors = []
        
        # Validate storage paths
        storage_validation = self.validate_storage_paths()
        if not storage_validation["valid"]:
            errors.extend(storage_validation["issues"])
        
        # Check GPU availability
        if not torch.cuda.is_available():
            errors.append("CUDA is not available")
        
        # Check minimum system requirements
        system_info = self.get_system_info()
        if system_info["memory_available_gb"] < self.MIN_FREE_MEMORY_GB:
            errors.append(f"Insufficient memory: {system_info['memory_available_gb']:.1f}GB < {self.MIN_FREE_MEMORY_GB}GB")
        
        # Check disk space
        for name, disk in system_info["disk_usage"].items():
            if disk and disk["free_gb"] < self.MIN_FREE_DISK_GB:
                errors.append(f"Insufficient disk space on {name}: {disk['free_gb']:.1f}GB < {self.MIN_FREE_DISK_GB}GB")
        
        return errors


# Initialize settings
settings = Settings()

# Apply optimizations
settings.optimize_for_g6e()

# Set HuggingFace token if available
if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN

# Environment-specific configurations
if settings.ENVIRONMENT == Environment.DEVELOPMENT:
    settings.API_RELOAD = True
    settings.LOG_LEVEL = "DEBUG"
    settings.API_WORKERS = 1
elif settings.ENVIRONMENT == Environment.PRODUCTION:
    settings.API_RELOAD = False
    settings.LOG_LEVEL = "INFO"
    settings.API_ACCESS_LOG = False  # Use nginx for access logs