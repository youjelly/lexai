"""
Basic middleware for LexAI API
Handles rate limiting, logging, and error handling
"""

import time
import json
import traceback
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from ..utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = defaultdict(lambda: {"count": 0, "reset_time": time.time() + 60})
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health check
        if request.url.path == "/api/health":
            return await call_next(request)
        
        # Check rate limit
        current_time = time.time()
        client_data = self.request_counts[client_ip]
        
        # Reset counter if time window passed
        if current_time > client_data["reset_time"]:
            client_data["count"] = 0
            client_data["reset_time"] = current_time + 60
        
        # Check if limit exceeded
        if client_data["count"] >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute",
                    "retry_after": int(client_data["reset_time"] - current_time)
                }
            )
        
        # Increment counter
        client_data["count"] += 1
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - client_data["count"]
        )
        response.headers["X-RateLimit-Reset"] = str(int(client_data["reset_time"]))
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs all API requests"""
    
    def __init__(self, app):
        super().__init__(app)
        self.log_path = Path(settings.API_LOG_PATH)
        self.log_path.mkdir(parents=True, exist_ok=True)
    
    async def dispatch(self, request: Request, call_next):
        # Start timer
        start_time = time.time()
        
        # Get request info
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": method,
            "path": path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "")
        }
        
        # Log to file (daily rotation)
        log_file = self.log_path / f"api_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")
        
        # Also log to console
        logger.info(
            f"{method} {path} - {response.status_code} - {duration_ms:.0f}ms - {client_ip}"
        )
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions (they have proper error responses)
            raise
            
        except Exception as e:
            # Log full traceback
            logger.error(f"Unhandled exception: {traceback.format_exc()}")
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e) if settings.LOG_LEVEL == "DEBUG" else "An unexpected error occurred",
                    "request_id": str(time.time())
                }
            )


def setup_cors(app):
    """Setup CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
    )


def setup_middleware(app):
    """Setup all middleware"""
    # Order matters - error handling should be outermost
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    setup_cors(app)


# Request validation helpers
async def validate_audio_file(file_size: int, content_type: str) -> None:
    """Validate uploaded audio file"""
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_TYPES = [
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mp3", "audio/mpeg",
        "audio/ogg", "audio/opus",
        "audio/webm", "audio/flac"
    ]
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // 1024 // 1024}MB"
        )
    
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_TYPES)}"
        )


async def validate_text_length(text: str, max_length: int = 5000) -> None:
    """Validate text length"""
    if len(text) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum length is {max_length} characters"
        )


async def validate_language_code(language: str) -> None:
    """Validate language code"""
    # Import here to avoid circular imports
    from ..tts import LANGUAGE_CODES
    
    valid_codes = list(LANGUAGE_CODES.values()) + list(LANGUAGE_CODES.keys())
    
    if language.lower() not in valid_codes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language code: {language}"
        )