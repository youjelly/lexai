"""
LexAI API - MVP FastAPI Application
Real-time multimodal AI voice assistant
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from .middleware import setup_middleware
from .routes import conversation, voice, audio, tts, sessions, system
from ..websocket import connection_manager
from ..models import model_manager
from ..utils.logging import get_logger, setup_logging
from config import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting LexAI API...")
    
    # Setup logging
    setup_logging()
    
    # Initialize services
    await connection_manager.initialize()
    
    # Preload models if configured
    if settings.ULTRAVOX_MODEL_NAME:
        logger.info("Preloading Ultravox model...")
        try:
            await model_manager.load_ultravox_model("ultravox")
            logger.info("Ultravox model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload Ultravox model: {e}")
    
    logger.info("LexAI API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LexAI API...")
    
    # Cleanup services
    await connection_manager.cleanup()
    model_manager.cleanup()
    
    logger.info("LexAI API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LexAI API",
    description="Real-time multimodal AI voice assistant API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Setup middleware
setup_middleware(app)

# Include routes
app.include_router(system.router, tags=["system"])
app.include_router(conversation.router, prefix="/conversations", tags=["conversations"])
app.include_router(voice.router, prefix="/voices", tags=["voices"])
app.include_router(audio.router, prefix="/audio", tags=["audio"])
app.include_router(tts.router, prefix="/tts", tags=["tts"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])

# WebSocket route (special handling)
app.include_router(conversation.ws_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "LexAI API",
        "version": "0.1.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return {
        "error": "Not found",
        "message": f"Path {request.url.path} not found",
        "status_code": 404
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal error: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }


# Optional: Mount static files for audio storage
# app.mount("/static", StaticFiles(directory="/mnt/storage/audio"), name="static")