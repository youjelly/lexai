#!/usr/bin/env python3
"""
LexAI Production Server Entry Point
"""

import os
import sys
import signal
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set HuggingFace cache to ephemeral storage early
os.environ['HF_HOME'] = '/opt/dlami/nvme/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/opt/dlami/nvme/cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/opt/dlami/nvme/cache/huggingface'

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from lexai.api.main import app as api_app
from fastapi import FastAPI
from monitoring import setup_logging, get_logger, metrics_collector

# Setup logging
log_config = setup_logging(
    app_name="lexai",
    log_level=settings.LOG_LEVEL,
    json_logs=settings.ENVIRONMENT.value == "production",
    console_logs=True
)
logger = get_logger("main")


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    # Stop metrics collection
    metrics_collector.stop()
    
    # Exit
    sys.exit(0)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)


def configure_app():
    """Configure the FastAPI application for production"""
    
    # Create a new app that wraps the API
    app = FastAPI(
        title="LexAI",
        description="Real-time multimodal AI voice assistant",
        version="0.1.0"
    )
    
    # Configure CORS for external access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )
    
    # Apply production settings
    if settings.ENVIRONMENT.value == "production":
        app.debug = False
        
        # Add production middleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )
        
        # Add security headers middleware
        if settings.SECURITY_HEADERS:
            from starlette.middleware import Middleware
            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.responses import Response
            
            class SecurityHeadersMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request, call_next):
                    response = await call_next(request)
                    
                    # Security headers for external access
                    response.headers["X-Content-Type-Options"] = "nosniff"
                    response.headers["X-Frame-Options"] = "DENY"
                    response.headers["X-XSS-Protection"] = "1; mode=block"
                    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                    
                    return response
            
            app.add_middleware(SecurityHeadersMiddleware)
    
    # Mount static files if enabled
    if settings.SERVE_STATIC_FILES:
        static_path = Path(settings.STATIC_FILES_PATH)
        if not static_path.exists():
            static_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created static files directory: {static_path}")
        
        # Mount static files
        app.mount("/static", StaticFiles(directory=settings.STATIC_FILES_PATH), name="static")
        logger.info(f"Mounted static files from {settings.STATIC_FILES_PATH}")
        
        # Serve index.html at root
        @app.get("/")
        async def serve_index():
            index_file = static_path / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file))
            else:
                return {"message": "LexAI API Server", "status": "running", "docs": "/docs"}
        
        logger.info("Configured root route to serve index.html")
    
    # Mount the API app
    app.mount("/api", api_app)
    logger.info("Mounted API at /api")
    
    # Start metrics collection
    metrics_collector.start()
    
    return app


def main():
    """Main entry point"""
    
    logger.info(f"Starting LexAI server in {settings.ENVIRONMENT.value} mode")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Project root: {project_root}")
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Configure application
    app = configure_app()
    
    # Optimize for g6e instance if in production
    if settings.ENVIRONMENT.value == "production":
        settings.optimize_for_g6e()
        logger.info("Applied g6e instance optimizations")
    
    # Preload models for faster first request
    if settings.ENVIRONMENT.value == "production":
        logger.info("Preloading AI models...")
        try:
            from lexai.models.ultravox_service import UltravoxService
            from lexai.tts.tts_service import TTSService
            import asyncio
            
            # Initialize Ultravox model
            ultravox = UltravoxService()
            asyncio.run(ultravox.initialize())
            logger.info("Ultravox model preloaded")
            
            # Initialize TTS model
            tts = TTSService()
            tts.load_model("tts_models/multilingual/multi-dataset/xtts_v2")
            logger.info("TTS model preloaded")
            
        except Exception as e:
            logger.error(f"Failed to preload models: {e}")
            # Continue anyway - models will load on demand
    
    # Validate system
    errors = settings.validate_system()
    if errors:
        for error in errors:
            logger.error(f"System validation error: {error}")
        if settings.ENVIRONMENT.value == "production":
            logger.critical("System validation failed, exiting...")
            sys.exit(1)
    else:
        logger.info("System validation passed")
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host=settings.HOST,
        port=settings.PORT,
        loop="uvloop" if settings.ENVIRONMENT.value == "production" else "asyncio",
        log_config=None,  # Use our custom logging
        access_log=True,
        use_colors=False if settings.ENVIRONMENT.value == "production" else True,
        server_header=False,
        date_header=True,
        forwarded_allow_ips="*" if settings.ENVIRONMENT.value == "production" else None,
        
        # Workers and concurrency
        workers=1,  # Single worker for GPU model
        limit_concurrency=1000,  # High limit for concurrent connections
        limit_max_requests=10000 if settings.ENVIRONMENT.value == "production" else None,
        
        # Timeouts
        timeout_keep_alive=5,
        timeout_notify=30,
        
        # SSL/TLS (configure if certificates are provided)
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_version=3,  # TLS 1.2+
        ssl_ciphers="TLSv1.2:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4" if settings.ENVIRONMENT.value == "production" else None,
    )
    
    server = uvicorn.Server(config)
    
    logger.info(f"Server starting on {settings.HOST}:{settings.PORT}")
    logger.info(f"API documentation available at http://{settings.HOST}:{settings.PORT}/docs")
    
    # Start server
    try:
        server.run()
    except Exception as e:
        logger.exception(f"Server crashed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        metrics_collector.stop()
        logger.info("Server shutdown complete")


# Create app instance for uvicorn
app = configure_app()

if __name__ == "__main__":
    main()