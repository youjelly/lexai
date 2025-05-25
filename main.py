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

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from lexai.api.main import app as api_app
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
    
    # Configure CORS for external access
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )
    
    # Apply production settings
    if settings.ENVIRONMENT.value == "production":
        api_app.debug = False
        
        # Add production middleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        api_app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )
        
        # Add security headers middleware
        if settings.SECURITY_HEADERS:
            from fastapi.middleware.middleware import Middleware
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
            
            api_app.add_middleware(SecurityHeadersMiddleware)
    
    # Mount static files if enabled
    if settings.SERVE_STATIC_FILES:
        static_path = Path(settings.STATIC_FILES_PATH)
        if not static_path.exists():
            static_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created static files directory: {static_path}")
        
        # Mount static files
        api_app.mount("/static", StaticFiles(directory=settings.STATIC_FILES_PATH), name="static")
        logger.info(f"Mounted static files from {settings.STATIC_FILES_PATH}")
        
        # Serve index.html at root
        @api_app.get("/")
        async def serve_index():
            index_file = static_path / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file))
            else:
                return {"message": "LexAI API Server", "status": "running", "docs": "/docs"}
        
        logger.info("Configured root route to serve index.html")
    
    # Start metrics collection
    metrics_collector.start()
    
    return api_app


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
        limit_concurrency=settings.MAX_WORKERS,
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


if __name__ == "__main__":
    main()