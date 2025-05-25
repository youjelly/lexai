"""
Logging configuration for LexAI production deployment
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

# Log directory
LOG_DIR = Path("/var/log/lexai")
LOG_DIR.mkdir(parents=True, exist_ok=True)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", 
                          "funcName", "levelname", "levelno", "lineno", 
                          "module", "msecs", "pathname", "process", 
                          "processName", "relativeCreated", "thread", 
                          "threadName", "exc_info", "exc_text", "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    app_name: str = "lexai",
    log_level: str = "INFO",
    json_logs: bool = True,
    console_logs: bool = True
) -> Dict[str, Any]:
    """
    Setup comprehensive logging configuration
    
    Args:
        app_name: Application name for log files
        log_level: Logging level
        json_logs: Enable JSON formatted logs
        console_logs: Enable console output
        
    Returns:
        Dictionary with logger configuration
    """
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    handlers = []
    
    # File handler for application logs
    app_log_file = LOG_DIR / f"{app_name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        app_log_file,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10,
        encoding='utf-8'
    )
    
    if json_logs:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    handlers.append(file_handler)
    
    # Error file handler
    error_log_file = LOG_DIR / f"{app_name}_error.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
        )
    )
    handlers.append(error_handler)
    
    # Console handler
    if console_logs:
        console_handler = logging.StreamHandler(sys.stdout)
        if sys.stdout.isatty():
            console_handler.setFormatter(
                ColoredFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
            )
        else:
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        handlers.append(console_handler)
    
    # Add handlers to logger
    for handler in handlers:
        logger.addHandler(handler)
    
    # Configure other loggers
    configure_third_party_loggers(log_level)
    
    return {
        "logger": logger,
        "handlers": handlers,
        "log_dir": str(LOG_DIR),
        "app_log": str(app_log_file),
        "error_log": str(error_log_file)
    }


def configure_third_party_loggers(log_level: str):
    """Configure third-party library loggers"""
    
    # Reduce verbosity of some libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    
    # TTS and model libraries
    logging.getLogger("TTS").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # Set uvicorn loggers
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(f"lexai.{name}")


# Logging context manager for performance tracking
class LogTimer:
    """Context manager for timing operations"""
    
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.log(self.level, f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.log(
                self.level, 
                f"Completed {self.operation} in {duration:.3f}s"
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration:.3f}s: {exc_val}"
            )


# Structured logging helpers
def log_api_request(logger: logging.Logger, request_id: str, method: str, 
                   path: str, **kwargs):
    """Log API request with structured data"""
    logger.info(
        "API request received",
        extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            **kwargs
        }
    )


def log_api_response(logger: logging.Logger, request_id: str, status_code: int, 
                    duration: float, **kwargs):
    """Log API response with structured data"""
    logger.info(
        "API response sent",
        extra={
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
            **kwargs
        }
    )


def log_model_inference(logger: logging.Logger, model_name: str, 
                       duration: float, **kwargs):
    """Log model inference with metrics"""
    logger.info(
        f"Model inference completed: {model_name}",
        extra={
            "model": model_name,
            "duration_ms": round(duration * 1000, 2),
            **kwargs
        }
    )