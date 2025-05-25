"""
LexAI Monitoring Module

Provides logging and metrics collection for production deployment
"""

from .logging_config import (
    setup_logging,
    get_logger,
    LogTimer,
    log_api_request,
    log_api_response,
    log_model_inference
)

from .metrics import (
    MetricsCollector,
    metrics_collector,
    track_api_request,
    track_model_inference,
    SystemMetrics,
    ModelMetrics,
    APIMetrics
)

__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    'LogTimer',
    'log_api_request',
    'log_api_response',
    'log_model_inference',
    
    # Metrics
    'MetricsCollector',
    'metrics_collector',
    'track_api_request',
    'track_model_inference',
    'SystemMetrics',
    'ModelMetrics',
    'APIMetrics'
]