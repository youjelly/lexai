"""
Metrics collection and monitoring for LexAI
"""

import time
import psutil
import torch
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
from dataclasses import dataclass, asdict
import json
from pathlib import Path

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
except ImportError:
    # Prometheus client not installed, use dummy implementations
    class DummyMetric:
        def labels(self, **kwargs): return self
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def set(self, value): pass
        def observe(self, value): pass
    
    Counter = Histogram = Gauge = lambda *args, **kwargs: DummyMetric()
    CollectorRegistry = lambda: None
    generate_latest = lambda registry: b""


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: Dict[str, float]
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    timestamp: datetime
    model_name: str
    inference_time_ms: float
    tokens_generated: Optional[int] = None
    audio_duration_s: Optional[float] = None
    batch_size: int = 1


@dataclass
class APIMetrics:
    """API performance metrics"""
    timestamp: datetime
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None


class MetricsCollector:
    """Centralized metrics collection and monitoring"""
    
    def __init__(self, 
                 app_name: str = "lexai",
                 collect_interval: int = 60,
                 history_size: int = 1440):  # 24 hours at 1-minute intervals
        
        self.app_name = app_name
        self.collect_interval = collect_interval
        self.history_size = history_size
        
        # Metrics storage
        self.system_metrics: deque = deque(maxlen=history_size)
        self.model_metrics: deque = deque(maxlen=history_size * 10)
        self.api_metrics: deque = deque(maxlen=history_size * 100)
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Collection thread
        self._stop_event = threading.Event()
        self._collector_thread = None
        
        # Performance tracking
        self.request_durations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.model_durations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        
        # System metrics
        self.cpu_usage = Gauge(
            'lexai_cpu_usage_percent', 
            'CPU usage percentage',
            registry=self.registry
        )
        self.memory_usage = Gauge(
            'lexai_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        self.gpu_usage = Gauge(
            'lexai_gpu_usage_percent',
            'GPU usage percentage',
            registry=self.registry
        )
        self.gpu_memory = Gauge(
            'lexai_gpu_memory_used_bytes',
            'GPU memory used in bytes',
            registry=self.registry
        )
        
        # API metrics
        self.api_requests = Counter(
            'lexai_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        self.api_duration = Histogram(
            'lexai_api_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Model metrics
        self.model_inferences = Counter(
            'lexai_model_inferences_total',
            'Total model inferences',
            ['model'],
            registry=self.registry
        )
        self.model_duration = Histogram(
            'lexai_model_inference_seconds',
            'Model inference duration',
            ['model'],
            registry=self.registry
        )
        
        # WebSocket metrics
        self.ws_connections = Gauge(
            'lexai_websocket_connections',
            'Active WebSocket connections',
            registry=self.registry
        )
        
    def start(self):
        """Start metrics collection"""
        if self._collector_thread is None or not self._collector_thread.is_alive():
            self._stop_event.clear()
            self._collector_thread = threading.Thread(
                target=self._collection_loop,
                daemon=True
            )
            self._collector_thread.start()
    
    def stop(self):
        """Stop metrics collection"""
        self._stop_event.set()
        if self._collector_thread:
            self._collector_thread.join(timeout=5)
    
    def _collection_loop(self):
        """Main collection loop"""
        while not self._stop_event.is_set():
            try:
                metrics = self.collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Update Prometheus gauges
                self.cpu_usage.set(metrics.cpu_percent)
                self.memory_usage.set(metrics.memory_percent)
                if metrics.gpu_utilization is not None:
                    self.gpu_usage.set(metrics.gpu_utilization)
                if metrics.gpu_memory_used_gb is not None:
                    self.gpu_memory.set(metrics.gpu_memory_used_gb * 1024**3)
                
            except Exception as e:
                print(f"Error collecting metrics: {e}")
            
            self._stop_event.wait(self.collect_interval)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk_usage = {}
        for path in ["/", "/mnt/storage", "/opt/dlami/nvme"]:
            try:
                usage = psutil.disk_usage(path)
                disk_usage[path] = usage.percent
            except:
                pass
        
        # GPU metrics
        gpu_util = None
        gpu_mem_used = None
        gpu_mem_total = None
        gpu_temp = None
        
        if torch.cuda.is_available():
            try:
                # Get GPU utilization using nvidia-ml-py
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used = mem_info.used / 1024**3
                gpu_mem_total = mem_info.total / 1024**3
                
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
            except:
                # Fallback to PyTorch
                gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / 1024**3,
            memory_available_gb=memory.available / 1024**3,
            disk_usage_percent=disk_usage,
            gpu_utilization=gpu_util,
            gpu_memory_used_gb=gpu_mem_used,
            gpu_memory_total_gb=gpu_mem_total,
            gpu_temperature=gpu_temp
        )
    
    def record_api_request(self, endpoint: str, method: str, status_code: int,
                          duration_ms: float, request_size: Optional[int] = None,
                          response_size: Optional[int] = None):
        """Record API request metrics"""
        
        metrics = APIMetrics(
            timestamp=datetime.utcnow(),
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=duration_ms,
            request_size_bytes=request_size,
            response_size_bytes=response_size
        )
        
        self.api_metrics.append(metrics)
        self.request_durations[endpoint].append(duration_ms)
        
        # Update Prometheus metrics
        self.api_requests.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.api_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration_ms / 1000)
    
    def record_model_inference(self, model_name: str, duration_ms: float,
                              tokens: Optional[int] = None,
                              audio_duration: Optional[float] = None):
        """Record model inference metrics"""
        
        metrics = ModelMetrics(
            timestamp=datetime.utcnow(),
            model_name=model_name,
            inference_time_ms=duration_ms,
            tokens_generated=tokens,
            audio_duration_s=audio_duration
        )
        
        self.model_metrics.append(metrics)
        self.model_durations[model_name].append(duration_ms)
        
        # Update Prometheus metrics
        self.model_inferences.labels(model=model_name).inc()
        self.model_duration.labels(model=model_name).observe(duration_ms / 1000)
    
    def get_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for the last N minutes"""
        
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        # System metrics summary
        recent_system = [m for m in self.system_metrics if m.timestamp > cutoff]
        system_summary = {}
        
        if recent_system:
            system_summary = {
                "cpu_avg": sum(m.cpu_percent for m in recent_system) / len(recent_system),
                "memory_avg": sum(m.memory_percent for m in recent_system) / len(recent_system),
                "gpu_avg": sum(m.gpu_utilization for m in recent_system if m.gpu_utilization) / 
                          len([m for m in recent_system if m.gpu_utilization]) if any(m.gpu_utilization for m in recent_system) else None
            }
        
        # API metrics summary
        recent_api = [m for m in self.api_metrics if m.timestamp > cutoff]
        api_summary = {
            "total_requests": len(recent_api),
            "avg_response_time": sum(m.response_time_ms for m in recent_api) / len(recent_api) if recent_api else 0,
            "error_rate": len([m for m in recent_api if m.status_code >= 400]) / len(recent_api) if recent_api else 0
        }
        
        # Model metrics summary
        recent_model = [m for m in self.model_metrics if m.timestamp > cutoff]
        model_summary = {
            "total_inferences": len(recent_model),
            "avg_inference_time": sum(m.inference_time_ms for m in recent_model) / len(recent_model) if recent_model else 0
        }
        
        return {
            "period_minutes": minutes,
            "system": system_summary,
            "api": api_summary,
            "models": model_summary,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def export_metrics(self, filepath: Path):
        """Export metrics to JSON file"""
        
        data = {
            "exported_at": datetime.utcnow().isoformat(),
            "system_metrics": [asdict(m) for m in self.system_metrics],
            "api_metrics": [asdict(m) for m in self.api_metrics],
            "model_metrics": [asdict(m) for m in self.model_metrics]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_prometheus_metrics(self) -> bytes:
        """Get Prometheus formatted metrics"""
        return generate_latest(self.registry)


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Decorators for easy metrics collection
def track_api_request(endpoint: str):
    """Decorator to track API request metrics"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = (time.time() - start_time) * 1000
                metrics_collector.record_api_request(
                    endpoint=endpoint,
                    method=kwargs.get('request', {}).get('method', 'GET'),
                    status_code=status_code,
                    duration_ms=duration
                )
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = (time.time() - start_time) * 1000
                metrics_collector.record_api_request(
                    endpoint=endpoint,
                    method='GET',
                    status_code=status_code,
                    duration_ms=duration
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def track_model_inference(model_name: str):
    """Decorator to track model inference metrics"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = await func(*args, **kwargs)
            
            duration = (time.time() - start_time) * 1000
            metrics_collector.record_model_inference(
                model_name=model_name,
                duration_ms=duration
            )
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            duration = (time.time() - start_time) * 1000
            metrics_collector.record_model_inference(
                model_name=model_name,
                duration_ms=duration
            )
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator