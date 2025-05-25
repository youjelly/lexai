"""
Streaming inference for real-time audio processing with Ultravox
Optimized for low-latency continuous audio stream processing
"""

import asyncio
import numpy as np
import torch
from typing import Optional, AsyncIterator, Tuple, Dict, Any, List
from dataclasses import dataclass, field
import time
from collections import deque
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from ..utils.logging import get_logger
from .audio_processor import AudioProcessor, AudioConfig
from .ultravox_service import UltravoxService, InferenceConfig
from config import settings

logger = get_logger(__name__)


@dataclass
class StreamConfig:
    """Configuration for streaming inference"""
    # Audio settings
    sample_rate: int = 16000
    chunk_duration_ms: int = 100  # Process every 100ms
    
    # Buffering settings
    min_audio_ms: int = 500  # Minimum audio before processing
    max_audio_ms: int = 3000  # Maximum audio to accumulate
    silence_timeout_ms: int = 1000  # Silence before processing
    
    # Model settings
    language_detection: bool = False  # Disabled due to audio placeholder conflict
    
    # Performance settings
    max_concurrent_inferences: int = 2
    inference_timeout_s: float = 10.0
    
    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000)
    
    @property
    def min_samples(self) -> int:
        return int(self.sample_rate * self.min_audio_ms / 1000)
    
    @property
    def max_samples(self) -> int:
        return int(self.sample_rate * self.max_audio_ms / 1000)


@dataclass
class AudioBuffer:
    """Buffer for accumulating audio data"""
    data: deque = field(default_factory=lambda: deque(maxlen=48000*30))  # 30s max
    timestamp: float = field(default_factory=time.time)
    
    def add(self, audio: np.ndarray):
        """Add audio to buffer"""
        self.data.extend(audio)
        self.timestamp = time.time()
    
    def get_array(self) -> np.ndarray:
        """Get buffer as numpy array"""
        if len(self.data) == 0:
            return np.array([], dtype=np.float32)
        return np.array(list(self.data), dtype=np.float32)
    
    def clear(self):
        """Clear buffer"""
        self.data.clear()
        self.timestamp = time.time()
    
    @property
    def duration_ms(self) -> float:
        """Get buffer duration in milliseconds"""
        return len(self.data) / 16  # Assuming 16kHz


class StreamingInference:
    """Handles streaming inference for continuous audio"""
    
    def __init__(
        self,
        ultravox_service: Optional[UltravoxService] = None,
        audio_processor: Optional[AudioProcessor] = None,
        config: Optional[StreamConfig] = None
    ):
        self.ultravox = ultravox_service or UltravoxService()
        self.audio_processor = audio_processor or AudioProcessor()
        self.config = config or StreamConfig()
        
        # Buffers and state
        self.audio_buffer = AudioBuffer()
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
        
        # The multimodal model handles speech patterns naturally
        
        # Executor for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Statistics
        self.stats = {
            "processed_chunks": 0,
            "inference_count": 0,
            "total_audio_ms": 0,
            "avg_latency_ms": 0
        }
    
    
    async def initialize(self):
        """Initialize streaming inference"""
        await self.ultravox.initialize()
        logger.info("Streaming inference initialized")
    
    async def process_audio_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        inference_config: Optional[InferenceConfig] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process continuous audio stream and yield responses"""
        
        if inference_config is None:
            inference_config = InferenceConfig()
        
        # Ensure models are initialized
        await self.initialize()
        
        inference_task = None
        
        try:
            async for audio_chunk in audio_stream:
                # Update statistics
                self.stats["processed_chunks"] += 1
                self.stats["total_audio_ms"] += len(audio_chunk) / 16
                
                # The multimodal model understands speech patterns and pauses
                
                # Add to buffer
                self.audio_buffer.add(audio_chunk)
                
                # Determine if we should process
                should_process = self._should_process_buffer()
                
                if should_process and not self.is_processing:
                    # Start processing in background
                    if inference_task is None or inference_task.done():
                        inference_task = asyncio.create_task(
                            self._process_buffer(inference_config)
                        )
                
                # Yield any completed results
                if inference_task and inference_task.done():
                    try:
                        result = inference_task.result()
                        if result:
                            yield result
                    except Exception as e:
                        logger.error(f"Inference error: {e}")
                        yield {
                            "type": "error",
                            "error": str(e),
                            "timestamp": time.time()
                        }
                    
                    inference_task = None
        
        finally:
            # Process any remaining audio
            if self.audio_buffer.duration_ms > 0:
                result = await self._process_buffer(inference_config)
                if result:
                    yield result
            
            # Cleanup
            self.executor.shutdown(wait=False)
    
    
    def _should_process_buffer(self) -> bool:
        """Determine if buffer should be processed"""
        duration_ms = self.audio_buffer.duration_ms
        
        # Don't process if too little audio
        if duration_ms < self.config.min_audio_ms:
            return False
        
        # Process if maximum duration reached
        if duration_ms >= self.config.max_audio_ms:
            return True
        
        # Process based on silence timeout
        time_since_update = (time.time() - self.audio_buffer.timestamp) * 1000
        if time_since_update > self.config.silence_timeout_ms:
            return True
        
        return False
    
    async def _process_buffer(
        self,
        inference_config: InferenceConfig
    ) -> Optional[Dict[str, Any]]:
        """Process audio buffer and return result"""
        
        async with self.processing_lock:
            if self.audio_buffer.duration_ms == 0:
                return None
            
            self.is_processing = True
            start_time = time.time()
            
            try:
                # Get audio data
                audio_data = self.audio_buffer.get_array()
                duration_ms = len(audio_data) / 16
                
                # Clear buffer for next accumulation
                self.audio_buffer.clear()
                
                logger.info(f"Processing {duration_ms:.0f}ms of audio, shape: {audio_data.shape}, range: [{audio_data.min():.3f}, {audio_data.max():.3f}], mean: {audio_data.mean():.3f}, std: {audio_data.std():.3f}")
                
                # Detect language if enabled
                language = None
                if self.config.language_detection and inference_config.language is None:
                    # Use first response to detect language
                    temp_config = InferenceConfig(
                        max_new_tokens=50,
                        stream=False
                    )
                    
                    response = ""
                    async for text in self.ultravox.process_audio(
                        audio_data,
                        self.config.sample_rate,
                        config=temp_config
                    ):
                        response += text
                    
                    language = self.ultravox.detect_language(response)
                    if language:
                        logger.info(f"Detected language: {language}")
                
                # Process audio with streaming
                result = {
                    "type": "transcription",
                    "audio_duration_ms": duration_ms,
                    "timestamp": start_time,
                    "language": language,
                    "segments": []
                }
                
                full_response = ""
                async for text in self.ultravox.process_audio(
                    audio_data,
                    self.config.sample_rate,
                    config=inference_config
                ):
                    full_response += text
                    
                    # Yield intermediate results
                    result["segments"].append({
                        "text": text,
                        "timestamp": time.time()
                    })
                
                result["text"] = full_response
                result["processing_time_ms"] = (time.time() - start_time) * 1000
                
                # Log the full response to understand what Ultravox returns
                logger.info(f"Ultravox audio processing response: '{full_response}'")
                logger.info(f"Response length: {len(full_response)} chars")
                
                # Update statistics
                self.stats["inference_count"] += 1
                latency = result["processing_time_ms"]
                self.stats["avg_latency_ms"] = (
                    (self.stats["avg_latency_ms"] * (self.stats["inference_count"] - 1) + latency)
                    / self.stats["inference_count"]
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing buffer: {e}")
                return {
                    "type": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
            finally:
                self.is_processing = False
    
    async def process_file_streaming(
        self,
        file_path: str,
        inference_config: Optional[InferenceConfig] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process audio file in streaming fashion"""
        
        # Load and process audio file
        audio_data, sample_rate = self.audio_processor.process_file(file_path)
        
        # Create chunks
        chunks = self.audio_processor.create_chunked_stream(
            audio_data,
            sample_rate
        )
        
        # Create async iterator from chunks
        async def chunk_iterator():
            for chunk in chunks:
                yield chunk
                # Simulate real-time streaming
                await asyncio.sleep(self.config.chunk_duration_ms / 1000)
        
        # Process stream
        async for result in self.process_audio_stream(
            chunk_iterator(),
            inference_config
        ):
            yield result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            **self.stats,
            "buffer_duration_ms": self.audio_buffer.duration_ms,
            "is_processing": self.is_processing,
            "config": {
                "chunk_duration_ms": self.config.chunk_duration_ms,
                "min_audio_ms": self.config.min_audio_ms,
                "max_audio_ms": self.config.max_audio_ms,
            }
        }
    
    async def reset(self):
        """Reset streaming state"""
        self.audio_buffer.clear()
        self.is_processing = False
        self.stats = {
            "processed_chunks": 0,
            "inference_count": 0,
            "total_audio_ms": 0,
            "avg_latency_ms": 0
        }
        logger.info("Streaming inference reset")


class BatchStreamingInference:
    """Handles multiple concurrent streaming sessions"""
    
    def __init__(self, max_sessions: int = 10):
        self.max_sessions = max_sessions
        self.sessions: Dict[str, StreamingInference] = {}
        self.ultravox_service = UltravoxService()
        self.session_lock = asyncio.Lock()
    
    async def create_session(self, session_id: str, config: Optional[StreamConfig] = None) -> StreamingInference:
        """Create a new streaming session"""
        async with self.session_lock:
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError(f"Maximum sessions ({self.max_sessions}) reached")
            
            if session_id in self.sessions:
                raise ValueError(f"Session {session_id} already exists")
            
            # Create new streaming inference instance
            stream = StreamingInference(
                ultravox_service=self.ultravox_service,
                config=config
            )
            
            self.sessions[session_id] = stream
            logger.info(f"Created streaming session: {session_id}")
            
            return stream
    
    async def get_session(self, session_id: str) -> Optional[StreamingInference]:
        """Get an existing session"""
        return self.sessions.get(session_id)
    
    async def remove_session(self, session_id: str):
        """Remove a streaming session"""
        async with self.session_lock:
            if session_id in self.sessions:
                await self.sessions[session_id].reset()
                del self.sessions[session_id]
                logger.info(f"Removed streaming session: {session_id}")
    
    async def cleanup_inactive_sessions(self, timeout_minutes: int = 30):
        """Clean up inactive sessions"""
        current_time = time.time()
        inactive_sessions = []
        
        async with self.session_lock:
            for session_id, stream in self.sessions.items():
                inactive_time = current_time - stream.audio_buffer.timestamp
                if inactive_time > timeout_minutes * 60:
                    inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            await self.remove_session(session_id)
        
        if inactive_sessions:
            logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")


# Global instances
streaming_inference = StreamingInference()
batch_streaming = BatchStreamingInference()