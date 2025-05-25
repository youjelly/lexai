"""
WebSocket audio streaming pipeline for real-time processing
Handles bidirectional audio streaming with Ultravox and TTS
"""

import asyncio
import json
import numpy as np
from typing import Optional, Dict, Any, AsyncIterator, Callable
from pathlib import Path
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import struct
import base64

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

from ..models import streaming_inference, StreamConfig
from ..tts import voice_manager, multilingual_tts
from ..utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    # Control messages
    INIT = "init"
    START = "start"
    STOP = "stop"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    
    # Audio messages
    AUDIO_DATA = "audio_data"
    AUDIO_CONFIG = "audio_config"
    
    # Response messages
    TRANSCRIPT = "transcript"
    AUDIO_RESPONSE = "audio_response"
    PROCESSING_STATUS = "processing_status"
    
    # Session messages
    SESSION_INFO = "session_info"
    SESSION_END = "session_end"


@dataclass
class StreamingConfig:
    """Configuration for audio streaming"""
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm16"  # pcm16, pcm32, float32
    chunk_size_ms: int = 100
    
    # Processing settings
    language: Optional[str] = None
    voice_id: Optional[str] = None
    enable_translation: bool = False
    target_language: Optional[str] = None
    
    # Stream settings
    bidirectional: bool = True
    store_conversation: bool = True
    
    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_size_ms / 1000)
    
    @property
    def bytes_per_sample(self) -> int:
        if self.encoding == "pcm16":
            return 2
        elif self.encoding in ["pcm32", "float32"]:
            return 4
        else:
            return 2


class AudioStreamer:
    """Main WebSocket audio streaming handler"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.config = StreamingConfig()
        
        # Paths
        self.stream_buffer_path = Path(settings.AUDIO_PROCESSING_PATH) / "audio_stream" / session_id
        self.temp_path = Path(settings.TEMP_FILES_PATH) / "websocket_temp" / session_id
        
        # Audio buffers
        self.input_buffer = asyncio.Queue(maxsize=100)
        self.output_buffer = asyncio.Queue(maxsize=100)
        
        # Processing state
        self.is_streaming = False
        self.processing_task = None
        self.synthesis_task = None
        
        # Streaming inference
        self.stream_config = StreamConfig(
            sample_rate=self.config.sample_rate,
            chunk_duration_ms=self.config.chunk_size_ms
        )
        
        # Statistics
        self.stats = {
            "chunks_received": 0,
            "chunks_sent": 0,
            "processing_time_ms": 0,
            "total_audio_ms": 0,
            "transcripts": 0,
            "errors": 0
        }
        
        # Initialize paths
        self._init_paths()
    
    def _init_paths(self):
        """Initialize required directories"""
        self.stream_buffer_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
    
    async def handle_connection(
        self,
        websocket: WebSocketServerProtocol,
        path: str
    ):
        """Handle WebSocket connection lifecycle"""
        logger.info(f"New WebSocket connection for session {self.session_id}")
        
        try:
            # Send initial session info
            await self._send_session_info(websocket)
            
            # Start message handler
            await self._handle_messages(websocket)
            
        except ConnectionClosed:
            logger.info(f"WebSocket connection closed for session {self.session_id}")
        except Exception as e:
            logger.error(f"WebSocket error for session {self.session_id}: {e}")
            await self._send_error(websocket, str(e))
        finally:
            await self._cleanup()
    
    async def _handle_messages(self, websocket: WebSocketServerProtocol):
        """Handle incoming WebSocket messages"""
        async for message in websocket:
            try:
                # Parse message
                if isinstance(message, bytes):
                    # Binary audio data
                    await self._handle_audio_data(message)
                else:
                    # JSON control message
                    data = json.loads(message)
                    message_type = MessageType(data.get("type"))
                    
                    if message_type == MessageType.INIT:
                        await self._handle_init(data, websocket)
                    elif message_type == MessageType.START:
                        await self._handle_start(websocket)
                    elif message_type == MessageType.STOP:
                        await self._handle_stop(websocket)
                    elif message_type == MessageType.PING:
                        await self._handle_ping(websocket)
                    elif message_type == MessageType.AUDIO_CONFIG:
                        await self._handle_audio_config(data)
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON")
            except Exception as e:
                logger.error(f"Message handling error: {e}")
                await self._send_error(websocket, str(e))
    
    async def _handle_init(self, data: Dict[str, Any], websocket: WebSocketServerProtocol):
        """Handle initialization message"""
        # Update configuration
        config_data = data.get("config", {})
        
        if "sample_rate" in config_data:
            self.config.sample_rate = config_data["sample_rate"]
        if "channels" in config_data:
            self.config.channels = config_data["channels"]
        if "encoding" in config_data:
            self.config.encoding = config_data["encoding"]
        if "language" in config_data:
            self.config.language = config_data["language"]
        if "voice_id" in config_data:
            self.config.voice_id = config_data["voice_id"]
            
            # Set active voice
            if self.config.voice_id:
                voice_manager.set_active_voice(self.config.voice_id)
        
        # Update stream config
        self.stream_config.sample_rate = self.config.sample_rate
        
        # Send acknowledgment
        await self._send_message(websocket, {
            "type": MessageType.SESSION_INFO.value,
            "session_id": self.session_id,
            "config": asdict(self.config)
        })
        
        logger.info(f"Session {self.session_id} initialized with config: {asdict(self.config)}")
    
    async def _handle_start(self, websocket: WebSocketServerProtocol):
        """Handle start streaming message"""
        if self.is_streaming:
            await self._send_error(websocket, "Already streaming")
            return
        
        self.is_streaming = True
        
        # Start processing tasks
        self.processing_task = asyncio.create_task(
            self._audio_processing_loop(websocket)
        )
        
        if self.config.bidirectional:
            self.synthesis_task = asyncio.create_task(
                self._audio_synthesis_loop(websocket)
            )
        
        # Send status
        await self._send_status(websocket, "streaming")
        
        logger.info(f"Started streaming for session {self.session_id}")
    
    async def _handle_stop(self, websocket: WebSocketServerProtocol):
        """Handle stop streaming message"""
        if not self.is_streaming:
            await self._send_error(websocket, "Not streaming")
            return
        
        self.is_streaming = False
        
        # Cancel tasks
        if self.processing_task:
            self.processing_task.cancel()
        if self.synthesis_task:
            self.synthesis_task.cancel()
        
        # Send status
        await self._send_status(websocket, "stopped")
        
        logger.info(f"Stopped streaming for session {self.session_id}")
    
    async def _handle_audio_data(self, data: bytes):
        """Handle incoming audio data"""
        if not self.is_streaming:
            return
        
        try:
            # Decode audio based on encoding
            if self.config.encoding == "pcm16":
                # Convert bytes to int16 array
                audio = np.frombuffer(data, dtype=np.int16)
                # Convert to float32 normalized
                audio = audio.astype(np.float32) / 32768.0
            elif self.config.encoding == "pcm32":
                audio = np.frombuffer(data, dtype=np.int32)
                audio = audio.astype(np.float32) / 2147483648.0
            elif self.config.encoding == "float32":
                audio = np.frombuffer(data, dtype=np.float32)
            else:
                logger.error(f"Unsupported encoding: {self.config.encoding}")
                return
            
            # Handle multi-channel
            if self.config.channels > 1:
                audio = audio.reshape(-1, self.config.channels).mean(axis=1)
            
            # Add to input buffer
            await self.input_buffer.put(audio)
            
            # Update stats
            self.stats["chunks_received"] += 1
            self.stats["total_audio_ms"] += len(audio) / self.config.sample_rate * 1000
            
        except Exception as e:
            logger.error(f"Audio data handling error: {e}")
            self.stats["errors"] += 1
    
    async def _handle_audio_config(self, data: Dict[str, Any]):
        """Handle audio configuration update"""
        config_update = data.get("config", {})
        
        # Update relevant configs
        if "language" in config_update:
            self.config.language = config_update["language"]
        if "voice_id" in config_update:
            self.config.voice_id = config_update["voice_id"]
            voice_manager.set_active_voice(self.config.voice_id)
        
        logger.info(f"Updated audio config for session {self.session_id}")
    
    async def _handle_ping(self, websocket: WebSocketServerProtocol):
        """Handle ping message"""
        await self._send_message(websocket, {
            "type": MessageType.PONG.value,
            "timestamp": time.time()
        })
    
    async def _audio_processing_loop(self, websocket: WebSocketServerProtocol):
        """Process incoming audio stream"""
        try:
            # Create audio stream from queue
            async def audio_generator():
                while self.is_streaming:
                    try:
                        audio = await asyncio.wait_for(
                            self.input_buffer.get(),
                            timeout=1.0
                        )
                        yield audio
                    except asyncio.TimeoutError:
                        continue
            
            # Process audio stream
            async for result in streaming_inference.process_audio_stream(
                audio_generator(),
                inference_config=None
            ):
                if result["type"] == "transcription":
                    # Send transcript
                    await self._send_transcript(websocket, result)
                    
                    # If bidirectional, generate response
                    if self.config.bidirectional and result.get("text"):
                        await self._generate_response(result["text"])
                
                elif result["type"] == "error":
                    await self._send_error(websocket, result["error"])
                
        except asyncio.CancelledError:
            logger.info(f"Audio processing cancelled for session {self.session_id}")
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            await self._send_error(websocket, str(e))
    
    async def _audio_synthesis_loop(self, websocket: WebSocketServerProtocol):
        """Synthesize and stream audio responses"""
        try:
            while self.is_streaming:
                try:
                    # Get text to synthesize
                    text = await asyncio.wait_for(
                        self.output_buffer.get(),
                        timeout=1.0
                    )
                    
                    # Synthesize audio
                    start_time = time.time()
                    
                    audio = await multilingual_tts.synthesize(
                        text,
                        language=self.config.language,
                        voice_id=self.config.voice_id
                    )
                    
                    synthesis_time = (time.time() - start_time) * 1000
                    self.stats["processing_time_ms"] += synthesis_time
                    
                    # Stream audio chunks
                    await self._stream_audio_response(websocket, audio)
                    
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Audio synthesis cancelled for session {self.session_id}")
        except Exception as e:
            logger.error(f"Audio synthesis error: {e}")
            await self._send_error(websocket, str(e))
    
    async def _generate_response(self, transcript: str):
        """Generate response for transcript"""
        # This is a placeholder - in real implementation,
        # you would process the transcript and generate appropriate response
        
        # For now, just echo back
        response = f"I heard you say: {transcript}"
        
        # Add to synthesis queue
        await self.output_buffer.put(response)
    
    async def _stream_audio_response(self, websocket: WebSocketServerProtocol, audio: np.ndarray):
        """Stream audio response in chunks"""
        # Convert to appropriate format
        if self.config.encoding == "pcm16":
            audio_int = (audio * 32768).astype(np.int16)
            audio_bytes = audio_int.tobytes()
        elif self.config.encoding == "pcm32":
            audio_int = (audio * 2147483648).astype(np.int32)
            audio_bytes = audio_int.tobytes()
        else:  # float32
            audio_bytes = audio.astype(np.float32).tobytes()
        
        # Calculate chunk size in bytes
        chunk_bytes = self.config.chunk_samples * self.config.bytes_per_sample
        
        # Stream chunks
        for i in range(0, len(audio_bytes), chunk_bytes):
            chunk = audio_bytes[i:i + chunk_bytes]
            
            # Send as binary message
            await websocket.send(chunk)
            
            # Also send metadata
            await self._send_message(websocket, {
                "type": MessageType.AUDIO_RESPONSE.value,
                "chunk_index": i // chunk_bytes,
                "total_chunks": (len(audio_bytes) + chunk_bytes - 1) // chunk_bytes,
                "sample_rate": 22050,  # TTS output rate
                "duration_ms": len(chunk) / self.config.bytes_per_sample / 22050 * 1000
            })
            
            self.stats["chunks_sent"] += 1
            
            # Small delay to simulate real-time
            await asyncio.sleep(self.config.chunk_size_ms / 1000)
    
    async def _send_transcript(self, websocket: WebSocketServerProtocol, result: Dict[str, Any]):
        """Send transcript result"""
        await self._send_message(websocket, {
            "type": MessageType.TRANSCRIPT.value,
            "text": result.get("text", ""),
            "language": result.get("language"),
            "duration_ms": result.get("audio_duration_ms", 0),
            "processing_time_ms": result.get("processing_time_ms", 0),
            "timestamp": time.time()
        })
        
        self.stats["transcripts"] += 1
    
    async def _send_status(self, websocket: WebSocketServerProtocol, status: str):
        """Send processing status"""
        await self._send_message(websocket, {
            "type": MessageType.PROCESSING_STATUS.value,
            "status": status,
            "stats": self.stats
        })
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error: str):
        """Send error message"""
        await self._send_message(websocket, {
            "type": MessageType.ERROR.value,
            "error": error,
            "timestamp": time.time()
        })
        
        self.stats["errors"] += 1
    
    async def _send_session_info(self, websocket: WebSocketServerProtocol):
        """Send session information"""
        await self._send_message(websocket, {
            "type": MessageType.SESSION_INFO.value,
            "session_id": self.session_id,
            "config": asdict(self.config),
            "capabilities": {
                "languages": multilingual_tts.get_supported_languages(),
                "voices": [
                    {"id": v.profile.id, "name": v.profile.name}
                    for v in voice_manager.list_voices()
                ],
                "encodings": ["pcm16", "pcm32", "float32"],
                "sample_rates": [16000, 22050, 44100, 48000]
            }
        })
    
    async def _send_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Send JSON message"""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def _cleanup(self):
        """Clean up resources"""
        # Cancel tasks
        if self.processing_task:
            self.processing_task.cancel()
        if self.synthesis_task:
            self.synthesis_task.cancel()
        
        # Clean up temp files
        import shutil
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        
        logger.info(f"Cleaned up session {self.session_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            "session_id": self.session_id,
            "stats": self.stats,
            "config": asdict(self.config),
            "is_streaming": self.is_streaming
        }


async def create_audio_streamer(session_id: Optional[str] = None) -> AudioStreamer:
    """Create a new audio streamer instance"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    return AudioStreamer(session_id)