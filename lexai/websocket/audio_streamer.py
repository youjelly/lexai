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

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from ..models import streaming_inference, StreamConfig
from ..tts import voice_manager, multilingual_tts
from ..utils.logging import get_logger
from ..services import model_services
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
    chunk_size_ms: int = 200  # Larger chunks for more efficient streaming
    
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
        
        # Audio output queue for non-blocking streaming
        self.audio_output_queue = asyncio.Queue(maxsize=1000)
        self.audio_output_task = None
        
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
        websocket: WebSocket,
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
    
    async def _handle_messages(self, websocket: WebSocket):
        """Handle incoming WebSocket messages"""
        while True:
            try:
                # Receive message from WebSocket
                message = await websocket.receive()
                
                # Handle disconnection
                if message['type'] == 'websocket.disconnect':
                    break
                
                # Get actual message data
                if 'bytes' in message:
                    # Binary audio data
                    await self._handle_audio_data(message['bytes'])
                elif 'text' in message:
                    # JSON control message
                    data = json.loads(message['text'])
                    logger.debug(f"Received JSON message: {data}")
                    
                    # Check for text message first (not in enum)
                    if data.get("type") == "text":
                        logger.info("Routing to text message handler")
                        await self._handle_text_message(data, websocket)
                    else:
                        # Try to parse as enum
                        try:
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
                        except ValueError:
                            logger.warning(f"Unknown message type: {data.get('type')}")
                
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON")
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Message handling error: {e}")
                await self._send_error(websocket, str(e))
    
    async def _handle_init(self, data: Dict[str, Any], websocket: WebSocket):
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
    
    async def _handle_start(self, websocket: WebSocket):
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
        
        # Start audio output task for non-blocking streaming
        self.audio_output_task = asyncio.create_task(
            self._audio_output_loop(websocket)
        )
        
        # Send status
        await self._send_status(websocket, "streaming")
        
        logger.info(f"Started streaming for session {self.session_id}")
    
    async def _handle_stop(self, websocket: WebSocket):
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
        if self.audio_output_task:
            self.audio_output_task.cancel()
        
        # Send status
        await self._send_status(websocket, "stopped")
        
        logger.info(f"Stopped streaming for session {self.session_id}")
    
    async def _handle_audio_data(self, data: bytes):
        """Handle incoming audio data"""
        if not self.is_streaming:
            return
        
        try:
            # Log data info for debugging
            logger.debug(f"Received audio data: {len(data)} bytes")
            
            # Since we know the browser sends raw PCM16, decode directly
            # Decode audio based on encoding
            if self.config.encoding == "pcm16":
                # Check if data length is valid for int16
                if len(data) % 2 != 0:
                    logger.warning(f"Audio data length {len(data)} not aligned for int16, padding")
                    data = data + b'\x00'  # Pad with zero byte
                
                # Convert bytes to int16 array (little-endian)
                audio = np.frombuffer(data, dtype='<i2')  # '<' for little-endian, 'i2' for int16
                # Convert to float32 normalized
                audio = audio.astype(np.float32) / 32768.0
                
                # Debug: log some statistics
                logger.debug(f"Audio stats - samples: {len(audio)}, min: {audio.min():.3f}, max: {audio.max():.3f}, mean: {audio.mean():.3f}, std: {audio.std():.3f}")
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
    
    async def _handle_ping(self, websocket: WebSocket):
        """Handle ping message"""
        await self._send_message(websocket, {
            "type": MessageType.PONG.value,
            "timestamp": time.time()
        })
    
    async def _handle_text_message(self, data: Dict[str, Any], websocket: WebSocket):
        """Handle text message for LLM processing"""
        text = data.get("text", "").strip()
        if not text:
            await self._send_error(websocket, "Empty text message")
            return
        
        # Check if TTS is requested (default to True for backward compatibility)
        enable_tts = data.get("enable_tts", True)
        
        logger.info(f"Received text message: {text[:50]}... (TTS: {enable_tts})")
        
        try:
            # Get Ultravox service singleton
            ultravox_service = await model_services.get_ultravox()
            
            # Process text through Ultravox LLM
            logger.info("Processing text through Ultravox LLM...")
            response_text = ""
            
            # Use Ultravox to generate a response
            async for chunk in ultravox_service.process_text(text):
                response_text += chunk
            
            logger.info(f"LLM response: {response_text[:100]}...")
            
            # Send the LLM response back to client
            await self._send_message(websocket, {
                "type": MessageType.TRANSCRIPT.value,
                "text": response_text,
                "timestamp": time.time()
            })
            
            # Synthesize speech if TTS is enabled
            if enable_tts and self.config.bidirectional:
                logger.info(f"Synthesizing TTS for response: {response_text[:50]}...")
                
                # Ensure synthesis loop is running
                if not self.is_streaming:
                    logger.info("Starting audio synthesis for text response...")
                    self.is_streaming = True
                    if not self.synthesis_task or self.synthesis_task.done():
                        self.synthesis_task = asyncio.create_task(
                            self._audio_synthesis_loop(websocket)
                        )
                
                # Ensure audio output task is running
                if not self.audio_output_task or self.audio_output_task.done():
                    logger.info("Starting audio output task...")
                    self.audio_output_task = asyncio.create_task(
                        self._audio_output_loop(websocket)
                    )
                
                # Add text to synthesis queue
                await self.output_buffer.put(response_text)
            else:
                logger.info("TTS disabled for this message")
                
        except Exception as e:
            logger.error(f"Error processing text message: {e}")
            await self._send_error(websocket, str(e))
    
    async def _audio_processing_loop(self, websocket: WebSocket):
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
                    # Log what we received from Ultravox
                    text = result.get("text", "").strip()
                    logger.info(f"Received response from Ultravox: '{text}'")
                    logger.info(f"Response type: {type(text)}, length: {len(text)}")
                    
                    # For audio input, Ultravox returns a conversational response, not a transcription
                    # We should send this as the AI response and synthesize it
                    
                    # Send as AI response (not user transcript)
                    await self._send_message(websocket, {
                        "type": MessageType.TRANSCRIPT.value,  # AI response
                        "text": text,
                        "timestamp": time.time()
                    })
                    
                    # Check if this is meaningful response to synthesize
                    # Filter out silence indicator and common noise/filler responses
                    noise_patterns = [".", "de", "de.", "...", "mm", "um", "uh", "ah", "hm", "hmm", "huh", "oh", "aa", "eh", "hh"]
                    is_noise = text.strip() in noise_patterns or len(text.strip()) < 2
                    
                    # Queue response for TTS synthesis if meaningful
                    if self.config.bidirectional and text and not is_noise:
                        logger.info(f"Queueing Ultravox response for TTS: {text[:50]}...")
                        
                        # Ensure synthesis loop is running
                        if not self.synthesis_task or self.synthesis_task.done():
                            self.synthesis_task = asyncio.create_task(
                                self._audio_synthesis_loop(websocket)
                            )
                        
                        # Ensure audio output task is running
                        if not self.audio_output_task or self.audio_output_task.done():
                            self.audio_output_task = asyncio.create_task(
                                self._audio_output_loop(websocket)
                            )
                        
                        # Add response text to synthesis queue
                        await self.output_buffer.put(text)
                
                elif result["type"] == "error":
                    await self._send_error(websocket, result["error"])
                
        except asyncio.CancelledError:
            logger.info(f"Audio processing cancelled for session {self.session_id}")
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            await self._send_error(websocket, str(e))
    
    async def _audio_synthesis_loop(self, websocket: WebSocket):
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
                    logger.info(f"Generating speech...")
                    
                    audio = await multilingual_tts.synthesize(
                        text,
                        language=self.config.language,
                        voice_id=self.config.voice_id
                    )
                    
                    synthesis_time = (time.time() - start_time) * 1000
                    self.stats["processing_time_ms"] += synthesis_time
                    logger.info(f"Speech generation done ({synthesis_time:.1f}ms)")
                    
                    # Stream audio chunks
                    await self._stream_audio_response(websocket, audio)
                    
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Audio synthesis cancelled for session {self.session_id}")
        except Exception as e:
            logger.error(f"Audio synthesis error: {e}")
            await self._send_error(websocket, str(e))
    
    async def _generate_response(self, transcript: str, websocket: WebSocket):
        """Generate response for transcript"""
        try:
            # Get Ultravox service singleton
            ultravox_service = await model_services.get_ultravox()
            
            # Generate a conversational response to the transcript
            logger.info(f"Generating response for transcript: {transcript[:50]}...")
            response_text = ""
            
            # Use Ultravox to generate a response
            async for chunk in ultravox_service.process_text(transcript):
                response_text += chunk
            
            logger.info(f"Generated response: {response_text[:100]}...")
            
            # Send the response back to client as AI response
            await self._send_message(websocket, {
                "type": MessageType.TRANSCRIPT.value,
                "text": response_text,
                "timestamp": time.time()
            })
            
            # Queue response for TTS synthesis
            if self.config.bidirectional and response_text:
                logger.info(f"Queueing response for TTS: {response_text[:50]}...")
                
                # Ensure synthesis loop is running
                if not self.synthesis_task or self.synthesis_task.done():
                    self.synthesis_task = asyncio.create_task(
                        self._audio_synthesis_loop(websocket)
                    )
                
                # Ensure audio output task is running
                if not self.audio_output_task or self.audio_output_task.done():
                    self.audio_output_task = asyncio.create_task(
                        self._audio_output_loop(websocket)
                    )
                
                # Add response text to synthesis queue
                await self.output_buffer.put(response_text)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
    
    async def _audio_output_loop(self, websocket: WebSocket):
        """Handle audio output streaming without blocking"""
        logger.info(f"Audio output loop started for session {self.session_id}")
        try:
            while self.is_streaming:
                try:
                    # Get audio chunk from output queue
                    chunk_data = await asyncio.wait_for(
                        self.audio_output_queue.get(),
                        timeout=0.1  # Short timeout for responsive streaming
                    )
                    
                    logger.debug(f"Sending audio chunk: {len(chunk_data['audio'])} bytes")
                    
                    # Send audio chunk
                    await websocket.send_bytes(chunk_data['audio'])
                    
                    # Send metadata
                    if chunk_data.get('metadata'):
                        await self._send_message(websocket, chunk_data['metadata'])
                    
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.debug(f"Audio output loop cancelled for session {self.session_id}")
        except Exception as e:
            logger.error(f"Audio output error: {e}")
    
    async def _stream_audio_response(self, websocket: WebSocket, audio: np.ndarray):
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
        
        # Queue chunks for non-blocking streaming
        total_chunks = (len(audio_bytes) + chunk_bytes - 1) // chunk_bytes
        logger.info(f"Queueing {total_chunks} audio chunks for streaming")
        
        for i in range(0, len(audio_bytes), chunk_bytes):
            chunk = audio_bytes[i:i + chunk_bytes]
            
            # Queue audio chunk with metadata
            chunk_data = {
                'audio': chunk,
                'metadata': {
                    "type": MessageType.AUDIO_RESPONSE.value,
                    "chunk_index": i // chunk_bytes,
                    "total_chunks": total_chunks,
                    "sample_rate": 22050,  # TTS output rate
                    "duration_ms": len(chunk) / self.config.bytes_per_sample / 22050 * 1000
                }
            }
            
            # Put in queue without blocking
            try:
                self.audio_output_queue.put_nowait(chunk_data)
                logger.debug(f"Queued chunk {i // chunk_bytes + 1}/{total_chunks}")
            except asyncio.QueueFull:
                # If queue is full, wait a bit
                logger.debug("Audio output queue full, waiting...")
                await asyncio.sleep(0.01)
                await self.audio_output_queue.put(chunk_data)
            
            self.stats["chunks_sent"] += 1
            
            # Reduced delay for faster streaming
            await asyncio.sleep(self.config.chunk_size_ms / 2000)  # Half the delay
    
    async def _send_transcript(self, websocket: WebSocket, result: Dict[str, Any]):
        """Send transcript result"""
        # Send as 'transcription' type for user speech
        await self._send_message(websocket, {
            "type": "transcription",  # Changed from TRANSCRIPT to match frontend
            "text": result.get("text", ""),
            "language": result.get("language"),
            "duration_ms": result.get("audio_duration_ms", 0),
            "processing_time_ms": result.get("processing_time_ms", 0),
            "timestamp": time.time()
        })
        
        self.stats["transcripts"] += 1
    
    async def _send_status(self, websocket: WebSocket, status: str):
        """Send processing status"""
        await self._send_message(websocket, {
            "type": MessageType.PROCESSING_STATUS.value,
            "status": status,
            "stats": self.stats
        })
    
    async def _send_error(self, websocket: WebSocket, error: str):
        """Send error message"""
        await self._send_message(websocket, {
            "type": MessageType.ERROR.value,
            "error": error,
            "timestamp": time.time()
        })
        
        self.stats["errors"] += 1
    
    async def _send_session_info(self, websocket: WebSocket):
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
    
    async def _send_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send JSON message"""
        try:
            await websocket.send_text(json.dumps(data))
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