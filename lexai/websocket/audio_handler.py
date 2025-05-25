"""
Audio processing handler for WebSocket streaming
Handles audio format conversion, buffering, and real-time processing
"""

import numpy as np
import asyncio
import struct
from typing import Optional, Union, List, Tuple, AsyncIterator
from dataclasses import dataclass
import time
import wave
import io
from pathlib import Path
from collections import deque
import soundfile as sf

from ..models import audio_processor, AudioConfig
from ..utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass
class AudioFormat:
    """Audio format specification"""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"  # float32, int16, int32
    encoding: str = "pcm"  # pcm, opus, mp3
    
    @property
    def bytes_per_sample(self) -> int:
        if self.dtype == "int16":
            return 2
        elif self.dtype in ["int32", "float32"]:
            return 4
        elif self.dtype == "float64":
            return 8
        else:
            return 2
    
    @property
    def numpy_dtype(self):
        if self.dtype == "int16":
            return np.int16
        elif self.dtype == "int32":
            return np.int32
        elif self.dtype == "float32":
            return np.float32
        elif self.dtype == "float64":
            return np.float64
        else:
            return np.float32


class AudioBuffer:
    """Circular audio buffer for streaming"""
    
    def __init__(self, max_duration_seconds: float = 30.0, format: Optional[AudioFormat] = None):
        self.format = format or AudioFormat()
        self.max_samples = int(max_duration_seconds * self.format.sample_rate)
        self.buffer = deque(maxlen=self.max_samples)
        self.total_samples = 0
        self.lock = asyncio.Lock()
    
    async def add(self, audio: np.ndarray):
        """Add audio to buffer"""
        async with self.lock:
            # Ensure correct shape
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            
            # Add to buffer
            self.buffer.extend(audio)
            self.total_samples += len(audio)
    
    async def get(self, num_samples: Optional[int] = None) -> np.ndarray:
        """Get audio from buffer"""
        async with self.lock:
            if num_samples is None or num_samples > len(self.buffer):
                # Get all available
                audio = np.array(self.buffer)
                self.buffer.clear()
            else:
                # Get specified amount
                audio = np.array([self.buffer.popleft() for _ in range(num_samples)])
            
            return audio
    
    async def get_duration_seconds(self) -> float:
        """Get current buffer duration in seconds"""
        async with self.lock:
            return len(self.buffer) / self.format.sample_rate
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


class AudioHandler:
    """Handles audio processing for WebSocket streaming"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Audio format
        self.input_format = AudioFormat()
        self.output_format = AudioFormat(sample_rate=22050)  # TTS output
        
        # Buffers
        self.input_buffer = AudioBuffer(format=self.input_format)
        self.output_buffer = AudioBuffer(format=self.output_format)
        
        # Processing
        self.processor = audio_processor
        self.processing_config = AudioConfig()
        
        # Recording
        self.recording_enabled = False
        self.recording_path = Path(settings.AUDIO_SESSIONS_PATH) / session_id
        self.recording_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "bytes_received": 0,
            "bytes_sent": 0,
            "samples_processed": 0,
            "format_conversions": 0,
            "buffer_overruns": 0
        }
    
    def set_input_format(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        encoding: str = "pcm"
    ):
        """Set input audio format"""
        self.input_format = AudioFormat(
            sample_rate=sample_rate,
            channels=channels,
            dtype=dtype,
            encoding=encoding
        )
        
        # Update buffer
        self.input_buffer = AudioBuffer(format=self.input_format)
        
        # Update processor config
        self.processing_config.target_sample_rate = sample_rate
        
        logger.info(f"Set input format: {sample_rate}Hz, {channels}ch, {dtype}")
    
    def set_output_format(
        self,
        sample_rate: int = 22050,
        channels: int = 1,
        dtype: str = "float32"
    ):
        """Set output audio format"""
        self.output_format = AudioFormat(
            sample_rate=sample_rate,
            channels=channels,
            dtype=dtype
        )
        
        # Update buffer
        self.output_buffer = AudioBuffer(format=self.output_format)
        
        logger.info(f"Set output format: {sample_rate}Hz, {channels}ch, {dtype}")
    
    async def process_incoming_audio(self, data: bytes) -> np.ndarray:
        """Process incoming audio data"""
        self.stats["bytes_received"] += len(data)
        
        # Decode based on encoding
        if self.input_format.encoding == "pcm":
            audio = self._decode_pcm(data)
        elif self.input_format.encoding == "opus":
            audio = await self._decode_opus(data)
        elif self.input_format.encoding == "mp3":
            audio = await self._decode_mp3(data)
        else:
            raise ValueError(f"Unsupported encoding: {self.input_format.encoding}")
        
        # Add to buffer
        await self.input_buffer.add(audio)
        
        # Record if enabled
        if self.recording_enabled:
            await self._record_audio(audio, "input")
        
        self.stats["samples_processed"] += len(audio)
        
        return audio
    
    def _decode_pcm(self, data: bytes) -> np.ndarray:
        """Decode PCM audio data"""
        # Convert bytes to numpy array based on dtype
        if self.input_format.dtype == "int16":
            audio = np.frombuffer(data, dtype=np.int16)
            # Normalize to float32
            audio = audio.astype(np.float32) / 32768.0
        elif self.input_format.dtype == "int32":
            audio = np.frombuffer(data, dtype=np.int32)
            audio = audio.astype(np.float32) / 2147483648.0
        elif self.input_format.dtype == "float32":
            audio = np.frombuffer(data, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {self.input_format.dtype}")
        
        # Handle multi-channel
        if self.input_format.channels > 1:
            audio = audio.reshape(-1, self.input_format.channels)
            audio = audio.mean(axis=1)  # Convert to mono
        
        return audio
    
    async def _decode_opus(self, data: bytes) -> np.ndarray:
        """Decode Opus audio data"""
        # This would require an Opus decoder
        # For now, raise not implemented
        raise NotImplementedError("Opus decoding not yet implemented")
    
    async def _decode_mp3(self, data: bytes) -> np.ndarray:
        """Decode MP3 audio data"""
        # This would require an MP3 decoder
        # For now, raise not implemented
        raise NotImplementedError("MP3 decoding not yet implemented")
    
    async def prepare_output_audio(self, audio: np.ndarray) -> bytes:
        """Prepare audio for output streaming"""
        # Resample if needed
        if self.output_format.sample_rate != 22050:  # TTS output rate
            audio = await self._resample_audio(
                audio,
                22050,
                self.output_format.sample_rate
            )
            self.stats["format_conversions"] += 1
        
        # Convert to output format
        if self.output_format.dtype == "int16":
            audio_int = (audio * 32768).astype(np.int16)
            data = audio_int.tobytes()
        elif self.output_format.dtype == "int32":
            audio_int = (audio * 2147483648).astype(np.int32)
            data = audio_int.tobytes()
        elif self.output_format.dtype == "float32":
            data = audio.astype(np.float32).tobytes()
        else:
            raise ValueError(f"Unsupported output dtype: {self.output_format.dtype}")
        
        # Add to output buffer
        await self.output_buffer.add(audio)
        
        # Record if enabled
        if self.recording_enabled:
            await self._record_audio(audio, "output")
        
        self.stats["bytes_sent"] += len(data)
        
        return data
    
    async def _resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
        
        # Use processor for resampling
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.processor._resample,
            audio,
            orig_sr,
            target_sr
        )
    
    async def get_audio_chunk(
        self,
        duration_ms: int = 100,
        timeout: float = 1.0
    ) -> Optional[np.ndarray]:
        """Get audio chunk from input buffer"""
        num_samples = int(self.input_format.sample_rate * duration_ms / 1000)
        
        # Wait for enough audio
        start_time = time.time()
        while await self.input_buffer.get_duration_seconds() < duration_ms / 1000:
            if time.time() - start_time > timeout:
                return None
            await asyncio.sleep(0.01)
        
        # Get chunk
        return await self.input_buffer.get(num_samples)
    
    async def create_audio_stream(
        self,
        chunk_duration_ms: int = 100
    ) -> AsyncIterator[np.ndarray]:
        """Create audio stream from input buffer"""
        while True:
            chunk = await self.get_audio_chunk(chunk_duration_ms)
            if chunk is not None and len(chunk) > 0:
                yield chunk
            else:
                # Small delay when no audio available
                await asyncio.sleep(0.01)
    
    def enable_recording(self, enabled: bool = True):
        """Enable/disable audio recording"""
        self.recording_enabled = enabled
        
        if enabled:
            logger.info(f"Recording enabled for session {self.session_id}")
        else:
            logger.info(f"Recording disabled for session {self.session_id}")
    
    async def _record_audio(self, audio: np.ndarray, direction: str = "input"):
        """Record audio to file"""
        try:
            # Create filename with timestamp
            timestamp = int(time.time() * 1000)
            filename = f"{direction}_{timestamp}.wav"
            filepath = self.recording_path / filename
            
            # Save audio
            await asyncio.get_event_loop().run_in_executor(
                None,
                sf.write,
                str(filepath),
                audio,
                self.input_format.sample_rate if direction == "input" else self.output_format.sample_rate
            )
            
        except Exception as e:
            logger.error(f"Failed to record audio: {e}")
    
    async def save_conversation_audio(self) -> Optional[Path]:
        """Save full conversation audio"""
        try:
            # Get all audio from buffers
            input_audio = await self.input_buffer.get()
            output_audio = await self.output_buffer.get()
            
            if len(input_audio) == 0 and len(output_audio) == 0:
                return None
            
            # Save input audio
            if len(input_audio) > 0:
                input_file = self.recording_path / "conversation_input.wav"
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    sf.write,
                    str(input_file),
                    input_audio,
                    self.input_format.sample_rate
                )
            
            # Save output audio
            if len(output_audio) > 0:
                output_file = self.recording_path / "conversation_output.wav"
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    sf.write,
                    str(output_file),
                    output_audio,
                    self.output_format.sample_rate
                )
            
            return self.recording_path
            
        except Exception as e:
            logger.error(f"Failed to save conversation audio: {e}")
            return None
    
    def create_wav_header(
        self,
        sample_rate: int,
        channels: int,
        bits_per_sample: int,
        data_size: int
    ) -> bytes:
        """Create WAV file header"""
        # WAV file header
        header = b'RIFF'
        header += struct.pack('<I', 36 + data_size)  # File size - 8
        header += b'WAVE'
        header += b'fmt '
        header += struct.pack('<I', 16)  # Subchunk size
        header += struct.pack('<H', 1)   # Audio format (PCM)
        header += struct.pack('<H', channels)
        header += struct.pack('<I', sample_rate)
        header += struct.pack('<I', sample_rate * channels * bits_per_sample // 8)
        header += struct.pack('<H', channels * bits_per_sample // 8)
        header += struct.pack('<H', bits_per_sample)
        header += b'data'
        header += struct.pack('<I', data_size)
        
        return header
    
    def encode_audio_chunk(
        self,
        audio: np.ndarray,
        include_header: bool = False
    ) -> bytes:
        """Encode audio chunk for streaming"""
        # Convert to output format
        if self.output_format.dtype == "int16":
            audio_data = (audio * 32768).astype(np.int16).tobytes()
            bits_per_sample = 16
        elif self.output_format.dtype == "int32":
            audio_data = (audio * 2147483648).astype(np.int32).tobytes()
            bits_per_sample = 32
        else:
            audio_data = audio.astype(np.float32).tobytes()
            bits_per_sample = 32
        
        if include_header:
            # Add WAV header
            header = self.create_wav_header(
                self.output_format.sample_rate,
                self.output_format.channels,
                bits_per_sample,
                len(audio_data)
            )
            return header + audio_data
        
        return audio_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audio handler statistics"""
        return {
            "session_id": self.session_id,
            "input_format": {
                "sample_rate": self.input_format.sample_rate,
                "channels": self.input_format.channels,
                "dtype": self.input_format.dtype,
                "encoding": self.input_format.encoding
            },
            "output_format": {
                "sample_rate": self.output_format.sample_rate,
                "channels": self.output_format.channels,
                "dtype": self.output_format.dtype
            },
            "stats": self.stats,
            "recording_enabled": self.recording_enabled,
            "input_buffer_duration": asyncio.run(self.input_buffer.get_duration_seconds()),
            "output_buffer_duration": asyncio.run(self.output_buffer.get_duration_seconds())
        }
    
    async def cleanup(self):
        """Clean up audio handler resources"""
        # Save any remaining audio
        if self.recording_enabled:
            await self.save_conversation_audio()
        
        # Clear buffers
        self.input_buffer.clear()
        self.output_buffer.clear()
        
        logger.info(f"Audio handler cleanup complete for session {self.session_id}")