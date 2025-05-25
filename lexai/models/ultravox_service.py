"""
Ultravox model service for real-time multimodal processing
Handles loading and inference for fixie-ai/ultravox-v0_5-llama-3_1-8b
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, Any, List, AsyncIterator
import asyncio
from pathlib import Path
import logging
from dataclasses import dataclass
from queue import Queue
import threading
from contextlib import asynccontextmanager

from transformers import (
    AutoModel,
    AutoProcessor,
    TextIteratorStreamer
)
import langdetect
from langdetect import detect_langs

from ..utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for Ultravox inference"""
    max_new_tokens: int = 128  # Reduced for faster response
    temperature: float = 0.5    # Lower for more focused output
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.5  # Higher to prevent repetition
    language: Optional[str] = None
    stream: bool = True


class UltravoxService:
    """Service for managing Ultravox model inference"""
    
    def __init__(self):
        self.model_path = settings.ULTRAVOX_MODEL_PATH
        self.cache_dir = settings.ULTRAVOX_CACHE_PATH
        self.model = None
        self.processor = None
        self.device = None
        self.model_lock = threading.Lock()
        self._initialized = False
        
        # Streaming components
        self.streamer = None
        self.generation_thread = None
        
        # Set environment variables for caching
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir
    
    async def initialize(self):
        """Initialize the Ultravox model and processor"""
        if self._initialized:
            return
        
        logger.info("Initializing Ultravox service...")
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Ultravox model not found at {self.model_path}. "
                "Please run scripts/download_models.py first."
            )
        
        # Load model and processor
        await asyncio.get_event_loop().run_in_executor(
            None, self._load_model
        )
        
        self._initialized = True
        logger.info("Ultravox service initialized successfully")
    
    def _load_model(self):
        """Load the Ultravox model and processor"""
        with self.model_lock:
            logger.info(f"Loading Ultravox model from {self.model_path}")
            
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU (will be slow)")
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir,
                local_files_only=False,  # Allow downloading missing components
                trust_remote_code=True,
                token=settings.HF_TOKEN if settings.HF_TOKEN else None
            )
            
            # Load model with optimizations
            logger.info("Loading model (this may take a minute)...")
            
            if self.device.type == "cuda":
                # GPU loading with 8-bit quantization to save memory
                try:
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16,
                        bnb_8bit_use_double_quant=True,
                        bnb_8bit_quant_type="nf4"
                    )
                    
                    logger.info("Loading model with 8-bit quantization...")
                    self.model = AutoModel.from_pretrained(
                        self.model_path,
                        cache_dir=self.cache_dir,
                        local_files_only=False,  # Allow downloading missing components
                        torch_dtype=torch.float16,
                        quantization_config=quantization_config,
                        device_map="auto",  # Required for quantization
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        token=settings.HF_TOKEN if settings.HF_TOKEN else None
                    )
                    logger.info("Model loaded with 8-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not installed, loading without quantization")
                    self.model = AutoModel.from_pretrained(
                        self.model_path,
                        cache_dir=self.cache_dir,
                        local_files_only=False,  # Allow downloading missing components
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        token=settings.HF_TOKEN if settings.HF_TOKEN else None
                    )
                    # Move model to GPU after loading
                    self.model = self.model.to(self.device)
            else:
                # CPU loading
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    cache_dir=self.cache_dir,
                    local_files_only=False,  # Allow downloading missing components
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    token=settings.HF_TOKEN if settings.HF_TOKEN else None
                )
                self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Log memory usage
            if self.device.type == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect language from text"""
        if not text or len(text) < 10:
            return None
        
        try:
            langs = detect_langs(text)
            if langs and langs[0].prob > 0.8:
                return langs[0].lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
        
        return None
    
    async def process_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None,
        config: Optional[InferenceConfig] = None
    ) -> AsyncIterator[str]:
        """Process audio with Ultravox model and stream response"""
        if not self._initialized:
            await self.initialize()
        
        if config is None:
            config = InferenceConfig()
        
        # Prepare inputs
        inputs = await self._prepare_inputs(audio_data, sample_rate, prompt)
        
        # Log what we're about to process
        logger.info(f"Processing audio - sample_rate: {sample_rate}, audio_length: {len(audio_data)} samples, prompt: '{prompt}'")
        
        # Stream generation
        if config.stream:
            full_response = ""
            async for token in self._stream_generation(inputs, config):
                full_response += token
                yield token
            logger.info(f"Ultravox audio response (streaming): '{full_response}'")
        else:
            # Non-streaming generation
            response = await self._generate(inputs, config)
            logger.info(f"Ultravox audio response (non-streaming): '{response}'")
            yield response
    
    async def _prepare_inputs(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model"""
        
        # Validate audio data
        if len(audio_data) == 0:
            logger.warning("Empty audio data received")
            audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        
        # Ensure audio is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        elif max_val < 0.01:  # Very quiet audio
            logger.warning(f"Very quiet audio detected (max amplitude: {max_val})")
        
        # Create turns format as expected by Ultravox
        turns = []
        
        # Add system message for conversational responses
        turns.append({
            "role": "system", 
            "content": "You are a helpful AI assistant in a voice conversation. IMPORTANT: If you hear silence, very quiet sounds, or background noise without clear speech, respond ONLY with a single period '.'. When you hear clear speech, respond concisely in 1-2 sentences maximum. Stay on topic and avoid rambling. Always respond in English unless explicitly asked to use another language."
        })
        
        # Add user turn with audio
        user_content = prompt if prompt else "<|audio|>"
        if "<|audio|>" not in user_content:
            user_content = f"<|audio|> {user_content}"
            
        turns.append({
            "role": "user",
            "content": user_content
        })
        
        # Apply chat template
        text = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.processor.tokenizer.apply_chat_template(
                turns,
                add_generation_prompt=True,
                tokenize=False
            )
        )
        
        # Process inputs with formatted text
        inputs = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.processor(
                text=text,
                audio=audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    async def _stream_generation(
        self,
        inputs: Dict[str, torch.Tensor],
        config: InferenceConfig
    ) -> AsyncIterator[str]:
        """Stream model generation token by token"""
        
        # Create streamer
        self.streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generation kwargs
        generation_kwargs = {
            **inputs,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "do_sample": config.do_sample,
            "repetition_penalty": config.repetition_penalty,
            "streamer": self.streamer,
            "pad_token_id": self.processor.tokenizer.eos_token_id
        }
        
        # Start generation in separate thread
        self.generation_thread = threading.Thread(
            target=self._generate_thread,
            args=(generation_kwargs,)
        )
        self.generation_thread.start()
        
        # Stream tokens
        for token in self.streamer:
            if token:
                yield token
        
        # Wait for generation to complete
        self.generation_thread.join()
    
    def _generate_thread(self, generation_kwargs: Dict[str, Any]):
        """Run generation in separate thread for streaming"""
        with torch.no_grad():
            self.model.generate(**generation_kwargs)
    
    async def _generate(
        self,
        inputs: Dict[str, torch.Tensor],
        config: InferenceConfig
    ) -> str:
        """Generate response without streaming"""
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "do_sample": config.do_sample,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": self.processor.tokenizer.eos_token_id
        }
        
        # Generate
        output_ids = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.generate(**generation_kwargs)
        )
        
        # Decode
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]
        
        return response
    
    async def process_text(
        self,
        text: str,
        config: Optional[InferenceConfig] = None
    ) -> AsyncIterator[str]:
        """Process text input through the LLM (text-only mode)"""
        if not self._initialized:
            await self.initialize()
        
        if config is None:
            config = InferenceConfig()
        
        # Create turns format with system prompt for consistency
        turns = []
        
        # Add system message (same as audio processing for consistency)
        turns.append({
            "role": "system", 
            "content": "You are a helpful AI assistant. When responding, be direct and conversational. Never start responses with phrases like 'I heard you say' or 'You said' - just respond naturally to the content."
        })
        
        # Add user message
        turns.append({
            "role": "user",
            "content": text
        })
        
        # Apply chat template
        prompt = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.processor.tokenizer.apply_chat_template(
                turns,
                add_generation_prompt=True,
                tokenize=False
            )
        )
        
        logger.info(f"Processing text prompt: {prompt[:100]}...")
        
        # Tokenize the text
        inputs = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        if config.stream:
            async for token in self._stream_generation(inputs, config):
                yield token
        else:
            response = await self._generate(inputs, config)
            yield response
    
    async def process_continuous_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        sample_rate: int,
        config: Optional[InferenceConfig] = None
    ) -> AsyncIterator[str]:
        """Process continuous audio stream"""
        if not self._initialized:
            await self.initialize()
        
        if config is None:
            config = InferenceConfig()
        
        # Buffer for accumulating audio
        audio_buffer = []
        min_audio_length = int(sample_rate * 0.5)  # 0.5 seconds minimum
        
        async for audio_chunk in audio_stream:
            audio_buffer.append(audio_chunk)
            
            # Check if we have enough audio
            total_samples = sum(len(chunk) for chunk in audio_buffer)
            
            if total_samples >= min_audio_length:
                # Concatenate audio
                audio_data = np.concatenate(audio_buffer)
                audio_buffer = []  # Clear buffer
                
                # Process audio
                async for response in self.process_audio(
                    audio_data, sample_rate, config=config
                ):
                    yield response
    
    def cleanup(self):
        """Clean up model resources"""
        logger.info("Cleaning up Ultravox service...")
        
        with self.model_lock:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._initialized = False
        
        logger.info("Ultravox service cleaned up")
    
    @asynccontextmanager
    async def session(self):
        """Context manager for model session"""
        try:
            await self.initialize()
            yield self
        finally:
            # Could implement session-specific cleanup here
            pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        info = {
            "status": "initialized",
            "model_path": self.model_path,
            "device": str(self.device),
            "model_type": "ultravox-v0_5-llama-3_1-8b"
        }
        
        if self.device.type == "cuda":
            info["gpu_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3
            }
        
        return info


# Global service instance
ultravox_service = UltravoxService()