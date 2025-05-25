#!/usr/bin/env python3
"""
Model download script for LexAI
Downloads and validates Ultravox and TTS models
"""

import os
import sys
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Optional, Dict
import subprocess
import logging
import json
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are already set

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoModel, AutoProcessor
    from TTS.utils.manage import ModelManager
    # Note: download_model_files might not exist in all TTS versions
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please ensure you have activated the virtual environment:")
    print("  source venv/bin/activate")
    print("And installed requirements:")
    print("  pip install huggingface-hub transformers TTS")
    sys.exit(1)

# Model configurations
MODELS = {
    "ultravox": {
        "repo_id": "fixie-ai/ultravox-v0_5-llama-3_1-8b",
        "path": "/mnt/storage/models/ultravox",
        "cache_dir": "/opt/dlami/nvme/cache/ultravox",
        "size_gb": 16,
        "description": "Ultravox v0.5 - Llama 3.1 8B multimodal model"
    },
    "tts": {
        "models": [
            # Best multilingual model for general use
            {
                "name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "size_gb": 2.0,
                "description": "XTTS v2 - Best multilingual TTS with voice cloning",
                "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
            },
            # English models
            {
                "name": "tts_models/en/vctk/vits",
                "size_gb": 0.5,
                "description": "VITS - Fast English multi-speaker model",
                "languages": ["en"]
            },
            {
                "name": "tts_models/en/jenny/jenny",
                "size_gb": 0.3,
                "description": "Jenny - High quality English female voice",
                "languages": ["en"]
            },
            # Voice cloning models
            {
                "name": "tts_models/multilingual/multi-dataset/your_tts",
                "size_gb": 1.5,
                "description": "YourTTS - Multilingual voice cloning model",
                "languages": ["en", "pt", "fr", "it", "pl", "es", "de", "nl", "cs", "ar", "tr", "ru", "zh-cn"]
            },
            # Speaker encoder for voice cloning
            {
                "name": "encoder_models/universal/libri-tts/wavegrad",
                "size_gb": 0.5,
                "description": "Universal speaker encoder for voice cloning",
                "type": "encoder"
            },
            # Vocoder models for high quality synthesis
            {
                "name": "vocoder_models/universal/libri-tts/wavegrad",
                "size_gb": 0.8,
                "description": "WaveGrad vocoder for high quality synthesis",
                "type": "vocoder"
            },
            {
                "name": "vocoder_models/en/ljspeech/hifigan_v2",
                "size_gb": 0.3,
                "description": "HiFiGAN v2 - Fast neural vocoder",
                "type": "vocoder"
            }
        ],
        "path": "/mnt/storage/models/tts",
        "cache_dir": "/opt/dlami/nvme/cache/tts",
        "total_size_gb": 7.4,
        "description": "Coqui TTS models for multilingual synthesis and voice cloning"
    }
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_disk_space(path: str, required_gb: float) -> bool:
    """Check if there's enough disk space for the model"""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    
    if free_gb < required_gb * 1.2:  # 20% buffer
        logger.error(f"Insufficient disk space: {free_gb:.1f}GB available, {required_gb * 1.2:.1f}GB required")
        return False
    
    logger.info(f"Disk space check passed: {free_gb:.1f}GB available")
    return True


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment or prompt user"""
    # Check multiple environment variables
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        logger.info("HuggingFace token not found in environment")
        logger.info("For gated models like Ultravox, you need a HuggingFace token")
        response = input("Enter your HuggingFace token (required for Ultravox): ").strip()
        if response:
            token = response
            logger.info("Using provided HuggingFace token")
        else:
            logger.warning("No token provided - Ultravox download may fail")
    else:
        logger.info("Using HuggingFace token from environment")
    
    return token


def download_ultravox_model(token: Optional[str] = None) -> bool:
    """Download Ultravox model from HuggingFace"""
    model_info = MODELS["ultravox"]
    
    logger.info(f"Downloading {model_info['description']}...")
    logger.info(f"Repository: {model_info['repo_id']}")
    logger.info(f"Destination: {model_info['path']}")
    
    # Check disk space
    if not check_disk_space(model_info['path'], model_info['size_gb']):
        return False
    
    # Create directories
    os.makedirs(model_info['path'], exist_ok=True)
    os.makedirs(model_info['cache_dir'], exist_ok=True)
    
    try:
        # Download model using snapshot_download for the entire repository
        logger.info("Starting model download (this may take a while)...")
        
        snapshot_download(
            repo_id=model_info['repo_id'],
            local_dir=model_info['path'],
            cache_dir=model_info['cache_dir'],
            token=token,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        
        logger.info("Download completed successfully")
        
        # Validate model can be loaded
        logger.info("Validating model integrity...")
        
        # Set environment variables for model loading
        os.environ['HF_HOME'] = model_info['cache_dir']
        os.environ['TRANSFORMERS_CACHE'] = model_info['cache_dir']
        
        # Try to load processor to validate
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_info['path'],
            cache_dir=model_info['cache_dir'],
            local_files_only=True,
            trust_remote_code=True  # Required for Ultravox custom code
        )
        
        logger.info("Model validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Ultravox model: {e}")
        return False


def download_tts_models() -> bool:
    """Download all TTS models from Coqui"""
    model_info = MODELS["tts"]
    
    logger.info(f"Downloading {model_info['description']}...")
    logger.info(f"Total size: {model_info['total_size_gb']}GB")
    logger.info(f"Destination: {model_info['path']}")
    
    # Check disk space
    if not check_disk_space(model_info['path'], model_info['total_size_gb']):
        return False
    
    # Create directories
    os.makedirs(model_info['path'], exist_ok=True)
    os.makedirs(model_info['cache_dir'], exist_ok=True)
    
    # Set TTS paths
    os.environ['TTS_HOME'] = model_info['path']
    
    # Initialize model manager
    from TTS.utils.manage import ModelManager as TTS_ModelManager
    manager = TTS_ModelManager(
        models_file=None,
        output_prefix=model_info['path'],
        progress_bar=True
    )
    
    successful_downloads = []
    failed_downloads = []
    
    # Download each model
    for tts_model in model_info['models']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading: {tts_model['description']}")
        logger.info(f"Model: {tts_model['name']}")
        logger.info(f"Size: {tts_model['size_gb']}GB")
        
        try:
            model_name = tts_model['name']
            
            # Different download methods based on model type
            if tts_model.get('type') == 'encoder':
                # Download encoder model
                model_type, dataset, model_name_short = model_name.split("/")
                download_path = manager.download_encoder(dataset, model_name_short)
                
                if download_path and os.path.exists(download_path):
                    logger.info(f"Encoder downloaded: {download_path}")
                    successful_downloads.append(tts_model['name'])
                else:
                    raise Exception("Encoder download failed")
                    
            elif tts_model.get('type') == 'vocoder':
                # Download vocoder model
                model_type, lang, dataset, model_name_short = model_name.split("/")
                model_path, config_path = manager.download_vocoder(lang, dataset, model_name_short)
                
                if model_path and os.path.exists(model_path):
                    logger.info(f"Vocoder downloaded: {model_path}")
                    successful_downloads.append(tts_model['name'])
                else:
                    raise Exception("Vocoder download failed")
                    
            else:
                # Download TTS model
                parts = model_name.split("/")
                if len(parts) == 4:
                    model_type, lang, dataset, model_name_short = parts
                else:
                    # Handle special cases like multilingual models
                    model_type = parts[0]
                    lang = parts[1]
                    dataset = parts[2]
                    model_name_short = parts[3] if len(parts) > 3 else dataset
                
                model_path, config_path, model_item = manager.download_model(
                    model_type, lang, dataset, model_name_short
                )
                
                # Verify downloaded files
                if model_path and os.path.exists(model_path):
                    logger.info(f"Model downloaded: {model_path}")
                    if config_path and os.path.exists(config_path):
                        logger.info(f"Config downloaded: {config_path}")
                    successful_downloads.append(tts_model['name'])
                else:
                    raise Exception("Model download failed")
                    
        except Exception as e:
            logger.error(f"Failed to download {tts_model['name']}: {e}")
            failed_downloads.append(tts_model['name'])
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TTS Model Download Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {len(successful_downloads)}/{len(model_info['models'])}")
    
    if successful_downloads:
        logger.info("\nSuccessfully downloaded:")
        for model in successful_downloads:
            logger.info(f"  ✓ {model}")
    
    if failed_downloads:
        logger.error("\nFailed downloads:")
        for model in failed_downloads:
            logger.error(f"  ✗ {model}")
    
    # Save model registry
    save_tts_model_registry(model_info['path'], successful_downloads)
    
    return len(failed_downloads) == 0


def save_tts_model_registry(tts_path: str, downloaded_models: list):
    """Save a registry of downloaded TTS models"""
    registry_path = Path(tts_path) / "model_registry.json"
    
    registry = {
        "downloaded_at": datetime.now().isoformat(),
        "models": {}
    }
    
    # Get model info for each downloaded model
    for model_name in downloaded_models:
        model_data = next(
            (m for m in MODELS["tts"]["models"] if m["name"] == model_name),
            None
        )
        if model_data:
            registry["models"][model_name] = {
                "description": model_data["description"],
                "languages": model_data.get("languages", []),
                "type": model_data.get("type", "tts"),
                "size_gb": model_data["size_gb"]
            }
    
    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Model registry saved to {registry_path}")


def calculate_model_hash(path: str) -> str:
    """Calculate SHA256 hash of model directory"""
    sha256_hash = hashlib.sha256()
    
    for root, _, files in os.walk(path):
        for file in sorted(files):  # Sort for consistency
            if file.endswith(('.bin', '.safetensors', '.pth', '.pt')):
                filepath = os.path.join(root, file)
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def save_model_info(model_type: str, info: Dict[str, str]) -> None:
    """Save model information for validation"""
    info_path = Path(MODELS[model_type]['path']) / "model_info.txt"
    
    with open(info_path, 'w') as f:
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Repository/Name: {info.get('repo_id', info.get('model_name', 'N/A'))}\n")
        f.write(f"Download Date: {info['date']}\n")
        f.write(f"Directory Hash: {info['hash']}\n")


def main():
    parser = argparse.ArgumentParser(description="Download models for LexAI")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["ultravox", "tts", "all"],
        default=["all"],
        help="Models to download"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip model validation after download"
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace API token"
    )
    
    args = parser.parse_args()
    
    # Determine which models to download
    if "all" in args.models:
        models_to_download = ["ultravox", "tts"]
    else:
        models_to_download = args.models
    
    # Get HF token
    token = args.hf_token or get_hf_token()
    
    # Warn if downloading Ultravox without token
    if "ultravox" in models_to_download and not token:
        logger.warning("=" * 60)
        logger.warning("WARNING: Ultravox requires a HuggingFace token!")
        logger.warning("The model is gated and requires authentication.")
        logger.warning("Get your token from: https://huggingface.co/settings/tokens")
        logger.warning("=" * 60)
        
        proceed = input("Continue without token? (y/N): ").strip().lower()
        if proceed != 'y':
            logger.info("Aborting download")
            return 1
    
    # Track results
    results = {}
    
    # Download models
    for model in models_to_download:
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading {model.upper()} model")
        logger.info(f"{'='*60}\n")
        
        if model == "ultravox":
            success = download_ultravox_model(token)
        elif model == "tts":
            success = download_tts_models()
        else:
            logger.error(f"Unknown model: {model}")
            success = False
        
        results[model] = success
        
        if success and not args.skip_validation:
            # Calculate and save model info
            from datetime import datetime
            
            logger.info(f"Calculating model hash for {model}...")
            model_hash = calculate_model_hash(MODELS[model]['path'])
            
            save_model_info(model, {
                'repo_id': MODELS[model].get('repo_id', MODELS[model].get('model_name')),
                'date': datetime.now().isoformat(),
                'hash': model_hash
            })
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Download Summary")
    logger.info(f"{'='*60}")
    
    for model, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"{model.upper()}: {status}")
    
    # Return success if all downloads succeeded
    if all(results.values()):
        logger.info("\nAll models downloaded successfully!")
        return 0
    else:
        logger.error("\nSome models failed to download.")
        return 1


if __name__ == "__main__":
    sys.exit(main())