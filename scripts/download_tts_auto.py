#!/usr/bin/env python3
"""
Automated TTS model download script
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set TTS to use our storage location BEFORE importing TTS
os.environ['TTS_HOME'] = '/mnt/storage/models/tts'
os.environ['TTS_CACHE'] = '/opt/dlami/nvme/cache/tts'

# Create directories
os.makedirs("/mnt/storage/models/tts", exist_ok=True)
os.makedirs("/opt/dlami/nvme/cache/tts", exist_ok=True)

print("Setting up TTS model download...")
print(f"TTS_HOME: {os.environ['TTS_HOME']}")
print(f"TTS_CACHE: {os.environ['TTS_CACHE']}")

# Now import TTS
from TTS.utils.manage import ModelManager
from TTS.config import load_config
import json

print("\nDownloading XTTS v2 model...")
print("This will download ~2GB to /mnt/storage/models/tts")

# Create model manager
manager = ModelManager(models_file=None, output_prefix="/mnt/storage/models/tts", progress_bar=True)

# Download XTTS v2
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

try:
    # Set agreement environment variable to bypass prompt
    os.environ['COQUI_TOS_AGREED'] = '1'
    
    print(f"\nDownloading {model_name}...")
    model_path, config_path = manager.download_model(model_name)
    
    print(f"✅ Model downloaded to: {model_path}")
    print(f"✅ Config downloaded to: {config_path}")
    
    # Create a marker file
    marker_file = Path("/mnt/storage/models/tts/download_complete.txt")
    marker_file.write_text(f"XTTS v2 downloaded successfully\nModel: {model_path}\nConfig: {config_path}")
    
except Exception as e:
    print(f"❌ Failed to download: {e}")
    sys.exit(1)

print("\n✅ TTS model download complete!")
print("Models are stored in: /mnt/storage/models/tts")
print("You can now start the LexAI server.")