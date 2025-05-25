#!/usr/bin/env python3
"""
Simple TTS model download script
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set TTS to use our storage location
os.environ['TTS_HOME'] = '/mnt/storage/models/tts'
os.environ['TTS_CACHE'] = '/opt/dlami/nvme/cache/tts'

# Import TTS
from TTS.api import TTS

# Create directories
os.makedirs("/mnt/storage/models/tts", exist_ok=True)
os.makedirs("/opt/dlami/nvme/cache/tts", exist_ok=True)

print("Downloading XTTS v2 model...")
print("This will download ~2GB and may take a while...")

# Initialize TTS with XTTS v2
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
    print("✅ XTTS v2 model downloaded successfully!")
    
    # Create a marker file to indicate successful download
    marker_file = Path("/mnt/storage/models/tts/xtts_v2_downloaded.txt")
    marker_file.write_text("XTTS v2 model downloaded successfully")
    
except Exception as e:
    print(f"❌ Failed to download XTTS v2: {e}")
    exit(1)

print("\nTTS model download complete!")
print("You can now start the LexAI server.")