#!/bin/bash

# Setup Python virtual environment for LexAI

echo "Setting up Python virtual environment for LexAI..."

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $PYTHON_VERSION is installed, but Python $REQUIRED_VERSION or higher is required."
    exit 1
fi

# Create virtual environment
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old environment..."
    rm -rf venv
fi

echo "Creating virtual environment with Python $PYTHON_VERSION..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support for g6e instance (Ada Lovelace architecture)
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install pydantic-settings separately as it's not in requirements.txt
pip install pydantic-settings

# Install additional dependencies for model support
echo "Installing additional dependencies..."
pip install accelerate GPUtil

# Create necessary directories
echo "Creating project directories..."

# Persistent storage directories
sudo mkdir -p /mnt/storage/{models/{ultravox,tts},mongodb,voices,audio/{conversations,sessions,voice_samples},logs/{mongodb,api,app}}

# Fast ephemeral storage directories (NVMe)
sudo mkdir -p /opt/dlami/nvme/{audio_processing,cache/{ultravox,tts,mongodb_temp},temp}

# Set appropriate permissions (assuming ubuntu user)
sudo chown -R ubuntu:ubuntu /mnt/storage
sudo chown -R ubuntu:ubuntu /opt/dlami/nvme

# Make scripts executable
chmod +x main.py
chmod +x scripts/download_models.py

# Download models (optional - can be skipped for faster setup)
echo ""
echo "Do you want to download the AI models now? (y/n)"
echo "Note: This will download ~17GB of model data and may take a while."
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Downloading models..."
    python scripts/download_models.py --models all
    
    if [ $? -eq 0 ]; then
        echo "Models downloaded successfully!"
    else
        echo "Model download failed. You can run 'python scripts/download_models.py' manually later."
    fi
else
    echo "Skipping model download. You can run 'python scripts/download_models.py' later."
fi

echo ""
echo "Setup complete! To activate the virtual environment, run:"
echo "    source venv/bin/activate"
echo ""
echo "To start the server, run:"
echo "    python main.py"
echo ""
echo "Don't forget to:"
echo "1. Copy .env.example to .env and configure your settings"
echo "2. Ensure MongoDB is running on localhost:27017"
echo "3. Download models if you haven't: python scripts/download_models.py"