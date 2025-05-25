# LexAI - Ultravox.ai Clone

Real-time multimodal AI voice assistant built with PyTorch, FastAPI, and MongoDB.

## Requirements

- **Python 3.10** (required for TTS compatibility)
- CUDA-compatible GPU (tested on AWS g6e instances)
- Ubuntu 22.04 or later
- 16GB+ RAM recommended
- 50GB+ storage for models

## Features

- Real-time speech-to-text and text-to-speech
- Multimodal AI processing with Ultravox models
- WebSocket streaming for low-latency communication
- MongoDB integration for conversation history
- GPU-accelerated inference on AWS g6e instances

## Installation

**⚠️ Important**: Clone this repository to ephemeral storage on AWS EC2 to avoid filling the root partition.

See [INSTALLATION.md](INSTALLATION.md) for detailed storage setup instructions.

### Quick Start

```bash
# Clone to ephemeral storage (AWS EC2)
cd /opt/dlami/nvme
git clone https://github.com/yourusername/lexai.git
cd lexai

# Run setup
./setup_venv.sh
source venv/bin/activate

# Configure and start
cp .env.example .env
nano .env  # Add HuggingFace token
python main.py
```

## Project Structure

- `lexai/models/` - Ultravox model integration
- `lexai/tts/` - Coqui TTS implementation
- `lexai/api/` - FastAPI routes
- `lexai/websocket/` - Real-time streaming
- `lexai/database/` - MongoDB integration
- `lexai/utils/` - Helper utilities

## License

MIT License