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

1. Create and activate a virtual environment:
```bash
./setup_venv.sh
source venv/bin/activate
```

2. Install the package:
```bash
pip install -e .
```

3. Copy environment configuration:
```bash
cp .env.example .env
```

4. Start the server:
```bash
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