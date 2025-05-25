# LexAI - Real-time Multimodal Voice AI Assistant

## 🎯 Project Overview

LexAI is a production-ready real-time multimodal voice AI assistant that serves as an Ultravox.ai clone. It enables natural voice conversations with AI using state-of-the-art speech recognition, language modeling, and text-to-speech synthesis.

**Key Features:**
- 🎤 Real-time voice input with WebSocket streaming
- 🤖 Ultravox v0.5 Llama 3.1 8B multimodal model integration
- 🗣️ Coqui TTS with voice cloning capabilities
- 🌐 Web-based interface accessible from external browsers
- 🔄 Live conversation management with persistent storage
- 🌍 Multilingual support (10+ languages)
- 📱 Mobile-responsive design

## 🏗️ System Architecture

### **Core Components:**
1. **FastAPI Backend** - REST API and WebSocket server
2. **Ultravox Model Service** - Speech-to-speech AI model
3. **Coqui TTS Service** - Text-to-speech with voice cloning
4. **MongoDB Database** - Conversation and session storage
5. **Web Frontend** - Modern React-like interface
6. **WebSocket Audio Streaming** - Real-time bidirectional audio

### **Deployment Environment:**
- **Platform**: AWS EC2 g6e instances (GPU-optimized)
- **OS**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA Ada Lovelace architecture (RTX 4090/L4)
- **Storage**: Dual-tier (persistent SSD + ephemeral NVMe)
- **External Access**: HTTP/WebSocket on public IP

## 📁 Project Structure

```
lexai/
├── 📁 lexai/                    # Core application package
│   ├── 📁 api/                  # FastAPI routes and middleware
│   │   ├── main.py              # FastAPI app with static file serving
│   │   ├── middleware/          # CORS, security, rate limiting
│   │   └── routes/              # API endpoints (health, audio, tts, etc.)
│   ├── 📁 database/             # MongoDB connection and operations
│   ├── 📁 models/               # AI model management and inference
│   │   ├── ultravox_service.py  # Ultravox model without quantization
│   │   └── model_manager.py     # Model loading and caching
│   ├── 📁 tts/                  # Text-to-speech services
│   │   ├── coqui_tts.py        # XTTS v2 implementation
│   │   └── voice_cloning.py    # Voice cloning functionality
│   ├── 📁 websocket/           # Real-time audio streaming
│   │   ├── connection_manager.py # WebSocket connection handling
│   │   ├── audio_streamer.py   # Audio processing pipeline
│   │   └── session_manager.py  # Session state management
│   └── 📁 utils/               # Utilities and logging
├── 📁 static/                   # Web frontend files
│   ├── index.html              # Main web interface
│   ├── 📁 js/                  # JavaScript modules
│   │   ├── app.js              # Main app with WebSocket handling
│   │   ├── audio-handler.js    # Audio capture and processing
│   │   └── voice-manager.js    # Voice cloning interface
│   └── 📁 css/                 # Responsive styling
├── 📁 scripts/                 # Deployment and management scripts
│   ├── install.sh              # Complete system setup
│   ├── start_server.sh         # Server startup with external access
│   ├── health_check.sh         # System health monitoring
│   └── backup.sh               # Data backup automation
├── 📁 monitoring/              # Production monitoring
│   ├── logging_config.py       # Structured logging setup
│   └── metrics.py              # Performance metrics collection
├── 📁 mongodb/                 # Database configuration
├── config.py                   # Environment-based configuration
├── main.py                     # Production server entry point
└── requirements.txt            # Python dependencies
```

## 🔧 Configuration

### **Environment Variables (.env):**
```bash
# External Access
HOST=0.0.0.0
PORT=8000
EXTERNAL_HOST=3.129.5.177
CORS_ORIGINS=["http://3.129.5.177:8000", "http://localhost:8000"]

# AI Models
HF_TOKEN=hf_YOUR_TOKEN_HERE
MODEL_DEVICE=cuda
MODEL_PRECISION=float16

# Performance
GPU_MEMORY_FRACTION=0.95
MIXED_PRECISION=true
MAX_WORKERS=4
```

### **Key Configuration Features:**
- ✅ External access configured for public IP
- ✅ CORS enabled for internet browsers
- ✅ GPU optimization for g6e instances
- ✅ HuggingFace token for gated models
- ✅ Security headers for production

## 🚀 Deployment

### **Quick Start:**
```bash
# 1. Install system dependencies and setup
sudo ./scripts/install.sh

# 2. Configure environment
cp .env.example .env
# Edit .env with your HuggingFace token

# 3. Start the server
./scripts/start_server.sh

# 4. Access web interface
# External: http://3.129.5.177:8000
# Local: http://localhost:8000
```

### **Management Commands:**
```bash
./scripts/health_check.sh --full    # System health check
./scripts/backup.sh --full          # Create full backup
./scripts/restart_server.sh         # Graceful restart
./scripts/stop_server.sh            # Stop services
```

## 🌐 External Access

### **Public URLs:**
- **Web Interface**: http://3.129.5.177:8000
- **API Documentation**: http://3.129.5.177:8000/docs
- **WebSocket**: ws://3.129.5.177:8000/ws/audio/{session_id}
- **Health Check**: http://3.129.5.177:8000/api/health

### **AWS Security Group Requirements:**
- Port 8000 (HTTP) - Source: 0.0.0.0/0
- Port 22 (SSH) - Source: Your IP
- Port 27017 (MongoDB) - Source: 127.0.0.1/32 (localhost only)

## 🎤 Usage

### **Voice Conversation Flow:**
1. Open web interface in browser
2. Click "Start Recording" or press Spacebar
3. Speak naturally to the AI
4. Receive real-time AI voice responses
5. View conversation history in chat interface

### **Voice Cloning:**
1. Upload 10-30 seconds of clear speech samples
2. Name your custom voice
3. Wait for processing (2-5 minutes)
4. Select custom voice for AI responses

### **API Usage:**
```javascript
// WebSocket connection for real-time audio
const ws = new WebSocket('ws://3.129.5.177:8000/ws/audio/session123');

// REST API for voice management
await fetch('http://3.129.5.177:8000/api/tts/voices', {
    method: 'POST',
    body: formData
});
```

## 📊 Performance

### **System Requirements:**
- **GPU Memory**: 8GB+ VRAM (for Ultravox model)
- **System RAM**: 16GB+ recommended
- **Storage**: 50GB+ for models and data
- **Bandwidth**: 1Mbps+ for real-time audio

### **Model Specifications:**
- **Ultravox**: 8B parameters, fp16 precision, no quantization
- **TTS**: XTTS v2 multilingual with voice cloning
- **Audio**: 16kHz sampling, opus codec, real-time streaming

## 🔒 Security

### **Production Security Features:**
- ✅ CORS configured for specific origins
- ✅ Security headers (XSS, CSRF protection)
- ✅ Rate limiting per IP address
- ✅ Input validation and sanitization
- ✅ Secure secret management via environment variables

### **Network Security:**
- ✅ Firewall configured (ufw)
- ✅ MongoDB bound to localhost only
- ✅ Non-privileged user execution
- ✅ Secure file permissions

## 📈 Monitoring

### **Built-in Monitoring:**
- **Health Checks**: System, GPU, storage, database
- **Metrics Collection**: API performance, model inference times
- **Logging**: Structured JSON logs with rotation
- **Backup**: Automated conversation and configuration backup

### **Log Locations:**
- Application: `/var/log/lexai/lexai.log`
- Errors: `/var/log/lexai/lexai_error.log`
- System: `journalctl -u lexai -f`

## 🛠️ Development

### **Local Development:**
```bash
# Start in development mode
./scripts/start_server.sh --dev

# This enables:
# - Auto-reload on code changes
# - Debug logging
# - CORS for localhost
```

### **Model Development:**
- Ultravox model: No quantization for full quality
- Voice cloning: 10-30 second samples recommended
- Language support: 10+ languages with auto-detection

## 🔄 Maintenance

### **Regular Tasks:**
- **Daily**: Check logs for errors
- **Weekly**: Run health check and backup
- **Monthly**: Update dependencies and models
- **Quarterly**: Security audit and performance review

### **Common Issues:**
- **GPU Memory**: Adjust `GPU_MEMORY_FRACTION` in .env
- **Audio Quality**: Check microphone settings and network
- **Model Loading**: Verify HuggingFace token and connectivity

## 📋 Version History

- **v0.1.0** (Current): Initial production release
  - Ultravox v0.5 Llama 3.1 8B integration
  - Coqui TTS with voice cloning
  - Web interface with external access
  - Production deployment on AWS g6e

## 🎯 Future Roadmap

- [ ] HTTPS/WSS support for encrypted connections
- [ ] User authentication and multi-tenancy
- [ ] Advanced voice cloning with fewer samples
- [ ] Mobile app development
- [ ] Integration with external AI services
- [ ] Performance optimization and caching