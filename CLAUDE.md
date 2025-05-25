# LexAI Project - Claude Session Documentation

## 🎯 Project Context

This is **LexAI**, a production-ready real-time multimodal voice AI assistant built as an Ultravox.ai clone. The project is deployed on AWS EC2 g6e instances with external internet access.

**Critical Information:**
- **External Public IP**: 3.129.5.177
- **Environment**: Production deployment on Ubuntu 22.04
- **Hardware**: AWS g6e instance with NVIDIA Ada Lovelace GPU
- **Status**: Fully functional and accessible from external browsers

## 🏗️ Complete Architecture

### **Core Technologies:**
- **AI Model**: Ultravox v0.5 Llama 3.1 8B (fixie-ai/ultravox-v0_5-llama-3_1-8b)
- **TTS**: Coqui TTS XTTS v2 with voice cloning
- **Backend**: FastAPI with WebSocket support
- **Database**: MongoDB for conversation storage
- **Frontend**: Modern web interface (HTML/CSS/JS)
- **Deployment**: Production-ready with monitoring and backup

### **Key URLs:**
- **Web Interface**: http://3.129.5.177:8000
- **API Docs**: http://3.129.5.177:8000/docs
- **WebSocket**: ws://3.129.5.177:8000/ws/audio/{session_id}
- **Health Check**: http://3.129.5.177:8000/api/health

## 🚨 Critical Storage Information

**The project is located in ephemeral storage** to avoid filling the root partition:
- **Project Location**: `/opt/dlami/nvme/lexai_new/`
- **Original Location**: `/home/ubuntu/lexai` (moved due to disk space issues)
- **Python Version**: Python 3.10 (required for TTS compatibility)

### Storage Configuration:
- **Ephemeral NVMe** (`/opt/dlami/nvme/`): Application, venv, pip cache, temp files
- **Persistent EBS** (`/mnt/storage/`): Models, MongoDB data, logs
- **Root Partition**: Only 40GB - DO NOT install large packages here!

## 📁 Project Structure

```
/opt/dlami/nvme/lexai_new/
├── 🎯 CORE APPLICATION
│   ├── lexai/                   # Main Python package
│   │   ├── api/                 # FastAPI application
│   │   │   ├── main.py          # FastAPI app with CORS and static files
│   │   │   ├── middleware/      # Security, CORS, rate limiting
│   │   │   └── routes/          # API endpoints
│   │   ├── models/              # AI model services
│   │   │   ├── ultravox_service.py  # Ultravox without quantization
│   │   │   └── model_manager.py     # Model loading/caching
│   │   ├── tts/                 # Text-to-speech
│   │   │   ├── coqui_tts.py     # XTTS v2 implementation
│   │   │   └── voice_cloning.py # Voice cloning functionality
│   │   ├── websocket/           # Real-time audio streaming
│   │   │   ├── connection_manager.py # WebSocket handling
│   │   │   ├── audio_streamer.py     # Audio processing
│   │   │   └── session_manager.py    # Session state
│   │   ├── database/            # MongoDB operations
│   │   └── utils/               # Logging and utilities
│   ├── config.py                # Environment-based configuration
│   ├── main.py                  # Production server entry point
│   └── requirements.txt         # Python dependencies
│
├── 🌐 WEB FRONTEND
│   └── static/                  # Web interface files
│       ├── index.html           # Main voice AI testing interface
│       ├── js/
│       │   ├── app.js           # WebSocket + core functionality
│       │   ├── audio-handler.js # Audio capture/processing
│       │   └── voice-manager.js # Voice cloning interface
│       └── css/
│           └── style.css        # Responsive modern styling
│
├── 🚀 DEPLOYMENT
│   ├── scripts/                 # Management scripts
│   │   ├── install.sh           # Complete system setup
│   │   ├── start_server.sh      # External access startup
│   │   ├── stop_server.sh       # Graceful shutdown
│   │   ├── restart_server.sh    # Service restart
│   │   ├── health_check.sh      # System monitoring
│   │   └── backup.sh            # Data backup
│   ├── monitoring/              # Production monitoring
│   │   ├── logging_config.py    # Structured logging
│   │   └── metrics.py           # Performance metrics
│   └── mongodb/                 # Database configuration
│
└── 📄 DOCUMENTATION
    ├── .env                     # Environment configuration
    ├── .env.example             # Template with options
    ├── .gitignore               # Git ignore patterns
    ├── PROJECT_SUMMARY.md       # Comprehensive overview
    └── CLAUDE.md                # This file
```

## ⚙️ Configuration Status

### **Environment Variables (.env):**
```bash
# EXTERNAL ACCESS CONFIGURED ✅
HOST=0.0.0.0                     # Binds to all interfaces
PORT=8000                        # Main HTTP port
EXTERNAL_HOST=3.129.5.177        # Public IP for WebSocket URLs
CORS_ORIGINS=["http://3.129.5.177:8000", "http://localhost:8000"]

# AI MODEL CONFIGURATION ✅
HF_TOKEN=hf_YOUR_TOKEN_HERE  # Replace with your HuggingFace token
MODEL_DEVICE=cuda                # GPU acceleration
MODEL_PRECISION=float16          # Memory optimization

# GPU OPTIMIZATION ✅
GPU_MEMORY_FRACTION=0.95         # Use 95% of GPU memory
MIXED_PRECISION=true             # TF32 optimization for Ada Lovelace

# PRODUCTION SETTINGS ✅
ENVIRONMENT=production           # Production mode
SECURITY_HEADERS=true           # Security headers enabled
CORS_ALLOW_CREDENTIALS=true     # For WebSocket auth
```

### **Critical Settings Applied:**
- ✅ **External Access**: Server binds to 0.0.0.0 for internet access
- ✅ **CORS Configuration**: Allows external browser connections
- ✅ **WebSocket URLs**: Use public IP instead of localhost
- ✅ **GPU Optimization**: TF32 enabled for g6e instances
- ✅ **Security**: Production headers and rate limiting
- ✅ **No Quantization**: Full model quality for Ultravox

## 🔧 Development Context

### **Important Implementation Details:**

1. **No Quantization Used**: Ultravox model runs at full fp16 precision for quality
2. **No VAD Required**: Multimodal model naturally handles speech pauses
3. **External WebSocket**: Fixed hardcoded localhost in system.py endpoints
4. **Dual Storage**: Persistent SSD (/mnt/storage) + ephemeral NVMe (/opt/dlami/nvme)
5. **Voice Cloning**: 10-30 second samples, multilingual support
6. **Real-time Audio**: WebSocket binary streaming with opus codec

### **Key Dependencies:**
```python
# Core AI
torch                    # PyTorch with CUDA support
transformers            # HuggingFace transformers
TTS                     # Coqui TTS library

# Web Framework
fastapi                 # API framework
uvicorn                 # ASGI server
websockets              # WebSocket support

# Database
motor                   # Async MongoDB driver
pymongo                 # MongoDB operations

# Audio Processing
librosa                 # Audio analysis
soundfile               # Audio I/O
```

## 🚀 Deployment Status

### **Services Running:**
- ✅ **FastAPI Server**: Port 8000, external access configured
- ✅ **WebSocket Server**: Integrated with FastAPI, external URLs
- ✅ **MongoDB**: Localhost-only for security
- ✅ **Static Files**: Web interface served at root (/)
- ✅ **Models**: Ultravox and TTS models loaded on startup

### **AWS Configuration:**
- ✅ **Security Group**: Port 8000 open for HTTP traffic
- ✅ **Instance Type**: g6e (GPU-optimized)
- ✅ **Storage**: EBS + Instance Store NVMe
- ✅ **Firewall**: ufw configured with proper rules

### **External Access Verified:**
- ✅ **Web Interface**: Accessible from any browser worldwide
- ✅ **API Endpoints**: All routes accept external requests
- ✅ **WebSocket**: Real-time audio streaming works externally
- ✅ **CORS**: Properly configured for cross-origin requests

## 🛠️ Common Commands

### **Service Management:**
```bash
# Start server (external access)
./scripts/start_server.sh

# Development mode with auto-reload
./scripts/start_server.sh --dev

# Health check
./scripts/health_check.sh --full

# Backup data
./scripts/backup.sh --full

# Restart service
./scripts/restart_server.sh
```

### **System Monitoring:**
```bash
# Check service status
systemctl status lexai

# View logs
journalctl -u lexai -f

# Check GPU usage
nvidia-smi

# Check disk space
df -h /mnt/storage /opt/dlami/nvme
```

### **Configuration Changes:**
```bash
# Edit environment
nano .env

# Restart after config change
./scripts/restart_server.sh

# Test configuration
python -c "from config import settings; print(settings.validate_system())"
```

## 🎯 User Workflows

### **Voice Conversation:**
1. User opens http://3.129.5.177:8000 in browser
2. Clicks "Start Recording" or presses Spacebar
3. Speaks to the AI naturally
4. Receives real-time voice responses
5. Views conversation history in chat interface

### **Voice Cloning:**
1. User clicks "Manage Voices" in settings
2. Uploads 10-30 seconds of clear speech samples
3. Names the custom voice
4. Waits for processing (2-5 minutes)
5. Selects custom voice for AI responses

### **API Integration:**
```javascript
// WebSocket for real-time audio
const ws = new WebSocket('ws://3.129.5.177:8000/ws/audio/new');

// REST API for voice cloning
const formData = new FormData();
formData.append('voice_name', 'MyVoice');
formData.append('audio_files', audioFile);

fetch('http://3.129.5.177:8000/api/tts/voices/clone', {
    method: 'POST',
    body: formData
});
```

## 🔍 Troubleshooting Guide

### **Common Issues:**

1. **"Cannot connect to server"**
   - Check AWS Security Group has port 8000 open
   - Verify server is running: `systemctl status lexai`
   - Check firewall: `ufw status`

2. **"WebSocket connection failed"**
   - Verify CORS origins in .env include client domain
   - Check external host setting: `EXTERNAL_HOST=3.129.5.177`
   - Test WebSocket endpoint manually

3. **"Model loading failed"**
   - Verify HuggingFace token is valid
   - Check GPU memory: `nvidia-smi`
   - Review logs: `journalctl -u lexai -f`

4. **"Audio not working"**
   - Check browser microphone permissions
   - Verify HTTPS not required (using HTTP)
   - Test audio capture in browser console

### **Performance Issues:**
- **GPU Memory**: Adjust `GPU_MEMORY_FRACTION` in .env
- **Audio Latency**: Check network connection quality
- **Model Speed**: Verify TF32 enabled for Ada Lovelace GPU

## 📊 Monitoring & Metrics

### **Health Monitoring:**
```bash
# Full system health check
./scripts/health_check.sh --full

# Returns status for:
# - System services (MongoDB, LexAI)
# - API endpoints responsiveness
# - System resources (CPU, Memory, GPU)
# - Storage availability
# - Database connectivity
# - Performance metrics
```

### **Log Analysis:**
```bash
# Application logs
tail -f /var/log/lexai/lexai.log

# Error logs
tail -f /var/log/lexai/lexai_error.log

# System service logs
journalctl -u lexai -f

# MongoDB logs
tail -f /var/log/mongodb/mongod.log
```

## 🔄 Next Session Guidelines

### **When Claude Starts Next Session:**

1. **Read this CLAUDE.md file first** to understand the complete project
2. **Check Python version**: Ensure using Python 3.10 (required for TTS)
   ```bash
   cd /opt/dlami/nvme/lexai_new
   source venv/bin/activate
   python --version  # Should show Python 3.10.x
   ```
3. **Check current status**: Run `./scripts/health_check.sh --full`
4. **Review logs**: Look for any errors or issues since last session
5. **Verify external access**: Test http://3.129.5.177:8000 works
6. **Understand environment**: Check .env file for current configuration

### **Key Files to Check:**
- `config.py` - Main configuration with external access settings
- `.env` - Environment variables including public IP and tokens
- `lexai/api/main.py` - FastAPI app with static file serving
- `static/index.html` - Web interface entry point
- `scripts/health_check.sh` - System validation tool

### **Common Next Steps:**
- Add new features to the voice AI
- Improve performance or add optimizations
- Add security enhancements (HTTPS, authentication)
- Scale the system or add new models
- Debug any issues that arose during operation

## ⚠️ Critical Notes

1. **Project Location**: `/opt/dlami/nvme/lexai_new/` (NOT in home directory)
2. **Python Version**: Python 3.10 REQUIRED (TTS won't work with 3.11+)
3. **Storage**: Use ephemeral NVMe for pip cache to avoid filling root partition
4. **External Access is FULLY CONFIGURED**: Anyone on the internet can access http://3.129.5.177:8000
2. **No Authentication**: Currently open access for testing/demo purposes
3. **HTTP Only**: No HTTPS/SSL configured (user preference)
4. **Production Ready**: All services configured for stability and monitoring
5. **GPU Optimized**: Specifically tuned for AWS g6e Ada Lovelace architecture
6. **Full Quality**: No model quantization - using fp16 for best quality

## 🎯 Project Status: COMPLETE & OPERATIONAL

✅ **All Systems Operational**
✅ **External Access Configured**
✅ **Web Interface Functional**
✅ **Voice AI Working**
✅ **Voice Cloning Available**
✅ **Production Monitoring Active**
✅ **Documentation Complete**

The LexAI system is fully deployed and accessible at **http://3.129.5.177:8000** for real-time voice AI conversations with external browser access.

## 📝 Recent Updates (Session 5/25/2025)

### **Major Improvements:**

1. **Text Input Testing Interface**
   - Added text input field with Send button for testing LLM + TTS pipeline
   - Implemented full text → Ultravox LLM → TTS pipeline
   - Added TTS toggle checkbox for enabling/disabling speech synthesis
   - Fixed message routing between connection_manager and audio_streamer

2. **TTS Model Compatibility**
   - Fixed XTTS v2 speaker issue (requires voice samples for synthesis)
   - Added fallback to VITS model for English when no voice sample available
   - Implemented automatic model switching based on requirements
   - Only one TTS model kept in memory at a time to save GPU RAM

3. **Performance Optimizations**
   - Added 8-bit quantization support for Ultravox (reduces from ~11GB to ~5GB)
   - Implemented singleton pattern for model services (prevents duplicate loading)
   - Added PyTorch expandable segments for better memory management
   - Reduced WebSocket/console logging for faster streaming
   - Larger audio chunks (200ms) for more efficient streaming

4. **Streaming Audio Playback**
   - Implemented Web Audio API for real-time audio streaming
   - Audio plays as chunks arrive instead of waiting for complete synthesis
   - Fixed PCM16 to Float32 conversion with proper little-endian byte order
   - Added audio queue management for smooth playback

### **Known Issues & TODOs:**

1. **Audio Quality**
   - Poor transcription quality from speech input (getting random characters)
   - Need to investigate audio capture/encoding from browser

2. **Memory Management**
   - Still tight on GPU memory with both models loaded
   - Consider using CPU for TTS or further quantization

3. **Voice Features**
   - XTTS v2 requires voice samples - need to implement default voice handling
   - Voice cloning interface needs testing

### **Configuration Updates:**
- Added `bitsandbytes>=0.41.0` to requirements for quantization
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for better memory
- Reduced WebSocket chunk delays for faster streaming
- Disabled verbose logging from uvicorn WebSocket implementation

### **Next Steps:**
1. Fix audio capture quality for speech input
2. Implement better default voice handling for XTTS v2
3. Add voice selection UI integration
4. Test and optimize memory usage further
5. Add error recovery for GPU OOM situations