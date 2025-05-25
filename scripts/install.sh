#!/bin/bash

# LexAI Installation Script
# Complete system setup for production deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/venv"
LOG_FILE="/var/log/lexai_install.log"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
}

# Check system requirements
check_system() {
    log "Checking system requirements..."
    
    # Check Ubuntu version
    if ! grep -q "Ubuntu" /etc/os-release; then
        error "This script is designed for Ubuntu systems"
    fi
    
    # Check GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        error "NVIDIA GPU drivers not found. Please install CUDA drivers first."
    fi
    
    # Check available disk space
    STORAGE_FREE=$(df -BG /mnt/storage | awk 'NR==2 {print $4}' | sed 's/G//')
    NVME_FREE=$(df -BG /opt/dlami/nvme | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "$STORAGE_FREE" -lt 50 ]; then
        error "Insufficient space in /mnt/storage. Need at least 50GB free."
    fi
    
    if [ "$NVME_FREE" -lt 20 ]; then
        error "Insufficient space in /opt/dlami/nvme. Need at least 20GB free."
    fi
    
    log "System requirements met"
}

# Install system dependencies
install_dependencies() {
    log "Installing system dependencies..."
    
    apt-get update
    apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        ffmpeg \
        libsndfile1 \
        libportaudio2 \
        libportaudiocpp0 \
        portaudio19-dev \
        libopenblas-dev \
        liblapack-dev \
        libasound2-dev \
        libssl-dev \
        libffi-dev \
        sox \
        libsox-fmt-mp3 \
        espeak-ng \
        mongodb-org \
        supervisor \
        nginx \
        htop \
        iotop \
        nvtop
    
    # Install Python build dependencies
    apt-get install -y \
        liblzma-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev
    
    log "System dependencies installed"
}

# Setup Python virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment
    python3.10 -m venv "$VENV_PATH"
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log "Python virtual environment created"
}

# Install Python packages
install_python_packages() {
    log "Installing Python packages..."
    
    source "$VENV_PATH/bin/activate"
    
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install project requirements
    cd "$PROJECT_ROOT"
    pip install -r requirements.txt
    
    # Install additional production packages
    pip install \
        gunicorn \
        uvloop \
        httptools \
        python-multipart \
        prometheus-client \
        psutil
    
    log "Python packages installed"
}

# Create directory structure
create_directories() {
    log "Creating directory structure..."
    
    # Storage directories
    mkdir -p /mnt/storage/{models,voices,audio,conversations,logs,backups}
    mkdir -p /mnt/storage/models/{ultravox,tts}
    mkdir -p /mnt/storage/voices/{samples,processed,embeddings}
    mkdir -p /mnt/storage/audio/{uploads,transcriptions,synthesized}
    
    # NVMe cache directories
    mkdir -p /opt/dlami/nvme/lexai/{cache,temp,sessions}
    
    # Log directories
    mkdir -p /var/log/lexai
    mkdir -p /var/log/mongodb
    
    # Set permissions
    chown -R ubuntu:ubuntu /mnt/storage
    chown -R ubuntu:ubuntu /opt/dlami/nvme/lexai
    chown -R ubuntu:ubuntu /var/log/lexai
    
    log "Directory structure created"
}

# Configure MongoDB
configure_mongodb() {
    log "Configuring MongoDB..."
    
    # Copy MongoDB configuration
    cp "$PROJECT_ROOT/mongodb/mongod.conf" /etc/mongod.conf
    
    # Create MongoDB data directory
    mkdir -p /mnt/storage/mongodb/data
    chown -R mongodb:mongodb /mnt/storage/mongodb
    
    # Enable and start MongoDB
    systemctl enable mongod
    systemctl restart mongod
    
    # Wait for MongoDB to start
    sleep 5
    
    # Initialize MongoDB
    "$SCRIPT_DIR/init_mongodb.sh"
    
    log "MongoDB configured"
}

# Download models
download_models() {
    log "Downloading AI models..."
    
    source "$VENV_PATH/bin/activate"
    cd "$PROJECT_ROOT"
    
    # Run model download script
    python scripts/download_models.py
    
    log "AI models downloaded"
}

# Configure firewall
configure_firewall() {
    log "Configuring firewall..."
    
    # Allow SSH
    ufw allow 22/tcp
    
    # Allow HTTP/HTTPS
    ufw allow 80/tcp
    ufw allow 443/tcp
    
    # Allow API port
    ufw allow 8000/tcp
    
    # Allow MongoDB (localhost only)
    ufw allow from 127.0.0.1 to any port 27017
    
    # Enable firewall
    ufw --force enable
    
    log "Firewall configured"
}

# Setup systemd service
setup_systemd() {
    log "Setting up systemd service..."
    
    cat > /etc/systemd/system/lexai.service << EOF
[Unit]
Description=LexAI Voice Assistant Service
After=network.target mongodb.service
Requires=mongodb.service

[Service]
Type=notify
User=ubuntu
Group=ubuntu
WorkingDirectory=$PROJECT_ROOT
Environment="PATH=$VENV_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=$PROJECT_ROOT"
ExecStart=$VENV_PATH/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
KillSignal=SIGTERM
Restart=always
RestartSec=10
StandardOutput=append:/var/log/lexai/lexai.log
StandardError=append:/var/log/lexai/lexai_error.log

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/mnt/storage /opt/dlami/nvme/lexai /var/log/lexai

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable lexai.service
    
    log "Systemd service configured"
}

# Setup log rotation
setup_logrotate() {
    log "Setting up log rotation..."
    
    cat > /etc/logrotate.d/lexai << EOF
/var/log/lexai/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ubuntu ubuntu
    sharedscripts
    postrotate
        systemctl reload lexai > /dev/null 2>&1 || true
    endscript
}
EOF
    
    log "Log rotation configured"
}

# Create environment file
create_env_file() {
    log "Creating environment configuration..."
    
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        
        # Note: SECRET_KEY not needed for HTTP-only deployment
        # Uncomment below if you plan to add authentication later:
        # SECRET_KEY=$(openssl rand -hex 32)
        # sed -i "s/your-secret-key-here/$SECRET_KEY/" "$PROJECT_ROOT/.env"
        
        warning "Please edit $PROJECT_ROOT/.env to add your HuggingFace token and other credentials"
    fi
    
    chown ubuntu:ubuntu "$PROJECT_ROOT/.env"
    chmod 600 "$PROJECT_ROOT/.env"
    
    log "Environment configuration created"
}

# Final setup
final_setup() {
    log "Performing final setup..."
    
    # Set ownership
    chown -R ubuntu:ubuntu "$PROJECT_ROOT"
    
    # Create run script
    cat > "$PROJECT_ROOT/run.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONPATH="$PWD"
exec python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
EOF
    
    chmod +x "$PROJECT_ROOT/run.sh"
    
    log "Final setup completed"
}

# Main installation flow
main() {
    log "Starting LexAI installation..."
    
    check_root
    check_system
    install_dependencies
    create_directories
    setup_venv
    install_python_packages
    configure_mongodb
    download_models
    configure_firewall
    setup_systemd
    setup_logrotate
    create_env_file
    final_setup
    
    log "Installation completed successfully!"
    log ""
    log "Next steps:"
    log "1. Edit /home/ubuntu/lexai/.env to add your HuggingFace token"
    log "2. Start the service: systemctl start lexai"
    log "3. Check status: systemctl status lexai"
    log "4. View logs: journalctl -u lexai -f"
    log ""
    log "API will be available at: http://localhost:8000"
    log "API documentation: http://localhost:8000/docs"
}

# Run main function
main "$@"