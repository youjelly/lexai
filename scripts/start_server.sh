#!/bin/bash

# LexAI Start Server Script
# Start the LexAI service with proper checks

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
PID_FILE="/var/run/lexai.pid"
LOG_FILE="/var/log/lexai/startup.log"

# Logging functions
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

# Check if service is already running
check_running() {
    if systemctl is-active --quiet lexai; then
        warning "LexAI service is already running"
        systemctl status lexai
        exit 0
    fi
}

# Verify environment
verify_environment() {
    log "Verifying environment..."
    
    # Check virtual environment
    if [ ! -d "$VENV_PATH" ]; then
        error "Virtual environment not found. Run install.sh first."
    fi
    
    # Check .env file
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        error ".env file not found. Copy .env.example and configure it."
    fi
    
    # Check if HF token is set
    if ! grep -q "HF_TOKEN=hf_" "$PROJECT_ROOT/.env"; then
        warning "HuggingFace token not found in .env file"
    fi
    
    # Check MongoDB
    if ! systemctl is-active --quiet mongod; then
        log "Starting MongoDB..."
        sudo systemctl start mongod
        sleep 3
    fi
    
    # Check storage directories
    for dir in /mnt/storage/models /mnt/storage/voices /opt/dlami/nvme/lexai; do
        if [ ! -d "$dir" ]; then
            error "Required directory $dir not found. Run install.sh first."
        fi
    done
    
    # Check models
    if [ ! -f "/mnt/storage/models/ultravox/config.json" ]; then
        warning "Ultravox model not found. Models will be downloaded on first run."
    fi
    
    log "Environment verification completed"
}

# Check system resources
check_resources() {
    log "Checking system resources..."
    
    # Check GPU
    if ! nvidia-smi &> /dev/null; then
        error "NVIDIA GPU not accessible"
    fi
    
    # Check available memory
    AVAILABLE_MEM=$(free -g | awk 'NR==2{print $7}')
    if [ "$AVAILABLE_MEM" -lt 8 ]; then
        warning "Low available memory: ${AVAILABLE_MEM}GB. Recommended: 8GB+"
    fi
    
    # Check GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEM" -lt 8000 ]; then
        warning "Low GPU memory: ${GPU_MEM}MB. Recommended: 8GB+"
    fi
    
    # Check disk space
    STORAGE_FREE=$(df -BG /mnt/storage | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$STORAGE_FREE" -lt 10 ]; then
        warning "Low disk space on /mnt/storage: ${STORAGE_FREE}GB"
    fi
    
    log "Resource check completed"
}

# Start service using systemd
start_systemd() {
    log "Starting LexAI service via systemd..."
    
    sudo systemctl start lexai
    
    # Wait for service to start
    sleep 5
    
    # Check if service started successfully
    if systemctl is-active --quiet lexai; then
        log "LexAI service started successfully"
        systemctl status lexai --no-pager
    else
        error "Failed to start LexAI service"
    fi
}

# Start service manually (development mode)
start_manual() {
    log "Starting LexAI in development mode..."
    
    cd "$PROJECT_ROOT"
    source "$VENV_PATH/bin/activate"
    
    # Export environment variables
    export PYTHONPATH="$PROJECT_ROOT"
    export ENVIRONMENT="development"
    
    # Start with auto-reload for external access
    log "Starting development server with auto-reload enabled (external access)..."
    log "Server will be accessible externally on all interfaces (0.0.0.0:8000)"
    log "Make sure EC2 security group allows inbound traffic on port 8000"
    exec python -m uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --reload-dir lexai \
        --log-level info \
        --access-log \
        --use-colors
}

# Start service in production mode (no systemd)
start_production() {
    log "Starting LexAI in production mode..."
    
    cd "$PROJECT_ROOT"
    source "$VENV_PATH/bin/activate"
    
    # Export environment variables
    export PYTHONPATH="$PROJECT_ROOT"
    export ENVIRONMENT="production"
    
    # Load .env file
    if [ -f "$PROJECT_ROOT/.env" ]; then
        export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)
    fi
    
    # Start without auto-reload for production
    log "Starting production server..."
    log "Server will be accessible externally on all interfaces (0.0.0.0:8000)"
    exec python "$PROJECT_ROOT/main.py"
}

# Health check
health_check() {
    log "Performing health check..."
    
    # Wait a bit more for full initialization
    sleep 5
    
    # Check API health endpoint
    if curl -f -s http://localhost:8000/health > /dev/null; then
        log "API health check passed"
        
        # Get detailed health status
        HEALTH_STATUS=$(curl -s http://localhost:8000/health | python3 -m json.tool)
        log "Health status:"
        echo "$HEALTH_STATUS"
    else
        warning "API health check failed - service may still be initializing"
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [--dev|--production|--systemd]"
    echo ""
    echo "Options:"
    echo "  --dev         Start in development mode with auto-reload"
    echo "  --production  Start in production mode (no systemd, no auto-reload)"
    echo "  --systemd     Start using systemd service (default)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Start using systemd service"
    echo "  $0 --dev            # Start development server with auto-reload"
    echo "  $0 --production     # Start production server without systemd"
    echo "  $0 --systemd        # Explicitly use systemd"
}

# Main function
main() {
    # Parse arguments
    MODE="systemd"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                MODE="dev"
                shift
                ;;
            --systemd)
                MODE="systemd"
                shift
                ;;
            --production|--prod)
                MODE="production"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    log "Starting LexAI server (mode: $MODE)..."
    
    # Perform checks
    if [ "$MODE" == "systemd" ]; then
        check_running
    fi
    
    verify_environment
    check_resources
    
    # Start service
    case $MODE in
        systemd)
            start_systemd
            health_check
            
            log ""
            log "LexAI service started successfully!"
            log "Local API URL: http://localhost:8000"
            log "External API URL: http://\$EC2_PUBLIC_IP:8000 (replace with your EC2 public IP)"
            log "API Docs: http://localhost:8000/docs"
            log "Web Interface: http://\$EC2_PUBLIC_IP:8000/"
            log "Logs: journalctl -u lexai -f"
            log ""
            log "IMPORTANT: Ensure EC2 Security Group allows inbound traffic on port 8000"
            log "AWS Console > EC2 > Security Groups > Add Inbound Rule:"
            log "  Type: HTTP, Port: 8000, Source: 0.0.0.0/0 (or your IP range)"
            ;;
        dev)
            log ""
            log "Starting development server (external access enabled)..."
            log "Local API URL: http://localhost:8000"
            log "External API URL: http://\$EC2_PUBLIC_IP:8000 (replace with your EC2 public IP)"
            log "API Docs: http://localhost:8000/docs"
            log "Web Interface: http://\$EC2_PUBLIC_IP:8000/"
            log "Press Ctrl+C to stop"
            log ""
            log "IMPORTANT: Ensure EC2 Security Group allows inbound traffic on port 8000"
            start_manual
            ;;
        production)
            log ""
            log "Starting production server (no auto-reload)..."
            log "Local API URL: http://localhost:8000"
            log "External API URL: http://\$EC2_PUBLIC_IP:8000 (replace with your EC2 public IP)"
            log "API Docs: http://localhost:8000/docs"
            log "Web Interface: http://\$EC2_PUBLIC_IP:8000/"
            log "Press Ctrl+C to stop"
            log ""
            start_production
            ;;
    esac
}

# Run main function
main "$@"