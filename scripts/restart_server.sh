#!/bin/bash

# LexAI Restart Server Script
# Gracefully restart the LexAI service

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/lexai/restart.log"

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

# Show usage
usage() {
    echo "Usage: $0 [--force|--dev|--reload]"
    echo ""
    echo "Options:"
    echo "  --force    Force restart without saving sessions"
    echo "  --dev      Restart development server"
    echo "  --reload   Reload configuration without full restart"
    echo ""
    echo "Examples:"
    echo "  $0              # Graceful restart with session saving"
    echo "  $0 --force      # Force restart immediately"
    echo "  $0 --reload     # Reload configuration only"
}

# Reload configuration
reload_config() {
    log "Reloading LexAI configuration..."
    
    if systemctl is-active --quiet lexai; then
        # Send HUP signal to reload config
        sudo systemctl reload lexai
        
        if [ $? -eq 0 ]; then
            log "Configuration reloaded successfully"
            return 0
        else
            warning "Configuration reload failed, performing full restart..."
            return 1
        fi
    else
        warning "Service not running, starting instead..."
        "$SCRIPT_DIR/start_server.sh"
        exit 0
    fi
}

# Main function
main() {
    # Parse arguments
    FORCE=false
    DEV=false
    RELOAD=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE=true
                shift
                ;;
            --dev)
                DEV=true
                shift
                ;;
            --reload)
                RELOAD=true
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
    
    log "Restarting LexAI server..."
    
    # Handle reload
    if [ "$RELOAD" = true ]; then
        if reload_config; then
            log "Reload completed successfully"
            exit 0
        fi
        # If reload failed, continue with restart
    fi
    
    # Prepare stop arguments
    STOP_ARGS=""
    if [ "$FORCE" = true ]; then
        STOP_ARGS="--force"
    fi
    if [ "$DEV" = true ]; then
        STOP_ARGS="$STOP_ARGS --dev"
    fi
    
    # Stop the server
    log "Stopping server..."
    "$SCRIPT_DIR/stop_server.sh" $STOP_ARGS
    
    # Wait a moment
    sleep 2
    
    # Start the server
    log "Starting server..."
    START_ARGS=""
    if [ "$DEV" = true ]; then
        START_ARGS="--dev"
    fi
    
    "$SCRIPT_DIR/start_server.sh" $START_ARGS
    
    log ""
    log "LexAI server restarted successfully"
}

# Run main function
main "$@"