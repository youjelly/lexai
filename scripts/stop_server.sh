#!/bin/bash

# LexAI Stop Server Script
# Gracefully stop the LexAI service

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/var/log/lexai/shutdown.log"

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

# Check if service is running
check_service() {
    if ! systemctl is-active --quiet lexai; then
        warning "LexAI service is not running"
        return 1
    fi
    return 0
}

# Save active sessions
save_sessions() {
    log "Saving active sessions..."
    
    # Call API endpoint to save sessions
    if curl -f -s -X POST http://localhost:8000/system/save-sessions > /dev/null 2>&1; then
        log "Active sessions saved"
    else
        warning "Could not save sessions - API may be unresponsive"
    fi
}

# Stop service gracefully
stop_service() {
    log "Stopping LexAI service..."
    
    # Send graceful shutdown signal
    sudo systemctl stop lexai
    
    # Wait for service to stop (max 30 seconds)
    local count=0
    while systemctl is-active --quiet lexai && [ $count -lt 30 ]; do
        sleep 1
        ((count++))
        echo -n "."
    done
    echo
    
    if systemctl is-active --quiet lexai; then
        warning "Service did not stop gracefully, forcing shutdown..."
        sudo systemctl kill lexai
        sleep 2
    fi
    
    if ! systemctl is-active --quiet lexai; then
        log "LexAI service stopped successfully"
    else
        error "Failed to stop LexAI service"
    fi
}

# Clean up temporary files
cleanup_temp() {
    log "Cleaning up temporary files..."
    
    # Clean NVMe cache (keep structure)
    if [ -d "/opt/dlami/nvme/lexai/temp" ]; then
        find /opt/dlami/nvme/lexai/temp -type f -mtime +1 -delete 2>/dev/null || true
    fi
    
    # Clean old session data
    if [ -d "/opt/dlami/nvme/lexai/sessions" ]; then
        find /opt/dlami/nvme/lexai/sessions -type f -mtime +7 -delete 2>/dev/null || true
    fi
    
    log "Temporary files cleaned"
}

# Stop development server
stop_dev_server() {
    log "Stopping development server..."
    
    # Find uvicorn processes
    PIDS=$(pgrep -f "uvicorn main:app" || true)
    
    if [ -z "$PIDS" ]; then
        warning "No development server processes found"
        return
    fi
    
    # Send SIGTERM to processes
    for pid in $PIDS; do
        log "Stopping process $pid..."
        kill -TERM "$pid" 2>/dev/null || true
    done
    
    # Wait for processes to stop
    sleep 2
    
    # Check if any processes remain
    REMAINING=$(pgrep -f "uvicorn main:app" || true)
    if [ -n "$REMAINING" ]; then
        warning "Some processes did not stop gracefully, forcing..."
        for pid in $REMAINING; do
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi
    
    log "Development server stopped"
}

# Show usage
usage() {
    echo "Usage: $0 [--force|--dev]"
    echo ""
    echo "Options:"
    echo "  --force    Force stop without saving sessions"
    echo "  --dev      Stop development server"
    echo ""
    echo "Examples:"
    echo "  $0              # Graceful stop with session saving"
    echo "  $0 --force      # Force stop immediately"
    echo "  $0 --dev        # Stop development server"
}

# Main function
main() {
    # Parse arguments
    FORCE=false
    DEV=false
    
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
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    log "Stopping LexAI server..."
    
    if [ "$DEV" = true ]; then
        stop_dev_server
    else
        # Check if service is running
        if check_service; then
            # Save sessions unless forced
            if [ "$FORCE" = false ]; then
                save_sessions
            else
                warning "Skipping session save (forced stop)"
            fi
            
            # Stop service
            stop_service
        fi
    fi
    
    # Cleanup
    cleanup_temp
    
    log ""
    log "LexAI server stopped"
}

# Run main function
main "$@"