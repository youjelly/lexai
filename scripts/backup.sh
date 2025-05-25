#!/bin/bash

# LexAI Backup Script
# Backup critical data and configurations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_ROOT="/mnt/storage/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/lexai_backup_$TIMESTAMP"
MAX_BACKUPS=30  # Keep last 30 backups

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Create backup directory
create_backup_dir() {
    log "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"/{config,database,voices,logs,models}
}

# Backup configuration files
backup_config() {
    log "Backing up configuration files..."
    
    # Copy configuration files
    cp "$PROJECT_ROOT/.env" "$BACKUP_DIR/config/" 2>/dev/null || warning ".env file not found"
    cp "$PROJECT_ROOT/config.py" "$BACKUP_DIR/config/"
    cp -r "$PROJECT_ROOT/scripts" "$BACKUP_DIR/config/"
    
    # MongoDB configuration
    cp /etc/mongod.conf "$BACKUP_DIR/config/" 2>/dev/null || warning "MongoDB config not found"
    
    # Systemd service files
    cp /etc/systemd/system/lexai.service "$BACKUP_DIR/config/" 2>/dev/null || true
    
    log "Configuration backup completed"
}

# Backup database
backup_database() {
    log "Backing up MongoDB database..."
    
    # Check if MongoDB is running
    if ! systemctl is-active --quiet mongod; then
        warning "MongoDB is not running, skipping database backup"
        return
    fi
    
    # Dump database
    mongodump \
        --db lexai \
        --out "$BACKUP_DIR/database" \
        --quiet \
        2>/dev/null || warning "MongoDB backup failed"
    
    # Get database stats
    mongo lexai --quiet --eval "db.stats()" > "$BACKUP_DIR/database/stats.json" 2>/dev/null || true
    
    log "Database backup completed"
}

# Backup voice files
backup_voices() {
    log "Backing up voice files..."
    
    # Copy voice samples and embeddings
    if [ -d "/mnt/storage/voices" ]; then
        rsync -a --info=progress2 \
            /mnt/storage/voices/ \
            "$BACKUP_DIR/voices/" \
            --exclude="*.tmp" \
            --exclude="processing/*"
    else
        warning "Voice directory not found"
    fi
    
    log "Voice backup completed"
}

# Backup important logs
backup_logs() {
    log "Backing up recent logs..."
    
    # Copy last 7 days of logs
    find /var/log/lexai -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/logs/" \; 2>/dev/null || true
    
    # Compress logs
    if [ "$(ls -A "$BACKUP_DIR/logs")" ]; then
        tar -czf "$BACKUP_DIR/logs.tar.gz" -C "$BACKUP_DIR" logs/
        rm -rf "$BACKUP_DIR/logs"
    fi
    
    log "Log backup completed"
}

# Backup model metadata
backup_model_metadata() {
    log "Backing up model metadata..."
    
    # Don't backup actual models (too large), just metadata
    if [ -d "/mnt/storage/models" ]; then
        # Find and copy config files
        find /mnt/storage/models -name "config.json" -o -name "*.yaml" -o -name "*.yml" | while read -r file; do
            relative_path="${file#/mnt/storage/models/}"
            mkdir -p "$BACKUP_DIR/models/$(dirname "$relative_path")"
            cp "$file" "$BACKUP_DIR/models/$relative_path"
        done
    fi
    
    # Save model inventory
    cat > "$BACKUP_DIR/models/inventory.txt" << EOF
Model Inventory - $(date)
========================

Ultravox Model:
$(ls -la /mnt/storage/models/ultravox 2>/dev/null || echo "Not found")

TTS Models:
$(ls -la /mnt/storage/models/tts 2>/dev/null || echo "Not found")

Total Size:
$(du -sh /mnt/storage/models 2>/dev/null || echo "N/A")
EOF
    
    log "Model metadata backup completed"
}

# Create backup manifest
create_manifest() {
    log "Creating backup manifest..."
    
    cat > "$BACKUP_DIR/manifest.json" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "1.0",
    "hostname": "$(hostname)",
    "backup_type": "$1",
    "components": {
        "config": true,
        "database": $([ -d "$BACKUP_DIR/database" ] && echo "true" || echo "false"),
        "voices": $([ -d "$BACKUP_DIR/voices" ] && echo "true" || echo "false"),
        "logs": $([ -f "$BACKUP_DIR/logs.tar.gz" ] && echo "true" || echo "false"),
        "models": $([ -d "$BACKUP_DIR/models" ] && echo "true" || echo "false")
    },
    "size": "$(du -sh "$BACKUP_DIR" | cut -f1)",
    "files": $(find "$BACKUP_DIR" -type f | wc -l)
}
EOF
}

# Compress backup
compress_backup() {
    log "Compressing backup..."
    
    cd "$BACKUP_ROOT"
    tar -czf "lexai_backup_$TIMESTAMP.tar.gz" "lexai_backup_$TIMESTAMP"
    
    # Remove uncompressed directory
    rm -rf "$BACKUP_DIR"
    
    BACKUP_SIZE=$(du -h "$BACKUP_ROOT/lexai_backup_$TIMESTAMP.tar.gz" | cut -f1)
    log "Backup compressed to $BACKUP_SIZE"
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Count existing backups
    BACKUP_COUNT=$(find "$BACKUP_ROOT" -name "lexai_backup_*.tar.gz" -type f | wc -l)
    
    if [ "$BACKUP_COUNT" -gt "$MAX_BACKUPS" ]; then
        # Calculate how many to delete
        DELETE_COUNT=$((BACKUP_COUNT - MAX_BACKUPS))
        
        # Delete oldest backups
        find "$BACKUP_ROOT" -name "lexai_backup_*.tar.gz" -type f -printf '%T+ %p\n' | \
            sort | head -n "$DELETE_COUNT" | cut -d' ' -f2- | \
            while read -r backup; do
                log "Removing old backup: $(basename "$backup")"
                rm -f "$backup"
            done
    fi
}

# Verify backup
verify_backup() {
    log "Verifying backup..."
    
    BACKUP_FILE="$BACKUP_ROOT/lexai_backup_$TIMESTAMP.tar.gz"
    
    if [ ! -f "$BACKUP_FILE" ]; then
        error "Backup file not found: $BACKUP_FILE"
    fi
    
    # Test archive integrity
    if tar -tzf "$BACKUP_FILE" > /dev/null 2>&1; then
        log "Backup verification passed"
    else
        error "Backup verification failed - archive is corrupted"
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [--full|--quick|--config-only]"
    echo ""
    echo "Options:"
    echo "  --full         Full backup including database and voices"
    echo "  --quick        Quick backup (config + logs only)"
    echo "  --config-only  Configuration files only"
    echo ""
    echo "Examples:"
    echo "  $0              # Default full backup"
    echo "  $0 --quick      # Quick backup for daily use"
    echo "  $0 --config-only # Before configuration changes"
}

# Main function
main() {
    # Parse arguments
    BACKUP_TYPE="full"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                BACKUP_TYPE="full"
                shift
                ;;
            --quick)
                BACKUP_TYPE="quick"
                shift
                ;;
            --config-only)
                BACKUP_TYPE="config"
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
    
    log "Starting LexAI backup (type: $BACKUP_TYPE)"
    
    # Create backup directory
    create_backup_dir
    
    # Perform backup based on type
    case $BACKUP_TYPE in
        full)
            backup_config
            backup_database
            backup_voices
            backup_logs
            backup_model_metadata
            ;;
        quick)
            backup_config
            backup_logs
            ;;
        config)
            backup_config
            ;;
    esac
    
    # Create manifest
    create_manifest "$BACKUP_TYPE"
    
    # Compress backup
    compress_backup
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Verify backup
    verify_backup
    
    log ""
    log "Backup completed successfully!"
    log "Location: $BACKUP_ROOT/lexai_backup_$TIMESTAMP.tar.gz"
    log "Size: $(du -h "$BACKUP_ROOT/lexai_backup_$TIMESTAMP.tar.gz" | cut -f1)"
}

# Run main function
main "$@"