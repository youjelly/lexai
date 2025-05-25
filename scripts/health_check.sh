#!/bin/bash

# LexAI Health Check Script
# Comprehensive system health validation

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
API_URL="http://localhost:8000"

# Status tracking
TOTAL_CHECKS=0
PASSED_CHECKS=0
WARNINGS=0

# Output functions
success() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

failure() {
    echo -e "${RED}✗${NC} $1"
    ((TOTAL_CHECKS++))
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

header() {
    echo
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check system services
check_services() {
    header "System Services"
    
    # Check MongoDB
    if systemctl is-active --quiet mongod; then
        success "MongoDB is running"
    else
        failure "MongoDB is not running"
    fi
    
    # Check LexAI service
    if systemctl is-active --quiet lexai; then
        success "LexAI service is running"
        
        # Get service uptime
        UPTIME=$(systemctl show lexai --property=ActiveEnterTimestamp | cut -d'=' -f2)
        if [ -n "$UPTIME" ]; then
            info "Service started: $UPTIME"
        fi
    else
        failure "LexAI service is not running"
    fi
}

# Check API endpoints
check_api() {
    header "API Health"
    
    # Basic health check
    if curl -f -s "$API_URL/health" > /dev/null; then
        success "API health endpoint responding"
        
        # Get detailed health
        HEALTH=$(curl -s "$API_URL/health")
        STATUS=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")
        
        if [ "$STATUS" = "healthy" ]; then
            success "API status: healthy"
        else
            warning "API status: $STATUS"
        fi
    else
        failure "API health endpoint not responding"
    fi
    
    # Check models endpoint
    if curl -f -s "$API_URL/models" > /dev/null; then
        success "Models endpoint responding"
        
        # Check loaded models
        MODELS=$(curl -s "$API_URL/models")
        ULTRAVOX_LOADED=$(echo "$MODELS" | grep -q "ultravox.*loaded.*true" && echo "yes" || echo "no")
        TTS_LOADED=$(echo "$MODELS" | grep -q "tts.*loaded.*true" && echo "yes" || echo "no")
        
        if [ "$ULTRAVOX_LOADED" = "yes" ]; then
            success "Ultravox model loaded"
        else
            warning "Ultravox model not loaded"
        fi
        
        if [ "$TTS_LOADED" = "yes" ]; then
            success "TTS model loaded"
        else
            warning "TTS model not loaded"
        fi
    else
        failure "Models endpoint not responding"
    fi
    
    # Check WebSocket endpoint
    if curl -f -s -o /dev/null -w "%{http_code}" "$API_URL/ws" | grep -q "426"; then
        success "WebSocket endpoint available (upgrade required)"
    else
        warning "WebSocket endpoint may not be properly configured"
    fi
}

# Check system resources
check_resources() {
    header "System Resources"
    
    # Check CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    if (( $(echo "$CPU_USAGE < 80" | bc -l) )); then
        success "CPU usage: ${CPU_USAGE}%"
    else
        warning "High CPU usage: ${CPU_USAGE}%"
    fi
    
    # Check memory
    MEM_INFO=$(free -m | awk 'NR==2{printf "%.1f", $3/$2*100}')
    MEM_AVAILABLE=$(free -g | awk 'NR==2{print $7}')
    if (( $(echo "$MEM_INFO < 80" | bc -l) )); then
        success "Memory usage: ${MEM_INFO}% (${MEM_AVAILABLE}GB available)"
    else
        warning "High memory usage: ${MEM_INFO}% (${MEM_AVAILABLE}GB available)"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1)
        GPU_UTIL=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
        GPU_MEM_USED=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
        GPU_MEM_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)
        GPU_MEM_PERCENT=$(echo "scale=1; $GPU_MEM_USED / $GPU_MEM_TOTAL * 100" | bc)
        
        success "GPU utilization: ${GPU_UTIL}%"
        if (( $(echo "$GPU_MEM_PERCENT < 90" | bc -l) )); then
            success "GPU memory: ${GPU_MEM_USED}MB / ${GPU_MEM_TOTAL}MB (${GPU_MEM_PERCENT}%)"
        else
            warning "High GPU memory usage: ${GPU_MEM_PERCENT}%"
        fi
    else
        failure "GPU not available"
    fi
}

# Check storage
check_storage() {
    header "Storage"
    
    # Check /mnt/storage
    STORAGE_USAGE=$(df -h /mnt/storage | awk 'NR==2 {print $5}' | sed 's/%//')
    STORAGE_FREE=$(df -BG /mnt/storage | awk 'NR==2 {print $4}')
    if [ "$STORAGE_USAGE" -lt 90 ]; then
        success "/mnt/storage: ${STORAGE_USAGE}% used (${STORAGE_FREE} free)"
    else
        warning "/mnt/storage: ${STORAGE_USAGE}% used (${STORAGE_FREE} free)"
    fi
    
    # Check /opt/dlami/nvme
    if [ -d "/opt/dlami/nvme" ]; then
        NVME_USAGE=$(df -h /opt/dlami/nvme | awk 'NR==2 {print $5}' | sed 's/%//')
        NVME_FREE=$(df -BG /opt/dlami/nvme | awk 'NR==2 {print $4}')
        if [ "$NVME_USAGE" -lt 90 ]; then
            success "/opt/dlami/nvme: ${NVME_USAGE}% used (${NVME_FREE} free)"
        else
            warning "/opt/dlami/nvme: ${NVME_USAGE}% used (${NVME_FREE} free)"
        fi
    fi
    
    # Check model files
    if [ -f "/mnt/storage/models/ultravox/config.json" ]; then
        success "Ultravox model files present"
    else
        failure "Ultravox model files missing"
    fi
    
    if [ -d "/mnt/storage/models/tts" ] && [ "$(ls -A /mnt/storage/models/tts)" ]; then
        success "TTS model files present"
    else
        failure "TTS model files missing"
    fi
}

# Check logs
check_logs() {
    header "Logs"
    
    # Check for recent errors
    LOG_FILE="/var/log/lexai/lexai_error.log"
    if [ -f "$LOG_FILE" ]; then
        RECENT_ERRORS=$(tail -n 100 "$LOG_FILE" | grep -i "error" | wc -l)
        if [ "$RECENT_ERRORS" -eq 0 ]; then
            success "No recent errors in logs"
        else
            warning "Found $RECENT_ERRORS recent errors in logs"
            info "Check: tail -n 50 $LOG_FILE"
        fi
    else
        info "Error log file not found"
    fi
    
    # Check log rotation
    if [ -f "/etc/logrotate.d/lexai" ]; then
        success "Log rotation configured"
    else
        warning "Log rotation not configured"
    fi
}

# Check database
check_database() {
    header "Database"
    
    # Check MongoDB connection
    if mongo --eval "db.stats()" >/dev/null 2>&1; then
        success "MongoDB connection successful"
        
        # Check collections
        DB_STATUS=$(mongo lexai --eval "db.getCollectionNames()" --quiet 2>/dev/null || echo "[]")
        if echo "$DB_STATUS" | grep -q "conversations"; then
            success "Database collections exist"
            
            # Get document counts
            CONV_COUNT=$(mongo lexai --eval "db.conversations.count()" --quiet 2>/dev/null || echo "0")
            info "Conversations in database: $CONV_COUNT"
        else
            warning "Database collections not initialized"
        fi
    else
        failure "Cannot connect to MongoDB"
    fi
}

# Performance test
performance_test() {
    header "Performance Test"
    
    # Test API response time
    if command -v curl &> /dev/null; then
        RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' "$API_URL/health" 2>/dev/null || echo "999")
        if (( $(echo "$RESPONSE_TIME < 1.0" | bc -l) )); then
            success "API response time: ${RESPONSE_TIME}s"
        else
            warning "Slow API response time: ${RESPONSE_TIME}s"
        fi
    fi
    
    # Test model inference endpoint
    if [ "$1" = "--full" ]; then
        info "Running full performance test..."
        
        # Test transcription
        TEST_RESULT=$(curl -s -X POST "$API_URL/audio/test-transcribe" 2>/dev/null || echo "{}")
        if echo "$TEST_RESULT" | grep -q "success"; then
            success "Transcription test passed"
        else
            warning "Transcription test failed or unavailable"
        fi
    fi
}

# Security check
check_security() {
    header "Security"
    
    # Check firewall
    if command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then
        success "Firewall is active"
    else
        warning "Firewall is not active"
    fi
    
    # Check .env permissions
    if [ -f "$PROJECT_ROOT/.env" ]; then
        PERM=$(stat -c %a "$PROJECT_ROOT/.env")
        if [ "$PERM" = "600" ]; then
            success ".env file permissions correct (600)"
        else
            warning ".env file permissions too permissive ($PERM)"
        fi
    else
        failure ".env file not found"
    fi
    
    # Check SSL/TLS
    if [ -f "/etc/nginx/sites-enabled/lexai" ]; then
        if grep -q "ssl_certificate" /etc/nginx/sites-enabled/lexai; then
            success "SSL/TLS configured"
        else
            info "SSL/TLS not configured (HTTP only)"
        fi
    fi
}

# Summary
show_summary() {
    header "Health Check Summary"
    
    echo
    echo "Total checks: $TOTAL_CHECKS"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Failed: ${RED}$((TOTAL_CHECKS - PASSED_CHECKS))${NC}"
    echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
    echo
    
    HEALTH_PERCENT=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
    
    if [ "$HEALTH_PERCENT" -gt 90 ]; then
        echo -e "${GREEN}System health: EXCELLENT ($HEALTH_PERCENT%)${NC}"
    elif [ "$HEALTH_PERCENT" -gt 70 ]; then
        echo -e "${YELLOW}System health: GOOD ($HEALTH_PERCENT%)${NC}"
    elif [ "$HEALTH_PERCENT" -gt 50 ]; then
        echo -e "${YELLOW}System health: FAIR ($HEALTH_PERCENT%)${NC}"
    else
        echo -e "${RED}System health: POOR ($HEALTH_PERCENT%)${NC}"
    fi
}

# Usage
usage() {
    echo "Usage: $0 [--full|--quick]"
    echo ""
    echo "Options:"
    echo "  --full     Run complete health check including performance tests"
    echo "  --quick    Run quick health check (default)"
    echo ""
    echo "Examples:"
    echo "  $0              # Quick health check"
    echo "  $0 --full       # Full health check with performance tests"
}

# Main function
main() {
    # Parse arguments
    MODE="quick"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                MODE="full"
                shift
                ;;
            --quick)
                MODE="quick"
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
    
    echo -e "${BLUE}LexAI Health Check${NC}"
    echo -e "${BLUE}==================${NC}"
    echo "Timestamp: $(date)"
    echo "Mode: $MODE"
    
    # Run checks
    check_services
    check_api
    check_resources
    check_storage
    check_logs
    check_database
    
    if [ "$MODE" = "full" ]; then
        performance_test --full
    else
        performance_test
    fi
    
    check_security
    
    # Show summary
    show_summary
}

# Run main function
main "$@"