#!/bin/bash

# Install systemd service for LexAI
# Run with sudo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$PROJECT_DIR/lexai.service"
SYSTEMD_DIR="/etc/systemd/system"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing LexAI systemd service...${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root (use sudo)${NC}" 
   exit 1
fi

# Check if service file exists
if [[ ! -f "$SERVICE_FILE" ]]; then
    echo -e "${RED}Service file not found at $SERVICE_FILE${NC}"
    exit 1
fi

# Check if .env file exists
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    echo -e "${YELLOW}Warning: .env file not found at $PROJECT_DIR/.env${NC}"
    echo -e "${YELLOW}Make sure to create it before starting the service${NC}"
fi

# Stop service if it's running
if systemctl is-active --quiet lexai; then
    echo -e "${YELLOW}Stopping existing LexAI service...${NC}"
    systemctl stop lexai
fi

# Copy service file to systemd directory
echo -e "${GREEN}Copying service file to systemd...${NC}"
cp "$SERVICE_FILE" "$SYSTEMD_DIR/lexai.service"

# Set correct permissions
chmod 644 "$SYSTEMD_DIR/lexai.service"

# Reload systemd daemon
echo -e "${GREEN}Reloading systemd daemon...${NC}"
systemctl daemon-reload

# Enable service to start on boot
echo -e "${GREEN}Enabling LexAI service...${NC}"
systemctl enable lexai

echo -e "${GREEN}✅ LexAI service installed successfully!${NC}"
echo ""
echo -e "Available commands:"
echo -e "  ${GREEN}sudo systemctl start lexai${NC}    - Start the service"
echo -e "  ${GREEN}sudo systemctl stop lexai${NC}     - Stop the service"
echo -e "  ${GREEN}sudo systemctl restart lexai${NC}  - Restart the service"
echo -e "  ${GREEN}sudo systemctl status lexai${NC}   - Check service status"
echo -e "  ${GREEN}sudo journalctl -u lexai -f${NC}   - View service logs"
echo ""
echo -e "To start the service now, run:"
echo -e "  ${GREEN}sudo systemctl start lexai${NC}"