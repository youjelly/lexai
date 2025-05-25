#!/bin/bash

# MongoDB initialization script for LexAI
# This script sets up MongoDB with proper directories, permissions, and users

set -e

echo "=== MongoDB Initialization for LexAI ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if running as root or with sudo
check_sudo() {
    if [ "$EUID" -ne 0 ]; then 
        echo -e "${RED}Please run this script with sudo${NC}"
        exit 1
    fi
}

# Function to create directories
create_directories() {
    echo -e "${YELLOW}Creating MongoDB directories...${NC}"
    
    # Create persistent storage directories
    mkdir -p /mnt/storage/mongodb
    mkdir -p /mnt/storage/logs/mongodb
    
    # Create fast ephemeral storage directories
    mkdir -p /opt/dlami/nvme/cache/mongodb_temp
    
    # Set ownership to mongodb user
    chown -R mongodb:mongodb /mnt/storage/mongodb
    chown -R mongodb:mongodb /mnt/storage/logs/mongodb
    chown -R mongodb:mongodb /opt/dlami/nvme/cache/mongodb_temp
    
    # Set proper permissions
    chmod 755 /mnt/storage/mongodb
    chmod 755 /mnt/storage/logs/mongodb
    chmod 755 /opt/dlami/nvme/cache/mongodb_temp
    
    echo -e "${GREEN}Directories created successfully${NC}"
}

# Function to backup existing config
backup_config() {
    if [ -f /etc/mongod.conf ]; then
        echo -e "${YELLOW}Backing up existing MongoDB configuration...${NC}"
        cp /etc/mongod.conf /etc/mongod.conf.backup.$(date +%Y%m%d_%H%M%S)
        echo -e "${GREEN}Backup created${NC}"
    fi
}

# Function to install new config
install_config() {
    echo -e "${YELLOW}Installing MongoDB configuration...${NC}"
    
    # Copy the new configuration
    cp /home/ubuntu/lexai/mongodb/mongod.conf /etc/mongod.conf
    
    # Set proper ownership and permissions
    chown root:root /etc/mongod.conf
    chmod 644 /etc/mongod.conf
    
    echo -e "${GREEN}Configuration installed${NC}"
}

# Function to stop MongoDB if running
stop_mongodb() {
    echo -e "${YELLOW}Stopping MongoDB service...${NC}"
    systemctl stop mongod || true
    sleep 2
}

# Function to start MongoDB with new config
start_mongodb() {
    echo -e "${YELLOW}Starting MongoDB with new configuration...${NC}"
    
    # Ensure MongoDB starts on boot
    systemctl enable mongod
    
    # Start MongoDB
    systemctl start mongod
    
    # Wait for MongoDB to be ready
    echo "Waiting for MongoDB to be ready..."
    for i in {1..30}; do
        if mongosh --quiet --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
            echo -e "${GREEN}MongoDB is ready${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    echo -e "${RED}MongoDB failed to start${NC}"
    journalctl -u mongod -n 50
    exit 1
}

# Function to create MongoDB users
create_users() {
    echo -e "${YELLOW}Creating MongoDB users...${NC}"
    
    # Check if authentication is already enabled
    if mongosh --quiet --eval "db.adminCommand('ping')" 2>&1 | grep -q "authentication"; then
        echo "Authentication already enabled, skipping user creation"
        return 0
    fi
    
    # Create admin user
    mongosh admin <<EOF
db.createUser({
    user: "lexai_admin",
    pwd: "$(openssl rand -base64 32)",
    roles: [
        { role: "userAdminAnyDatabase", db: "admin" },
        { role: "dbAdminAnyDatabase", db: "admin" },
        { role: "readWriteAnyDatabase", db: "admin" },
        { role: "clusterAdmin", db: "admin" }
    ]
})
EOF
    
    # Create application user
    mongosh lexai <<EOF
db.createUser({
    user: "lexai_app",
    pwd: "$(openssl rand -base64 32)",
    roles: [
        { role: "readWrite", db: "lexai" },
        { role: "dbAdmin", db: "lexai" }
    ]
})
EOF
    
    echo -e "${GREEN}Users created successfully${NC}"
    echo -e "${YELLOW}IMPORTANT: Save the generated passwords from the output above!${NC}"
}

# Function to initialize database
init_database() {
    echo -e "${YELLOW}Initializing LexAI database...${NC}"
    
    cd /home/ubuntu/lexai
    
    # Create initialization Python script
    cat > /tmp/init_db.py <<'EOF'
import asyncio
import sys
sys.path.append('/home/ubuntu/lexai')

from lexai.database.connection import init_db
from lexai.database.migrations import initialize_database

async def main():
    try:
        await init_db()
        await initialize_database()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF
    
    # Run initialization
    if [ -f "venv/bin/python" ]; then
        sudo -u ubuntu venv/bin/python /tmp/init_db.py
    else
        echo -e "${YELLOW}Virtual environment not found. Please run setup_venv.sh first${NC}"
    fi
    
    rm -f /tmp/init_db.py
}

# Function to verify installation
verify_installation() {
    echo -e "${YELLOW}Verifying MongoDB installation...${NC}"
    
    # Check service status
    if systemctl is-active --quiet mongod; then
        echo -e "${GREEN}✓ MongoDB service is running${NC}"
    else
        echo -e "${RED}✗ MongoDB service is not running${NC}"
        return 1
    fi
    
    # Check connectivity
    if mongosh --quiet --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ MongoDB is accessible${NC}"
    else
        echo -e "${RED}✗ Cannot connect to MongoDB${NC}"
        return 1
    fi
    
    # Check directories
    if [ -d "/mnt/storage/mongodb" ] && [ -d "/mnt/storage/logs/mongodb" ]; then
        echo -e "${GREEN}✓ Storage directories exist${NC}"
    else
        echo -e "${RED}✗ Storage directories missing${NC}"
        return 1
    fi
    
    # Check log file
    if [ -f "/mnt/storage/logs/mongodb/mongod.log" ]; then
        echo -e "${GREEN}✓ Log file created${NC}"
    else
        echo -e "${YELLOW}! Log file not yet created (will be created on first write)${NC}"
    fi
    
    echo -e "${GREEN}MongoDB setup completed successfully!${NC}"
}

# Main execution
main() {
    check_sudo
    
    echo -e "${YELLOW}This script will configure MongoDB for LexAI${NC}"
    echo "It will:"
    echo "  - Create storage directories"
    echo "  - Install custom MongoDB configuration"
    echo "  - Restart MongoDB service"
    echo "  - Create database users (optional)"
    echo "  - Initialize the database schema (optional)"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 0
    fi
    
    create_directories
    backup_config
    stop_mongodb
    install_config
    start_mongodb
    
    # Optional: Create users
    read -p "Create MongoDB users? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_users
    fi
    
    # Optional: Initialize database
    read -p "Initialize LexAI database schema? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        init_database
    fi
    
    verify_installation
    
    echo ""
    echo "Next steps:"
    echo "1. If you created users, update .env with the MongoDB credentials"
    echo "2. Restart the MongoDB service if you make any config changes: sudo systemctl restart mongod"
    echo "3. Monitor logs at: /mnt/storage/logs/mongodb/mongod.log"
}

# Run main function
main