#!/bin/bash

# Script to configure MongoDB systemd service for optimal performance
# This ensures MongoDB uses our custom configuration and storage paths

set -e

echo "=== MongoDB Service Configuration for LexAI ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run this script with sudo${NC}"
    exit 1
fi

# Create systemd override directory
echo -e "${YELLOW}Creating systemd override directory...${NC}"
mkdir -p /etc/systemd/system/mongod.service.d

# Create override configuration
echo -e "${YELLOW}Creating service override configuration...${NC}"
cat > /etc/systemd/system/mongod.service.d/override.conf <<'EOF'
[Unit]
Description=MongoDB Database Server for LexAI
After=network-online.target
Wants=network-online.target

[Service]
# Performance tuning
LimitNOFILE=64000
LimitNPROC=64000
TasksMax=infinity

# Memory settings
LimitMEMLOCK=infinity
LimitAS=infinity

# Ensure proper environment
Environment="MONGODB_MEMORY_LIMIT=8G"

# Restart policy
Restart=always
RestartSec=10

# Create runtime directory
RuntimeDirectory=mongodb
RuntimeDirectoryMode=0755

# Working directory
WorkingDirectory=/mnt/storage/mongodb

# Additional security
PrivateTmp=false
ProtectHome=false
ProtectSystem=false

# Allow access to storage paths
ReadWritePaths=/mnt/storage/mongodb
ReadWritePaths=/mnt/storage/logs/mongodb
ReadWritePaths=/opt/dlami/nvme/cache/mongodb_temp

[Install]
WantedBy=multi-user.target
EOF

# Create MongoDB environment file
echo -e "${YELLOW}Creating MongoDB environment configuration...${NC}"
cat > /etc/default/mongod <<'EOF'
# MongoDB environment configuration for LexAI

# Disable transparent huge pages
TRANSPARENT_HUGEPAGE=never

# NUMA settings (for multi-CPU systems)
NUMA_INTERLEAVE=1

# Additional MongoDB options
MONGO_OPTS="--quiet"
EOF

# Create startup script for system tuning
echo -e "${YELLOW}Creating system tuning script...${NC}"
cat > /usr/local/bin/mongodb-tuning.sh <<'EOF'
#!/bin/bash
# System tuning for MongoDB

# Disable transparent huge pages
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# Set swappiness low for database workloads
echo 1 > /proc/sys/vm/swappiness

# Increase file descriptor limits
ulimit -n 64000
ulimit -u 64000

# Set dirty ratio for better write performance
echo 15 > /proc/sys/vm/dirty_ratio
echo 10 > /proc/sys/vm/dirty_background_ratio

# Enable memory overcommit
echo 1 > /proc/sys/vm/overcommit_memory
EOF

chmod +x /usr/local/bin/mongodb-tuning.sh

# Create systemd service for tuning
echo -e "${YELLOW}Creating system tuning service...${NC}"
cat > /etc/systemd/system/mongodb-tuning.service <<'EOF'
[Unit]
Description=MongoDB System Tuning
Before=mongod.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/mongodb-tuning.sh
RemainAfterExit=yes

[Install]
WantedBy=mongod.service
EOF

# Create log rotation configuration
echo -e "${YELLOW}Creating log rotation configuration...${NC}"
cat > /etc/logrotate.d/mongodb <<'EOF'
/mnt/storage/logs/mongodb/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 640 mongodb mongodb
    sharedscripts
    postrotate
        /bin/kill -SIGUSR1 $(cat /var/run/mongodb/mongod.pid 2>/dev/null) 2>/dev/null || true
    endscript
}
EOF

# Reload systemd
echo -e "${YELLOW}Reloading systemd configuration...${NC}"
systemctl daemon-reload

# Enable services
echo -e "${YELLOW}Enabling services...${NC}"
systemctl enable mongodb-tuning.service
systemctl enable mongod.service

# Apply tuning immediately
echo -e "${YELLOW}Applying system tuning...${NC}"
/usr/local/bin/mongodb-tuning.sh

# Restart MongoDB with new configuration
echo -e "${YELLOW}Restarting MongoDB service...${NC}"
systemctl restart mongod

# Wait for MongoDB to be ready
echo "Waiting for MongoDB to be ready..."
for i in {1..30}; do
    if mongosh --quiet --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
        echo -e "${GREEN}MongoDB is ready${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Show service status
echo -e "${YELLOW}MongoDB service status:${NC}"
systemctl status mongod --no-pager

echo -e "${GREEN}MongoDB service configuration completed!${NC}"
echo ""
echo "The following optimizations have been applied:"
echo "  ✓ Increased file descriptor limits"
echo "  ✓ Disabled transparent huge pages"
echo "  ✓ Optimized VM settings for database workloads"
echo "  ✓ Configured automatic log rotation"
echo "  ✓ Set up automatic restart on failure"
echo ""
echo "To check MongoDB status: systemctl status mongod"
echo "To view logs: journalctl -u mongod -f"
echo "To view MongoDB logs: tail -f /mnt/storage/logs/mongodb/mongod.log"