# LexAI Installation Guide

## Storage Considerations for AWS EC2

When installing LexAI on AWS EC2, storage management is critical to avoid filling the root partition.

### Recommended Setup

**Clone the repository to ephemeral storage** to avoid root partition issues:

```bash
# For AWS EC2 with ephemeral NVMe storage
cd /opt/dlami/nvme
git clone https://github.com/yourusername/lexai.git
cd lexai

# For systems with limited root storage, use any large partition
# cd /mnt/storage  # or wherever you have space
# git clone https://github.com/yourusername/lexai.git
# cd lexai
```

### Storage Layout

LexAI uses a dual-storage approach:

1. **Ephemeral Storage** (`/opt/dlami/nvme/`) - Fast NVMe for:
   - Application code
   - Virtual environment
   - Pip cache
   - Temporary files
   - Audio processing cache

2. **Persistent Storage** (`/mnt/storage/`) - EBS/persistent for:
   - AI models (17GB+)
   - MongoDB data
   - Voice samples
   - Conversation history
   - Logs

### Quick Installation

```bash
# 1. Ensure you're on ephemeral storage
cd /opt/dlami/nvme/lexai

# 2. Run the setup script (uses ephemeral storage for pip)
./setup_venv.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Configure environment
cp .env.example .env
nano .env  # Add your HuggingFace token and configure settings

# 5. Start MongoDB (if not running)
sudo systemctl start mongodb

# 6. Download models (optional, 17GB)
python scripts/download_models.py

# 7. Start the server
python main.py
```

### Storage Requirements

- **Root partition**: 5GB minimum free (for system packages)
- **Ephemeral storage**: 50GB+ recommended (for venv, cache, temp)
- **Persistent storage**: 50GB+ recommended (for models, data)

### Troubleshooting Storage Issues

If you encounter "No space left on device" errors:

1. **Check disk usage**:
   ```bash
   df -h
   ```

2. **Clear pip cache**:
   ```bash
   pip cache purge
   ```

3. **Move installation to ephemeral storage**:
   ```bash
   cd /opt/dlami/nvme
   mv /home/ubuntu/lexai .
   ```

4. **Configure pip to use ephemeral storage**:
   ```bash
   export PIP_CACHE_DIR=/opt/dlami/nvme/pip_cache
   export TMPDIR=/opt/dlami/nvme/tmp
   ```

### Python Version

**Python 3.10 is required** for TTS compatibility. The setup script will check for this.

If Python 3.10 is not installed:
```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
```

### Full System Installation

For a complete system setup (requires root):
```bash
sudo ./scripts/install.sh
```

This will:
- Install system dependencies
- Set up MongoDB
- Configure systemd service
- Create all necessary directories
- Set proper permissions