#!/bin/bash
# ============================================
# AI FX Trading Bot - Install Script
# ============================================
# Usage:
#   chmod +x deploy/install.sh
#   sudo ./deploy/install.sh
#
# After install:
#   1. Edit /opt/ai-fx-bot/.env with your API keys
#   2. sudo systemctl start ai-fx-bot
#   3. sudo systemctl status ai-fx-bot
#   4. journalctl -u ai-fx-bot -f   (view logs)

set -e

INSTALL_DIR="/opt/ai-fx-bot"
SERVICE_NAME="ai-fx-bot"
CURRENT_USER="${SUDO_USER:-$USER}"

echo "============================================"
echo "  AI FX Trading Bot - Installer"
echo "============================================"

# Check root
if [ "$EUID" -ne 0 ]; then
    echo "Error: Please run with sudo"
    echo "  sudo ./deploy/install.sh"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python: $PYTHON_VERSION"

# Create install directory
echo ""
echo "[1/5] Creating install directory..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/data"

# Copy project files
echo "[2/5] Copying project files..."
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cp -r "$SCRIPT_DIR/main.py" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/src" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/config" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"

# Copy .env if not exists (don't overwrite existing config)
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$INSTALL_DIR/.env"
    echo "  Created .env from template - EDIT THIS FILE with your API keys!"
else
    echo "  .env already exists - keeping existing configuration"
fi

# Create virtual environment
echo "[3/5] Setting up Python virtual environment..."
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"
echo "  Dependencies installed"

# Set ownership
chown -R "$CURRENT_USER:$CURRENT_USER" "$INSTALL_DIR"

# Install systemd service
echo "[4/5] Installing systemd service..."
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=AI FX Trading Bot - Paper Trading Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python main.py
Restart=always
RestartSec=30

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# Security
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# Summary
echo "[5/5] Installation complete!"
echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "  Install dir: $INSTALL_DIR"
echo "  Service:     $SERVICE_NAME"
echo ""
echo "  NEXT STEPS:"
echo "  1. Edit API keys:"
echo "     sudo nano $INSTALL_DIR/.env"
echo ""
echo "  2. Start the bot:"
echo "     sudo systemctl start $SERVICE_NAME"
echo ""
echo "  3. Check status:"
echo "     sudo systemctl status $SERVICE_NAME"
echo ""
echo "  4. View live logs:"
echo "     journalctl -u $SERVICE_NAME -f"
echo ""
echo "  5. Stop the bot:"
echo "     sudo systemctl stop $SERVICE_NAME"
echo ""
echo "============================================"
