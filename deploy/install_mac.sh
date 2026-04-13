#!/bin/bash
# ============================================
# AI FX Trading Bot - macOS Install Script
# ============================================
# Usage:
#   chmod +x deploy/install_mac.sh
#   ./deploy/install_mac.sh
#
# After install:
#   1. Edit ~/.ai-fx-bot/.env with your API keys
#   2. launchctl load ~/Library/LaunchAgents/com.ai-fx-bot.plist
#   3. tail -f ~/.ai-fx-bot/logs/trading.log

set -e

INSTALL_DIR="$HOME/.ai-fx-bot"
PLIST_NAME="com.ai-fx-bot"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

echo "============================================"
echo "  AI FX Trading Bot - macOS Installer"
echo "============================================"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON=$(command -v python3)
elif command -v python &> /dev/null; then
    PYTHON=$(command -v python)
else
    echo "Error: Python 3 is required"
    echo "  Install: brew install python"
    exit 1
fi

PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python: $PYTHON_VERSION ($PYTHON)"

# Create install directory
echo ""
echo "[1/5] Creating install directory..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/data"

# Copy project files
echo "[2/5] Copying project files..."
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cp "$SCRIPT_DIR/main.py" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/src" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/config" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"

# Copy .env if not exists
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$INSTALL_DIR/.env"
    echo "  Created .env from template - EDIT THIS FILE!"
else
    echo "  .env already exists - keeping existing configuration"
fi

# Create virtual environment
echo "[3/5] Setting up Python virtual environment..."
$PYTHON -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"
echo "  Dependencies installed"

# Create launchd plist
echo "[4/5] Creating launchd service..."
mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DIR}/venv/bin/python</string>
        <string>${INSTALL_DIR}/main.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${INSTALL_DIR}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>StandardOutPath</key>
    <string>${INSTALL_DIR}/logs/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${INSTALL_DIR}/logs/stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${INSTALL_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

echo "  Created: $PLIST_PATH"

# Summary
echo "[5/5] Installation complete!"
echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "  Install dir: $INSTALL_DIR"
echo "  Service:     $PLIST_NAME"
echo ""
echo "  NEXT STEPS:"
echo ""
echo "  1. Edit API keys & email settings:"
echo "     nano $INSTALL_DIR/.env"
echo ""
echo "  2. Start the bot:"
echo "     launchctl load $PLIST_PATH"
echo ""
echo "  3. Check if running:"
echo "     launchctl list | grep ai-fx-bot"
echo ""
echo "  4. View logs:"
echo "     tail -f $INSTALL_DIR/logs/trading.log"
echo ""
echo "  5. Stop the bot:"
echo "     launchctl unload $PLIST_PATH"
echo ""
echo "============================================"
