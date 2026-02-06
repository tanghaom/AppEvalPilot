#!/bin/bash
# Run OSAgent with Xvfb virtual display

# Check if Xvfb is installed
if ! command -v Xvfb &> /dev/null; then
    echo "Error: Xvfb is not installed."
    echo "Please install it with: sudo apt-get install -y xvfb"
    exit 1
fi

# Check if Xvfb is already running on display :99
if ! ps aux | grep -i "[X]vfb :99" > /dev/null; then
    echo "Starting Xvfb on display :99..."
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    sleep 2
    if ps aux | grep -i "[X]vfb :99" > /dev/null; then
        echo "Xvfb started successfully on display :99"
    else
        echo "Warning: Failed to start Xvfb"
    fi
else
    echo "Xvfb is already running on display :99"
fi

# Set DISPLAY environment variable
export DISPLAY=:99
echo "DISPLAY=$DISPLAY"

# Set CONFIG_ROOT environment variable
export CONFIG_ROOT=/root/.metagpt
echo "CONFIG_ROOT=$CONFIG_ROOT"

# Fix Anaconda libffi conflict for pyatspi/gi
if [ -f /lib/x86_64-linux-gnu/libffi.so.7 ]; then
    export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
    echo "LD_PRELOAD=$LD_PRELOAD (fix for Anaconda libffi conflict)"
fi

# Change to project directory
cd "$(dirname "$0")/.." || exit 1

# Activate virtual environment and run the script
source venv/bin/activate
python scripts/run_osagent.py "$@"
