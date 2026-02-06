#!/bin/bash
# Start Xvfb virtual display for headless Linux servers

# Check if Xvfb is installed
if ! command -v Xvfb &> /dev/null; then
    echo "Xvfb is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y xvfb
fi

# Check if Xvfb is already running on display :99
if ps aux | grep -i "[X]vfb :99" > /dev/null; then
    echo "Xvfb is already running on display :99"
    export DISPLAY=:99
else
    echo "Starting Xvfb on display :99..."
    Xvfb :99 -screen 0 1024x768x24 &
    sleep 2
    export DISPLAY=:99
    echo "Xvfb started. DISPLAY=$DISPLAY"
fi

