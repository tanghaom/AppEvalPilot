#!/bin/bash
# Quick test script for TextAgent (text-only, no screenshots sent to LLM)
# Uses accessibility tree / DOM tree for perception instead of images.
# Much cheaper (text-only LLM) and faster (no OCR, no image encoding).

export DISPLAY=:99
export CONFIG_ROOT=/root/.metagpt
export BROWSER=/usr/bin/google-chrome
# Fix Anaconda libffi conflict for pyatspi/gi
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7

# Prevent OpenCV Qt plugin from crashing under Xvfb
export QT_QPA_PLATFORM=offscreen

# AT-SPI accessibility support
export GTK_MODULES=gail:atk-bridge
export GNOME_ACCESSIBILITY=1
export NO_AT_BRIDGE=0

# Start Xvfb virtual display (if not already running)
if ! pgrep -f "Xvfb :99" > /dev/null 2>&1; then
    echo "🖥️  Starting Xvfb on display :99 ..."
    Xvfb :99 -screen 0 1920x1080x24 &
    sleep 2
    echo "✅ Xvfb started (PID: $!)"
else
    echo "✅ Xvfb already running on :99"
fi

source venv/bin/activate
echo "🤖 Running TextAgent (text-only, a11y tree mode)..."
python main_text.py --platform Linux --max_iters 10

