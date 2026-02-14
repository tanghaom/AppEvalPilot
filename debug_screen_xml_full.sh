#!/bin/bash
# Full debug script for screen XML extraction with Xvfb and Chrome

set -e

DISPLAY_NUM=999
CHROME_PORT=9999
TEST_URL="https://example.com"

echo "=========================================="
echo "Debug Screen XML Extraction (Full Setup)"
echo "=========================================="
echo

# Fix Anaconda libffi conflict for pyatspi/gi
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7

# Kill any existing processes
echo "Cleaning up old processes..."
pkill -f "Xvfb :$DISPLAY_NUM" 2>/dev/null || true
pkill -f "remote-debugging-port=$CHROME_PORT" 2>/dev/null || true
pkill -f "at-spi-bus-launcher" 2>/dev/null || true
sleep 1

# Start Xvfb
echo "Starting Xvfb on :$DISPLAY_NUM..."
Xvfb :$DISPLAY_NUM -screen 0 1920x1080x24 &
XVFB_PID=$!
export DISPLAY=:$DISPLAY_NUM
echo "✅ Xvfb started (PID: $XVFB_PID)"
sleep 2

# Start D-Bus session bus (required for AT-SPI)
echo "Starting D-Bus session bus..."
eval $(dbus-launch --sh-syntax)
echo "✅ D-Bus started (PID: $DBUS_SESSION_BUS_PID)"

# Start AT-SPI bus launcher (required for accessibility tree)
echo "Starting AT-SPI bus launcher..."
/usr/libexec/at-spi-bus-launcher &
ATSPI_PID=$!
echo "✅ AT-SPI bus launcher started (PID: $ATSPI_PID)"
sleep 2

# Start AT-SPI registryd
echo "Starting AT-SPI registry daemon..."
/usr/libexec/at-spi2-registryd &
REGISTRYD_PID=$!
echo "✅ AT-SPI registryd started (PID: $REGISTRYD_PID)"
sleep 1

# Enable ATK bridge so Chrome registers with AT-SPI
export GTK_MODULES=gail:atk-bridge
export GNOME_ACCESSIBILITY=1
export NO_AT_BRIDGE=0

echo "Environment:"
echo "  DISPLAY=$DISPLAY"
echo "  DBUS_SESSION_BUS_ADDRESS=$DBUS_SESSION_BUS_ADDRESS"
echo "  GTK_MODULES=$GTK_MODULES"
echo "  GNOME_ACCESSIBILITY=$GNOME_ACCESSIBILITY"

# Start Chrome
echo "Starting Chrome..."
google-chrome \
    --no-sandbox \
    --force-renderer-accessibility \
    --remote-debugging-port=$CHROME_PORT \
    --user-data-dir=/tmp/chrome_debug_atspi \
    --no-first-run --no-default-browser-check \
    --disable-gpu \
    --disable-software-rasterizer \
    --disable-dev-shm-usage \
    --window-size=1920,1080 \
    --start-maximized \
    "$TEST_URL" &
CHROME_PID=$!
echo "✅ Chrome started (PID: $CHROME_PID)"
echo "Waiting for Chrome to load..."
sleep 8

# Verify AT-SPI can see Chrome
echo
echo "Verifying AT-SPI desktop..."
python -c "
import ctypes, os
os.environ['DISPLAY'] = ':$DISPLAY_NUM'
ctypes.CDLL('/lib/x86_64-linux-gnu/libffi.so.7', mode=ctypes.RTLD_GLOBAL)
import pyatspi
desktop = pyatspi.Registry.getDesktop(0)
print(f'Desktop child count: {desktop.childCount}')
for i in range(desktop.childCount):
    c = desktop.getChildAtIndex(i)
    if c: print(f'  Child {i}: name=\"{c.name}\", role={c.getRoleName()}')
" 2>&1 || echo "AT-SPI verification failed"

# Run the debug script
echo
echo "Running debug script..."
echo "=========================================="
cd /data/hongsirui/AppEvalPilot
source venv/bin/activate 2>/dev/null || true
python debug_screen_xml.py

# Cleanup
echo
echo "=========================================="
echo "Cleanup..."
kill $CHROME_PID 2>/dev/null || true
kill $REGISTRYD_PID 2>/dev/null || true
kill $ATSPI_PID 2>/dev/null || true
kill $DBUS_SESSION_BUS_PID 2>/dev/null || true
kill $XVFB_PID 2>/dev/null || true
sleep 1
pkill -f "Xvfb :$DISPLAY_NUM" 2>/dev/null || true
pkill -f "remote-debugging-port=$CHROME_PORT" 2>/dev/null || true
echo "✅ Cleanup done"

