#!/bin/bash
# Start Orca screen reader in background to enable Firefox accessibility
# Firefox will detect Orca and automatically enable AT-SPI support

# Check if Orca is already running
if pgrep -x "orca" > /dev/null; then
    echo "✓ Orca is already running"
    exit 0
fi

echo "Starting Orca screen reader in background..."
echo "This enables Firefox accessibility support (AT-SPI)"

# Start Orca in background, suppress output
nohup orca > /dev/null 2>&1 &
disown

# Wait a moment for Orca to initialize
sleep 2

if pgrep -x "orca" > /dev/null; then
    echo "✓ Orca started successfully"
    echo "Firefox accessibility should now be available"
else
    echo "✗ Failed to start Orca"
    echo "Install it with: sudo apt-get install orca"
    exit 1
fi

