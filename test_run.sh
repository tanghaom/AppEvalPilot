#!/bin/bash
# Quick test script
export DISPLAY=:99
export CONFIG_ROOT=/root/.metagpt
export BROWSER=/usr/bin/google-chrome
# Fix Anaconda libffi conflict for pyatspi/gi
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
source venv/bin/activate
python main.py --platform Linux --max_iters 5
