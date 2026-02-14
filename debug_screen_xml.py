#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug script for testing get_screen_xml functionality.
Supports both AT-SPI and CDP modes.

Usage:
  python debug_screen_xml.py              # default: CDP mode
  python debug_screen_xml.py --mode cdp   # CDP mode (lightweight)
  python debug_screen_xml.py --mode atspi # AT-SPI mode (needs D-Bus)
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Set up environment
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":800"
    print(f"Setting DISPLAY={os.environ['DISPLAY']}")

# Set up Python path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from loguru import logger
from appeval.tools.device_controller import PCController

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cdp", "atspi"], default="cdp",
                        help="Accessibility tree mode: cdp (lightweight) or atspi (needs D-Bus)")
    parser.add_argument("--port", type=int, default=9999,
                        help="Chrome remote debugging port")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Debug: get_screen_xml (mode={args.mode})")
    print("=" * 60)
    print(f"DISPLAY: {os.environ.get('DISPLAY')}")
    print(f"a11y_mode: {args.mode}")
    if args.mode == "cdp":
        print(f"Chrome port: {args.port}")
    print()
    
    # Create controller
    print("Creating PCController...")
    controller = PCController(
        pc_type="linux",
        max_tokens=8000,
        a11y_mode=args.mode,
        remote_debugging_port=args.port,
    )
    print("✅ Controller created")
    print()
    
    # First: raw AT-SPI tree dump
    print("=" * 60)
    print("Step 1: Raw AT-SPI Tree Dump")
    print("=" * 60)
    try:
        import ctypes
        if os.path.exists("/lib/x86_64-linux-gnu/libffi.so.7"):
            ctypes.CDLL("/lib/x86_64-linux-gnu/libffi.so.7", mode=ctypes.RTLD_GLOBAL)
        import pyatspi

        desktop = pyatspi.Registry.getDesktop(0)
        print(f"Desktop: childCount={desktop.childCount}")
        for i in range(desktop.childCount):
            app = desktop.getChildAtIndex(i)
            if app is None:
                print(f"  App[{i}]: None")
                continue
            app_children = getattr(app, "childCount", 0)
            print(f"  App[{i}]: name='{app.name}', role='{app.getRoleName()}', children={app_children}")
            for j in range(app_children):
                try:
                    win = app.getChildAtIndex(j)
                    if win:
                        role = win.getRoleName()
                        name = win.name if hasattr(win, 'name') else '?'
                        win_children = getattr(win, "childCount", 0)
                        print(f"    Win[{j}]: name='{name}', role='{role}', children={win_children}")
                        # Show first few children of the window
                        for k in range(min(win_children, 5)):
                            try:
                                child = win.getChildAtIndex(k)
                                if child:
                                    print(f"      Child[{k}]: name='{child.name}', role='{child.getRoleName()}'")
                            except Exception:
                                pass
                    else:
                        print(f"    Win[{j}]: None")
                except Exception as e:
                    print(f"    Win[{j}]: Error: {e}")
    except Exception as e:
        print(f"❌ AT-SPI raw dump failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 2: Test get_screen_xml
    print()
    print("=" * 60)
    print("Step 2: get_screen_xml()")
    print("=" * 60)
    
    try:
        start_time = time.time()
        xml_results = controller.get_screen_xml(location_info="center")
        elapsed = time.time() - start_time
        
        print(f"✅ get_screen_xml completed in {elapsed:.3f}s")
        print(f"Number of elements: {len(xml_results)}")
        print()
        
        if xml_results:
            print("First 10 elements:")
            print("-" * 60)
            for i, element in enumerate(xml_results[:10]):
                print(f"{i+1}. {element}")
            
            if len(xml_results) > 10:
                print(f"... and {len(xml_results) - 10} more elements")
        else:
            print("⚠️ No elements found from get_screen_xml!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()

