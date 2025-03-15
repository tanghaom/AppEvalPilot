#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@Author  : tanghaoming
@File    : test_device_controller.py
@Desc    : Test script for device controller functionality
"""


from appeval.tools.device_controller import create_controller


def test_android():
    """Test Android device controller functionality"""
    controller = create_controller("Android")
    controller.get_screenshot()
    elements = controller.get_screen_xml()
    print("Android element info:", elements)


def test_pc():
    """Test PC device controller functionality"""
    controller = create_controller("Windows", search_keys=("win", "s"), ctrl_key="ctrl")
    controller.get_screenshot()
    elements = controller.get_screen_xml()
    print("PC element info:", elements)


if __name__ == "__main__":
    # print("Running Android controller test...")
    # test_android()
    print("\nRunning PC controller test...")
    test_pc()
