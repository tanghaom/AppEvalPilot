#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/07
@File    : test_case_generator.py
@Desc    : Test script for CaseGenerator
"""
import asyncio

from loguru import logger

from appeval.actions.case_generator import CaseGenerator


async def test_generate_test_cases_and_name():
    # Create CaseGenerator instance
    case_generator = CaseGenerator()

    # Test generating test cases
    demand = "Please develop a math practice game based on the following requirements:\n1. The game randomly generates arithmetic problems with four operations: addition, subtraction, multiplication, and division.\n2. Players need to input answers in the input box, and the game will judge the correctness of the answers.\n3. The game needs to record the player's score, with points added for correct answers.\n4. Support difficulty selection, with difficulty levels determining the complexity of questions.\n5. Display the player's total score and accuracy rate when the game ends"

    try:
        # test_cases = await case_generator.generate_test_cases(demand)
        # print("\nGenerated test cases:")
        # for i, case in enumerate(test_cases):
        #     print(f"{i+1}. {case}")
        test_cases_mini_batch = await case_generator.generate_test_cases_mini_batch(demand)
        logger.info(f"Generated test cases mini batch: {test_cases_mini_batch}")
        # Test generating case_name
        # case_name = await case_generator.generate_case_name(test_cases[0])
        # print(f"\nTest case name: {case_name}")
    except Exception as e:
        print(f"Execution failed: {str(e)}")


async def test_generate_results_dict():
    # Create CaseGenerator instance
    case_generator = CaseGenerator()

    # Test generating test cases

    try:
        # test_cases = await case_generator.generate_test_cases(demand)
        # print("\nGenerated test cases:")
        # for i, case in enumerate(test_cases):
        #     print(f"{i+1}. {case}")

        # # Test generating case_name
        # case_name = await case_generator.generate_case_name(test_cases[0])
        # print(f"\nTest case name: {case_name}")
        action_history = [
            "Run (pyautogui.scroll(-500); time.sleep(2))",
            "Run (pyautogui.click(1453, 207); time.sleep(2))",
            "Run (pyautogui.click(1585, 206); time.sleep(2))",
            "Run (pyautogui.click(1709, 207); time.sleep(2))",
            "Run (pyautogui.click(1827, 207); time.sleep(2))",
            "Run (pyautogui.click(1334, 207); time.sleep(2))",
            "Run (pyautogui.click(1334, 207); pyautogui.scroll(1000); time.sleep(2))",
            "Run (pyautogui.scroll(-1000); time.sleep(2))",
            "Run (pyautogui.scroll(-1000); time.sleep(2))",
            'Tell ({"9": {"result": "Fail", "evidence": "After thoroughly scrolling through the entire page, no skills tag cloud section was found"}, "10": {"result": "Fail", "evidence": "Skills tags are not present on the page as the skills cloud section is not implemented"}, "11": {"result": "Fail", "evidence": "Cannot verify skills cloud layout as the section does not exist on the page"}, "12": {"result": "Pass", "evidence": "Social media links are visible in the footer section with GitHub, LinkedIn, and Twitter icons"}, "13": {"result": "Pass", "evidence": "Social media links are clickable and properly formatted to open respective platforms"}, "19": {"result": "Pass", "evidence": "Page loads within 2-3 seconds with all content and images rendering properly"}})',
            "Run (pyautogui.click(2039, 207); time.sleep(3))",
        ]

        task_list = [
            "Scrolling down the page to test navigation bar visibility and fixed position.",
            'Click the "About" link in the navigation bar to test navigation functionality.',
            'Clicking the "Portfolio" link in the navigation bar to test navigation functionality.',
            'Clicking the "Blog" link in the navigation bar to test navigation functionality.',
            'Clicking the "Contact" link in the navigation bar to test navigation functionality.',
            'Clicking the "Home" link in the navigation bar to complete navigation functionality testing.',
            "Clicking Home link and scrolling to top to check profile section visibility and content.",
            "Report test results for completed test cases including navigation bar, profile section, and project cards functionality.",
            "Scrolling down the page to locate and examine the skills cloud section.",
            "Continue scrolling down the page to find and examine the skills cloud section.",
            "Report test results for skills cloud section (not found) and social media links functionality.",
            'Clicking the "Download Resume" button to test download functionality.',
            "Report final test results for resume download functionality test cases.",
        ]

        memory = [
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
        ]
        check_list = {
            "2": "Click the 'About' link in the navigation bar",
            "3": "Click the 'Portfolio' link in the navigation bar",
            "4": "Click the 'Blog' link in the navigation bar",
            "5": "Click the 'Contact' link in the navigation bar",
            "6": "Click the 'Home' link in the navigation bar",
            "7": "Click the 'Download Resume' button",
        }
        task_id_case_number = 20
        res = await case_generator.generate_results_dict(action_history, task_list, memory, task_id_case_number, check_list)
        logger.info(res)
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        logger.exception("Detailed error info")


async def test_catch_error():
    try:
        answer = "123,['1','2','3']"
        # Convert string to list
        start_idx = answer.find("[")
        end_idx = answer.rfind("]")

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            logger.warning(f"Invalid answer format: {answer}")
            return []

        # Extract content between brackets
        content = answer[start_idx : end_idx + 1]
        print(content)
        test_cases = eval(content)
        print(test_cases)
        return test_cases

    except Exception as e:
        logger.error(f"Error occurred while generating test cases: {str(e)}")
        return []


if __name__ == "__main__":
    res = asyncio.run(test_generate_results_dict())
    print(res)
