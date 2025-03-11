#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/07
@File    : test_case_generator.py
@Desc    : Test script for CaseGenerator
"""
import asyncio

from appeval.actions.case_generator import CaseGenerator


async def main():
    # Create CaseGenerator instance
    case_generator = CaseGenerator()

    # Test generating test cases
    demand = "Please develop a math practice game based on the following requirements:\n1. The game randomly generates arithmetic problems with four operations: addition, subtraction, multiplication, and division.\n2. Players need to input answers in the input box, and the game will judge the correctness of the answers.\n3. The game needs to record the player's score, with points added for correct answers.\n4. Support difficulty selection, with difficulty levels determining the complexity of questions.\n5. Display the player's total score and accuracy rate when the game ends"

    try:
        test_cases = await case_generator.generate_test_cases(demand)
        print("\nGenerated test cases:")
        for i, case in enumerate(test_cases):
            print(f"{i+1}. {case}")

        # Test generating case_name
        case_name = await case_generator.generate_case_name(test_cases[0])
        print(f"\nTest case name: {case_name}")

    except Exception as e:
        print(f"Execution failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
