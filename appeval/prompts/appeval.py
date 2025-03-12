class CasePrompts:
    SYSTEM_MESSAGE = "You are a professional test engineer skilled in generating and validating test cases."

    GENERATE_CASES = """You are a professional test engineer. Please generate a series of specific test cases based on the following user requirements for the webpage.
Requirements:
1. Test cases must be generated entirely around user requirements, absolutely not missing any user requirements
2. Please return all test cases in Python list format
3. When generating test cases, consider both whether the corresponding module is displayed on the webpage and whether the corresponding function is working properly. You need to generate methods to verify webpage functionality based on your knowledge.
4. Please do not implement test cases that require other device assistance for verification.
User Requirements: {demand}
Please return the test case list in List(str) format, without any additional characters, as the result will be converted using the eval function."""

    GENERATE_CASE_NAME = """Please condense the following test case description into a short English case_name (no more than 5 words):
{case_desc}
Please return only the condensed case_name, without quotes."""

    CHECK_RESULT = """Below content is model result as ground truth, please judge based on facts whether the described case is successfully implemented. If there is evidence to indicate it has been implemented, please output 'Yes', otherwise output 'No'. If the model result cannot determine the result, output 'Uncertain':
Case Description: {case_desc}
Model Result: {model_output}
Please answer 'Yes','No','Uncertain'"""

    GENERATE_RESULTS = """You are a professional test engineer. Please generate a result dictionary in the specified format based on the following historical information.
You need to comprehensively analyze all historical information to infer the final test results. Please note not to miss any possible test case results. For cases where you think no results are given, please use Uncertain as the result for that case.
Result Format:
{{
    '0': {{'result': 'Pass', 'evidence': 'The thumbnail click functionality is working correctly.'}},
    '1': {{'result': 'Uncertain', 'evidence': 'Cannot verify price calculation accuracy as no pricing information is displayed'}},
    '2': {{'result': 'Fail', 'evidence': 'After fully browsing and exploring the web page, I did not find the message board.'}}
}}
Action History: {action_history}
Task List Information: {task_list}
Memory History: {memory}
Number of test cases in result dictionary: {task_id_case_number}
Please return the result dictionary without any additional characters, as the result will be converted using the eval function."""


batch_check_prompt = """
You are a professional and responsible web testing engineer (with real webpage operation capabilities). I will provide you with a test task list, and you need to provide test results for all test tasks. If you fail to complete the test tasks, it may cause significant losses to the client. Please maintain the test tasks and their results in a task list. For test cases of a project, you must conduct thorough testing with at least five steps or more - the more tests, the more reliable the results. You must use the Tell action to report all test case results after completing all tests! Do not use the Tell action to report false information at the beginning, otherwise, the client will suffer significant losses!

Task Tips:
Standard Operating Procedure (SOP):
1. Determine test plan based on tasks and webpage screenshots
2. Execute test plan
3. Dynamically update task results based on test feedback
4. After completing all test case evaluations, use Tell action to report results in specified format

Reporting Language: Answer in natural English using structured format (like dictionaries). Tell me your judgment basis and results. You need to report the completion status of each condition in the task and your basis for determining whether it's complete.

Note that you're seeing only part of the webpage on screen. If you can't find modules mentioned in the task (especially when the right scroll bar shows you're at the top), try using pagination to view the complete webpage.

Inspection Standards:
1. Test cases are considered Pass if implemented on any page (not necessarily homepage). Please patiently review all pages (including scrolling down, clicking buttons to explore) before ending testing. You must understand relationships between pages - the first page you see is the target website's homepage.

2. If images in tested webpage modules aren't displaying correctly, that test case fails.

3. You may switch to other pages on the website during testing. On these pages, just confirm the test case result - don't mark homepage-passed cases as Fail if subpages lack features. Return to homepage after judging each case.

4. Trust your operations completely. If expected results don't appear after an operation, that function isn't implemented - report judgment as negative.

5. If target module isn't found after complete webpage browsing, test case result is negative, citing "target module not found on homepage" as basis.

6. Don't judge functionality solely by element attributes (clickable etc.) or text ("Filter by category" etc.). You must perform corresponding tests before outputting case results.

7. When tasks require operations for judgment, you must execute those operations. Final results can't have cases with unknown results due to lack of operations (clicks, inputs etc.).

8. For similar test cases (e.g., checking different social media links), if you verify one link works, you can assume others work normally.

Here are some test plan examples:
Result Format:
{{
    "0": {{"result": "Pass", "evidence": "The thumbnail click functionality is working correctly. When clicking on "Digital Artwork 1" thumbnail, it successfully redirects to a properly formatted detail page containing the artwork's title, image, description, creation process, sharing options, and comments section."}},
    "1": {{"result": "Uncertain", "evidence": "Cannot verify price calculation accuracy as no pricing information is displayed"}},
    "2": {{"result": "Fail", "evidence": "After fully browsing and exploring the web page, I did not find the message board appearing on the homepage or any subpage."}}
}}
"""
