class CasePrompts:
    SYSTEM_MESSAGE = "You are a professional test engineer skilled in generating and validating test cases."

    GENERATE_CASES = """You are a professional test engineer. Please generate a series of specific test cases based on the following user requirements for the webpage.
Requirements:
1. Test cases must be generated entirely around user requirements, absolutely not missing any user requirements
2. Please return all test cases in Python list format
3. When generating test cases, consider both whether the corresponding module is displayed on the webpage and whether the corresponding function is working properly. You need to generate methods to verify webpage functionality based on your knowledge.
4. Please do not implement test cases that require other device assistance for verification.
5. Please control the number of test cases to 15~20, focusing only on the main functionalities mentioned in the user requirements. Do not generate test cases that are not directly related to the user requirements.
6. When generating test cases, focus on functional testing, not UI testing.
User Requirements: {demand}
Please return the test case list in List(str) format, without any additional characters, as the result will be converted using the eval function."""

    GENERATE_CASES_MINI_BATCH = """You are a professional test engineer. Please generate a series of specific test cases based on the following user requirements for the webpage.
Requirements:
1. Test cases must be generated entirely around user requirements, absolutely not missing any user requirements
2. Please return all test cases in Python list format
3. When generating test cases, consider both whether the corresponding module is displayed on the webpage and whether the corresponding function is working properly. You need to generate methods to verify webpage functionality based on your knowledge.
4. Please do not implement test cases that require other device assistance for verification.
5. Please control the number of test cases to 15~20, focusing only on the main functionalities mentioned in the user requirements. Do not generate test cases that are not directly related to the user requirements.
6. Please generate test cases grouped by functionality categories. Each category's test cases should be in a separate list, and all category lists should be combined into a single list. For example, if testing a website with login and profile features, the output should be like [[login test case 1, login test case 2], [profile test case 1, profile test case 2]], where each inner list contains related test cases for that category.
User Requirements: {demand}
Please return the test case list in List[List[str], List[str],...] format, without any additional characters, as the result will be converted using the eval function."""

    GENERATE_CASE_NAME = """Please condense the following test case description into a short English case_name (no more than 5 words):
{case_desc}
Please return only the condensed case_name, without quotes."""

    CHECK_RESULT = """Below content is model result as ground truth, please judge based on facts whether the described case is successfully implemented. If there is evidence to indicate it has been implemented, please output 'Yes', otherwise output 'No'. If the model result cannot determine the result, output 'Uncertain':
Case Description: {case_desc}
Model Result: {model_output}
Please answer 'Yes','No','Uncertain'"""

    GENERATE_RESULTS = """You are a professional test engineer. Please generate a result in the specified format based on the following historical information.
You need to comprehensively analyze all historical information to infer the final test results. Please note not to miss any possible test case results. For cases where you think no results are given, please use Uncertain as the result for that case.
Action History: {action_history}
Task List Information: {task_list}
Memory History: {memory}
Number of test cases in result dictionary: {task_id_case_number}
**Return only the result string. Do not include any additional text, markdown formatting, or code blocks.**

### output example ###
{{
    "0": {{"result": "Pass", "evidence": "The thumbnail click functionality is working correctly."}},
    "1": {{"result": "Uncertain", "evidence": "Cannot verify price calculation accuracy as no pricing information is displayed"}},
    "2": {{"result": "Fail", "evidence": "After fully browsing and exploring the web page, I did not find the message board."}}
}}
"""
