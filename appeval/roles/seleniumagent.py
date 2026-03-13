#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/03/11
@File    : seleniumagent.py
@Desc    : Selenium-based Web Automation Agent
"""
import json
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleContext
from metagpt.schema import AIMessage,UserMessage,SystemMessage, Message
from metagpt.utils.common import encode_image
from PIL import Image
from pydantic import ConfigDict, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from appeval.tools.selenium_controller import SeleniumController
import logging

warnings.filterwarnings("ignore")

SYSTEM = """
You are a professional and responsible web testing engineer (with real operation capabilities). I will provide you with a user instruction, and you need to provide test results for all test tasks. If you fail to complete the test tasks, it may cause significant losses to the client. Please maintain the test tasks and their results in a task list. For test cases of a project, you must conduct thorough testing with at least five steps or more - the more tests, the more reliable the results.

[IMPORTANT]: You must test ALL test cases before providing your final report! Do not skip any test cases or fabricate results without actual testing! Failing to complete the entire task list will result in invalid test results and significant client losses.

Task Tips:
Standard Operating Procedure (SOP):
1. Determine test plan based on instruction
2. Execute test plan systematically - verify the task list one by one
3. After completing all test plan, you can use Stop action to report and finish

Reporting Language: Answer in natural English using structured format (like dictionaries). Tell me your judgment basis and results. You need to report the completion status of each condition in the task and your basis for determining whether it's complete.

Inspection Standards:
1. Test cases are considered Pass if implemented on any page (not necessarily homepage). Please patiently review all pages (including scrolling down, clicking buttons to explore) before ending testing. You must understand relationships between pages - the first page you see is the target app's homepage.

2. You may switch to other pages on the app(or webpage) during testing. On these pages, just confirm the test case result - don't mark other pages-passed cases as Fail if subpages lack features. Return to homepage after judging each case.

3. Trust your operations completely. If expected results don't appear after an operation, that function isn't implemented - report judgment as False.

4. If target module isn't found after complete app(or webpage) browsing, test case result is negative, citing "target module not found on any page" as basis.

5. Don't judge functionality solely by element attributes (clickable etc.) or text ("Filter by category" etc.). You must perform corresponding tests before outputting case results.

6. When tasks require operations for judgment, you must execute those operations. Final results can't have cases with unknown results due to lack of operations (clicks, inputs etc.).

7. For similar test cases (e.g., checking different social media links), **envn if you verify one link works, you can't assume others work normally, test them all**.

For each task case completion, you can use report action to report just that result:
report({"task_id":"<Number Of Task>","task_name":"<Currect Task Name>","result": "<Pass/Fail/Uncertain>", "evidence": "<Your Evidence>"})

Even in these failure cases, you must perform sufficient testing steps to prove your judgment before using the Report action to report result.

[VERIFICATION REQUIRED]: Before submitting your task report, verify that:
1. You have tested the task clearly
2. task has an explicit result (Pass/Fail/Uncertain)
3. report result has supporting evidence based on your actual testing

when finish all the tasks use stop action to quit:
stop()

**Return only the result string. Do not include any additional text, markdown formatting, or code blocks.**
"""

HINTS = """
There are hints to help you complete the user's instructions. The hints are as follow:

**Critical Rules:**
- **NEVER repeat a failed action**: If an operation failed in the previous step, you MUST try a different approach. Repeating the same action will likely fail again.
- **Check for patterns**: If you see the same operation attempted 2+ times in history without success, stop and try a completely different strategy (e.g., use keyboard shortcuts instead of clicking, try a different element, or break down the task differently).
- **When stuck**: If multiple attempts fail, consider whether the current goal is achievable with available elements, or if you need to take prerequisite steps first.

**Element Interaction:**
- Do not skip steps, please wait for the previous click action to finish.
- Pay attention to the history to verify that it has been completed and avoid duplicate operations.
"""


OUTPUT_FORMAT = """
Your output consists of the following Five parts. Please note that only one set of content should be output at a time, and do not repeat the output.

**IMPORTANT**: Each section title must be wrapped with `###` on both sides (e.g., `### Title ###`). Follow this exact format:

Be thorough and specific. This reflection will guide future decisions and help avoid repeating failed approaches.

### Reflection Thought ###
Write a comprehensive analysis of the last operation's outcome in one paragraph. Your analysis must include:
- **What was expected vs. what actually happened**: Compare the intended outcome with actual screen state
- **Success evaluation**: Clearly state if the operation succeeded, partially succeeded, or failed, with specific evidence from the screenshot
- **Failure analysis** (if applicable): Explain the root cause (e.g., wrong coordinates, element not clickable, timing issue, incorrect approach, element not available)
- **Important observations**: Note any UI changes, error messages, warnings, or unexpected behaviors

### Thought ###
Based on the reflection and history, plan your next action in one paragraph. Your thought process must include:
- **History review**: Check recent operations for any patterns of repeated failures with the same approach
- **Strategy validation**: If the last operation failed and you're considering a similar approach, you MUST explain why this time will be different, OR choose a completely different strategy
- **Approach viability**: Assess whether the current method can achieve the goal, or if you need to try a fundamentally different approach
- **Clear reasoning**: State your logic before deciding on the next action

If you see the same operation attempted multiple times without success, you MUST try a different method.

### Action ###
Avaliable actions as follow:

- switch_window_handle(step)
<description>
    Switches between browser windows or tabs based on a given step offset.
    This method retrieves the list of all open window handles, finds the index of the currently active window handle, and attempts to switch to the window handle at the calculated target index (current index + step). If the target index is valid (within the bounds of the handle list), it switches the driver's focus and returns a success message. If the target index is out of bounds, it returns an error message. All exceptions during the process are caught and reported.
    
    Args:
        step (int): The number of steps to move. A positive value (e.g., 1) switches to the next window; a negative value (e.g., -1) switches to the previous window.
    
    Returns:
        str: A string message describing the operation's result.
        On successful switch: Contains the previous handle, the new current handle, and the list of all handles.
        On failure (index out of bounds or an exception): Contains a description of the error.
    
    Raises:
        This method catches and suppresses all exceptions, returning them as a string message instead of re-raising.
    
    Notes:
        Window handle indices are zero-based.
        All results (success, out-of-bounds, exception) are logged via logger.error.
</description>

- close_window_handle()
<description>
    Closes the current browser window or tab and returns the status information.
    This method attempts to close the currently active browser window. If successful, it returns a confirmation message that includes the new current window handle (the handle of the window that becomes active after the close operation) and the updated list of all remaining window handles. If the close operation fails, an error message is returned, which also includes the then-current window handle and the list of all window handles.

    Returns:
        str: A string message containing the operation's result and window handle information.
        On successful close: Reports success and lists the new current handle and all remaining handles.
        On failure (if an exception occurs): Reports the failure and lists the then-current handle and all handles.

    Raises:
        This method catches and suppresses all exceptions, returning them as part of the string message instead of re-raising.

    Notes:
        After closing a window, the driver's focus automatically switches to one of the remaining windows. The "current page" in the return message refers to this new active window.
        The result is logged via logger.error in both success and failure cases.
</description>

- navigate(url)
<description>
    Starts a browser session and navigates to the provided URL.

    This method initializes a browser session using the configured driver
    and attempts to navigate to the specified `url`. It includes a brief
    wait to allow the page to load and logs the operation's outcome.

    Args:
        url(str): The target URL to navigate to. Must be a valid, well-formed
                web address.

    Returns:
        str: A message indicating success ("成功跳转URL: {url}") or failure
                ("跳转URL失败").
</description>

- type_text(text)
<description>
    Types the specified text into the currently selected element.

    This method attempts to input the provided text into the web element
    previously selected and stored in `self.current_element`. It performs
    necessary pre-checks for browser session and element selection status,
    clears any existing content in the element, and logs the operation
    outcome.

    Args:
        text(str): The text string to be typed into the element.

    Returns:
        str: A message indicating the operation status:
                - "浏览器没有运行" if no browser session is active.
                - "没有选择元素" if no element is currently selected.
                - "成功输入文本: {text} \n 输入元素: {data}" on success.
                - "输入文本失败: {e}" if an exception occurs during input.
</description>

- select_element(by,value)
<description>
    Finds and selects a web element using the specified selector strategy.

    This method attempts to locate a single web element on the current page
    using the provided selector type (`by`) and value (`value`). It waits for
    the element to become present on the page (up to the configured timeout)
    and, upon success, stores it as the `self.current_element` for subsequent
    operations. The method also extracts and logs information about the
    located element.

    Supported selector types (case-insensitive) include:
        - "id": Locate by element ID.
        - "css": Locate by CSS selector.
        - "xpath": Locate by XPATH(recommend).
        - "class": Locate by class name.
        - "name": Locate by name attribute.
        - "tag": Locate by tag name.

    Args:
        by: The selector strategy to use. Must be one of the supported types.
        value: The selector value corresponding to the chosen strategy.

    Returns:
        str | None: A message string indicating the operation outcome, or None
                    if the browser is not started or an invalid selector type
                    is provided. Specifically:
                    - None: if browser not started or invalid `by` argument.
                    - "成功选择元素: {data}": on successful element location.
                    - "选择元素失败: {e}": if the element cannot be found within
                      the timeout or another exception occurs.
</description>

- click_element()
<description>
    Performs a click action on the currently selected web element.

    This method executes a click operation on the element stored in
    `self.current_element`. It is intended to simulate a user's click
    interaction, such as following a link, submitting a form, or
    triggering a UI event. After the click, the method waits for a
    brief period (3 seconds) to allow for potential page transitions
    or dynamic updates, then captures and returns the current page
    state including the URL and open window handles.

    Note: This method assumes that a valid element has already been
            selected (e.g., via `find_element`). The behavior is undefined
            if `self.current_element` is not a valid, interactable
            WebElement.

    Returns:
        str: A message string indicating the operation outcome:
                - "点击元素：{data} \n 当前url: {url} \n 当前窗口: {handles}"
                on successful click, including element information,
                the current page URL, and the list of window handles.
                - "点击元素失败: {e}" if an exception occurs during the
                click operation.
</description>

- report(result)
<description>
    Reports a task execution record by parsing the input JSON string.

    The method attempts to parse the provided string as JSON. If successful,
    it logs the parsed dictionary as a successful submission. If parsing fails,
    it logs the exception and reports the failure. The return value is always
    a human-readable status message.

    Args:
        result(str): A JSON-formatted string containing the task record to submit.

    Returns:
        A string describing the submission outcome, either success with the
        parsed dictionary or failure with the error.

    Raises:
        json.JSONDecodeError: If the input string is not valid JSON. (Note:
        This exception is caught internally and converted to a return message.)
</description>

- stop()
<description>
    when finish all the tasks use this action.
</description>

Action Output Format Specification

Each response must output one and only one action, strictly adhering to the following format:

Format:
`FunctionName(argument1,argument2,...)`

Specifications:
*   FunctionName: Use the precise English function name, e.g., `click`, `type_text`, `navigate`.
*   Arguments: Multiple arguments are separated by commas (`,`). Do not include spaces after the commas within the parentheses.
*   Complete Expression: The function name and its argument list together form a complete call expression. Do not add any trailing punctuation.

Correct Examples:
   `switch_window_handle(1)`
   `navigate(https://www.baidu.com)`
   `type_text(hello world)`

You must adhere strictly to this format, **not use `"` to boxed str parameter**. Only one action can be chose in a round.

### Summary ###
This is a core summary for your thought, action and task.

### Task List ###
* **[Completed Tasks]:** (List the tasks that have been successfully completed so far)
    * <Task 1 Description>
    * <Task 2 Description>
    ...
* **[Current Task]:** <Current Task Description>.
* **[Next Operation]:** (Describe the immediate next operation in detail, including what needs to be done)
    * <Step 1 Description>
    * <Step 2 Description>
    ...
* **[Remaining Tasks]:** (List the remaining high-level tasks that need to be completed to achieve the user's objective, excluding the current and next operation. )
    * <Task 1 Description>
    * <Task 2 Description>
    ...
"""

ACTION_EXAMPLT = """
### Few-Shot Examples for Action

The following examples demonstrate how to interact with different web elements over multiple sequential rounds. You must strictly output **only one action** per round.

#### Example 1: Interacting with a Button
**Task Goal**: Click a submit button that has the ID `submit-btn`.

**Round 1**
**Agent**: `select_element(id,submit-btn)`

**Round 2**
**Agent**: `click_element()`

**Round 3**
**Agent**: `report({<Task Result Here>})`
---

#### Example 2: Interacting with a Textarea / Input
**Task Goal**: Enter the text "Hello World" into a textarea with the name `description`.

**Round 1**
**Agent**: `select_element(name,description)`

**Round 2**
**Agent**: `type_text(Hello World)`

**Round 3**
**Agent**: `report({<Task Result Here>})`
---

#### Example 3: Interacting with a Selector (Dropdown Menu)
**Task Goal**: Select the option "New York" from a dropdown menu with the class `city-dropdown`.

**Round 1**
**Agent**: `select_element(class,city-dropdown)`

**Round 2**
**Agent**: `click_element()`

**Round 3**
**Agent**: `select_element(xpath,//option[text()='New York'])`

**Round 4**
**Agent**: `click_element()`

**Round 5**
**Agent**: `report({<Task Result Here>})`

#### Example 4: Opening and Closing a New Window
**Task Goal**: Interacting when Open a new window, must to switch to it to watch and fianlly close it.

**Round 1**
**Agent**: `select_element(id,submit-btn)`

**Round 2**
**Agent: switch_window_handle(1)

**Round 3**
**Agent**: close_window_handle()

**Round 4**
**Agent**: report({<Task Result Here>})
"""

TASK_EXAMPLT = """
### Few-Shot Examples for [Remaining Tasks]
**Wrong Example** 
*   Test right-side channel links (a, b, c, d, e, f)

**Correct Example** 
*   Test right-side channel link a
*   Test right-side channel link b
*   Test right-side channel link c
*   Test right-side channel link d
*   Test right-side channel link e
*   Test right-side channel link f

The more detailed, the better. Each task must correspond to a single, well-defined interactive element, and all test tasks must be listed., don't use `etc.`
"""

ELEMENT_EXAMPLT = """
### Few-Shot Examples for Action `select_element(by,value)`, how to locate elements using multiple conditions.
*   find_element(css, "input[name='username'][type='text']")
*   find_element(css, "a:contains('登录')")
*   find_element(css, "#user-id.username-input")
*   find_element(css, ".btn.btn-primary[disabled]")
"""


class SeleniumAgentContext(RoleContext):
    """Runtime context for SeleniumAgent"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory: List[Dict[str,str]] = Field(default_factory=list) 
    thought: str = "First Round, No Last Turn Thought."
    reflection_thought: str = "First Round, No Last Turn Reflection"
    task_list: str = "First Round, No Planed Task."
    summary: str = "First Round, No Last Turn Summary"
    action: str = "First Round, No Last Turn Action"
    action_reponse: str = "First Round, No Last Turn Action_Reponse"
    iter: int = 0

    def reset(self) -> None:
        """Reset all states to initial values"""
        self.thought = ""
        self.reflection_thought = ""
        self.summary = ""
        self.action = ""
        self.task_list = ""
        self.memory = []
        self.iter = 0


class SeleniumAgent(Role):
    """Selenium-based Web Automation Agent"""

    name: str = "SeleniumAgent"
    profile: str = "Web Automation Agent"
    goal: str = "Execute web automation tasks"
    constraints: str = "Ensure task execution accuracy and efficiency"
    desc: str = "Selenium-based agent for web automation tasks"

    rc: SeleniumAgentContext = Field(default_factory=SeleniumAgentContext)

    def __init__(
        self,
        max_iters: int = 500,
        log_dirs: str = "workspace",
        headless: bool = True,
        **kwargs,
    ) -> None:
        """Initialize SeleniumAgent.

        Args:
            max_iters: Maximum number of iterations
            log_dirs: Log directory
            system_prompt: System prompt
            add_info: Additional information to add to the prompt
            chrome_path: Path to Chrome executable
            headless: Run browser in headless mode
        """
        super().__init__(**kwargs)

        self.max_iters = max_iters
        self.log_dirs = log_dirs
        self.headless = headless

        self._init_environment()
        self._init_tools()

    def _init_environment(self) -> None:
        """Initialize runtime environment"""
        self._get_timestamped_paths()
        self._setup_logs()

    def _init_tools(self) -> None:
        """Initialize tool components"""
        self.controller = SeleniumController(
            headless=self.headless
        )

    def _get_timestamped_paths(self) -> None:
        """Update file paths with timestamps"""
        current_time = time.strftime("%Y%m%d%H%M")
        log_dir = Path(self.log_dirs) / current_time
        self.save_info = str(log_dir / "info.txt")

    def _reset_state(self) -> None:
        """Reset state, clear previous records when running new tasks"""
        self.rc.reset()
        self._get_timestamped_paths()


    def _setup_logs(self) -> None:
        """Set up logging"""
        log_dir = Path(self.save_info).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.remove()
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module}:{function}:{line} - {message}"

        logger.add(
            self.save_info,
            level=logging.INFO,
            format=log_format,
            mode="w",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        logger.add(sys.stdout, level="DEBUG", format=log_format, colorize=True, enqueue=True)
        logger.info(f"Initialized logging, log file: {self.save_info}")

 
    async def _think(self) -> bool:
        """Generate operation decisions"""
        # Build prompt

        system_msg = SYSTEM+HINTS+OUTPUT_FORMAT+ACTION_EXAMPLT+TASK_EXAMPLT #+ELEMENT_EXAMPLT
        prompt = self._build_action_prompt()
        logger.info(f"\n\n######################## prompt_action:\n{prompt}\n\n########################\n\n")

        self.rc.memory.append({"role":"user","content":prompt})

        output_action = await self.llm.aask(
            system_msgs = [system_msg],
            msg=self.rc.memory,
            stream=True,
        )

        

        self.rc.reflection_thought = self._extract_between(output_action, "### Reflection Thought ###", "### Thought ###")
        self.rc.thought = self._extract_between(output_action, "### Thought ###", "### Action ###")
        self.rc.action = self._extract_between(output_action, "### Action ###", "### Summary ###")
        self.rc.summary = self._extract_between(output_action, "### Summary ###", "### Task List ###")
        self.rc.task_list = self._extract_between(output_action, "### Task List ###")



        if len(self.rc.memory)>=10:
            self.rc.memory.pop(0)
            while True:
                if self.rc.memory[0]["role"]!="assistant" and not self.rc.memory[0]["content"].startswith("### Reflection Thought ###"):
                    self.rc.memory.pop(0)
                else:
                    break

        self.rc.memory.append({"role":"assistant","content":output_action})
        logger.info(f"\n\n######################## output_action:\n{output_action}\n\n########################\n\n")


        return not self.rc.action.startswith("stop")

    def _extract_between(self, text: str, start: str, end: str = None) -> str:
        """Extract text between markers"""
        if start not in text:
            return ""
        start_idx = text.find(start) + len(start)
        if end is not None:
            end_idx = text.find(end, start_idx)
            if end_idx == -1:
                return ""
            content = text[start_idx:end_idx]
        else:
            content = text[start_idx:]
        return content.strip()



    def _build_action_prompt(self) -> str:
        """Build action prompt"""
        prompt = f"""You are controlling a web browser to complete the following task:
**User Instruction**:
- Instruction: {self.instruction}
- Start(Usually Home Page) URL: {self.start_url}

**Current Page Information**:
- URL: {self.controller.get_url()}
- Window Handlers: {self.controller.get_window_handles()}
- Core Elements: {self.controller.get_core_elements_info()}
"""

        prompt += """
Please provide your response in the following format:

### Reflection Thought ###
[Reflect here]

### Thought ###
[Your reasoning]

### Action ###
[The action to take]

### Summary ###
[Core summary]

### Task List ###
[Updated task list]
"""
        return prompt

    async def _act(self) -> Message:
        """Execute action step"""

        try:
            rsp = self._execute_action(self.rc.action)
            logger.info(rsp)
        except Exception as e:
            rsp = f"Action execution failed: {e}"
            logger.error(rsp)
        
        self.rc.memory.append({"role":"user","content":rsp})
        return AIMessage(content=rsp, cause_by=Action)

    def _execute_action(self, action: str) -> None:
        """Execute the parsed action"""
        action = action.strip()
        if action.startswith("navigate"):
            url = re.search(r"\((.*?)\)", action).group(1)    
            rsp = self.controller.navigate(url)
        elif action.startswith("select_element"):
            match = re.search(r"^\s*[^(]+\(\s*([^,]+?)\s*,\s*(.*?)\s*\)\s*$", action)
            by, value = match.group(1).strip(), match.group(2).strip()   
            rsp = self.controller.select_element(by, value)
        elif action.startswith("click_element"):
            rsp = self.controller.click_element()
        elif action.startswith("type_text"):
            text = re.search(r"\((.*?)\)", action).group(1)  
            rsp = self.controller.type_text(text)
        elif action.startswith("switch_window_handle"):
            step = re.search(r"\((.*?)\)", action).group(1)
            rsp = self.controller.switch_window_handle(int(step))
        elif action.startswith("close_window_handle"):
            rsp = self.controller.close_window_handle()
        elif action.startswith("report"):
            result = re.search(r"\((.*?)\)", action).group(1)          
            rsp = self.controller.report(result)
        else:
            rsp = f"Unknown action: {action}"
            logger.warning(rsp)
        
        msg = """### Action Response ###\n"""
        
        return msg+rsp

    async def _react(self) -> Message:
        """Main reaction loop"""
        self.rc.iter = 0
        rsp = AIMessage(content="No actions taken yet", cause_by=Action)

        while self.rc.iter < self.max_iters:
            self.rc.iter += 1
            logger.info(f"\n\n#### Iteration: {self.rc.iter}\n\n")

            has_todo = await self._think()
            if not has_todo:
                rsp = AIMessage(content="Selenium Agent has finished all tasks", cause_by=Action)
                break

            rsp = await self._act()

        return rsp

    async def run(self, instruction: str, start_url: str = None) -> Message:
        """Run main loop

        Args:
            instruction: User instruction
            start_url: Starting URL (optional)
        """
        self._reset_state()
        self._setup_logs()
        self.instruction = instruction



        if start_url:
            self.start_url = start_url
            self.controller.navigate(start_url)
    #         aim_page_information = f"""
    # **Aim Page Information**:
    # - URL: {start_url}
    # - Core Elements: {self.controller.get_core_elements_info()}
    #         """
    #         self.rc.memory.append({"role":"user","content":aim_page_information})
            time.sleep(5)
        
        rsp = await self.react()
        self.controller.stop()
        return rsp
