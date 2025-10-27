from typing import Dict, List

from pydantic import BaseModel


class ActionPromptContext(BaseModel):
    """Context data for Action prompts"""

    instruction: str  # User instruction
    clickable_infos: List[Dict]  # List of clickable element information
    width: int  # Screen width
    height: int  # Screen height
    thought_history: List[str]  # History of thoughts
    summary_history: List[str]  # History of operation summaries
    action_history: List[str]  # History of actions
    reflection_thought_history: List[str]  # History of reflection thoughts
    last_summary: str  # Last operation summary
    last_action: str  # Last executed action
    reflection_thought: str  # Current reflection content
    add_info: str  # Additional information
    error_flag: bool  # Error flag
    error_message: str  # Error message when action execution fails
    completed_content: str  # Completed content
    memory: List[str]  # Memory list
    task_list: str  # Task list
    use_som: bool  # Whether to use SOM
    location_info: str = "center"  # Location format, defaults to center coordinates
    icon_caption: bool = True  # Whether to use icon descriptions


class BasePrompt:
    """Base class for Prompt templates"""

    def __init__(self, platform: str):
        self.platform = platform

        # Base prompt template
        self.prompt_template = """
### Background ###
{background}

### Screenshot information ###
{screenshot_info}

### Hints ###
{hints}

{additional_info}

{history_operations}

{task_list}

{last_operation}

### Task requirements ###
{task_requirements}

### Output format ###
{output_format}
"""

        # Base background information template
        self.background_template = (
            """The {image_desc} width is {width} pixels and its height is {height} pixels. The user's instruction is: {instruction}."""
        )

        # Base screenshot information template
        self.screenshot_info_template = """
In order to help you better perceive the content in this screenshot, we extract some information {source_desc}.
This information consists of two parts: coordinates; content. 
{location_format}
{content_format}
The information is as follow:
{clickable_info}
Please note that this information is not necessarily accurate. You need to combine the screenshot to understand."""

        # Base hints template
        self.hints = """
There are hints to help you complete the user's instructions. The hints are as follow:

**Critical Rules:**
- **NEVER repeat a failed action**: If an operation failed in the previous step, you MUST try a different approach. Repeating the same action will likely fail again.
- **Check for patterns**: If you see the same operation attempted 2+ times in history without success, stop and try a completely different strategy (e.g., use keyboard shortcuts instead of clicking, try a different element, or break down the task differently).
- **When stuck**: If multiple attempts fail, consider whether the current goal is achievable with available elements, or if you need to take prerequisite steps first.

**Element Interaction:**
- When there is no direct element counterpart, you should guess the possible elements based on the task and its coordinates.
- Sometimes both shortcuts and clicking can accomplish the same action; in such cases, prioritize using shortcuts.
- Do not make a location estimate, if it does not pop up, please wait.
- You should not assume the position of an element you cannot see.
- Perform only one click at a time, Do not skip steps, please wait for the previous click action to finish.
- Pay attention to the history to verify that it has been completed and avoid duplicate operations."""

        # Base history operations template
        self.history_template = """
### History operations ###
Before reaching this page, some operations have been completed. You need to refer to the completed operations to decide the next operation. These history records include the following components:
- Memory: A concise description of the screen state at this step. This describes what was observed before deciding what to do.
- Operation: A textual description of the intended action, based on the observed screen state.
- Action: The action that was executed.
- Reflection_thought: Analysis of the screen state and outcome after this action was executed. It evaluates whether this step achieved its expected result. It's not necessarily correct, you should make your own judgment.

**IMPORTANT**: Carefully review the history to identify:
1. Which operations succeeded and which failed
2. Any repeated attempts with the same approach that didn't work
3. Patterns that suggest the current strategy needs to change
{history_details}"""

        # Base output format template
        self.output_format = """
Your output consists of the following six parts. Please note that only one set of content should be output at a time, and do not repeat the output.

**IMPORTANT**: Each section title must be wrapped with `###` on both sides (e.g., `### Title ###`). Follow this exact format:

### Image Description ###
Provide a concise description of the current screenshot using only observable information. If multiple screenshots are provided, summarize notable changes between the latest image and previous images.

### Reflection Thought ###
Write a comprehensive analysis of the last operation's outcome in one paragraph. Your analysis must include:
- **What was expected vs. what actually happened**: Compare the intended outcome with actual screen state
- **Success evaluation**: Clearly state if the operation succeeded, partially succeeded, or failed, with specific evidence from the screenshot
- **Failure analysis** (if applicable): Explain the root cause (e.g., wrong coordinates, element not clickable, timing issue, incorrect approach, element not available)
- **Important observations**: Note any UI changes, error messages, warnings, or unexpected behaviors

Be thorough and specific. This reflection will guide future decisions and help avoid repeating failed approaches.

### Thought ###
Based on the reflection and history, plan your next action in one paragraph. Your thought process must include:
- **History review**: Check recent operations for any patterns of repeated failures with the same approach
- **Strategy validation**: If the last operation failed and you're considering a similar approach, you MUST explain why this time will be different, OR choose a completely different strategy
- **Approach viability**: Assess whether the current method can achieve the goal, or if you need to try a fundamentally different approach
- **Clear reasoning**: State your logic before deciding on the next action

If you see the same operation attempted multiple times without success, you MUST try a different method.

### Action ###
{action_options}

### Operation ###
This is a one sentence summary of this operation.

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
* **[Remaining Tasks]:** (List the remaining high-level tasks that need to be completed to achieve the user's objective, excluding the current and next operation.)
    * <Task 1 Description>
    * <Task 2 Description>
    ...
"""

        # Package name prompt template
        self.package_name_template = """
### Background ###
There is an user's instruction which is: {app_name}. You are a {platform} operating assistant and are operating the user's {platform}.

{mapping_info}

### Response requirements ###
There is the list of all application package names on the target {platform}:{package_list}. If I want to use ADB to open the target app, please tell me the package name corresponding to the target app. You can only choose from those listed above.

### Output format ###
Your output format is:
Don't output the purpose of any operation. If there is one, output it as "package:package_name". If not, output "package:None".
(Please use English to output)"""

        # Mapping information template
        self.mapping_info_template = """
### Hints ###
The following is the correspondence between some common application names and their package names, which may be helpful for you to complete the task. It's not necessarily correct, you should make your own judgment.
{app_mapping}"""

    def _build_background(self, ctx: ActionPromptContext, device_type: str) -> str:
        """Build background information section"""
        image_desc = (
            f"first image is a clean {device_type} screenshot. And the second image is the annotated version of it, where icons are marked with numbers."
            if ctx.use_som
            else f"image is a {device_type} screenshot."
        )
        return self.background_template.format(image_desc=image_desc, width=ctx.width, height=ctx.height, instruction=ctx.instruction)

    def _build_screenshot_info(self, ctx: ActionPromptContext, source_desc: str) -> str:
        """Build screenshot information section"""
        # Build location format information
        location_format = {
            "center": "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from top to bottom;",
            "bbox": "The format of the coordinates is [x1, y1, x2, y2], x is the pixel from left to right and y is the pixel from top to bottom. (x1, y1) is the coordinates of the upper-left corner, (x2, y2) is the coordinates of the bottom-right corner;",
        }[ctx.location_info]

        # Build content format information
        content_format = """the content can be one of three types:
1. text from OCR
2. icon description or 'icon'
3. element information from device tree, which contains element attributes like type, text content, identifiers, accessibility descriptions and position information"""

        # Build clickable information
        clickable_info = "\n".join(
            f"{info['coordinates']}; {info['text']}"
            for info in ctx.clickable_infos
            if info["text"] != "" and info["text"] != "icon: None" and info["coordinates"] != (0, 0)
        )

        return self.screenshot_info_template.format(
            source_desc=source_desc,
            location_format=location_format,
            content_format=content_format,
            clickable_info=clickable_info,
        )

    def _build_history_operations(self, ctx: ActionPromptContext) -> str:
        """Build history operations section"""
        if len(ctx.action_history) == 0:
            return ""

        history_details = ""
        for i in range(len(ctx.action_history)):
            # Remove newlines from each field to avoid format disruption
            memory = (ctx.memory[i] if len(ctx.memory) > i else "None").replace("\n", " ").replace("\r", " ")
            operation = ctx.summary_history[i].replace("\n", " ").replace("\r", " ")
            action = ctx.action_history[i].replace("\n", " ").replace("\r", " ")

            history_details += f"Step-{i+1}:\n\tMemory: {memory}\n"
            history_details += f"\tOperation: {operation}\n"
            history_details += f"\tAction: {action}\n"
            # Reflection from next step (reflects on this step's outcome)
            reflection_idx = i + 1
            if reflection_idx < len(ctx.reflection_thought_history):
                reflection = ctx.reflection_thought_history[reflection_idx].replace("\n", " ").replace("\r", " ")
                history_details += f"\tReflection_thought: {reflection}\n"

        return self.history_template.format(history_details=history_details)

    def _build_task_list(self, ctx: ActionPromptContext) -> str:
        """Build task list section"""
        if not ctx.task_list:
            return ""

        return (
            f"### Last Task List ###\nHere is the task list generated in the previous step. Please use this information to update the task list in the current step. Specifically, you should:\n"
            f"1. Identify and move any completed tasks to the `[Completed Tasks]` section.\n"
            f"2. Determine the current task and place it in the `[Current Task]` section.\n"
            f"3. Plan the next immediate operation and detail its steps in the `[Next Operation]` section.\n"
            f"4. List the remaining tasks (excluding the current and next operation) in the `[Remaining Tasks]` section. Adjust or refine these tasks as needed based on the current context.\n"
            f"{ctx.task_list}"
        )

    def _build_last_operation(self, ctx: ActionPromptContext) -> str:
        """Build last operation section"""
        if not ctx.error_flag:
            return ""

        error_details = f"\n\nError details: {ctx.error_message}" if ctx.error_message else ""
        return (
            f'### Last operation ###\nYou previously wanted to perform the operation "{ctx.last_summary}" on this page '
            f'and executed the Action "{ctx.last_action}". However, the operation failed or did not meet your expectation.'
            f"{error_details}\n\nPlease analyze the error details above carefully and fix the issue in your next operation."
        )

    def get_action_prompt(self, ctx: ActionPromptContext) -> str:
        raise NotImplementedError

    def get_package_name_prompt(self, app_name: str, app_mapping: str, package_list: List[str]) -> str:
        """Build package name prompt (common implementation for all platforms)"""
        # Build mapping information
        mapping_info = self.mapping_info_template.format(app_mapping=app_mapping) if app_mapping else ""

        # Use template to build prompt
        return self.package_name_template.format(app_name=app_name, platform=self.platform, mapping_info=mapping_info, package_list=package_list)


class Android_prompt(BasePrompt):
    def __init__(self):
        super().__init__("Android phone")

        # Android-specific task requirements
        self.task_requirements = """
In order to meet the user's requirements, you need to select one of the following operations to operate on the current screen:
Note that to open an app, use the Open App action, rather than tapping the app's icon. 
For certain items that require selection, such as font and font size, direct input is more efficient than scrolling through choices.
You must choose one of the actions below:

- Open App (app name)
    Please prioritize using this action to open the app. You can use this action to open the app named "app name" under any circumstances, including when the target app is not on the current page.

- Run (your code)
    You can use this action to run python code. Your code will run on a mobile device for controlling the interface. You are required to use `self.device` (which is an instance of `uiautomator2`) to perform actions grounded to the observation. Do not use shell to execute adb commands.

    For actions:
    - Use `self.device.click(x, y)` for single clicks
    - Use `self.device.double_click(x, y)` for double clicks
    - Use `self.device.long_click(x, y)` for long presses
    - Use `self.device.swipe(sx, sy, ex, ey)` for swipes
    - Use `self.device.press("back")` for back button
    - Use `self.device.press("home")` for home button
    - Use `self.device.press("enter")` for enter key
    - Use `self.device.clear_text()` to clear text from input fields

    For text input:
    1. First click on the input box
    2. Use `self.device.clear_text()` if needed to clear existing text
    3. Use `self.device.send_keys(\"\"\"text\"\"\", clear=True)` for entering text
    4. Make sure the content of text is enclosed in triple quotes
    5. For search operations, prefer using `self.device.press("enter")` after input

    Each action must end with a `time.sleep(duration)` statement. The `duration` should be chosen based on the expected time for the action to complete and the UI to update:
    - Simple clicks: 0.5-1 second
    - Text input: 1-2 seconds
    - Opening apps: 5-10 seconds
    - Loading content: 2-5 seconds
    - System actions (back/home): 1-2 seconds
    You need to determine the appropriate duration based on the context.

    Example: Run (self.device.click(550, 173); time.sleep(0.5); self.device.send_keys(\"\"\"hello\"\"\", clear=True); time.sleep(1))

    Limit the execution to no more than 5 steps at a time to avoid errors.

- Wait
    If the interface is not fully loaded, use the 'wait' action to skip the current step. However, avoid using 'wait' consecutively multiple times. If you find yourself needing to 'wait' repeatedly, re-evaluate your assessment of the interface's readiness.

- Stop
    If you think all the requirements of user's instruction have been completed and no further operation is required, you can choose this action to terminate the operation process."""

    def get_action_prompt(self, ctx: ActionPromptContext) -> str:
        # Build all sections using base class methods
        background = self._build_background(ctx, "phone")
        screenshot_info = self._build_screenshot_info(ctx, "on the current screenshot through system files")
        history_operations = self._build_history_operations(ctx)
        task_list = self._build_task_list(ctx)
        last_operation = self._build_last_operation(ctx)

        # Use main template to build final prompt
        return self.prompt_template.format(
            background=background,
            screenshot_info=screenshot_info,
            hints=self.hints,
            additional_info=ctx.add_info,
            history_operations=history_operations,
            task_list=task_list,
            last_operation=last_operation,
            task_requirements=self.task_requirements,
            output_format=self.output_format.format(
                action_options="Open app () or Run () or Wait or Stop. Only one action can be output at one time."
            ),
        )


class PC_prompt(BasePrompt):
    def __init__(self):
        super().__init__("PC")

        # PC-specific hints
        self.hints += r"""
If Tell action was used in the previous round, it cannot be used again this time.
To fully view webpage content, you must use the 'pagedown' key to scroll. Note that you can only advance one page at a time.
If you need to change the size of the webpage, you can do so by simultaneously pressing the ctrl and + or - keys.
The webpage you need to test is already displayed in front of you, so you don't need to open a browser.
If the target application requires uploading an image for testing, please upload the image located in "C:/test_data".
If the target application needs to upload a video for testing, please upload the video located in "C:/test_data".
If the target application needs to upload other files, please use the files in the "C:/test_data" directory as the main source.
If the target application requires entering a password, please first register an account and then use the account to log in.
"""

        # PC-specific task requirements
        self.task_requirements = """
In order to meet the user's requirements, you need to select one of the following operations to operate on the current screen:
For certain items that require selection, such as font and font size, direct input is more efficient than scrolling through choices.
You must choose one of the actions below:

- Run (your code)
    You can use this action to run python code. Your code will run on the computer for controlling the mouse and keyboard. You are required to use `pyautogui` to perform the action grounded to the observation, but DO NOT use the `pyautogui.locateCenterOnScreen` function to locate the element you want to operate with since we have no image of the element you want to operate with. DO NOT USE `pyautogui.screenshot()` to make a screenshot. Return one line of python code to perform the action each time. When predicting multiple code statements, separate them with semicolons (;) within the same line. Each time you need to predict a complete code, no variables or functions can be shared from history.

    For actions:
    - Use `pyautogui.click(x, y)` for single clicks
    - Use `pyautogui.doubleClick(x, y)` for double clicks
    - Use `pyautogui.rightClick(x, y)` for right clicks
    - Use `pyautogui.moveTo(x, y)` to move mouse
    - Use `pyautogui.dragTo(x, y)` to drag mouse
    - Use `pyautogui.scroll(amount)` to scroll, positive numbers scroll up, negative scroll down
    - Use `pyautogui.hotkey(key1, key2)` for keyboard shortcuts

    For text input:
    1. First check if the text box already contains content
    2. If it does, determine whether it needs to be cleared before inputting new text
    3. Use `pyperclip.copy(text)` to copy the content
    4. Use `pyautogui.hotkey(self.ctrl_key, 'v')` to paste it
    5. Make sure the content of text is enclosed in triple quotes

    Each action must end with a `time.sleep(duration)` statement. The `duration` should be chosen based on the expected time for the action to complete and the UI to update:
    - Simple clicks: 0.5-1 second
    - Text input: 1-2 seconds  
    - Opening apps/pages: 5-10 seconds
    - Loading content: 2-5 seconds
    You need to determine the appropriate duration based on the context.

    Example: Run (pyperclip.copy(\"\"\"hi\"\"\"); time.sleep(0.5); pyautogui.hotkey('ctrl', 'v'); time.sleep(1))

    Limit the execution to no more than 8 steps at a time to avoid errors.

- Tell (your answer)
    If you think the user's instruction has been fully satisfied, use this action to answer the user's question in English, and tell me the final answer. The final answer must be inside the brackets. Do not reuse this action to output the same response. If the required content has already been output using this action, then do not use it again. For example: Tell (100) ; Tell (OSagent is the best GUI Agent.)

- Stop
    If all the operations to meet the user's requirements have been completed in ### History operation ###, use this operation to stop the whole process."""

    def get_action_prompt(self, ctx: ActionPromptContext) -> str:
        # Build all sections using base class methods
        background = self._build_background(ctx, "computer")
        screenshot_info = self._build_screenshot_info(ctx, "of the current screenshot")
        history_operations = self._build_history_operations(ctx)
        task_list = self._build_task_list(ctx)
        last_operation = self._build_last_operation(ctx)

        # Use main template to build final prompt
        return self.prompt_template.format(
            background=background,
            screenshot_info=screenshot_info,
            hints=self.hints,
            additional_info=ctx.add_info,
            history_operations=history_operations,
            task_list=task_list,
            last_operation=last_operation,
            task_requirements=self.task_requirements,
            output_format=self.output_format.format(
                action_options="Open App () or Run () or Tell () or Stop. Only one action can be output at one time."
            ),
        )


case_batch_check_system_prompt = """
You are a professional and responsible web testing engineer (with real operation capabilities). I will provide you with a test task list, and you need to provide test results for all test tasks. If you fail to complete the test tasks, it may cause significant losses to the client. Please maintain the test tasks and their results in a task list. For test cases of a project, you must conduct thorough testing with at least five steps or more - the more tests, the more reliable the results.

[IMPORTANT]: You must test ALL test cases before providing your final report! Do not skip any test cases or fabricate results without actual testing! Failing to complete the entire task list will result in invalid test results and significant client losses.

Task Tips:
Standard Operating Procedure (SOP):
1. Determine test plan based on tasks and screenshots
2. Execute test plan for each test case systematically - verify each case in the task list one by one
3. After completing each test case, you can use Tell action to report that individual test case result
4. After completing ALL test case evaluations, use Tell action to report the COMPLETE results in the specified format

Reporting Language: Answer in natural English using structured format (like dictionaries). Tell me your judgment basis and results. You need to report the completion status of each condition in the task and your basis for determining whether it's complete.

Note that you're seeing only part of the app(or webpage) on screen. If you can't find modules mentioned in the task (especially when the right scroll bar shows you're at the top), try using pagedown to view the complete app(or webpage).

Inspection Standards:
1. Test cases are considered Pass if implemented on any page (not necessarily homepage). Please patiently review all pages (including scrolling down, clicking buttons to explore) before ending testing. You must understand relationships between pages - the first page you see is the target app's homepage.

2. If images in tested app(or webpage) modules aren't displaying correctly, that test case fails.

3. You may switch to other pages on the app(or webpage) during testing. On these pages, just confirm the test case result - don't mark other pages-passed cases as Fail if subpages lack features. Return to homepage after judging each case.

4. Trust your operations completely. If expected results don't appear after an operation, that function isn't implemented - report judgment as False.

5. If target module isn't found after complete app(or webpage) browsing, test case result is negative, citing "target module not found on any page" as basis.

6. Don't judge functionality solely by element attributes (clickable etc.) or text ("Filter by category" etc.). You must perform corresponding tests before outputting case results.

7. When tasks require operations for judgment, you must execute those operations. Final results can't have cases with unknown results due to lack of operations (clicks, inputs etc.).

8. For similar test cases (e.g., checking different social media links), if you verify one link works, you can assume others work normally.

For each individual test case completion, you can use Tell action to report just that result:
Tell ({"case_number": {"result": "Pass/Fail/Uncertain", "evidence": "Your evidence here"}})

Even in these failure cases, you must perform sufficient testing steps to prove your judgment before using the Tell action to report all results.

[VERIFICATION REQUIRED]: Before submitting your final report, verify that:
1. You have tested EVERY test case in the task list
2. Each test case has an explicit result (Pass/Fail/Uncertain)
3. Each result has supporting evidence based on your actual testing

Final Result Format (must include ALL test cases):
{{
    "0": {{"result": "Pass", "evidence": "The thumbnail click functionality is working correctly. When clicking on "Digital Artwork 1" thumbnail, it successfully redirects to a properly formatted detail page containing the artwork's title, image, description, creation process, sharing options, and comments section."}},
    "1": {{"result": "Uncertain", "evidence": "Cannot verify price calculation accuracy as no pricing information is displayed"}},
    "2": {{"result": "Fail", "evidence": "After fully browsing and exploring the web page, I did not find the message board appearing on the homepage or any subpage."}}
}}
**Return only the result string. Do not include any additional text, markdown formatting, or code blocks.**
"""
