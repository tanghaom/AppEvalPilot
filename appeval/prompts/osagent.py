from typing import Dict, List

from pydantic import BaseModel


class ActionPromptContext(BaseModel):
    """Action提示的上下文数据"""

    instruction: str  # 用户指令
    clickable_infos: List[Dict]  # 可点击元素信息列表
    width: int  # 屏幕宽度
    height: int  # 屏幕高度
    thought_history: List[str]  # 历史思考记录
    summary_history: List[str]  # 历史操作摘要
    action_history: List[str]  # 历史动作记录
    reflection_thought_history: List[str]  # 历史反思记录
    last_summary: str  # 上一次操作摘要
    last_action: str  # 上一次执行的动作
    reflection_thought: str  # 当前反思内容
    add_info: str  # 额外补充信息
    error_flag: bool  # 错误标志
    completed_content: str  # 已完成的内容
    memory: List[str]  # 记忆列表
    task_list: str  # 任务列表
    use_som: bool  # 是否使用SOM
    location_info: str = "center"  # 位置信息格式,默认为中心点坐标
    icon_caption: bool = True  # 是否使用图标描述


class BasePrompt:
    """Prompt模板基类"""

    def __init__(self, platform: str):
        self.platform = platform

        # 基础提示模板
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

{reflection_thought}

### Task requirements ###
{task_requirements}

### Output format ###
{output_format}
"""

        # 基础背景信息模板
        self.background_template = """The {image_desc} width is {width} pixels and its height is {height} pixels. The user's instruction is: {instruction}."""

        # 基础截图信息模板
        self.screenshot_info_template = """
In order to help you better perceive the content in this screenshot, we extract some information {source_desc}.
This information consists of two parts: coordinates; content. 
{location_format}
{content_format}
The information is as follow:
{clickable_info}
Please note that this information is not necessarily accurate. You need to combine the screenshot to understand."""

        # 基础提示模板
        self.hints = """
There are hints to help you complete the user's instructions. The hints are as follow:
When there is no direct element counterpart, you should guess the possible elements based on the task and its coordinates.
Sometimes both shortcuts and clicking can accomplish the same action; in such cases, prioritize using shortcuts.
Do not make a location estimate, if it does not pop up, please wait.
You should not assume the position of an element you cannot see.
Perform only one click at a time, Do not skip steps, please wait for the previous click action to finish.
Pay attention to the history to verify that it has been completed and avoid duplicate operations."""

        # 基础历史操作模板
        self.history_template = """
### History operations ###
Before reaching this page, some operations have been completed. You need to refer to the completed operations to decide the next operation. These history records include the following components:
- Operation: A textual description of the intended action.
- Action: The action that was executed.
- Reflection_thought: Reflections made after executing the action. It's not necessarily correct, you should make your own judgment.
- Memory: Potentially useful content extracted from screenshots {memory_timing} the action was performed.
{history_details}"""

        # 基础输出格式模板
        self.output_format = """
Your output consists of the following four parts. Please note that only one set of content should be output at a time, and do not repeat the output.
### Thought ###
This is your thinking about how to proceed the next operation, please output the thoughts about the history operations explicitly.
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

        # 包名提示模板
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

        # 映射信息模板
        self.mapping_info_template = """
### Hints ###
The following is the correspondence between some common application names and their package names, which may be helpful for you to complete the task. It's not necessarily correct, you should make your own judgment.
{app_mapping}"""

    def get_action_prompt(self, ctx: ActionPromptContext) -> str:
        raise NotImplementedError

    def get_package_name_prompt(self, app_name: str, app_mapping: str, package_list: List[str]) -> str:
        raise NotImplementedError


class Android_prompt(BasePrompt):
    def __init__(self):
        super().__init__("Android phone")

        # Android特有的任务要求
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
        # 构建背景信息
        image_desc = (
            "first image is a clean phone screenshot. And the second image is the annotated version of it, where icons are marked with numbers."
            if ctx.use_som
            else "image is a phone screenshot."
        )
        background = self.background_template.format(
            image_desc=image_desc, width=ctx.width, height=ctx.height, instruction=ctx.instruction
        )

        # 构建位置格式信息
        location_format = {
            "center": "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from top to bottom;",
            "bbox": "The format of the coordinates is [x1, y1, x2, y2], x is the pixel from left to right and y is the pixel from top to bottom. (x1, y1) is the coordinates of the upper-left corner, (x2, y2) is the coordinates of the bottom-right corner;",
        }[ctx.location_info]

        # 构建内容格式信息
        content_format = """the content can be one of three types:
1. text from OCR
2. icon description or 'icon'
3. element information from device tree, which contains element attributes like type, text content, identifiers, accessibility descriptions and position information"""

        # 构建可点击信息
        clickable_info = "\n".join(
            f"{info['coordinates']}; {info['text']}"
            for info in ctx.clickable_infos
            if info["text"] != "" and info["text"] != "icon: None" and info["coordinates"] != (0, 0)
        )

        # 构建截图信息
        screenshot_info = self.screenshot_info_template.format(
            source_desc="on the current screenshot through system files",
            location_format=location_format,
            content_format=content_format,
            clickable_info=clickable_info,
        )

        # 构建历史操作信息
        history_operations = ""
        if len(ctx.action_history) > 0:
            history_details = ""
            for i in range(len(ctx.action_history)):
                history_details += (
                    f"Step-{i+1}:\n\tOperation: {ctx.summary_history[i]}\n\tAction: {ctx.action_history[i]}\n"
                )
                history_details += f"\tReflection_thought: {ctx.reflection_thought_history[i] if len(ctx.reflection_thought_history) == len(ctx.action_history) else 'None'}\n"
                history_details += (
                    f"\tMemory: {ctx.memory[i] if len(ctx.memory) == len(ctx.action_history) else 'None'}\n"
                )
            history_operations = self.history_template.format(memory_timing="after", history_details=history_details)

        # 构建任务列表信息
        task_list = (
            f"### Last Task List ###\nHere is the task list generated in the previous step. Please use this information to update the task list in the current step. Specifically, you should:\n1. Identify and move any completed tasks to the `[Completed Tasks]` section.\n2. Determine the current task and place it in the `[Current Task]` section.\n3. Plan the next immediate operation and detail its steps in the `[Next Operation]` section.\n4. List the remaining tasks (excluding the current and next operation) in the `[Remaining Tasks]` section. Adjust or refine these tasks as needed based on the current context.\n{ctx.task_list}"
            if ctx.task_list
            else ""
        )

        # 构建最后操作信息
        last_operation = ""
        if ctx.error_flag:
            last_operation = f'### Last operation ###\nYou previously wanted to perform the operation "{ctx.last_summary}" on this page and executed the Action "{ctx.last_action}". But you find that this operation does not meet your expectation. You need to reflect and revise your operation this time.'

        # 构建反思信息
        reflection_thought = (
            f"### The reflection thought of the last operation ###\n{ctx.reflection_thought}"
            if ctx.error_flag and ctx.reflection_thought
            else ""
        )

        # 使用主模板构建最终提示
        return self.prompt_template.format(
            background=background,
            screenshot_info=screenshot_info,
            hints=self.hints,
            additional_info=ctx.add_info,
            history_operations=history_operations,
            task_list=task_list,
            last_operation=last_operation,
            reflection_thought=reflection_thought,
            task_requirements=self.task_requirements,
            output_format=self.output_format.format(
                action_options="Open app () or Run () or Wait or Stop. Only one action can be output at one time."
            ),
        )

    def get_package_name_prompt(self, app_name: str, app_mapping: str, package_list: List[str]) -> str:
        # 构建映射信息
        mapping_info = self.mapping_info_template.format(app_mapping=app_mapping) if app_mapping else ""

        # 使用主模板构建提示
        return self.package_name_template.format(
            app_name=app_name, platform=self.platform, mapping_info=mapping_info, package_list=package_list
        )


class PC_prompt(BasePrompt):
    def __init__(self):
        super().__init__("PC")

        # PC特有的提示
        self.hints += """
If Tell action was used in the previous round, it cannot be used again this time.
To fully view webpage content, you must use the 'pagedown' key to scroll. Note that you can only advance one page at a time."""

        # PC特有的任务要求
        self.task_requirements = """
In order to meet the user's requirements, you need to select one of the following operations to operate on the current screen:
Note that to open an app, use the Open App action, rather than tapping the app's icon. 
For certain items that require selection, such as font and font size, direct input is more efficient than scrolling through choices.
You must choose one of the actions below:
- Open App (app name)
    If you want to open an app, you should use this action to open the app named 'app name'.

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
        # 构建背景信息
        image_desc = (
            "first image is a clean computer screenshot. And the second image is the annotated version of it, where icons are marked with numbers."
            if ctx.use_som == 1
            else "image is a computer screenshot."
        )
        background = self.background_template.format(
            image_desc=image_desc, width=ctx.width, height=ctx.height, instruction=ctx.instruction
        )

        # 构建位置格式信息
        location_format = {
            "center": "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from top to bottom;",
            "bbox": "The format of the coordinates is [x1, y1, x2, y2], x is the pixel from left to right and y is the pixel from top to bottom. (x1, y1) is the coordinates of the upper-left corner, (x2, y2) is the coordinates of the bottom-right corner;",
        }[ctx.location_info]

        # 构建内容格式信息
        content_format = """the content can be one of three types:
1. text from OCR
2. icon description or 'icon'
3. element information from device tree, which contains element attributes like type, text content, identifiers, accessibility descriptions and position information"""

        # 构建可点击信息
        clickable_info = "\n".join(
            f"{info['coordinates']}; {info['text']}"
            for info in ctx.clickable_infos
            if info["text"] != "" and info["text"] != "icon: None" and info["coordinates"] != (0, 0)
        )

        # 构建截图信息
        screenshot_info = self.screenshot_info_template.format(
            source_desc="of the current screenshot",
            location_format=location_format,
            content_format=content_format,
            clickable_info=clickable_info,
        )

        # 构建历史操作信息
        history_operations = ""
        if len(ctx.action_history) > 0:
            history_details = ""
            for i in range(len(ctx.action_history)):
                history_details += (
                    f"Step-{i+1}:\n\tOperation: {ctx.summary_history[i]}\n\tAction: {ctx.action_history[i]}\n"
                )
                history_details += f"\tReflection_thought: {ctx.reflection_thought_history[i] if len(ctx.reflection_thought_history) == len(ctx.action_history) else 'None'}\n"
                history_details += (
                    f"\tMemory: {ctx.memory[i] if len(ctx.memory) == len(ctx.action_history) else 'None'}\n"
                )
            history_operations = self.history_template.format(memory_timing="before", history_details=history_details)

        # 构建任务列表信息
        task_list = (
            f"### Last Task List ###\nHere is the task list generated in the previous step. Please use this information to update the task list in the current step. Specifically, you should:\n1. Identify and move any completed tasks to the `[Completed Tasks]` section.\n2. Determine the current task and place it in the `[Current Task]` section.\n3. Plan the next immediate operation and detail its steps in the `[Next Operation]` section.\n4. List the remaining tasks (excluding the current and next operation) in the `[Remaining Tasks]` section. Adjust or refine these tasks as needed based on the current context.\n{ctx.task_list}"
            if ctx.task_list
            else ""
        )

        # 构建最后操作信息
        last_operation = ""
        if ctx.error_flag:
            last_operation = f'### Last operation ###\nYou previously wanted to perform the operation "{ctx.last_summary}" on this page and executed the Action "{ctx.last_action}". But you find that this operation does not meet your expectation. You need to reflect and revise your operation this time.'

        # 构建反思信息
        reflection_thought = (
            f"### The reflection thought of the last operation ###\n{ctx.reflection_thought}"
            if ctx.error_flag and ctx.reflection_thought
            else ""
        )

        # 使用主模板构建最终提示
        return self.prompt_template.format(
            background=background,
            screenshot_info=screenshot_info,
            hints=self.hints,
            additional_info=ctx.add_info,
            history_operations=history_operations,
            task_list=task_list,
            last_operation=last_operation,
            reflection_thought=reflection_thought,
            task_requirements=self.task_requirements,
            output_format=self.output_format.format(
                action_options="Open App () or Run () or Tell () or Stop. Only one action can be output at one time."
            ),
        )

    def get_package_name_prompt(self, app_name: str, app_mapping: str, package_list: List[str]) -> str:
        # 构建映射信息
        mapping_info = self.mapping_info_template.format(app_mapping=app_mapping) if app_mapping else ""

        # 使用主模板构建提示
        return self.package_name_template.format(
            app_name=app_name, platform=self.platform, mapping_info=mapping_info, package_list=package_list
        )
