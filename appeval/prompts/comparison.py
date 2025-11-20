COMPARISON_PROMPT_TEMPLATE = """
You are analyzing two screenshots taken before and after an operation.

### Before the operation ###
This is the screenshot taken BEFORE the operation was executed.

### After the operation ###
This is the screenshot taken AFTER the operation was executed.

### Operation Context ###
User instruction: {instruction}
Operation thought: {operation_thought}
Operation action: {action}

### Analysis Requirements ###
Please carefully compare the two screenshots and identify ALL changes that occurred. **Pay special attention to changes directly related to the executed action.**

- What specific UI elements were directly affected by the action (clicked buttons, input fields, selected items, etc.)
- Immediate visual feedback from the action (button state changes, highlight effects, focus indicators)
- Elements that appeared or disappeared as a direct result of the action
- Changes in the target element's state (enabled/disabled, selected/unselected, active/inactive)


### Output format ###
Please provide a summary of the changes that occurred using a paragraph.
If there are no changes, please say "No changes occurred."
"""