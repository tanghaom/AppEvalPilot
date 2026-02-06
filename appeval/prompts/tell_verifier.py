#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/01/20
@File    : tell_verifier.py
@Desc    : Prompts for Tell action verification module
"""


class TellVerifierPrompts:
    """Prompts for verifying Tell action judgments against screenshot evidence"""

    SYSTEM_PROMPT = """You are an expert verification system for GUI automation agents. Your task is to verify whether an agent's reported test results (Tell action) accurately reflect what is shown in the screenshots.

You must detect the following types of errors:

## Hallucination Errors (Judgment Issues)

1. **Outcome Hallucination**: The agent fabricates or claims results that do not appear in the screenshots.
   - Example: Agent claims "text was successfully input" but the text field in the screenshot is empty
   - Example: Agent claims "cursor moved to next line" but status bar shows same position

2. **Confirmation Bias from Partial Cues**: The agent draws conclusions from weak or partial evidence, ignoring key requirements.
   - Example: Agent sees a counter change and claims "feature fully working" without verifying all related functionality
   - Example: Agent sees a loading spinner disappear and concludes "operation successful" without checking the actual result

3. **Perception Hallucination**: The page actually changed, but the agent claims there was no change.
   - Example: Agent claims "no change" but the page actually changed (e.g., new visualizations appeared, color changed, etc.).

## Action Errors (Operation Issues)

4. **Interaction Modality Mismatch (W4)**: The task requires a specific interaction method (hover/drag/keyboard/scroll), but the agent uses click or other substitutes, resulting in verifying a different functionality.
   - Example: Task requires hover to trigger a menu, but agent uses click
   - Example: Task requires drag-and-drop, but agent uses copy-paste or other methods
   - Example: Task requires keyboard shortcut (Ctrl+C), but agent uses right-click menu
   - Example: Task requires mouse scroll to test scroll behavior, but agent uses scrollbar click

5. **Mechanics & Focus Failures (W6)**: Basic operation mistakes that prevent reaching a verifiable state.
   - Example: Input field not focused before typing
   - Example: Unable to clear the address bar content
   - Example: Accidentally paused the game when it should be running
   - Example: Keyboard focus not on the canvas/target element
   - Example: Click coordinates off-target (missed the button)
   - Example: Wrong window or tab in focus
   - Example: Modal dialog blocking the target element

**IMPORTANT**: For Action Errors (W4 and W6), you must provide corrective guidance that instructs the agent on how to perform the correct operation to continue the task.

Your verification must be based ONLY on observable evidence in the screenshots and the agent's historical reflections. Do not assume or infer results that are not visually verifiable."""

    VERIFICATION_PROMPT_TEMPLATE = """
### Verification Task ###
You need to verify if the agent's judgment is accurate based on the screenshot evidence and check if the agent used correct interaction methods.

**Test Task and Expected Result:**
{test_cases}

**Agent's Judgment (to verify):**
{judgment}

**Agent's Execution History (step by step):**
Each step contains the action performed and the agent's reflection after observing the result.
{execution_history}

**Screenshots Provided:**
The images provided are screenshots from the agent's execution, ordered from oldest to newest (LATEST screenshot is LAST)

### Verification Steps ###
1. First, check if the agent used the CORRECT interaction method as required by the task (hover, drag, keyboard shortcut, scroll, etc.)
2. Then, verify if the agent's operations reached the correct state (focus, target element, etc.)
3. Finally, verify if the agent's judgment matches the screenshot evidence

### Output Format ###
Provide your verification result in the following JSON format:
```json
{{
    "verification_status": "VALID" | "OUTCOME_HALLUCINATION" | "CONFIRMATION_BIAS" | "PERCEPTION_HALLUCINATION" | "INTERACTION_MODALITY_MISMATCH" | "MECHANICS_FOCUS_FAILURE",
    "reasoning": "Detailed explanation of your verification analysis",
    "corrections": {{
        "case_id": {{
            "original_result": "Pass/Fail/Uncertain",
            "corrected_result": "Pass/Fail/Uncertain",
            "corrected_evidence": "Evidence based on actual screenshot observation"
        }}
    }},
    "corrected_tell_content": {{...}},  // Full corrected Tell content if any corrections needed, null if VALID
    "action_error": {{  // Required ONLY for INTERACTION_MODALITY_MISMATCH or MECHANICS_FOCUS_FAILURE
        "error_type": "W4" | "W6",  // W4 for interaction mismatch, W6 for mechanics/focus failure
        "error_description": "Specific description of what went wrong with the agent's operation",
        "required_action": "The correct interaction method or operation that should have been used",
        "corrective_guidance": "Step-by-step instructions for the agent to perform the correct operation and continue the task"
    }}
}}
```

### Important Notes ###
- For VALID status: "corrections" should be empty and "corrected_tell_content" should be null
- For hallucination errors (OUTCOME_HALLUCINATION, CONFIRMATION_BIAS, PERCEPTION_HALLUCINATION): provide "corrections" and "corrected_tell_content"
- For action errors (INTERACTION_MODALITY_MISMATCH, MECHANICS_FOCUS_FAILURE): MUST provide "action_error" with detailed corrective guidance to help the agent continue correctly
"""

    @classmethod
    def get_verification_prompt(
        cls,
        test_cases: str,
        judgment: str,
        execution_history: str,
    ) -> str:
        """Build the verification prompt with all context

        Args:
            test_cases: The original test cases being verified
            judgment: The agent's judgment (Tell action content) to verify
            execution_history: Agent's execution history with actions and reflections per step

        Returns:
            str: Complete verification prompt
        """
        return cls.VERIFICATION_PROMPT_TEMPLATE.format(
            test_cases=test_cases,
            judgment=judgment,
            execution_history=execution_history,
        )
