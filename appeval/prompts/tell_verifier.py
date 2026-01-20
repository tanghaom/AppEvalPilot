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

You must detect two types of hallucination errors:

1. **Outcome Hallucination**: The agent fabricates or claims results that do not appear in the screenshots.
   - Example: Agent claims "text was successfully input" but the text field in the screenshot is empty
   - Example: Agent claims "cursor moved to next line" but status bar shows same position

2. **Confirmation Bias from Partial Cues**: The agent draws conclusions from weak or partial evidence, ignoring key requirements.
   - Example: Agent sees a counter change and claims "feature fully working" without verifying all related functionality
   - Example: Agent sees a loading spinner disappear and concludes "operation successful" without checking the actual result

Your verification must be based ONLY on observable evidence in the screenshots. Do not assume or infer results that are not visually verifiable."""

    VERIFICATION_PROMPT_TEMPLATE = """
### Verification Task ###
You need to verify if the agent's judgment is accurate based on the screenshot evidence.

**Test Task and Expected Result:**
{test_cases}

**Agent's Judgment (to verify):**
{judgment}

**Agent's Historical Reflections (step by step):**
{reflection_history}

**Screenshots Provided:**
The images provided are screenshots from the agent's execution, ordered from oldest to newest (LATEST screenshot is LAST)

### Output Format ###
Provide your verification result in the following JSON format:
```json
{{
    "verification_status": "VALID" | "HALLUCINATION" | "PARTIAL_EVIDENCE",
    "reasoning": "Detailed explanation of your verification analysis",
    "corrections": {{
        "case_id": {{
            "original_result": "Pass/Fail/Uncertain",
            "corrected_result": "Pass/Fail/Uncertain",
            "corrected_evidence": "Evidence based on actual screenshot observation"
        }}
    }},
    "corrected_tell_content": {{...}} // Full corrected Tell content if any corrections needed, null if VALID
}}
```
"""

    @classmethod
    def get_verification_prompt(
        cls,
        test_cases: str,
        judgment: str,
        reflection_history: str,
    ) -> str:
        """Build the verification prompt with all context

        Args:
            test_cases: The original test cases being verified
            judgment: The agent's judgment (Tell action content) to verify
            reflection_history: Agent's reflection history

        Returns:
            str: Complete verification prompt
        """
        return cls.VERIFICATION_PROMPT_TEMPLATE.format(
            test_cases=test_cases,
            judgment=judgment,
            reflection_history=reflection_history,
        )
