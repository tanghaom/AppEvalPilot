#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/01/20
@File    : tell_verifier.py
@Desc    : Action for verifying Tell action judgments against screenshot evidence
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from metagpt.actions.action import Action
from metagpt.config2 import Config
from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.utils.common import encode_image
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from appeval.prompts.tell_verifier import TellVerifierPrompts


class ActionError(BaseModel):
    """Details of action error when agent uses wrong interaction method or has operation failures"""

    error_type: str = Field(
        description="Error type: 'W4' for interaction modality mismatch, 'W6' for mechanics/focus failure"
    )
    error_description: str = Field(
        default="",
        description="Specific description of what went wrong with the agent's operation"
    )
    required_action: str = Field(
        default="",
        description="The correct interaction method or operation that should have been used"
    )
    corrective_guidance: str = Field(
        default="",
        description="Step-by-step instructions for the agent to perform the correct operation"
    )


class VerificationResult(BaseModel):
    """Result of Tell action verification"""

    is_valid: bool = Field(
        description="Whether the original judgment is valid")
    verification_status: str = Field(
        default="VALID",
        description="Verification status: 'VALID', 'OUTCOME_HALLUCINATION', 'CONFIRMATION_BIAS', "
                    "'PERCEPTION_HALLUCINATION', 'INTERACTION_MODALITY_MISMATCH', or 'MECHANICS_FOCUS_FAILURE'"
    )
    original_judgment: str = Field(
        description="The original Tell action content")
    corrected_judgment: Optional[str] = Field(
        default=None,
        description="The corrected Tell action content if correction is needed"
    )
    reasoning: str = Field(
        default="", description="Explanation of the verification analysis")
    corrections: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-case corrections with original and corrected results"
    )
    needs_correction: bool = Field(
        default=False,
        description="Whether the original judgment needs to be corrected"
    )
    action_error: Optional[ActionError] = Field(
        default=None,
        description="Details of action error if verification_status is INTERACTION_MODALITY_MISMATCH or MECHANICS_FOCUS_FAILURE"
    )
    has_action_error: bool = Field(
        default=False,
        description="Whether the verification found an action error (W4 or W6)"
    )

    @property
    def corrected_action(self) -> str:
        """Get the corrected Tell action string"""
        if self.corrected_judgment:
            return f"Tell ({self.corrected_judgment})"
        return f"Tell ({self.original_judgment})"

    @property
    def is_action_error(self) -> bool:
        """Check if this is an action error (W4 or W6)"""
        return self.verification_status in ["INTERACTION_MODALITY_MISMATCH", "MECHANICS_FOCUS_FAILURE"]

    def get_corrective_guidance(self) -> Optional[str]:
        """Get corrective guidance for action errors"""
        if self.action_error:
            return self.action_error.corrective_guidance
        return None


class TellVerifier(Action):
    """Verifies Tell action judgments against screenshot evidence

    This action is triggered when an agent outputs a Tell action to report test results.
    It analyzes historical screenshots and agent reflections to detect hallucination errors:
    - Outcome hallucination: Agent fabricates results not shown in screenshots
    - Confirmation bias: Agent concludes from weak/partial evidence
    """

    name: str = "TellVerifier"
    desc: str = "Verifies Tell action judgments against screenshot evidence"

    def __init__(self, config_path: str = "config/config2.yaml", max_screenshots: int = 10):
        """Initialize TellVerifier

        Args:
            config_path: Path to configuration file containing task_validator LLM config
            max_screenshots: Maximum number of screenshots to include in verification
        """
        super().__init__()
        self.config_path = Path(config_path)
        self.max_screenshots = max_screenshots

        # Load tell_verifier configuration (uses gpt-4o by default)
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
                verifier_config = config_data.get("tell_verifier")

                if verifier_config:
                    self.config = Config.from_llm_config(verifier_config)
                    self.llm = LLM(self.config.llm)
                    logger.info(
                        f"TellVerifier initialized with tell_verifier config: {verifier_config.get('model')}")
                else:
                    # Fallback to default LLM config if tell_verifier not found
                    logger.warning(
                        "tell_verifier config not found, using default LLM config")
                    default_config = config_data.get("llm", {})
                    self.config = Config.from_llm_config(default_config)
                    self.llm = LLM(self.config.llm)
        except Exception as e:
            logger.error(f"Failed to load TellVerifier config: {str(e)}")
            raise

    def _collect_screenshots(self, screenshot_dir: str, current_iter: int) -> List[str]:
        """Collect screenshot paths for verification

        Args:
            screenshot_dir: Directory containing screenshots
            current_iter: Current iteration number

        Returns:
            List[str]: List of screenshot file paths (oldest to newest)
        """
        screenshot_paths = []
        screenshot_dir_path = Path(screenshot_dir)

        if not screenshot_dir_path.exists():
            logger.warning(
                f"Screenshot directory does not exist: {screenshot_dir}")
            return screenshot_paths

        # Calculate which screenshots to include (most recent ones up to max_screenshots)
        start_iter = max(0, current_iter - self.max_screenshots + 1)

        for i in range(start_iter, current_iter + 1):
            origin_path = screenshot_dir_path / f"origin_{i}.jpg"
            if origin_path.exists():
                screenshot_paths.append(str(origin_path))

        logger.info(
            f"Collected {len(screenshot_paths)} screenshots for verification (iters {start_iter}-{current_iter})")
        return screenshot_paths

    def _format_history(self, history_list: List[str], prefix: str = "Step") -> str:
        """Format a history list into a readable string

        Args:
            history_list: List of history items
            prefix: Prefix for each step

        Returns:
            str: Formatted history string
        """
        if not history_list:
            return "No history available"

        formatted = []
        for i, item in enumerate(history_list, 1):
            # Clean up the item - remove excessive whitespace
            clean_item = item.replace("\\n", " ").strip() if item else "N/A"
            formatted.append(f"{prefix}-{i}: {clean_item}")

        return "\n".join(formatted)

    def _format_execution_history(
        self,
        action_history: List[str],
        reflection_history: List[str]
    ) -> str:
        """Format action and reflection history into a step-by-step execution history

        Args:
            action_history: List of actions performed at each step
            reflection_history: List of reflections after each step

        Returns:
            str: Formatted execution history with actions and reflections per step
        """
        if not action_history and not reflection_history:
            return "No execution history available"

        # Determine the number of steps (use the longer list)
        num_steps = max(len(action_history), len(reflection_history))

        if num_steps == 0:
            return "No execution history available"

        formatted = []
        for i in range(num_steps):
            step_num = i + 1
            formatted.append(f"--- Step {step_num} ---")

            # Add action if available
            if i < len(action_history) and action_history[i]:
                action = action_history[i].replace("\\n", " ").strip()
                formatted.append(f"  Action: {action}")
            else:
                formatted.append(f"  Action: N/A")

            # Add reflection if available
            if i < len(reflection_history) and reflection_history[i]:
                reflection = reflection_history[i].replace("\\n", " ").strip()
                formatted.append(f"  Reflection: {reflection}")
            else:
                formatted.append(f"  Reflection: N/A")

            formatted.append("")  # Add empty line between steps

        return "\n".join(formatted)

    def _extract_tell_content(self, tell_action: str) -> str:
        """Extract content from Tell action string

        Args:
            tell_action: Full Tell action string like "Tell ({...})"

        Returns:
            str: Extracted content between parentheses
        """
        if not tell_action.startswith("Tell"):
            return tell_action

        start_idx = tell_action.find("(")
        end_idx = tell_action.rfind(")")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return tell_action[start_idx + 1:end_idx].strip()

        return tell_action

    def _parse_verification_result(self, llm_output: str, original_judgment: str) -> VerificationResult:
        """Parse LLM output into VerificationResult

        Args:
            llm_output: Raw LLM output
            original_judgment: Original Tell content

        Returns:
            VerificationResult: Parsed verification result
        """
        try:
            # Try to extract JSON from the output
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', llm_output)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_start = llm_output.find("{")
                json_end = llm_output.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = llm_output[json_start:json_end]
                else:
                    logger.warning(
                        "Could not extract JSON from LLM output, treating as valid")
                    return VerificationResult(
                        is_valid=True,
                        verification_status="VALID",
                        original_judgment=original_judgment,
                        reasoning="Could not parse verification result, assuming valid",
                        needs_correction=False,
                    )

            # Parse JSON
            result_dict = json.loads(json_str)

            verification_status = result_dict.get(
                "verification_status", "VALID")
            reasoning = result_dict.get("reasoning", "")
            corrections = result_dict.get("corrections", {})
            corrected_tell_content = result_dict.get("corrected_tell_content")

            is_valid = verification_status == "VALID"
            needs_correction = not is_valid and corrected_tell_content is not None

            # Format corrected judgment as string
            corrected_judgment = None
            if corrected_tell_content:
                corrected_judgment = json.dumps(
                    corrected_tell_content, ensure_ascii=False)

            # Parse action_error if present (for W4 and W6 errors)
            action_error = None
            has_action_error = verification_status in [
                "INTERACTION_MODALITY_MISMATCH", "MECHANICS_FOCUS_FAILURE"
            ]

            if has_action_error:
                action_error_dict = result_dict.get("action_error", {})
                if action_error_dict:
                    action_error = ActionError(
                        error_type=action_error_dict.get("error_type", "W6"),
                        error_description=action_error_dict.get(
                            "error_description", ""),
                        required_action=action_error_dict.get(
                            "required_action", ""),
                        corrective_guidance=action_error_dict.get(
                            "corrective_guidance", ""),
                    )
                    logger.info(
                        f"Detected action error ({action_error.error_type}): {action_error.error_description}"
                    )
                else:
                    # Create a default action error if the LLM didn't provide details
                    action_error = ActionError(
                        error_type="W4" if verification_status == "INTERACTION_MODALITY_MISMATCH" else "W6",
                        error_description=reasoning,
                        required_action="",
                        corrective_guidance="Please review the task requirements and retry with the correct interaction method.",
                    )
                    logger.warning(
                        f"Action error detected but no action_error details provided in LLM output"
                    )

            return VerificationResult(
                is_valid=is_valid,
                verification_status=verification_status,
                original_judgment=original_judgment,
                corrected_judgment=corrected_judgment,
                reasoning=reasoning,
                corrections=corrections,
                needs_correction=needs_correction,
                action_error=action_error,
                has_action_error=has_action_error,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse verification result JSON: {str(e)}")
            return VerificationResult(
                is_valid=True,
                verification_status="VALID",
                original_judgment=original_judgment,
                reasoning=f"JSON parsing error: {str(e)}, assuming valid",
                needs_correction=False,
            )
        except Exception as e:
            logger.error(f"Error parsing verification result: {str(e)}")
            return VerificationResult(
                is_valid=True,
                verification_status="VALID",
                original_judgment=original_judgment,
                reasoning=f"Parsing error: {str(e)}, assuming valid",
                needs_correction=False,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"TellVerifier LLM call failed, {retry_state.attempt_number}th retry: {str(retry_state.outcome.exception())}"
        ),
        reraise=True,
    )
    async def _call_llm(self, prompt: str, images: List[str]) -> str:
        """Call LLM with prompt and images

        Args:
            prompt: Verification prompt
            images: List of base64 encoded images

        Returns:
            str: LLM response
        """
        response = await self.llm.aask(
            prompt,
            system_msgs=[TellVerifierPrompts.SYSTEM_PROMPT],
            images=images,
            stream=False,
        )
        return response

    async def run(
        self,
        tell_content: str,
        action_history: List[str],
        reflection_history: List[str],
        screenshot_dir: str,
        current_iter: int,
        test_cases: str = "",
    ) -> VerificationResult:
        """Execute Tell action verification

        Args:
            tell_content: The Tell action content to verify (e.g., "Tell ({...})")
            action_history: List of actions performed at each step
            reflection_history: List of reflection thoughts for each step
            screenshot_dir: Directory containing screenshot files
            current_iter: Current iteration number
            test_cases: Original test cases being verified (optional)

        Returns:
            VerificationResult: Verification result with potential corrections
        """
        logger.info("Starting Tell action verification...")

        # Extract Tell content
        original_judgment = self._extract_tell_content(tell_content)
        logger.info(
            f"Original Tell content to verify: {original_judgment[:200]}...")

        # Collect screenshots
        screenshot_paths = self._collect_screenshots(
            screenshot_dir, current_iter)

        if not screenshot_paths:
            logger.warning(
                "No screenshots available for verification, skipping")
            return VerificationResult(
                is_valid=True,
                verification_status="VALID",
                original_judgment=original_judgment,
                reasoning="No screenshots available for verification",
                needs_correction=False,
            )

        # Encode screenshots
        encoded_images = []
        for path in screenshot_paths:
            try:
                encoded_images.append(encode_image(path))
            except Exception as e:
                logger.warning(f"Failed to encode screenshot {path}: {str(e)}")

        if not encoded_images:
            logger.warning(
                "No screenshots could be encoded, skipping verification")
            return VerificationResult(
                is_valid=True,
                verification_status="VALID",
                original_judgment=original_judgment,
                reasoning="No screenshots could be encoded for verification",
                needs_correction=False,
            )

        # Format execution history (action + reflection per step)
        formatted_execution_history = self._format_execution_history(
            action_history, reflection_history)

        # Build verification prompt
        prompt = TellVerifierPrompts.get_verification_prompt(
            test_cases=test_cases or "Not provided",
            judgment=original_judgment,
            execution_history=formatted_execution_history,
        )

        # Print the full verification prompt for debugging
        logger.info(
            f"\n\n######################## Tell Verifier Prompt:\n{prompt}\n"
            f"######################## Tell Verifier Prompt End\n"
            f"Screenshots count: {len(encoded_images)}\n\n"
        )

        # Call LLM for verification
        try:
            llm_output = await self._call_llm(prompt, encoded_images)
            logger.info(f"Verification LLM response: {llm_output[:500]}...")
        except Exception as e:
            logger.error(f"LLM verification call failed: {str(e)}")
            return VerificationResult(
                is_valid=True,
                verification_status="VALID",
                original_judgment=original_judgment,
                reasoning=f"LLM call failed: {str(e)}, assuming valid",
                needs_correction=False,
            )

        # Parse result
        result = self._parse_verification_result(llm_output, original_judgment)

        # Log verification outcome
        if result.has_action_error:
            # Action error detected (W4 or W6)
            error_type_name = (
                "Interaction Modality Mismatch (W4)"
                if result.verification_status == "INTERACTION_MODALITY_MISMATCH"
                else "Mechanics & Focus Failure (W6)"
            )
            logger.warning(
                f"Tell action verification detected ACTION ERROR: {error_type_name}\n"
                f"Reasoning: {result.reasoning}\n"
            )
            if result.action_error:
                logger.info(
                    f"Action Error Details:\n"
                    f"  - Error Type: {result.action_error.error_type}\n"
                    f"  - Description: {result.action_error.error_description}\n"
                    f"  - Required Action: {result.action_error.required_action}\n"
                    f"  - Corrective Guidance: {result.action_error.corrective_guidance}\n"
                )
        elif result.needs_correction:
            logger.warning(
                f"Tell action verification found hallucination issues: {result.verification_status}, "
                f"reasoning: {result.reasoning[:200]}..."
            )
        else:
            logger.info(
                f"Tell action verification passed: {result.verification_status}")

        return result
