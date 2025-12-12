"""
Analyze AgentRetryFail type cases, extract previous attempt information, and use LLM to generate possible retry suggestions
"""
import base64
import os
from typing import Any, List, Optional

from metagpt.logs import logger
from metagpt.utils.common import encode_image


async def generate_retry_action_online(
    task_description: str,
    action_history: List[str],
    summary_history: List[str],
    screenshot: Any = None,
    llm: Any = None,
) -> Optional[str]:
    """
    在线生成 retry 建议（不依赖文件系统，直接使用内存中的数据）

    Args:
        task_description: 任务描述
        action_history: 历史动作列表
        summary_history: 历史操作总结列表
        screenshot: 当前截图（可以是 bytes、base64 字符串或文件路径）
        llm: LLM 实例，用于调用 aask 方法

    Returns:
        LLM 生成的 retry 建议，如果失败则返回 None
    """
    if not action_history or not summary_history:
        logger.warning("No action history or summary history provided")
        return None

    if llm is None:
        logger.error("LLM instance is required for generate_retry_action_online")
        return None

    # 提取最后几个非 stop/tell 操作
    valid_operations = []
    for i, (action, summary) in enumerate(zip(action_history, summary_history)):
        action_lower = action.lower() if action else ""
        # 跳过 stop 和 tell 操作
        if action_lower == "stop" or action_lower.startswith("tell"):
            continue
        if summary and len(summary.strip()) > 0:
            valid_operations.append(summary)

    # 只取最后3个操作
    # valid_operations = valid_operations[-3:] if len(
    #     valid_operations) > 3 else valid_operations

    if not valid_operations:
        logger.warning("No valid operations found in history")
        return None

    # 构建 prompt
    attempt_logs_text = "\n\n".join(valid_operations)
    prompt_template = """Please generate a correct action to complete the task given the previous unsuccessful attempts. Output only one correct action, no other text.

## Task Description
{task_description}

## Previous attempts
{attempt_logs}
"""
    text_prompt = prompt_template.format(task_description=task_description, attempt_logs=attempt_logs_text)

    # 处理截图
    images = []
    if screenshot is not None:
        try:
            if isinstance(screenshot, bytes):
                screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")
                images.append(screenshot_base64)
            elif isinstance(screenshot, str) and os.path.exists(screenshot):
                images.append(encode_image(screenshot))
            elif isinstance(screenshot, str):
                # 假设已经是 base64 字符串
                images.append(screenshot)
        except Exception as e:
            logger.warning(f"Error processing screenshot for retry action: {e}")

    # 调用 LLM
    try:
        system_msg = (
            "You are a professional test analysis expert, skilled at analyzing test failure reasons and providing actionable retry suggestions."
        )
        retry_suggestion = await llm.aask(
            text_prompt,
            system_msgs=[system_msg],
            images=images if images else None,
            stream=False,
        )
        return retry_suggestion
    except Exception as e:
        logger.error(f"Error calling LLM for retry action: {e}")
        return None
