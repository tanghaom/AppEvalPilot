#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/01
@File    : hijack_osagent.py
@Author  : tanghaoming
@Desc    : OSAgent劫持工具 - 通过劫持llm.aask实现记录功能

使用说明:
1. 导入OSAgentHijacker类
2. 创建OSAgentHijacker实例
3. 创建OSAgent实例
4. 使用hijacker.hijack(agent)应用劫持
5. 运行OSAgent任务
6. 使用hijacker.cleanup()清理资源

配置选项:
- log_dir: 日志保存目录
- save_full_state: 是否保存完整状态（设为False可减少磁盘占用）
- save_images: 是否保存图片（设为False可减少磁盘占用）
- log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
- directory_structure: 自定义目录结构
- perf_stats_interval: 性能统计周期

示例用法:
```python
from hijack_osagent import OSAgentHijacker
from appeval.roles.osagent import OSAgent
import asyncio

async def main():
    # 创建劫持器
    hijacker = OSAgentHijacker(log_dir="logs", save_full_state=True)
    
    # 创建OSAgent
    agent = OSAgent(platform="Windows")
    
    # 应用劫持
    hijacker.hijack(agent)
    
    try:
        # 运行任务
        result = await agent.run("执行某个任务")
    finally:
        # 清理资源
        await hijacker.cleanup()
        
if __name__ == "__main__":
    asyncio.run(main())
```
"""
import asyncio
import json
import shutil
import sys
import time
import traceback
from functools import wraps
from pathlib import Path

import aiofiles
from loguru import logger

from appeval.roles.osagent import OSAgent


def async_error_handler(func):
    """异步函数错误处理装饰器

    Args:
        func: 要装饰的异步函数

    Returns:
        包装后的异步函数
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_details = {
                "function": func.__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            logger.error(f"函数 {func.__name__} 执行出错: {e}")
            logger.debug(f"详细错误信息: {error_details}")
            # 对于关键操作，返回None可能会导致后续流程出错
            # 所以只记录日志，然后继续抛出异常
            raise

    return wrapper


class OSAgentHijacker:
    """OSAgent劫持器，用于记录调用过程中的状态和输出"""

    def __init__(
        self,
        log_dir="osagent_logs",
        save_full_state=True,
        save_images=True,
        log_level="INFO",
        directory_structure=None,
        perf_stats_interval=5,
    ):
        """初始化劫持器

        Args:
            log_dir: 日志保存目录
            save_full_state: 是否保存完整状态，设为False可节省磁盘空间
            save_images: 是否保存图片，设为False可节省磁盘空间
            log_level: 日志级别，可选值为DEBUG、INFO、WARNING、ERROR
            directory_structure: 自定义目录结构，如果为None则使用默认结构
            perf_stats_interval: 性能统计周期（单位：次迭代），设为0则不启用周期统计
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.original_aask = None
        self.original_reflection = None
        self.original_reset_state = None
        self.directory_structure = directory_structure
        self.save_full_state = save_full_state
        self.save_images = save_images
        self.perf_stats_interval = perf_stats_interval
        self.log_level = log_level

        # 设置日志级别
        logger.remove()
        logger.add(sys.stderr, level=log_level)

        # 创建新的会话目录
        self._setup_session_dirs()

    def _setup_session_dirs(self):
        """创建新的会话目录结构"""
        # 生成新时间戳
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / self.timestamp
        self.session_dir.mkdir(exist_ok=True)

        # 添加session特定的日志文件
        logger.add(self.session_dir / "hijacker.log", level=self.log_level)

        # 性能统计重置
        self.perf_stats = {
            "aask_calls": 0,
            "reflection_calls": 0,
            "aask_total_time": 0,
            "reflection_total_time": 0,
            "aask_max_time": 0,
            "reflection_max_time": 0,
            "errors": 0,
            "start_time": time.time(),
        }

        # 创建子目录结构
        if self.directory_structure is None:
            # 使用默认目录结构
            self.task_dir = self.session_dir / "task"
            self.task_prompts_dir = self.task_dir / "prompts"
            self.task_outputs_dir = self.task_dir / "outputs"
            self.task_images_dir = self.task_dir / "images"
            self.think_dir = self.session_dir / "think"
            self.think_prompts_dir = self.think_dir / "prompts"
            self.think_states_dir = self.think_dir / "states"
            self.think_outputs_dir = self.think_dir / "outputs"
            self.think_images_dir = self.think_dir / "images"
            self.reflection_dir = self.session_dir / "reflection"
            self.reflection_states_dir = self.reflection_dir / "states"
            self.reflection_prompts_dir = self.reflection_dir / "prompts"
            self.reflection_outputs_dir = self.reflection_dir / "outputs"
            self.reflection_images_dir = self.reflection_dir / "images"
            self.performance_dir = self.session_dir / "performance"

            self.dirs_to_create = [
                self.task_dir,
                self.task_prompts_dir,
                self.task_outputs_dir,
                self.task_images_dir,
                self.think_dir,
                self.think_prompts_dir,
                self.think_states_dir,
                self.think_outputs_dir,
                self.think_images_dir,
                self.reflection_dir,
                self.reflection_states_dir,
                self.reflection_prompts_dir,
                self.reflection_outputs_dir,
                self.reflection_images_dir,
                self.performance_dir,
            ]
        else:
            # 使用自定义目录结构
            self.dirs_to_create = []
            for dir_name, path in self.directory_structure.items():
                full_path = self.session_dir / path
                setattr(self, dir_name, full_path)
                self.dirs_to_create.append(full_path)

            # 确保性能目录存在
            if not hasattr(self, "performance_dir"):
                self.performance_dir = self.session_dir / "performance"
                self.dirs_to_create.append(self.performance_dir)

        # 创建所有目录
        for d in self.dirs_to_create:
            d.mkdir(exist_ok=True, parents=True)

        logger.info(f"已创建日志目录: {self.session_dir}")
        logger.info(
            f"配置: save_full_state={self.save_full_state}, save_images={self.save_images}, perf_stats_interval={self.perf_stats_interval}"
        )

    @async_error_handler
    async def save_json_async(self, file_path, data):
        """异步保存JSON数据到文件

        Args:
            file_path: 保存路径
            data: 要保存的数据
        """
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=4))

    @async_error_handler
    async def save_text_async(self, file_path, text):
        """异步保存文本数据到文件

        Args:
            file_path: 保存路径
            text: 要保存的文本
        """
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(text)

    @async_error_handler
    async def save_image_async(self, source_path, target_path):
        """异步保存图片

        Args:
            source_path: 源图片路径
            target_path: 目标路径
        """
        if source_path and Path(source_path).exists():
            # 创建目标目录（如果不存在）
            target_path.parent.mkdir(exist_ok=True, parents=True)
            # 复制文件
            shutil.copy2(source_path, target_path)

    def extract_rc_data(self, rc):
        """从RunContext提取数据

        Args:
            rc: RunContext对象

        Returns:
            dict: 提取的数据字典
        """
        try:
            return {
                "perception_infos": getattr(rc, "perception_infos", []),
                "thought_history": getattr(rc, "thought_history", []),
                "summary_history": getattr(rc, "summary_history", []),
                "action_history": getattr(rc, "action_history", []),
                "reflection_thought_history": getattr(rc, "reflection_thought_history", []),
                "thought": getattr(rc, "thought", ""),
                "summary": getattr(rc, "summary", ""),
                "action": getattr(rc, "action", ""),
                "reflection_thought": getattr(rc, "reflection_thought", ""),
                "error_flag": getattr(rc, "error_flag", False),
                "memory": getattr(rc, "memory", []),
                "task_list": getattr(rc, "task_list", ""),
                "completed_requirements": getattr(rc, "completed_requirements", ""),
                "iter": getattr(rc, "iter", 0),
                "last_perception_infos": getattr(rc, "last_perception_infos", []),
                "width": getattr(rc, "width", 0),
                "height": getattr(rc, "height", 0),
                "webbrowser_console_logs": getattr(rc, "webbrowser_console_logs", []),
            }
        except Exception as e:
            logger.error(f"提取RunContext数据出错: {e}")
            return {}

    async def update_performance_stats(self, iteration=None):
        """更新并保存性能统计信息

        Args:
            iteration: 当前迭代次数，如果为None则不保存文件
        """
        try:
            # 计算平均时间
            aask_avg_time = self.perf_stats["aask_total_time"] / max(1, self.perf_stats["aask_calls"])
            reflection_avg_time = self.perf_stats["reflection_total_time"] / max(1, self.perf_stats["reflection_calls"])

            # 计算总运行时间
            total_runtime = time.time() - self.perf_stats["start_time"]

            # 更新完整统计信息
            full_stats = {
                **self.perf_stats,
                "aask_avg_time": aask_avg_time,
                "reflection_avg_time": reflection_avg_time,
                "total_runtime": total_runtime,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 保存到文件
            if iteration is not None:
                await self.save_json_async(self.performance_dir / f"perf_stats_iter_{iteration}.json", full_stats)
                logger.debug(f"已保存性能统计数据，迭代次数: {iteration}")

            return full_stats
        except Exception as e:
            logger.error(f"更新性能统计信息出错: {e}")
            return {}

    @async_error_handler
    async def save_agent_state(self, agent, output_dir, iteration, additional_data=None):
        """保存OSAgent的完整状态

        Args:
            agent: OSAgent实例
            output_dir: 输出目录
            iteration: 当前迭代次数
            additional_data: 额外的数据字典，用于合并到状态中

        Returns:
            bool: 是否成功保存
        """
        try:
            # 手动提取rc的属性
            rc_data = self.extract_rc_data(agent.rc)

            # 基本信息
            full_state = {
                "rc": rc_data,
                "platform": getattr(agent, "platform", "Unknown"),
                "use_som": getattr(agent, "use_som", False),
                "use_ocr": getattr(agent, "use_ocr", False),
                "use_icon_detect": getattr(agent, "use_icon_detect", False),
                "use_memory": getattr(agent, "use_memory", False),
                "use_reflection": getattr(agent, "use_reflection", False),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 合并额外数据
            if additional_data:
                full_state.update(additional_data)

            # 保存完整状态到文件
            await self.save_json_async(output_dir / f"full_state_iter_{iteration}.json", full_state)
            return True
        except Exception as e:
            logger.error(f"保存状态出错: {e}")
            self.perf_stats["errors"] += 1
            return False

    def hijack(self, agent):
        """应用劫持到OSAgent实例

        Args:
            agent: OSAgent实例

        Returns:
            OSAgent: 劫持后的OSAgent实例
        """
        if not isinstance(agent, OSAgent):
            raise TypeError("只能劫持OSAgent类型的实例")

        # 保存原始aask方法
        self.original_aask = agent.llm.aask

        # 保存原始reflection方法
        self.original_reflection = agent._reflection

        # 保存原始reset_state方法
        self.original_reset_state = agent._reset_state

        # 创建新的aask方法
        @wraps(self.original_aask)
        async def patched_aask(prompt, system_msgs=None, images=None, stream=False):
            # 获取调用者信息
            caller_frame = sys._getframe(1)
            caller_name = caller_frame.f_code.co_name
            iteration = 0

            # 只记录_think方法中的调用
            if caller_name == "_think":
                # 确保能获取到迭代次数
                try:
                    iteration = agent.rc.iter
                except Exception as e:
                    logger.warning(f"获取迭代次数出错: {e}")
                    iteration = 0

                # 记录当前rc状态及额外信息
                if self.save_full_state:
                    try:
                        # 收集额外数据
                        additional_data = {
                            "instruction": getattr(agent, "instruction", ""),
                            "width": getattr(agent, "width", 0),
                            "height": getattr(agent, "height", 0),
                            "add_info": getattr(agent, "add_info", ""),
                            "location_info": getattr(agent, "location_info", ""),
                            "screenshot_file": getattr(agent, "screenshot_file", ""),
                            "screenshot_som_file": getattr(agent, "screenshot_som_file", ""),
                            "last_screenshot_file": getattr(agent, "last_screenshot_file", ""),
                            "last_screenshot_som_file": getattr(agent, "last_screenshot_som_file", ""),
                        }

                        # 调用保存函数
                        await self.save_agent_state(agent, self.think_states_dir, iteration, additional_data)
                    except Exception as e:
                        logger.error(f"保存状态出错: {e}")
                        self.perf_stats["errors"] += 1
                else:
                    logger.debug("跳过保存完整状态（save_full_state=False）")

                # 保存提示词到文件
                try:
                    prompt_data = {"prompt": prompt, "system_msgs": system_msgs, "has_images": images is not None}
                    await self.save_json_async(self.think_prompts_dir / f"prompt_iter_{iteration}.json", prompt_data)
                except Exception as e:
                    logger.error(f"保存提示词出错: {e}")
                    self.perf_stats["errors"] += 1

                # 保存图片
                if self.save_images and images:
                    for i, img_data in enumerate(images):
                        try:
                            # 从base64提取图片数据并保存
                            img_dir = self.think_images_dir / f"iter_{iteration}"
                            img_dir.mkdir(exist_ok=True)
                            img_path = img_dir / "image.jpg"
                            img_som_path = img_dir / "image_som.jpg"

                            # 这里保存图片的代码需要根据images的实际格式调整
                            # 如果images已经是文件路径，则直接复制
                            if hasattr(agent, "screenshot_file") and i == 0:
                                await self.save_image_async(agent.screenshot_file, img_path)
                            elif hasattr(agent, "screenshot_som_file") and i == 1:
                                await self.save_image_async(agent.screenshot_som_file, img_som_path)
                        except Exception as e:
                            logger.error(f"保存图片出错: {e}")
                            self.perf_stats["errors"] += 1
                elif not self.save_images and images:
                    logger.debug("跳过保存图片（save_images=False）")

            # 处理初始任务列表生成的输入
            elif caller_name == "_generate_initial_task_list":
                # 保存提示词到文件
                try:
                    prompt_data = {
                        "prompt": prompt,
                        "system_msgs": system_msgs[0] if isinstance(system_msgs, list) and system_msgs else system_msgs,
                        "has_images": images is not None,
                    }
                    await self.save_json_async(self.task_prompts_dir / f"prompt_iter_{iteration}.json", prompt_data)
                except Exception as e:
                    logger.error(f"保存提示词出错: {e}")
                    self.perf_stats["errors"] += 1

                # 保存图片
                if self.save_images and images:
                    for i, img_data in enumerate(images):
                        try:
                            # 从base64提取图片数据并保存
                            img_dir = self.task_images_dir / f"iter_{iteration}"
                            img_dir.mkdir(exist_ok=True)
                            img_path = img_dir / "image.jpg"
                            img_som_path = img_dir / "image_som.jpg"

                            # 这里保存图片的代码需要根据images的实际格式调整
                            # 如果images已经是文件路径，则直接复制
                            if hasattr(agent, "screenshot_file") and i == 0:
                                await self.save_image_async(agent.screenshot_file, img_path)
                            elif hasattr(agent, "screenshot_som_file") and i == 1:
                                await self.save_image_async(agent.screenshot_som_file, img_som_path)
                        except Exception as e:
                            logger.error(f"保存图片出错: {e}")
                            self.perf_stats["errors"] += 1
                elif not self.save_images and images:
                    logger.debug("跳过保存图片（save_images=False）")

            # 调用原始方法
            start_time = time.time()
            try:
                output = await self.original_aask(prompt, system_msgs=system_msgs, images=images, stream=stream)
                self.perf_stats["aask_calls"] += 1
            except Exception as e:
                self.perf_stats["errors"] += 1
                logger.error(f"调用原始aask方法出错: {e}")
                # 重新抛出异常，确保错误传播
                raise
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                self.perf_stats["aask_total_time"] += execution_time
                self.perf_stats["aask_max_time"] = max(self.perf_stats["aask_max_time"], execution_time)

            # 处理_think中的调用
            if caller_name == "_think":
                try:
                    iteration = agent.rc.iter

                    # 保存原始输出到文件
                    await self.save_text_async(self.think_outputs_dir / f"llm_raw_output_iter_{iteration}.txt", output)

                    # 保存处理后的输出和元数据
                    meta_data = {
                        "execution_time": execution_time,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "parsed_output": {
                            "thought": getattr(agent.rc, "thought", ""),
                            "action": getattr(agent.rc, "action", ""),
                            "summary": getattr(agent.rc, "summary", ""),
                            "task_list": getattr(agent.rc, "task_list", ""),
                        },
                    }
                    await self.save_json_async(self.think_outputs_dir / f"output_meta_iter_{iteration}.json", meta_data)

                    # 如果启用性能统计并达到指定间隔，则保存性能统计
                    if self.perf_stats_interval > 0 and iteration % self.perf_stats_interval == 0:
                        await self.update_performance_stats(iteration)
                except Exception as e:
                    logger.error(f"保存输出出错: {e}")
                    self.perf_stats["errors"] += 1

            # 处理初始任务列表生成的输出
            elif caller_name == "_generate_initial_task_list":
                try:
                    # 保存原始输出到文件
                    await self.save_text_async(self.task_outputs_dir / f"llm_raw_output_iter_{iteration}.txt", output)

                    # 保存处理后的输出和元数据
                    meta_data = {
                        "execution_time": execution_time,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "parsed_output": {"task_list": getattr(agent.rc, "task_list", "")},
                    }
                    await self.save_json_async(self.task_outputs_dir / f"output_meta_iter_{iteration}.json", meta_data)

                    # 如果启用性能统计并达到指定间隔，则保存性能统计
                    if self.perf_stats_interval > 0:
                        await self.update_performance_stats(iteration)
                except Exception as e:
                    logger.error(f"保存任务列表输出出错: {e}")
                    self.perf_stats["errors"] += 1

            return output

        # 创建新的reflection方法
        @wraps(self.original_reflection)
        async def patched_reflection(
            instruction,
            last_perception_infos,
            perception_infos,
            width,
            height,
            summary,
            action,
            add_info,
            last_screenshot_file,
            screenshot_file,
        ):
            # 保存输入参数
            try:
                iteration = agent.rc.iter

                # 手动提取rc的属性
                if self.save_full_state:
                    try:
                        # 收集额外数据
                        additional_data = {
                            "instruction": instruction,
                            "width": width,
                            "height": height,
                            "summary": summary,
                            "action": action,
                            "add_info": add_info,
                            "last_perception_infos_count": len(last_perception_infos) if last_perception_infos else 0,
                            "perception_infos_count": len(perception_infos) if perception_infos else 0,
                            "last_screenshot_file": last_screenshot_file,
                            "screenshot_file": screenshot_file,
                        }

                        # 调用保存函数
                        await self.save_agent_state(agent, self.reflection_states_dir, iteration, additional_data)
                    except Exception as e:
                        logger.error(f"保存反思完整状态出错: {e}")
                        self.perf_stats["errors"] += 1

                # 根据配置决定是否保存图片
                if self.save_images:
                    # 保存截图
                    img_dir = self.reflection_images_dir / f"iter_{iteration}"
                    img_dir.mkdir(exist_ok=True)

                    if last_screenshot_file and Path(last_screenshot_file).exists():
                        await self.save_image_async(last_screenshot_file, img_dir / "last_screenshot.jpg")

                    if screenshot_file and Path(screenshot_file).exists():
                        await self.save_image_async(screenshot_file, img_dir / "screenshot.jpg")
                else:
                    logger.debug("跳过保存反思截图（save_images=False）")

                # 构建和保存反思提示词
                if hasattr(agent.reflection_action, "get_reflection_prompt"):
                    # 使用reflection_action的方法构建提示词
                    prompt = agent.reflection_action.get_reflection_prompt(
                        instruction, last_perception_infos, perception_infos, width, height, summary, action, add_info
                    )

                    # 保存提示词和系统消息为JSON格式，与aask格式对齐
                    system_msg = f"You are a helpful AI {'mobile phone' if agent.platform=='Android' else 'PC'} operating assistant."
                    prompt_data = {"prompt": prompt, "system_msgs": system_msg, "has_images": True}  # 反思总是包含图片
                    await self.save_json_async(
                        self.reflection_prompts_dir / f"reflection_prompt_iter_{iteration}.json", prompt_data
                    )

                else:
                    # 如果无法找到get_reflection_prompt方法，记录错误
                    logger.warning("无法找到reflection_action.get_reflection_prompt方法，无法保存反思提示词")

            except Exception as e:
                logger.error(f"保存反思输入数据出错: {e}")
                self.perf_stats["errors"] += 1

            # 调用原始方法
            start_time = time.time()
            try:
                reflect, reflection_thought = await self.original_reflection(
                    instruction,
                    last_perception_infos,
                    perception_infos,
                    width,
                    height,
                    summary,
                    action,
                    add_info,
                    last_screenshot_file,
                    screenshot_file,
                )
                self.perf_stats["reflection_calls"] += 1
            except Exception as e:
                self.perf_stats["errors"] += 1
                logger.error(f"调用原始reflection方法出错: {e}")
                # 重新抛出异常，确保错误传播
                raise
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                self.perf_stats["reflection_total_time"] += execution_time
                self.perf_stats["reflection_max_time"] = max(self.perf_stats["reflection_max_time"], execution_time)

            # 保存输出结果
            try:
                iteration = agent.rc.iter
                output_data = {
                    "reflect": reflect,
                    "reflection_thought": reflection_thought,
                    "execution_time": execution_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                await self.save_json_async(
                    self.reflection_outputs_dir / f"output_meta_iter_{iteration}.json", output_data
                )

                # 保存反思完整输出
                # 根据reflection_thought和reflect重建原始LLM输出
                reconstructed_output = f"### Thought ###\n{reflection_thought}\n\n### Answer ###\n{reflect}"
                await self.save_text_async(
                    self.reflection_outputs_dir / f"llm_raw_output_iter_{iteration}.txt", reconstructed_output
                )

                # 如果启用性能统计并达到指定间隔，则保存性能统计
                if self.perf_stats_interval > 0 and iteration % self.perf_stats_interval == 0:
                    await self.update_performance_stats(iteration)

            except Exception as e:
                logger.error(f"保存反思输出数据出错: {e}")
                self.perf_stats["errors"] += 1

            return reflect, reflection_thought

        # 创建新的reset_state方法
        @wraps(self.original_reset_state)
        def patched_reset_state(instruction=None):
            # 先调用原始的reset_state方法
            result = self.original_reset_state(instruction)

            # 重置劫持器的会话目录
            logger.info("检测到OSAgent状态重置，创建新的日志会话目录")

            # 保存当前性能统计
            try:
                asyncio.create_task(self.update_performance_stats())
            except Exception as e:
                logger.error(f"保存性能统计出错: {e}")

            # 创建新会话目录
            self._setup_session_dirs()

            return result

        # 应用猴子补丁
        agent.llm.aask = patched_aask
        agent._reflection = patched_reflection
        agent._reset_state = patched_reset_state
        logger.info("已成功劫持OSAgent的llm.aask、_reflection和_reset_state方法")

        return agent

    async def cleanup(self):
        """最终清理工作，保存完整性能统计"""
        logger.info("执行最终清理工作")
        stats = await self.update_performance_stats()

        # 保存最终性能报告
        await self.save_json_async(self.performance_dir / "final_performance_report.json", stats)

        # 生成可读的摘要
        summary = f"""性能统计摘要:
- 总运行时间: {stats.get('total_runtime', 0):.2f}秒
- aask调用次数: {stats.get('aask_calls', 0)}
- reflection调用次数: {stats.get('reflection_calls', 0)}
- aask平均耗时: {stats.get('aask_avg_time', 0):.2f}秒
- reflection平均耗时: {stats.get('reflection_avg_time', 0):.2f}秒
- aask最长耗时: {stats.get('aask_max_time', 0):.2f}秒
- reflection最长耗时: {stats.get('reflection_max_time', 0):.2f}秒
- 错误次数: {stats.get('errors', 0)}
- 时间戳: {stats.get('timestamp', '')}
"""
        await self.save_text_async(self.performance_dir / "performance_summary.txt", summary)
        logger.info("性能统计已保存，劫持器清理完成")


# 使用样例
async def run():
    """运行示例

    此函数展示了如何使用OSAgentHijacker进行完整的OSAgent劫持和执行过程。
    包括创建劫持器、应用劫持、执行任务和清理资源。

    Returns:
        任务执行的响应结果
    """
    # 创建劫持器
    hijacker = OSAgentHijacker(
        log_dir="osagent_logs",  # 日志保存目录
        save_full_state=True,  # 保存完整状态
        save_images=True,  # 保存图片
        log_level="INFO",  # 日志级别
        perf_stats_interval=1,  # 每次迭代都保存性能统计
    )

    # 创建OSAgent实例
    agent = OSAgent(
        platform="Windows",  # 平台类型，可以是"Windows"、"Mac"或"Android"
        max_iters=10,  # 最大迭代次数
        use_ocr=False,  # 是否使用OCR
        use_icon_detect=False,  # 是否使用图标检测
        use_memory=False,  # 是否使用记忆
        use_reflection=True,  # 是否使用反思
        use_chrome_debugger=False,  # 是否使用chrome调试器
        extend_xml_infos=True,  # 是否扩展xml信息
    )

    # 应用劫持
    agent = hijacker.hijack(agent)  # 返回劫持后的agent，确保链式调用

    try:
        # 运行任务
        instruction = "打开百度并搜索python"
        logger.info(f"开始执行任务: {instruction}")

        # 执行任务
        response = await agent.run(instruction)

        logger.info(f"任务执行完成，响应: {response.content}")
    finally:
        # 确保清理操作得到执行，即使出现异常
        await hijacker.cleanup()

    logger.info(f"所有日志已保存到: {hijacker.session_dir}")

    return response


# 可以直接运行的入口点
if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
    except Exception as e:
        logger.error(f"运行出错: {e}")
        logger.exception("详细错误信息:")
