#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/06
@Author  : tanghaoming
@File    : server.py
@Desc    : 任务管理服务器模块

该模块提供了一个基于FastAPI的任务管理系统，支持以下功能：
- 提交并管理不同类型的测试任务（URL、Python应用、Python Web应用）
- 异步任务处理
- Conda环境管理
- 进程管理
- 任务状态跟踪

主要类:
- TaskManager: 核心任务管理器
- OSAgent: 测试代理
- TaskType: 任务类型枚举
- TaskStatus: 任务状态枚举
"""

import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
import zipfile
from datetime import datetime
from typing import Dict, Optional

import psutil
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from appeval.roles.appeval import AppEvalRole


class MockAppEvalRole:
    def __init__(self, **kwargs):
        pass

    async def run(self, url=None, user_requirement=None, case_name=None, task_id=None):
        """模拟OSAgent的运行过程"""
        # 等待几秒钟模拟处理过程
        for i in range(5):
            logger.info(f"模拟OSAgent的运行过程: {i}")
            await asyncio.sleep(2)

        if url:
            # 模拟URL类型或Web应用的结果
            return {
                "success": True,
                "message": f"Tested {url} successfully",
                "details": {"case_name": case_name, "task_id": task_id, "requirement": user_requirement},
            }
        else:
            # 模拟Python应用的结果
            return {
                "success": True,
                "message": "Python application test completed",
                "details": {"case_name": case_name, "task_id": task_id, "requirement": user_requirement},
            }


class TaskType:
    """任务类型常量类"""

    URL = "url"  # URL类型任务
    PYTHON_APP = "python_app"  # 普通Python应用
    PYTHON_WEB = "python_web"  # Python Web应用
    STREAMLIT = "streamlit"  # Streamlit应用


class TaskStatus:
    """任务状态常量类"""

    PENDING = "pending"  # 等待处理
    RUNNING = "running"  # 正在运行
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败


class TaskManager:
    """任务管理器类，负责处理任务的完整生命周期"""

    def __init__(self):
        """初始化任务管理器"""
        self.tasks: Dict[str, dict] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.is_worker_running: bool = False
        self.app = FastAPI(title="Task Manager API", description="管理和执行测试任务的API服务", version="1.0.0")
        self.appeval = AppEvalRole(
            data_path="data/temp.json",
            planner_model="claude-3-5-sonnet-v2",
            os_type="Windows",
            use_ocr=True,
            quad_split_ocr=True,
            use_memory=True,
            use_reflection=True,
            use_chrome_debugger=True,
            enable_result_check=False,
            extend_xml_infos=True,
        )
        self.setup_routes()

    def setup_routes(self):
        """设置API路由"""
        self.app.post("/submit_task")(self.submit_task)
        self.app.get("/task_status/{task_id}")(self.get_task_status)
        self.app.get("/task_result/{task_id}")(self.get_task_result)

    async def create_conda_env(self, env_name: str, requirements_path: str) -> bool:
        """
        创建conda环境并安装依赖

        Args:
            env_name: 环境名称
            requirements_path: requirements.txt文件路径

        Returns:
            bool: 环境创建是否成功
        """
        try:
            conda_path = self._get_conda_path()
            if not conda_path:
                raise Exception("无法找到conda可执行文件")

            # 创建环境
            logger.info(f"开始创建conda环境: {env_name}")
            if not await self._execute_conda_create(conda_path, env_name):
                return False

            # 安装依赖
            logger.info(f"开始安装依赖: {requirements_path}")
            if not await self._install_requirements(conda_path, env_name, requirements_path):
                return False

            return True

        except Exception as e:
            logger.error(f"创建conda环境失败: {str(e)}")
            logger.exception("详细错误信息")
            return False

    def _get_conda_path(self) -> Optional[str]:
        """
        获取conda可执行文件路径

        Returns:
            str: conda可执行文件路径，如果未找到则返回None
        """
        if os.name == "nt":
            possible_paths = [
                os.environ.get("CONDA_EXE", ""),
                os.path.join(os.environ.get("CONDA_PREFIX", ""), "Scripts", "conda.exe"),
                os.path.join(os.environ.get("USERPROFILE", ""), "Anaconda3", "Scripts", "conda.exe"),
                os.path.join(os.environ.get("USERPROFILE", ""), "miniconda3", "Scripts", "conda.exe"),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    return path
            return "conda.exe"
        return "conda"

    async def run_background_service(self, env_name: str, work_dir: str, start_path: str, task_type: str) -> int:
        """在后台运行服务，返回进程ID"""
        try:
            if os.name == "nt":  # Windows系统
                # 获取conda安装路径
                conda_path = self._get_conda_path()
                if not conda_path:
                    raise Exception("Cannot find conda executable")

                # 根据任务类型选择启动命令
                if task_type == TaskType.STREAMLIT:
                    cmd = f'"{conda_path}" run -n {env_name} streamlit run {start_path}'
                else:
                    cmd = f'"{conda_path}" run -n {env_name} python {start_path}'

                with open("service.log", "w") as log_file:
                    process = subprocess.Popen(
                        cmd,
                        cwd=work_dir,
                        stdout=log_file,
                        stderr=log_file,
                        shell=True,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                return process.pid
            else:  # Linux/Unix系统
                # 根据任务类型选择启动命令
                if task_type == TaskType.STREAMLIT:
                    cmd = f"nohup conda run -n {env_name} streamlit run {start_path} > service.log 2>&1 & echo $!"
                else:
                    cmd = f"nohup conda run -n {env_name} python {start_path} > service.log 2>&1 & echo $!"

                process = await asyncio.create_subprocess_shell(
                    cmd, cwd=work_dir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                pid = int(stdout.decode().strip())
                return pid
        except Exception as e:
            logger.error(f"Error running background service: {str(e)}")
            return -1

    async def kill_process(self, pid: int) -> bool:
        """终止指定的进程"""
        try:
            if os.name == "nt":  # Windows系统
                # 使用psutil来确保进程及其子进程都被终止
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            else:  # Linux/Unix系统
                cmd = f"kill {pid}"
                process = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
            logger.info(f"Process {pid} killed")
            return True
        except Exception as e:
            logger.error(f"Error killing process: {str(e)}")
            return False

    async def process_tasks(self):
        """后台任务处理器"""
        while True:
            task_id = await self.task_queue.get()
            task = self.tasks[task_id]
            work_dir = f"work_dir_{task_id}"

            try:
                self.tasks[task_id]["status"] = TaskStatus.RUNNING

                if task["type"] in [TaskType.PYTHON_APP, TaskType.PYTHON_WEB, TaskType.STREAMLIT]:
                    # 解压文件
                    with zipfile.ZipFile(task["zip_path"], "r") as zip_ref:
                        zip_ref.extractall(work_dir)

                    # 创建conda环境
                    env_name = f"env_{task_id}"
                    requirements_path = os.path.join(work_dir, "requirements.txt").replace("\\", "/")

                    if not await self.create_conda_env(env_name, requirements_path):
                        raise Exception("Failed to create conda environment")

                    # 在后台运行服务
                    pid = await self.run_background_service(env_name, work_dir, task["start_path"], task["type"])
                    if pid == -1:
                        raise Exception("Failed to start background service")

                    # 等待服务启动
                    await asyncio.sleep(5)  # 给予足够的启动时间

                    # 创建并运行OSAgent
                    service_url = None
                    if task["type"] == TaskType.PYTHON_WEB:
                        # 从任务参数中获取端口号，默认为8000
                        port = task["params"].get("port", 8000)
                        service_url = f"http://localhost:{port}"
                    elif task["type"] == TaskType.STREAMLIT:
                        service_url = "http://localhost:8501"  # Streamlit默认端口

                    result = await self.appeval.run(
                        url=service_url,
                        user_requirement=task["user_requirement"],
                        case_name=task["case_name"],
                    )
                    result = json.loads(result.content)

                    # 关闭后台服务
                    await self.kill_process(pid)

                    time.sleep(4)

                    # 清理conda环境
                    process = await asyncio.create_subprocess_shell(
                        f"conda env remove -n {env_name} -y",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    # 等待进程完成
                    await process.communicate()
                    logger.info(f"Conda environment {env_name} removed")

                    self.tasks[task_id].update(
                        {"status": TaskStatus.COMPLETED, "result": result, "end_time": datetime.now().isoformat()}
                    )

                elif task["type"] == TaskType.URL:
                    # 创建并运行OSAgent
                    result = await self.appeval.run(
                        url=task["url"],
                        user_requirement=task["user_requirement"],
                        case_name=task["case_name"],
                    )
                    result = json.loads(result.content)

                    self.tasks[task_id].update(
                        {"status": TaskStatus.COMPLETED, "result": result, "end_time": datetime.now().isoformat()}
                    )

            except Exception as e:
                self.tasks[task_id].update(
                    {"status": TaskStatus.FAILED, "error": str(e), "end_time": datetime.now().isoformat()}
                )

            finally:
                try:
                    # 清理临时文件
                    if task["type"] in [TaskType.PYTHON_APP, TaskType.PYTHON_WEB]:
                        if os.path.exists(task["zip_path"]):
                            os.remove(task["zip_path"])
                        if os.path.exists(work_dir):
                            shutil.rmtree(work_dir)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary files: {str(e)}")

                self.task_queue.task_done()

    async def submit_task(self, file: Optional[UploadFile] = None, params: str = Form(...)) -> Dict[str, str]:
        """
        提交新任务

        Args:
            file: 上传的ZIP文件（Python应用所需）
            params: JSON格式的任务参数

        Returns:
            包含task_id的字典

        Raises:
            HTTPException: 当参数无效或缺失时
        """
        try:
            params_dict = json.loads(params)
            self._validate_task_params(params_dict, file)

            task_id = str(uuid.uuid4())
            task_info = self._create_task_info(task_id, params_dict, file)

            self.tasks[task_id] = task_info
            await self.task_queue.put(task_id)

            self._ensure_worker_running()

            return {"task_id": task_id}

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="无效的JSON参数")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    def _validate_task_params(self, params: dict, file: Optional[UploadFile]) -> None:
        """验证任务参数的有效性"""
        if "type" not in params:
            raise ValueError("缺少任务类型参数")

        task_type = params["type"]
        if task_type not in [TaskType.URL, TaskType.PYTHON_APP, TaskType.PYTHON_WEB, TaskType.STREAMLIT]:
            raise ValueError("无效的任务类型")

        if task_type in [TaskType.PYTHON_APP, TaskType.PYTHON_WEB, TaskType.STREAMLIT] and not file:
            raise ValueError("Python应用需要上传ZIP文件")

        if task_type == TaskType.URL and "url" not in params:
            raise ValueError("URL类型任务需要提供url参数")

    def _create_task_info(self, task_id: str, params: dict, file: Optional[UploadFile]) -> dict:
        """创建任务信息字典"""
        task_info = {
            "id": task_id,
            "type": params["type"],
            "case_name": params.get("case_name"),
            "user_requirement": params.get("user_requirement"),
            "status": TaskStatus.PENDING,
            "create_time": datetime.now().isoformat(),
        }

        if task_info["type"] in [TaskType.PYTHON_APP, TaskType.PYTHON_WEB]:
            if not file:
                raise ValueError("Zip file required for Python application")

            zip_path = f"upload_{task_id}.zip"
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            task_info.update({"zip_path": zip_path, "start_path": params.get("start_path", "main.py")})
        else:  # URL type
            if "url" not in params:
                raise ValueError("URL required for url type")
            task_info["url"] = params["url"]

        return task_info

    def _ensure_worker_running(self):
        """确保后台任务处理器正在运行"""
        if not self.is_worker_running:
            self.is_worker_running = True
            asyncio.create_task(self.process_tasks())

    async def get_task_status(self, task_id: str):
        """查询任务状态接口"""
        task = self.tasks.get(task_id)
        if not task:
            return JSONResponse(status_code=404, content={"error": "Task not found"})

        return {
            "task_id": task_id,
            "status": task["status"],
            "create_time": task["create_time"],
            "end_time": task.get("end_time"),
        }

    async def get_task_result(self, task_id: str):
        """查询任务结果接口"""
        task = self.tasks.get(task_id)
        if not task:
            return JSONResponse(status_code=404, content={"error": "Task not found"})

        if task["status"] not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return JSONResponse(status_code=400, content={"error": "Task is not completed yet"})

        return {"task_id": task_id, "status": task["status"], "result": task.get("result"), "error": task.get("error")}

    async def _execute_conda_create(self, conda_path: str, env_name: str) -> bool:
        """
        执行conda create命令创建新环境

        Args:
            conda_path: conda可执行文件路径
            env_name: 要创建的环境名称

        Returns:
            bool: 环境创建是否成功
        """
        try:
            process = await asyncio.create_subprocess_exec(
                conda_path,
                "create",
                "-n",
                env_name,
                "python=3.10",
                "-y",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                stderr_text = self._decode_output(stderr)
                logger.error(f"创建conda环境失败: {stderr_text}")
                return False

            logger.info("conda环境创建成功")
            return True

        except Exception as e:
            logger.error(f"执行conda create命令时发生错误: {str(e)}")
            return False

    async def _install_requirements(self, conda_path: str, env_name: str, requirements_path: str) -> bool:
        """
        在指定的conda环境中安装依赖

        Args:
            conda_path: conda可执行文件路径
            env_name: 环境名称
            requirements_path: requirements.txt文件路径

        Returns:
            bool: 依赖安装是否成功
        """
        try:
            process = await asyncio.create_subprocess_exec(
                conda_path,
                "run",
                "-n",
                env_name,
                "python",
                "-m",
                "pip",
                "install",
                "-r",
                requirements_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            # 检查安装结果
            if stderr:
                stderr_text = self._decode_output(stderr)
                logger.warning(f"安装依赖时出现警告: {stderr_text}")

            logger.info("依赖安装完成")
            return True

        except Exception as e:
            logger.error(f"安装依赖时发生错误: {str(e)}")
            return False

    def _decode_output(self, output: bytes) -> str:
        """
        解码命令输出，处理不同编码

        Args:
            output: 命令输出的字节串

        Returns:
            str: 解码后的文本
        """
        try:
            return output.decode("utf-8", errors="replace")
        except:
            try:
                return output.decode("gbk", errors="replace")
            except:
                return str(output)

    def run(self, host="0.0.0.0", port=8888):
        """启动服务器"""
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    task_manager = TaskManager()
    task_manager.run()
