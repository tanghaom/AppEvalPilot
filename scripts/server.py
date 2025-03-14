#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/06
@Author  : tanghaoming
@File    : server.py
@Desc    : Task Management Server Module

This module provides a FastAPI-based task management system with the following features:
- Submit and manage different types of test tasks (URL, Python app, Python Web app)
- Asynchronous task processing
- Conda environment management
- Process management
- Task status tracking

Main classes:
- TaskManager: Core task manager
- OSAgent: Test agent
- TaskType: Task type enumeration
- TaskStatus: Task status enumeration
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

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from appeval.roles.eval_runner import AppEvalRole
from appeval.utils.window_utils import kill_process


class MockAppEvalRole:
    def __init__(self, **kwargs):
        pass

    async def run(self, url=None, user_requirement=None, case_name=None, task_id=None):
        """Simulate the running process of OSAgent"""
        # Wait a few seconds to simulate the processing
        for i in range(5):
            logger.info(f"Simulating OSAgent running process: {i}")
            await asyncio.sleep(2)

        if url:
            # Simulate results for URL type or Web application
            return {
                "success": True,
                "message": f"Tested {url} successfully",
                "details": {"case_name": case_name, "task_id": task_id, "requirement": user_requirement},
            }
        else:
            # Simulate results for Python application
            return {
                "success": True,
                "message": "Python application test completed",
                "details": {"case_name": case_name, "task_id": task_id, "requirement": user_requirement},
            }


class TaskType:
    """Task type constants class"""

    URL = "url"  # URL type task
    PYTHON_APP = "python_app"  # Regular Python application
    PYTHON_WEB = "python_web"  # Python Web application
    STREAMLIT = "streamlit"  # Streamlit application


class TaskStatus:
    """Task status constants class"""

    PENDING = "pending"  # Waiting for processing
    RUNNING = "running"  # Currently running
    COMPLETED = "completed"  # Completed
    FAILED = "failed"  # Failed


class TaskManager:
    """Task manager class, responsible for handling the complete lifecycle of tasks"""

    def __init__(self):
        """Initialize the task manager"""
        self.tasks: Dict[str, dict] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.is_worker_running: bool = False
        self.app = FastAPI(
            title="Task Manager API", description="API service for managing and executing test tasks", version="1.0.0"
        )
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
        """Set up API routes"""
        self.app.post("/submit_task")(self.submit_task)
        self.app.get("/task_status/{task_id}")(self.get_task_status)
        self.app.get("/task_result/{task_id}")(self.get_task_result)

    async def create_conda_env(self, env_name: str, requirements_path: str) -> bool:
        """
        Create conda environment and install dependencies

        Args:
            env_name: Environment name
            requirements_path: Path to requirements.txt file

        Returns:
            bool: Whether the environment creation was successful
        """
        try:
            conda_path = self._get_conda_path()
            if not conda_path:
                raise Exception("Cannot find conda executable")

            # Create environment
            logger.info(f"Starting to create conda environment: {env_name}")
            if not await self._execute_conda_create(conda_path, env_name):
                return False

            # Install dependencies
            logger.info(f"Starting to install dependencies: {requirements_path}")
            if not await self._install_requirements(conda_path, env_name, requirements_path):
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to create conda environment: {str(e)}")
            logger.exception("Detailed error information")
            return False

    def _get_conda_path(self) -> Optional[str]:
        """
        Get the path to the conda executable

        Returns:
            str: Path to conda executable, or None if not found
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
        """Run service in the background, return process ID

        Args:
            env_name: Environment name
            work_dir: Working directory
            start_path: Start path
            task_type: Task type

        Returns:
            int: Process ID (PID) of the started service
        """
        try:
            if os.name == "nt":  # Windows system
                # Get conda installation path
                conda_path = self._get_conda_path()
                if not conda_path:
                    raise Exception("Cannot find conda executable")

                # Choose startup command based on task type
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
            else:  # Linux/Unix system
                # Choose startup command based on task type
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

    async def process_tasks(self):
        """Background task processor"""
        while True:
            task_id = await self.task_queue.get()
            task = self.tasks[task_id]
            work_dir = f"work_dir_{task_id}"

            try:
                self.tasks[task_id]["status"] = TaskStatus.RUNNING

                if task["type"] in [TaskType.PYTHON_APP, TaskType.PYTHON_WEB, TaskType.STREAMLIT]:
                    # Extract files
                    with zipfile.ZipFile(task["zip_path"], "r") as zip_ref:
                        zip_ref.extractall(work_dir)

                    # Create conda environment
                    env_name = f"env_{task_id}"
                    requirements_path = os.path.join(work_dir, "requirements.txt").replace("\\", "/")

                    if not await self.create_conda_env(env_name, requirements_path):
                        raise Exception("Failed to create conda environment")

                    # Run service in the background
                    pid = await self.run_background_service(env_name, work_dir, task["start_path"], task["type"])
                    if pid == -1:
                        raise Exception("Failed to start background service")

                    # Wait for service to start
                    await asyncio.sleep(5)  # Give enough time to start

                    # Create and run AppEval
                    service_url = None
                    if task["type"] == TaskType.PYTHON_WEB:
                        # Get port number from task parameters, default to 8000
                        port = task["params"].get("port", 8000)
                        service_url = f"http://localhost:{port}"
                    elif task["type"] == TaskType.STREAMLIT:
                        service_url = "http://localhost:8501"  # Streamlit default port

                    result = await self.appeval.run(
                        url=service_url,
                        user_requirement=task["user_requirement"],
                        case_name=task["case_name"],
                    )
                    result = json.loads(result.content)

                    # Shut down background service
                    await kill_process(pid)

                    time.sleep(4)

                    # Clean up conda environment
                    process = await asyncio.create_subprocess_shell(
                        f"conda env remove -n {env_name} -y",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    # Wait for process to complete
                    await process.communicate()
                    logger.info(f"Conda environment {env_name} removed")

                    self.tasks[task_id].update(
                        {"status": TaskStatus.COMPLETED, "result": result, "end_time": datetime.now().isoformat()}
                    )

                elif task["type"] == TaskType.URL:
                    # Create and run OSAgent
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
                    # Clean up temporary files
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
        Submit a new task

        Args:
            file: Uploaded ZIP file (required for Python applications)
            params: Task parameters in JSON format

        Returns:
            Dictionary containing task_id

        Raises:
            HTTPException: When parameters are invalid or missing
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
            raise HTTPException(status_code=400, detail="Invalid JSON parameters")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    def _validate_task_params(self, params: dict, file: Optional[UploadFile]) -> None:
        """Validate the task parameters

        Args:
            params: Task parameters
            file: Uploaded ZIP file (required for Python applications)

        Raises:
            ValueError: When parameters are invalid or missing
        """
        if "type" not in params:
            raise ValueError("Missing task type parameter")

        task_type = params["type"]
        if task_type not in [TaskType.URL, TaskType.PYTHON_APP, TaskType.PYTHON_WEB, TaskType.STREAMLIT]:
            raise ValueError("Invalid task type")

        if task_type in [TaskType.PYTHON_APP, TaskType.PYTHON_WEB, TaskType.STREAMLIT] and not file:
            raise ValueError("Python application requires a ZIP file upload")

        if task_type == TaskType.URL and "url" not in params:
            raise ValueError("URL type task requires a url parameter")

    def _create_task_info(self, task_id: str, params: dict, file: Optional[UploadFile]) -> dict:
        """Create task information dictionary

        Args:
            task_id: Task ID
            params: Task parameters
            file: Uploaded ZIP file (required for Python applications)

        Returns:
            Dictionary containing task information
        """
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
        """Ensure the background task processor is running"""
        if not self.is_worker_running:
            self.is_worker_running = True
            asyncio.create_task(self.process_tasks())

    async def get_task_status(self, task_id: str):
        """Task status query interface

        Args:
            task_id: Task ID

        Returns:
            Dictionary containing task status information
        """
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
        """Task result query interface

        Args:
            task_id: Task ID

        Returns:
            Dictionary containing task result information
        """
        task = self.tasks.get(task_id)
        if not task:
            return JSONResponse(status_code=404, content={"error": "Task not found"})

        if task["status"] not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return JSONResponse(status_code=400, content={"error": "Task is not completed yet"})

        return {"task_id": task_id, "status": task["status"], "result": task.get("result"), "error": task.get("error")}

    async def _execute_conda_create(self, conda_path: str, env_name: str) -> bool:
        """
        Execute conda create command to create a new environment

        Args:
            conda_path: Path to conda executable
            env_name: Name of the environment to create

        Returns:
            bool: Whether the environment creation was successful
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
                logger.error(f"Failed to create conda environment: {stderr_text}")
                return False

            logger.info("Conda environment created successfully")
            return True

        except Exception as e:
            logger.error(f"Error executing conda create command: {str(e)}")
            return False

    async def _install_requirements(self, conda_path: str, env_name: str, requirements_path: str) -> bool:
        """
        Install dependencies in the specified conda environment

        Args:
            conda_path: Path to conda executable
            env_name: Environment name
            requirements_path: Path to requirements.txt file

        Returns:
            bool: Whether the dependency installation was successful
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

            # Check installation results
            if stderr:
                stderr_text = self._decode_output(stderr)
                logger.warning(f"Warnings during dependency installation: {stderr_text}")

            logger.info("Dependencies installation completed")
            return True

        except Exception as e:
            logger.error(f"Error installing dependencies: {str(e)}")
            return False

    def _decode_output(self, output: bytes) -> str:
        """
        Decode command output, handling different encodings

        Args:
            output: Command output bytes

        Returns:
            str: Decoded text
        """
        try:
            return output.decode("utf-8", errors="replace")
        except:
            try:
                return output.decode("gbk", errors="replace")
            except:
                return str(output)

    def run(self, host="0.0.0.0", port=8888):
        """Start the server

        Args:
            host: Host address
            port: Port number
        """
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    task_manager = TaskManager()
    task_manager.run()
