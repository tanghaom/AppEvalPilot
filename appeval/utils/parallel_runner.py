#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/01/29
@File    : parallel_runner.py
@Desc    : Parallel test execution utilities for Linux environment using multiprocessing
"""
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional, Tuple

from metagpt.logs import logger


@dataclass
class WorkerConfig:
    """Configuration for a parallel worker"""
    worker_id: int
    display: int          # Xvfb display number (e.g., 99, 100, 101)
    debug_port: int       # Chrome remote debugging port
    user_data_dir: str    # Chrome user data directory
    
    @property
    def display_str(self) -> str:
        return f":{self.display}"


class XvfbManager:
    """Manage Xvfb virtual display instances"""
    
    def __init__(self, base_display: int = 99, screen_size: str = "1920x1080x24"):
        self.base_display = base_display
        self.screen_size = screen_size
        self.running_displays: Dict[int, subprocess.Popen] = {}
    
    def start_display(self, display_num: int) -> bool:
        """Start an Xvfb display (synchronous)"""
        if display_num in self.running_displays:
            logger.warning(f"Display :{display_num} already running")
            return True
        
        try:
            cmd = ["Xvfb", f":{display_num}", "-screen", "0", self.screen_size]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1.5)  # Wait for Xvfb to fully initialize
            
            if process.poll() is None:
                self.running_displays[display_num] = process
                logger.info(f"Started Xvfb on display :{display_num}")
                return True
            else:
                logger.error(f"Failed to start Xvfb on display :{display_num}")
                return False
        except Exception as e:
            logger.error(f"Error starting Xvfb: {e}")
            return False
    
    def stop_display(self, display_num: int) -> bool:
        """Stop an Xvfb display"""
        if display_num not in self.running_displays:
            return True
        
        try:
            process = self.running_displays[display_num]
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            del self.running_displays[display_num]
            logger.info(f"Stopped Xvfb on display :{display_num}")
            return True
        except Exception as e:
            logger.error(f"Error stopping Xvfb: {e}")
            return False
    
    def cleanup_all(self):
        """Stop all running Xvfb displays"""
        displays = list(self.running_displays.keys())
        for display_num in displays:
            self.stop_display(display_num)


def _worker_process(
    worker_id: int,
    display: int,
    debug_port: int,
    user_data_dir: str,
    task_args: Dict[str, Any],
    result_queue: Queue
):
    """
    Worker process function - runs in a separate process with its own DISPLAY.
    
    This ensures pyautogui connects to the correct Xvfb display.
    """
    try:
        # Set DISPLAY before importing anything that uses X11
        os.environ["DISPLAY"] = f":{display}"
        
        # Now import and run the test
        import asyncio
        from appeval.roles.eval_runner import AppEvalRole
        
        async def run_test():
            task_name = task_args.get("task_name", f"task_{worker_id}")
            test_cases = task_args.get("test_cases", {})
            start_func = task_args.get("start_func", "")
            log_dir = task_args.get("log_dir", f"work_dirs/worker_{worker_id}")
            
            # Create worker-specific log directory
            worker_log_dir = f"{log_dir}/worker_{worker_id}"
            os.makedirs(worker_log_dir, exist_ok=True)
            
            # Initialize AppEvalRole with worker-specific settings
            appeval = AppEvalRole(
                json_file=task_args.get("json_file"),
                use_ocr=task_args.get("use_ocr", False),
                quad_split_ocr=task_args.get("quad_split_ocr", False),
                use_memory=task_args.get("use_memory", False),
                use_reflection=task_args.get("use_reflection", True),
                use_chrome_debugger=task_args.get("use_chrome_debugger", False),
                extend_xml_infos=task_args.get("extend_xml_infos", True),
                max_iters=task_args.get("max_iters", 20),
                remote_debugging_port=debug_port,
                user_data_dir=user_data_dir,
            )
            
            result, executability = await appeval.run_api(
                task_name=f"{task_name}_worker_{worker_id}",
                test_cases=test_cases,
                start_func=start_func,
                log_dir=worker_log_dir
            )
            
            return result, executability
        
        result, executability = asyncio.run(run_test())
        result_queue.put((worker_id, result, executability, None))
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        result_queue.put((worker_id, {}, False, error_msg))


class ParallelTestRunner:
    """
    Run multiple test tasks in parallel with isolated environments.
    
    Each worker runs in a SEPARATE PROCESS with:
    - Unique Xvfb display (DISPLAY=:100, :101, :102, ...)
    - Unique Chrome debugging port (9300, 9301, 9302, ...)
    - Unique Chrome user data directory
    
    Using multiprocessing ensures pyautogui connects to the correct DISPLAY.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        base_display: int = 100,
        base_port: int = 9300,
        screen_size: str = "1920x1080x24"
    ):
        self.max_workers = max_workers
        self.base_display = base_display
        self.base_port = base_port
        self.xvfb_manager = XvfbManager(base_display, screen_size)
        self.temp_dirs: List[str] = []
    
    def _create_worker_config(self, worker_id: int) -> WorkerConfig:
        """Create configuration for a worker"""
        user_data_dir = tempfile.mkdtemp(prefix=f"chrome_worker_{worker_id}_")
        self.temp_dirs.append(user_data_dir)
        
        return WorkerConfig(
            worker_id=worker_id,
            display=self.base_display + worker_id,
            debug_port=self.base_port + worker_id,
            user_data_dir=user_data_dir
        )
    
    def run_parallel(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Tuple[int, Any, bool, Optional[str]]]:
        """
        Run multiple tasks in parallel using multiprocessing.
        
        Args:
            tasks: List of task argument dictionaries
            
        Returns:
            List of (worker_id, result, executability, error) tuples
        """
        # Limit concurrent workers
        num_workers = min(len(tasks), self.max_workers)
        results = []
        
        # Process tasks in batches
        for batch_start in range(0, len(tasks), num_workers):
            batch_end = min(batch_start + num_workers, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            
            logger.info(f"Starting batch {batch_start//num_workers + 1}: tasks {batch_start} to {batch_end-1}")
            
            # Start Xvfb displays for this batch
            worker_configs = []
            for i, _ in enumerate(batch_tasks):
                worker_id = batch_start + i
                config = self._create_worker_config(worker_id)
                worker_configs.append(config)
                
                if not self.xvfb_manager.start_display(config.display):
                    logger.error(f"Failed to start Xvfb for worker {worker_id}")
                    results.append((worker_id, {}, False, "Failed to start Xvfb"))
                    continue
            
            # Create result queue and processes
            result_queue = Queue()
            processes = []
            
            for config, task_args in zip(worker_configs, batch_tasks):
                logger.info(f"Worker {config.worker_id}: DISPLAY=:{config.display}, port={config.debug_port}")
                
                p = Process(
                    target=_worker_process,
                    args=(
                        config.worker_id,
                        config.display,
                        config.debug_port,
                        config.user_data_dir,
                        task_args,
                        result_queue
                    )
                )
                processes.append(p)
                p.start()
            
            # Wait for all processes to complete
            for p in processes:
                p.join()
            
            # Collect results
            while not result_queue.empty():
                worker_id, result, executability, error = result_queue.get()
                results.append((worker_id, result, executability, error))
            
            # Cleanup Xvfb displays for this batch
            for config in worker_configs:
                self.xvfb_manager.stop_display(config.display)
        
        self.cleanup()
        return results
    
    def cleanup(self):
        """Clean up all resources"""
        self.xvfb_manager.cleanup_all()
        
        # Clean up temp directories
        import shutil
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to remove temp dir {temp_dir}: {e}")
        self.temp_dirs.clear()


def run_parallel_tests(
    tasks: List[Dict[str, Any]],
    max_workers: int = 4,
    base_display: int = 100,
    base_port: int = 9300,
    **appeval_kwargs
) -> List[Tuple[str, dict, bool, Optional[str]]]:
    """
    High-level function to run multiple AppEval tests in parallel.
    
    Args:
        tasks: List of task configs, each containing:
            - task_name: str
            - test_cases: dict
            - start_func: str (URL)
            - log_dir: str
        max_workers: Maximum number of concurrent workers
        base_display: Starting Xvfb display number
        base_port: Starting Chrome debug port
        **appeval_kwargs: Additional arguments for AppEvalRole
        
    Returns:
        List of (task_name, result, executability, error) tuples
        
    Example:
        tasks = [
            {
                "task_name": "Test1",
                "test_cases": {...},
                "start_func": "https://example1.com",
                "log_dir": "work_dirs/test1"
            },
            {
                "task_name": "Test2", 
                "test_cases": {...},
                "start_func": "https://example2.com",
                "log_dir": "work_dirs/test2"
            }
        ]
        
        results = run_parallel_tests(tasks, max_workers=2)
    """
    runner = ParallelTestRunner(
        max_workers=max_workers,
        base_display=base_display,
        base_port=base_port
    )
    
    # Merge appeval_kwargs into each task
    task_args_list = []
    for task in tasks:
        task_args_list.append({
            **task,
            **appeval_kwargs
        })
    
    # Run in parallel (synchronous call using multiprocessing)
    raw_results = runner.run_parallel(task_args_list)
    
    # Format results
    formatted_results = []
    for worker_id, result, executability, error in raw_results:
        # Find the task name
        task_idx = worker_id
        if task_idx < len(tasks):
            task_name = tasks[task_idx].get("task_name", f"task_{task_idx}")
        else:
            task_name = f"task_{worker_id}"
        
        formatted_results.append((task_name, result, executability, error))
    
    return formatted_results
