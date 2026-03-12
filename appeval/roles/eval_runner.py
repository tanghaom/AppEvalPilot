#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/10/24
@File    : eval_runner.py
@Author  : tanghaoming
@Desc    : Refactored automated testing role for AppEval
"""
import asyncio
import copy
import datetime
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml as _yaml
from loguru import logger
from metagpt.config2 import Config
from metagpt.roles.role import Role, RoleContext
from metagpt.utils.common import read_json_file, write_json_file
from pydantic import ConfigDict, Field

from appeval.actions.case_generator import CaseGenerator, OperationType
from appeval.prompts.osagent import case_batch_check_system_prompt
from appeval.prompts.text_agent import text_agent_system_prompt
from appeval.judges.supervisor_judge import analyze_trajectory_async as supervisor_analyze_trajectory_async
from appeval.judges.supervisor_judge import select_plans_async as supervisor_select_plans_async
from appeval.roles.osagent import OSAgent
from appeval.roles.osagent import (
    CHECKPOINT_ACTION_TAIL_LEN,
    CHECKPOINT_SUMMARY_TAIL_LEN,
)
from appeval.utils.excel_json_converter import (
    convert_json_to_excel,
    list_to_json,
    make_json_single,
    mini_list_to_excel,
    mini_list_to_json,
    update_project_excel_iters,
)
from appeval.utils.window_utils import kill_process, kill_windows, start_windows

# Constants for sleep times
SLEEP_AFTER_START_WEB = 10
SLEEP_AFTER_START_APP = 20
SLEEP_AFTER_CLEANUP = 5
SLEEP_BEFORE_EXECUTE = 30
SLEEP_BETWEEN_RETRIES = 5


async def _wait_for_cdp(port: int, max_wait: float = 20.0, interval: float = 1.0) -> bool:
    """Poll Chrome CDP /json until it returns a valid list or timeout. Used when a11y_mode=cdp."""
    import requests
    url = f"http://127.0.0.1:{port}/json"
    deadline = time.perf_counter() + max_wait
    while time.perf_counter() < deadline:
        try:
            r = requests.get(url, timeout=2, proxies={"http": None, "https": None})
            if r.status_code == 200 and r.content:
                data = r.json()
                if isinstance(data, list):
                    logger.debug(f"CDP on port {port} ready with {len(data)} tab(s)")
                    return True
        except Exception as e:
            logger.debug(f"CDP port {port} not ready yet: {e}")
        await asyncio.sleep(interval)
    logger.warning(f"CDP on port {port} did not become ready within {max_wait}s")
    return False


def _save_resume_checkpoint(
    osagent,
    save_dir: str,
    chrome_profile_src: str = "",
    run_id: str = "",
    worker_id: str = "",
    remote_debugging_port: int = 0,
) -> str:
    """Serialize current osagent state into a resume_checkpoint.json file.

    The saved profile strategy is always ``base_to_work_copy``:  the live
    ``chrome_profile_src`` directory is copied to
    ``{save_dir}/chrome_profile_round1_base`` and stored as the base.  Round-2
    startup should copy that base into a fresh working directory before
    launching Chrome (so the original base is never mutated).

    Args:
        osagent:                  OSAgent instance after a completed run.
        save_dir:                 Directory where the checkpoint file is written.
        chrome_profile_src:       Path to the live user_data_dir for this run.
        run_id:                   Run identifier (for concurrent-worker diagnostics).
        worker_id:                Worker identifier.
        remote_debugging_port:    Chrome DevTools port (for diagnostics).

    Returns:
        Absolute path to the written checkpoint JSON, or "" on failure.
    """
    try:
        rc = osagent.rc
        controller = getattr(osagent, "controller", None)

        def _safe_list(val):
            try:
                return [str(x) for x in (val or [])]
            except Exception:
                return []

        # ── Page state ──────────────────────────────────────────────
        current_url = ""
        scroll = {"x": 0, "y": 0}
        viewport = {"w": 0, "h": 0, "dpr": 1}
        if controller:
            if hasattr(controller, "get_current_tab_url"):
                current_url = controller.get_current_tab_url()
            if hasattr(controller, "get_scroll"):
                scroll = controller.get_scroll()
            if hasattr(controller, "get_viewport"):
                viewport = controller.get_viewport()

        # ── Screenshot ──────────────────────────────────────────────
        last_iter = int(rc.iter)
        save_img = getattr(osagent, "save_img", "")
        last_screenshot_path = ""
        if save_img:
            candidate = Path(save_img) / f"origin_{last_iter}.jpg"
            if candidate.exists():
                last_screenshot_path = str(candidate)

        # ── History: decision payload only (action_tail + summary_tail), no thought/reflection/memory ───
        raw_history = _safe_list(rc.action_history)
        action_only = [a for a in raw_history if not a.startswith("Tell ")]
        result_history = []
        for i, a in enumerate(raw_history):
            if a.startswith("Tell "):
                result_history.append({"iter": i + 1, "action": a})
        action_tail = action_only[-CHECKPOINT_ACTION_TAIL_LEN:] if action_only else []
        summary_tail = _safe_list(rc.summary_history)[-CHECKPOINT_SUMMARY_TAIL_LEN:]

        # ── Chrome profile (base_to_work_copy strategy) ─────────────
        base_path = ""
        profile_copy_ok = False
        if chrome_profile_src and Path(chrome_profile_src).exists():
            base_path = str(Path(save_dir) / "chrome_profile_round1_base")

            def _ignore_singleton_and_lock(_dir: str, names: list) -> list:
                ignored = []
                for n in names:
                    if (
                        n in {"SingletonLock", "SingletonCookie", "SingletonSocket", "LOCK"}
                        or n.endswith(".tmp")
                        or n.endswith(".TMP")
                    ):
                        ignored.append(n)
                return ignored

            try:
                if Path(base_path).exists():
                    shutil.rmtree(base_path, ignore_errors=True)
                shutil.copytree(chrome_profile_src, base_path, ignore=_ignore_singleton_and_lock)
                profile_copy_ok = True
                logger.info(f"Chrome profile base saved: {base_path}")
            except Exception as copy_exc:
                base_path = ""
                logger.warning(f"Failed to copy Chrome profile: {copy_exc}")

        # ── Integrity metadata ───────────────────────────────────────
        path_checks = []
        if last_screenshot_path:
            path_checks.append({"path": last_screenshot_path, "must_exist": True, "exists": Path(last_screenshot_path).exists()})
        if base_path:
            path_checks.append({"path": base_path, "must_exist": True, "exists": Path(base_path).exists()})

        resume_degraded = not bool(current_url) or (bool(chrome_profile_src) and not profile_copy_ok)
        degrade_reason = ""
        if not current_url:
            degrade_reason += "resume_from_url is empty; "
        if chrome_profile_src and not profile_copy_ok:
            degrade_reason += "profile copy failed; "

        replay_core = {
            "mode": "resume",
            "run_id": run_id or "",
            "worker_id": str(worker_id) if worker_id else "",
            "remote_debugging_port": remote_debugging_port or 0,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "resume_from_url": current_url,
            "guard_policy": {
                "expected_url_mode": "set_to_resume_from_url",
                "expected_domain_mode": "restrict_to_resume_domain",
            },
            "viewport": viewport,
            "scroll": scroll,
            "restore_order": ["copy_profile", "start_chrome", "navigate", "wait_stable", "set_scroll", "confirm_screenshot"],
            "last_completed_iter": last_iter,
            "last_screenshot_path": last_screenshot_path,
            "resume_confirm_screenshot_path": "",
            "profile": {
                "strategy": "base_to_work_copy",
                "base_path": base_path,
                "work_path_round2": "",
            },
        }
        decision_payload = {
            "action_tail": action_tail,
            "summary_tail": summary_tail,
            "result_history": result_history,
        }
        checkpoint = {
            "replay_core": replay_core,
            "decision_payload": decision_payload,
            "integrity": {
                "required_fields": ["resume_from_url", "last_completed_iter", "profile.base_path"],
                "path_checks": path_checks,
                "resume_degraded": resume_degraded,
                "degrade_reason": degrade_reason.strip(),
            },
        }

        out_path = Path(save_dir) / "resume_checkpoint.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        logger.info(f"Resume checkpoint saved: {out_path} | degraded={resume_degraded}")
        return str(out_path)
    except Exception as exc:
        logger.warning(f"Failed to save resume checkpoint: {exc}")
        return ""


def _prune_checkpoints_keep_only_restart(checkpoint_dir_path: str, restart_from_iter: Optional[int]) -> None:
    """After case completes and LLM gives restart_recommendation: keep only step_{restart_from_iter}.json, delete other step_*.json."""
    if restart_from_iter is None:
        return
    step_dir = Path(checkpoint_dir_path) / "checkpoints"
    if not step_dir.is_dir():
        return
    restart_from_iter = int(restart_from_iter)
    # restart_from_iter == 0 means restart from scratch: no step checkpoint should be kept.
    if restart_from_iter == 0:
        removed = 0
        for f in step_dir.glob("step_*.json"):
            try:
                f.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to prune checkpoint {f}: {e}")
        index_path = step_dir / "checkpoint_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    idx = json.load(f)
                idx["steps"] = {}
                idx["latest_step"] = 0
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(idx, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to prune checkpoint index {index_path}: {e}")
        if removed:
            logger.info(f"Pruned {removed} checkpoint(s) for restart_from_iter=0 in {step_dir}")
        return
    keep_name = f"step_{int(restart_from_iter):03d}.json"
    keep_file = step_dir / keep_name
    if not keep_file.exists():
        logger.debug(f"Recommended step file {keep_name} not found in {step_dir}, skip pruning checkpoints")
        return
    removed = 0
    for f in step_dir.glob("step_*.json"):
        if f.name != keep_name:
            try:
                f.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to prune checkpoint {f}: {e}")
    # Keep checkpoint index consistent with remaining step file.
    index_path = step_dir / "checkpoint_index.json"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            steps = idx.get("steps", {}) if isinstance(idx, dict) else {}
            keep_key = str(int(restart_from_iter))
            kept_step = steps.get(keep_key)
            idx["steps"] = {keep_key: kept_step} if kept_step else {}
            idx["latest_step"] = int(restart_from_iter)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(idx, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to prune checkpoint index {index_path}: {e}")
    if removed:
        logger.info(f"Pruned {removed} checkpoint(s), kept only {keep_name} in {step_dir}")


def _build_trajectory_tail(checkpoint: dict, restart_iter: int, tail_steps: int = 6) -> str:
    """Extract and format the action/summary steps near the retry node.

    Includes up to tail_steps steps ending at the last recorded step.
    The restart node step is marked with an arrow so the judge can orient itself.

    Args:
        checkpoint: Loaded resume_checkpoint dict.
        restart_iter: 0-based step index recommended as the retry node.
        tail_steps: How many steps to include in the tail.
    Returns:
        Human-readable multi-line string, or "(no trajectory available)".
    """
    # Prefer decision_payload tails (what we actually persist in current checkpoints),
    # then fall back to replay_core/full-history legacy fields.
    payload = checkpoint.get("decision_payload") or {}
    rc = checkpoint.get("replay_core") or {}
    actions = payload.get("action_tail") or rc.get("action_history") or []
    summaries = payload.get("summary_tail") or rc.get("summary_history") or []
    if not actions:
        # Fallback: look one level up
        actions = checkpoint.get("action_history") or checkpoint.get("action_history_prefix") or []
        summaries = checkpoint.get("summary_history") or checkpoint.get("summary_history_prefix") or []
    n = len(actions)
    if n == 0:
        return "(no trajectory available)"
    start = max(0, n - tail_steps)
    lines = []
    for i in range(start, n):
        action = actions[i] if i < len(actions) else ""
        summary = summaries[i] if i < len(summaries) else ""
        marker = "  ← retry node" if i == restart_iter else ""
        lines.append(f"  Step {i}: {action} | {summary}{marker}")
    return "\n".join(lines)


def _load_resume_checkpoint(path: str) -> Optional[dict]:
    """Load and validate a resume checkpoint JSON.

    Always returns a dict (never None) so the caller can inspect
    ``integrity.resume_degraded`` and decide whether to fallback.
    Returns None only if the file is completely unreadable.

    Degradation rules (sets ``integrity.resume_degraded = True``):
    - Missing top-level required fields
    - ``profile.base_path`` missing or directory does not exist
    - ``resume_from_url`` empty
    """
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cp = json.load(f)
    except Exception as exc:
        logger.warning(f"Cannot load resume checkpoint '{path}': {exc}")
        return None

    _apply_replay_core_to_checkpoint(cp)

    integrity = cp.setdefault("integrity", {})
    degrade_reasons = []

    # ── Required top-level fields ────────────────────────────────────
    for field in ("resume_from_url", "last_completed_iter"):
        if not cp.get(field) and cp.get(field) != 0:
            degrade_reasons.append(f"missing or empty required field: {field}")

    # ── Profile validation ───────────────────────────────────────────
    profile = cp.get("profile", {})
    base_path = profile.get("base_path", "")
    if not base_path:
        degrade_reasons.append("profile.base_path is empty")
    elif not Path(base_path).exists():
        degrade_reasons.append(f"profile.base_path does not exist: {base_path}")

    # ── Path existence checks ────────────────────────────────────────
    for pc in integrity.get("path_checks", []):
        if pc.get("must_exist") and not Path(pc.get("path", "")).exists():
            pc["exists"] = False
            degrade_reasons.append(f"required path missing: {pc.get('path')}")

    if degrade_reasons:
        integrity["resume_degraded"] = True
        integrity["degrade_reason"] = "; ".join(degrade_reasons)
        logger.warning(
            f"Resume checkpoint degraded ({len(degrade_reasons)} issue(s)): "
            + integrity["degrade_reason"]
            + " — will fall back to baseline mode."
        )
    else:
        integrity.setdefault("resume_degraded", False)
        integrity.setdefault("degrade_reason", "")
        logger.info(
            f"Resume checkpoint OK: url={cp.get('resume_from_url')} "
            f"iter={cp.get('last_completed_iter')} "
            f"profile={base_path or '(none)'}"
        )

    return cp


def _apply_replay_core_to_checkpoint(cp: dict) -> None:
    """Copy replay_core fields to top level so existing code can use cp.get('resume_from_url') etc."""
    if "replay_core" not in cp:
        return
    core = cp["replay_core"]
    cp["resume_from_url"] = core.get("resume_from_url")
    cp["last_completed_iter"] = core.get("last_completed_iter")
    cp["guard_policy"] = core.get("guard_policy", {})
    cp["viewport"] = core.get("viewport", {})
    cp["scroll"] = core.get("scroll", {})
    cp["restore_order"] = core.get("restore_order", [])
    cp["last_screenshot_path"] = core.get("last_screenshot_path", "")
    cp["profile"] = core.get("profile", {})


class AppEvalContext(RoleContext):
    """AppEval Runtime Context"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    json_file: str = "data/default_results.json"
    env_process: Optional[Any] = None
    agent_params: Dict = Field(default_factory=dict)
    test_generator: Optional[CaseGenerator] = None
    test_cases: Optional[List[str]] = None


class AppEvalRole(Role):
    """Automated Testing Role"""

    name: str = "AppEvalRole"
    profile: str = "Automated Test Executor"
    goal: str = "Execute automated testing tasks"
    constraints: str = "Ensure accuracy and efficiency of test execution"

    rc: AppEvalContext = Field(default_factory=AppEvalContext)
    # NOTE: osagent is stored via object.__setattr__ to prevent pydantic from
    # coercing TextAgent subclass back to OSAgent during model validation.
    # Access via self.osagent still works normally.

    def __init__(self, json_file: str = "data/default_results.json", **kwargs):
        super().__init__()
        self.rc.json_file = json_file

        # Chrome launch params for multi-worker isolation
        self._remote_debugging_port = kwargs.get("remote_debugging_port", 9222)
        self._user_data_dir = kwargs.get("user_data_dir", "")
        self._run_id = kwargs.get("run_id", "")
        self._worker_id = kwargs.get("worker_id", "")
        self._config_file = kwargs.get("config_file", "")

        # Initialize agent_params
        self.rc.agent_params = {
            "use_ocr": kwargs.get("use_ocr", True),
            "quad_split_ocr": kwargs.get("quad_split_ocr", True),
            "use_memory": kwargs.get("use_memory", True),
            "use_reflection": kwargs.get("use_reflection", True),
            "use_chrome_debugger": kwargs.get("use_chrome_debugger", True),
            "extend_xml_infos": kwargs.get("extend_xml_infos", True),
            "a11y_mode": kwargs.get("a11y_mode", "atspi"),
            "use_tell_verifier": kwargs.get("use_tell_verifier", True),
            "log_dirs": kwargs.get("log_dirs", "work_dirs"),
            "use_timestamp_log_dir": kwargs.get("use_timestamp_log_dir", True),
            "max_iters": kwargs.get("max_iters", 20),
            "save_checkpoint_per_step": kwargs.get("save_checkpoint_per_step", False),
            "save_profile_per_step": kwargs.get("save_profile_per_step", False),
            "branching_n_candidates": int(kwargs.get("branching_n_candidates", 0)),
            "branching_k": int(kwargs.get("branching_k", 1)),
        }

        # Store agent_class for _init_osagent
        self._agent_class = kwargs.get("agent_class", "osagent")

        # Initialize CaseGenerator Action
        self.test_generator = CaseGenerator(
            config_path=kwargs.get("config_file", "config/config2.yaml")
        )

        # Initialize OSAgent (or TextAgent)
        self._init_osagent(**kwargs)

    def _init_osagent(self, **kwargs) -> None:
        """Initialize OSAgent or TextAgent based on agent_class parameter.

        Args (via kwargs):
            agent_class: "osagent" (default, VLM + screenshots) or "text_agent" (text-only, a11y tree)
        """
        add_info = """**[CRITICAL - Login Credentials]** If the application requires login or registration (e.g. you see a login page, "Welcome Back", "Sign Up", or "Create your account"), use these credentials to LOG IN directly:
  - Email: algo_020@qq.com
  - Password: 123456
Click the email field, type the email using pyautogui.write(), then click the password field and type the password. Then click the "Log in" button. Do NOT register a new account. Do NOT click "Create your account" or "Sign up".
If you see an "Authorize Application" page requesting permissions (OpenID, Email etc.), click the "Allow" button immediately.
If a "Save password?" popup appears from Chrome, click "Never" to dismiss it and continue testing.

Before interacting with any web page, first browse the page completely from top to bottom by pressing Page Down to page through the content, so you get an overall understanding and can locate the required elements. If after a full scan you still cannot find the element, press Ctrl+F to search by visible keywords such as labels, button text, or field names. Clear the search and continue once the element is located.
If you need to interact with elements outside of a web popup, such as calendar or time selection popups, make sure to close the popup first. If the content in a text box is entered incorrectly, use the select all and delete actions to clear it, then re-enter the correct information.
To open a folder in File Explorer, please use a double-click.
If there is a problem with opening the web page, please do not keep trying to refresh the page or click repeatedly. After an attempt, please proceed directly to the remaining tasks.
Pay attention not to use shortcut keys to change the window size when testing on the web page.
If it involves the display effect of a web page on mobile devices, you can open the developer mode of the web page by pressing F12, and then use the shortcut key Ctrl+Shift+M to switch to the mobile view.
When testing game-related content, please pay close attention to judge whether the game functions are abnormal. If you find that no expected changes occur after certain operations, directly exit and mark this feature as negative.
Please use the Tell action to report the results of all test cases before executing Stop"""

        # Auto-detect platform if not specified
        import platform as platform_module
        # Check environment variable first, then kwargs, then auto-detect
        platform_from_env = os.environ.get('PLATFORM')
        if platform_from_env:
            default_platform = platform_from_env
        elif os.name == "nt":
            default_platform = "Windows"
        elif platform_module.system() == "Linux":
            default_platform = "Linux"
        elif platform_module.system() == "Darwin":
            default_platform = "Mac"
        else:
            default_platform = "Linux"  # Default to Linux for other Unix-like systems

        # 从 config_file 读取 llm 配置，显式传给 OSAgent，
        # 避免被 metagpt 默认的 config/config2.yaml 覆盖 run_config 的模型设置
        osagent_config = None
        config_file = kwargs.get("config_file", "")
        if config_file and os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                _cfg = _yaml.safe_load(f)
            _llm_cfg = _cfg.get("llm") if _cfg else None
            if _llm_cfg:
                osagent_config = Config.from_llm_config(_llm_cfg)
                logger.info(f"OSAgent LLM config from {config_file}: {_llm_cfg.get('model')} @ {_llm_cfg.get('base_url')}")

        agent_kwargs = dict(
            platform=kwargs.get("os_type", kwargs.get("platform", default_platform)),
            max_iters=self.rc.agent_params["max_iters"],
            extend_xml_infos=self.rc.agent_params["extend_xml_infos"],
            a11y_mode=self.rc.agent_params["a11y_mode"],
            remote_debugging_port=self._remote_debugging_port,
            user_data_dir=self._user_data_dir,
            location_info="center",
            log_dirs=self.rc.agent_params["log_dirs"],
            use_timestamp_log_dir=self.rc.agent_params["use_timestamp_log_dir"],
            config_file=kwargs.get("config_file", ""),
            add_info=add_info,
            run_id=kwargs.get("run_id", ""),
            worker_id=kwargs.get("worker_id", ""),
        )
        if osagent_config:
            agent_kwargs["config"] = osagent_config

        if self._agent_class == "text_agent":
            # Text-only agent: uses a11y tree / DOM tree, no screenshots
            from appeval.roles.text_agent import create_text_agent
            agent_kwargs.update(
                debug_screenshots=kwargs.get("debug_screenshots", True),
                system_prompt=text_agent_system_prompt,
            )
            object.__setattr__(self, 'osagent', create_text_agent(**agent_kwargs))
        else:
            # Default: VLM-based OSAgent with screenshots
            agent_kwargs.update(
                use_ocr=self.rc.agent_params["use_ocr"],
                quad_split_ocr=self.rc.agent_params["quad_split_ocr"],
                use_icon_detect=False,
                use_icon_caption=True,
                use_memory=self.rc.agent_params["use_memory"],
                use_reflection=self.rc.agent_params["use_reflection"],
                use_som=False,
                use_chrome_debugger=self.rc.agent_params["use_chrome_debugger"],
                use_tell_verifier=self.rc.agent_params["use_tell_verifier"],
                save_checkpoint_per_step=self.rc.agent_params["save_checkpoint_per_step"],
                save_profile_per_step=self.rc.agent_params["save_profile_per_step"],
                draw_text_box=False,
                system_prompt=case_batch_check_system_prompt,
            )
            object.__setattr__(self, 'osagent', OSAgent(**agent_kwargs))

    # ==================== Core Helper Methods ====================

    async def _start_environment(self, url: str = None, work_path: str = None) -> Optional[int]:
        """Start test environment (browser or application)"""
        if url:
            return await start_windows(
                target_url=url,
                remote_debugging_port=self._remote_debugging_port,
                user_data_dir=self._user_data_dir,
            )
        if work_path:
            return await start_windows(
                work_path=work_path,
                remote_debugging_port=self._remote_debugging_port,
                user_data_dir=self._user_data_dir,
            )
        return None

    async def _cleanup_environment(self, is_web: bool, pid: Optional[int] = None) -> None:
        """Clean up test environment"""
        # If we have a dedicated user_data_dir, only kill that Chrome instance
        if self._user_data_dir and os.name != "nt":
            try:
                cmd_kill = f"pkill -f 'user-data-dir={self._user_data_dir}'"
                proc = await asyncio.create_subprocess_shell(
                    cmd_kill, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()
            except Exception:
                pass
        else:
            processes = ["Chrome"] if is_web else [
                "Chrome", "cmd", "npm", "projectapp", "Edge"]
            await kill_windows(processes)
        if pid:
            await kill_process(pid)

    @staticmethod
    def _find_matching_key(key: Any, target_dict: dict) -> Optional[Any]:
        """Find matching key in target dict, handling both string and int key types.

        Args:
            key: The key to match (can be string or int)
            target_dict: The dictionary to search in

        Returns:
            The matching key if found, None otherwise
        """
        if key in target_dict:
            return key

        str_key = str(key)
        if str_key in target_dict:
            return str_key

        if str_key.isdigit():
            int_key = int(str_key)
            if int_key in target_dict:
                return int_key

        return None

    def _parse_results_from_tell(self, action_history: List[str]) -> Optional[dict]:
        """Parse test results from Tell action in action history

        Note: If use_tell_verifier is enabled, the Tell action content in action_history
        has already been verified and potentially corrected by TellVerifier in OSAgent._act().
        This ensures that hallucinated judgments (outcome hallucination, confirmation bias)
        are detected and corrected before being stored in history and parsed here.
        """
        if not action_history:
            return None

        content = action_history[-1]
        if not content.startswith("Tell ("):
            return None

        # Extract content between parentheses
        start_idx = content.find("(")
        end_idx = content.rfind(")")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None

        answer = content[start_idx + 1: end_idx]

        # Try direct eval
        try:
            return eval(answer)
        except Exception:
            pass

        # Try extracting dict from answer
        dict_start = answer.find("{")
        dict_end = answer.rfind("}")
        if dict_start != -1 and dict_end != -1 and dict_end > dict_start:
            try:
                return eval(answer[dict_start: dict_end + 1])
            except Exception as e:
                logger.error(f"Result parsing failed: {str(e)}")

        return None

    async def _execute_test_with_retry(
        self,
        task_id: str,
        task_id_case_number: int,
        check_list: dict,
        max_retries: int = 2,
        soft_reset: bool = False,
    ) -> tuple[List[str], str, List[str], str]:
        """Execute test with retry mechanism.

        Args:
            soft_reset: Passed through to osagent.run(); True in resume mode so
                        history / iter prefix loaded from checkpoint are preserved.
        """
        instruction = (
            "Please complete the following tasks，And after completion, use the Tell action to "
            f"inform me of the results of all the test cases at once: {check_list}\n"
        )

        for attempt in range(max_retries + 1):
            try:
                await self.osagent.run(instruction, soft_reset=(soft_reset and attempt == 0))
                return (self.osagent.rc.action_history, self.osagent.rc.task_list, self.osagent.rc.memory, self.osagent.rc.iter)
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1} failed for task {task_id}, retrying... Error: {str(e)}")
                    await asyncio.sleep(SLEEP_BETWEEN_RETRIES)
                else:
                    logger.error(
                        f"All {max_retries + 1} attempts failed for task {task_id}. Error: {str(e)}")
                    raise

    async def _process_test_results(
        self,
        task_id: str,
        task_id_case_number: int,
        action_history: List[str],
        task_list: str,
        memory: List[str],
        iter_num: str,
        check_list: dict = None,
        return_dict: bool = False,
    ) -> Optional[dict]:
        """Process test results and either save to JSON or return as dict"""
        # Parse results from Tell action
        results_dict = self._parse_results_from_tell(action_history)

        # Generate results if parsing failed or incomplete
        if not results_dict or len(results_dict) != task_id_case_number:
            results_dict = await self.test_generator.generate_results_dict(action_history, task_list, memory, task_id_case_number, check_list)

        # Return dict for API mode
        if return_dict:
            return results_dict

        # Write to JSON for batch mode
        data = read_json_file(self.rc.json_file)
        data[task_id]["iters"] = iter_num
        for key, value in results_dict.items():
            matching_key = self._find_matching_key(key, data[task_id]["test_cases"])
            if matching_key is not None:
                if isinstance(value, dict):
                    data[task_id]["test_cases"][matching_key].update({
                        "result": value.get("result", ""),
                        "evidence": value.get("evidence", "")
                    })
                else:
                    data[task_id]["test_cases"][matching_key].update({
                        "result": str(value),
                        "evidence": ""
                    })
        write_json_file(self.rc.json_file, data, indent=4)
        return None

    async def execute_batch_check(self, task_id: str, task_id_case_number: int, check_list: dict) -> None:
        """Execute test and write results to JSON file"""
        logger.info(
            f"Start testing project {task_id}, log_dirs: {self.osagent.log_dirs}")

        try:
            action_history, task_list, memory, iter_num = await self._execute_test_with_retry(task_id, task_id_case_number, check_list)
            await self._process_test_results(task_id, task_id_case_number, action_history, task_list, memory, iter_num)
        except Exception as e:
            # Write failed result to JSON
            try:
                await self._process_test_results(task_id, task_id_case_number, ["Failed after all retries"], "Failed", [f"Error: {str(e)}"], "0")
            except Exception as write_error:
                logger.error(f"Failed to write error result to JSON: {str(write_error)}")
                raise

    async def execute_api_check(
        self,
        task_id: str,
        task_id_case_number: int,
        check_list: dict,
        soft_reset: bool = False,
    ) -> dict:
        """Execute test and return results as dictionary."""
        logger.info(f"Start testing project {task_id}, log_dirs: {self.osagent.log_dirs}")

        action_history, task_list, memory, iter_num = await self._execute_test_with_retry(
            task_id, task_id_case_number, check_list, soft_reset=soft_reset
        )
        return await self._process_test_results(
            task_id, task_id_case_number, action_history, task_list, memory, iter_num, check_list, return_dict=True
        )

    async def _execute_task_batch(self, test_cases: dict, max_retry_uncertain: int = 1) -> None:
        """Execute batch of test tasks with retry mechanism"""
        for task_id, task_info in test_cases.items():
            if "test_cases" not in task_info:
                continue

            start_func = (task_info.get("url") or task_info.get("work_path") or "").strip()
            if not start_func:
                logger.warning(f"No valid url or work_path for task {task_id}, skipping...")
                continue

            logger.info(f"Executing task: {task_id}")

            try:
                final_test_cases, _ = await self._run_test_with_retry(
                    task_name=task_id,
                    test_cases=task_info["test_cases"],
                    start_func=start_func,
                    log_dir="batch",
                    max_retry_uncertain=max_retry_uncertain,
                    save_to_file=False,
                )
                task_info["test_cases"] = final_test_cases

            except Exception as e:
                logger.error(f"Failed to execute task {task_id}: {str(e)}")
                for case_id in task_info["test_cases"]:
                    task_info["test_cases"][case_id]["result"] = "Failed"
                    task_info["test_cases"][case_id]["evidence"] = f"Error: {str(e)}"

    async def _prepare_batch_test_cases(self, project_excel_path: str, operation_type: OperationType, converter_func) -> Optional[Any]:
        """Prepare test cases from Excel file"""
        if not project_excel_path:
            raise ValueError("project_excel_path must be provided for batch run.")

        logger.info("Start generating automated test cases...")
        await self.test_generator.process_excel_file(project_excel_path, operation_type)

        logger.info("Start converting to JSON format...")
        return converter_func(project_excel_path, self.rc.json_file)

    async def _retry_uncertain_api_mode(self, task_name: str, uncertain_cases: dict, is_web: bool, start_func: str, retry_count: int) -> dict:
        """Retry uncertain cases in API mode"""
        uncertain_test_cases = uncertain_cases[task_name]["test_cases"]

        logger.info(f"Restarting environment for retry {retry_count}...")
        try:
            if hasattr(self.osagent, "controller") and hasattr(self.osagent.controller, "set_expected_url"):
                self.osagent.controller.set_expected_url(start_func if is_web else "")
        except Exception as e:
            logger.debug(f"Failed to bind expected URL before retry: {e}")
        await self._start_environment(url=start_func if is_web else None, work_path=start_func if not is_web else None)
        await asyncio.sleep(SLEEP_BEFORE_EXECUTE)

        task_id_case_number = len(uncertain_test_cases)
        logger.info(f"Executing retry {retry_count} for {task_id_case_number} uncertain cases...")
        retry_result_dict = await self.execute_api_check(task_name, task_id_case_number, uncertain_test_cases)

        logger.info(f"Cleaning up environment after retry {retry_count}...")
        await self._cleanup_environment(is_web)
        await asyncio.sleep(SLEEP_AFTER_CLEANUP)

        # Construct and return retry result; value may be dict or bare str
        normalized = {}
        for key, value in retry_result_dict.items():
            if isinstance(value, dict):
                normalized[key] = {"result": value.get("result", ""), "evidence": value.get("evidence", "")}
            else:
                normalized[key] = {"result": str(value), "evidence": ""}
        return {
            task_name: {
                "test_cases": normalized
            }
        }

    async def _retry_uncertain_single_mode(self, uncertain_cases: dict, json_path: str, result: dict, retry_count: int) -> dict:
        """Retry uncertain cases in Single mode"""
        retry_json_path = str(Path(json_path).parent / f"{Path(json_path).stem}_retry_{retry_count}.json")
        write_json_file(retry_json_path, uncertain_cases, indent=4)
        self.rc.json_file = retry_json_path

        await self._execute_test_cases(uncertain_cases, log_dir_suffix=f"retry_{retry_count}")

        retry_result = read_json_file(retry_json_path)

        # Update original JSON file
        self.rc.json_file = json_path
        write_json_file(json_path, result, indent=4)

        # Clean up temporary file
        Path(retry_json_path).unlink(missing_ok=True)

        return retry_result

    async def _retry_uncertain_cases(
        self, result: dict, max_retry: int, task_name: str = None, start_func: str = None, json_path: str = None
    ) -> dict:
        """Retry uncertain cases (unified for both API and Single modes)"""
        retry_count = 0
        previous_uncertain_count = float("inf")
        original_log_dir = self.osagent.log_dirs
        is_api_mode = task_name is not None and start_func is not None
        is_web = start_func.startswith("http") if start_func else False

        while retry_count < max_retry:
            uncertain_cases = self._extract_uncertain_cases(result)
            should_retry, current_count = self._should_retry_uncertain(uncertain_cases, retry_count, max_retry, previous_uncertain_count)

            if not should_retry:
                break

            previous_uncertain_count = current_count
            self.osagent.log_dirs = f"{original_log_dir}/retry_{retry_count}"
            logger.info(f"Setting log_dirs to: {self.osagent.log_dirs} (retry)")

            # Execute retry based on mode
            if is_api_mode:
                retry_result = await self._retry_uncertain_api_mode(task_name, uncertain_cases, is_web, start_func, retry_count)
            else:
                retry_result = await self._retry_uncertain_single_mode(uncertain_cases, json_path, result, retry_count)

            result = self._merge_results(result, retry_result)
            self.osagent.log_dirs = original_log_dir
            retry_count += 1

        return result

    # ==================== Uncertain Cases Retry Methods ====================

    def _extract_uncertain_cases(self, test_result: dict) -> dict:
        """Extract test cases with uncertain results and clear result/evidence fields for retry"""
        uncertain_cases = {}

        for task_id, task_info in test_result.items():
            if "test_cases" not in task_info:
                continue

            uncertain_test_cases = {}
            for case_id, case_info in task_info["test_cases"].items():
                if str(case_info.get("result") or "").strip().lower() == "uncertain":
                    # Deep copy and clear result/evidence to treat as fresh test
                    clean_case = copy.deepcopy(case_info)
                    clean_case.pop("result", None)
                    clean_case.pop("evidence", None)
                    uncertain_test_cases[case_id] = clean_case

            if uncertain_test_cases:
                uncertain_cases[task_id] = copy.deepcopy(task_info)
                uncertain_cases[task_id]["test_cases"] = uncertain_test_cases

        return uncertain_cases

    def _merge_results(self, original_result: dict, retry_result: dict) -> dict:
        """Merge retry results into original results"""
        merged_result = copy.deepcopy(original_result)

        for task_id, task_info in retry_result.items():
            if task_id not in merged_result or "test_cases" not in task_info:
                continue

            for case_id, case_info in task_info["test_cases"].items():
                if case_id in merged_result[task_id]["test_cases"]:
                    merged_result[task_id]["test_cases"][case_id]["result"] = case_info.get("result", "")
                    merged_result[task_id]["test_cases"][case_id]["evidence"] = case_info.get("evidence", "")
                    logger.info(f"Updated case {case_id} in task {task_id} with retry result: {case_info.get('result', '')}")

        return merged_result

    def _count_uncertain_cases(self, test_result: dict) -> int:
        """Count total number of uncertain cases"""
        return sum(len(task_info["test_cases"]) for task_info in test_result.values() if "test_cases" in task_info)

    def _should_retry_uncertain(self, uncertain_cases: dict, retry_count: int, max_retry: int, previous_count: int) -> tuple[bool, int]:
        """Determine if uncertain cases should be retried"""
        if not uncertain_cases:
            logger.info("No uncertain cases found, skipping retry")
            return False, 0

        if retry_count >= max_retry:
            return False, 0

        current_count = self._count_uncertain_cases(uncertain_cases)

        if current_count >= previous_count:
            logger.warning(
                f"No improvement in uncertain cases ({current_count} cases still uncertain), " f"stopping retry to avoid redundant testing"
            )
            return False, current_count

        logger.info(f"Found {current_count} uncertain cases, starting retry {retry_count + 1}/{max_retry}...")
        return True, current_count

    async def _execute_test_cases(self, test_cases: dict, log_dir_suffix: str = "") -> None:
        """Execute test cases with environment setup and cleanup"""
        for task_id, task_info in test_cases.items():
            if "test_cases" not in task_info:
                continue

            # Handle log directory
            original_log_dir = None
            if log_dir_suffix:
                original_log_dir = self.osagent.log_dirs
                self.osagent.log_dirs = f"{original_log_dir}/{log_dir_suffix}"
                logger.info(f"Setting log_dirs to: {self.osagent.log_dirs} (retry)")

            # Start environment and wait
            is_web = "url" in task_info
            pid = await self._start_environment(url=task_info.get("url"), work_path=task_info.get("work_path"))
            await asyncio.sleep(SLEEP_AFTER_START_APP if not is_web else 0)
            await asyncio.sleep(SLEEP_AFTER_START_WEB)

            # Execute tests - pass only test_cases dict to maintain consistency
            await self.execute_batch_check(task_id, len(task_info["test_cases"]), task_info["test_cases"])

            # Restore and cleanup
            if original_log_dir:
                self.osagent.log_dirs = original_log_dir
            await self._cleanup_environment(is_web, pid)

    async def execute_batch_check(self, task_id: str, task_id_case_number: int, check_list: dict) -> None:
        """Execute test and write results to JSON file"""
        logger.info(
            f"Start testing project {task_id}, log_dirs: {self.osagent.log_dirs}")

        try:
            action_history, task_list, memory, iter_num = await self._execute_test_with_retry(task_id, task_id_case_number, check_list)
            await self._process_test_results(task_id, task_id_case_number, action_history, task_list, memory, iter_num)
        except Exception as e:
            # Write failed result to JSON
            try:
                await self._process_test_results(task_id, task_id_case_number, ["Failed after all retries"], "Failed", [f"Error: {str(e)}"], "0")
            except Exception as write_error:
                logger.error(
                    f"Failed to write error result to JSON: {str(write_error)}")
                raise

    async def execute_api_check(
        self,
        task_id: str,
        task_id_case_number: int,
        check_list: dict,
        soft_reset: bool = False,
    ) -> dict:
        """Execute test and return results as dictionary"""
        logger.info(
            f"Start testing project {task_id}, log_dirs: {self.osagent.log_dirs}")

        action_history, task_list, memory, iter_num = await self._execute_test_with_retry(
            task_id, task_id_case_number, check_list, soft_reset=soft_reset
        )
        return await self._process_test_results(
            task_id, task_id_case_number, action_history, task_list, memory, iter_num, check_list, return_dict=True
        )

    async def _execute_task_batch(self, test_cases: dict, max_retry_uncertain: int = 1, sequential_mode: bool = False) -> None:
        """Execute batch of test tasks with retry mechanism

        Args:
            test_cases: Dictionary of test cases to execute
            max_retry_uncertain: Maximum retries for uncertain cases
            sequential_mode: If True, execute test cases one by one without browser cleanup between cases,
                           only reset osagent state. If False, execute all test cases at once (default).
        """
        for task_id, task_info in test_cases.items():
            if "test_cases" not in task_info:
                continue

            start_func = (task_info.get("url") or task_info.get(
                "work_path") or "").strip()
            if not start_func:
                logger.warning(
                    f"No valid url or work_path for task {task_id}, skipping...")
                continue

            logger.info(f"Executing task: {task_id}")

            try:
                final_test_cases, _ = await self._run_test_with_retry(
                    task_name=task_id,
                    test_cases=task_info["test_cases"],
                    start_func=start_func,
                    log_dir="batch",
                    max_retry_uncertain=max_retry_uncertain,
                    save_to_file=False,
                    sequential_mode=sequential_mode,
                )
                task_info["test_cases"] = final_test_cases

            except Exception as e:
                logger.error(f"Failed to execute task {task_id}: {str(e)}")
                for case_id in task_info["test_cases"]:
                    task_info["test_cases"][case_id]["result"] = "Failed"
                    task_info["test_cases"][case_id]["evidence"] = f"Error: {str(e)}"

    async def _prepare_batch_test_cases(self, project_excel_path: str, operation_type: OperationType, converter_func) -> Optional[Any]:
        """Prepare test cases from Excel file"""
        if not project_excel_path:
            raise ValueError(
                "project_excel_path must be provided for batch run.")

        logger.info("Start generating automated test cases...")
        await self.test_generator.process_excel_file(project_excel_path, operation_type)

        logger.info("Start converting to JSON format...")
        return converter_func(project_excel_path, self.rc.json_file)

    async def _retry_uncertain_api_mode(self, task_name: str, uncertain_cases: dict, is_web: bool, start_func: str, retry_count: int) -> dict:
        """Retry uncertain cases in API mode"""
        uncertain_test_cases = uncertain_cases[task_name]["test_cases"]

        logger.info(f"Restarting environment for retry {retry_count}...")
        try:
            if hasattr(self.osagent, "controller") and hasattr(self.osagent.controller, "set_expected_url"):
                self.osagent.controller.set_expected_url(start_func if is_web else "")
        except Exception as e:
            logger.debug(f"Failed to bind expected URL before retry: {e}")
        await self._start_environment(url=start_func if is_web else None, work_path=start_func if not is_web else None)
        await asyncio.sleep(SLEEP_BEFORE_EXECUTE)

        task_id_case_number = len(uncertain_test_cases)
        logger.info(
            f"Executing retry {retry_count} for {task_id_case_number} uncertain cases...")
        retry_result_dict = await self.execute_api_check(task_name, task_id_case_number, uncertain_test_cases)

        logger.info(f"Cleaning up environment after retry {retry_count}...")
        await self._cleanup_environment(is_web)
        await asyncio.sleep(SLEEP_AFTER_CLEANUP)

        # Construct and return retry result; value may be dict or bare str
        normalized = {}
        for key, value in retry_result_dict.items():
            if isinstance(value, dict):
                normalized[key] = {"result": value.get("result", ""), "evidence": value.get("evidence", "")}
            else:
                normalized[key] = {"result": str(value), "evidence": ""}
        return {
            task_name: {
                "test_cases": normalized
            }
        }

    async def _retry_uncertain_single_mode(self, uncertain_cases: dict, json_path: str, result: dict, retry_count: int) -> dict:
        """Retry uncertain cases in Single mode"""
        retry_json_path = str(Path(json_path).parent /
                              f"{Path(json_path).stem}_retry_{retry_count}.json")
        write_json_file(retry_json_path, uncertain_cases, indent=4)
        self.rc.json_file = retry_json_path

        await self._execute_test_cases(uncertain_cases, log_dir_suffix=f"retry_{retry_count}")

        retry_result = read_json_file(retry_json_path)

        # Update original JSON file
        self.rc.json_file = json_path
        write_json_file(json_path, result, indent=4)

        # Clean up temporary file
        Path(retry_json_path).unlink(missing_ok=True)

        return retry_result

    async def _retry_uncertain_cases(
        self, result: dict, max_retry: int, task_name: str = None, start_func: str = None, json_path: str = None
    ) -> dict:
        """Retry uncertain cases (unified for both API and Single modes)"""
        retry_count = 0
        previous_uncertain_count = float("inf")
        original_log_dir = self.osagent.log_dirs
        is_api_mode = task_name is not None and start_func is not None
        is_web = (start_func.startswith("http://")
                  or start_func.startswith("https://")) if start_func else False

        while retry_count < max_retry:
            uncertain_cases = self._extract_uncertain_cases(result)
            should_retry, current_count = self._should_retry_uncertain(
                uncertain_cases, retry_count, max_retry, previous_uncertain_count)

            if not should_retry:
                break

            previous_uncertain_count = current_count
            self.osagent.log_dirs = f"{original_log_dir}/retry_{retry_count}"
            logger.info(
                f"Setting log_dirs to: {self.osagent.log_dirs} (retry)")

            # Execute retry based on mode
            if is_api_mode:
                retry_result = await self._retry_uncertain_api_mode(task_name, uncertain_cases, is_web, start_func, retry_count)
            else:
                retry_result = await self._retry_uncertain_single_mode(uncertain_cases, json_path, result, retry_count)

            result = self._merge_results(result, retry_result)
            self.osagent.log_dirs = original_log_dir
            retry_count += 1

        return result

    # ==================== Uncertain Cases Retry Methods ====================

    def _extract_uncertain_cases(self, test_result: dict) -> dict:
        """Extract test cases with uncertain results and clear result/evidence fields for retry"""
        uncertain_cases = {}

        for task_id, task_info in test_result.items():
            if "test_cases" not in task_info:
                continue

            uncertain_test_cases = {}
            for case_id, case_info in task_info["test_cases"].items():
                if str(case_info.get("result") or "").strip().lower() == "uncertain":
                    # Deep copy and clear result/evidence to treat as fresh test
                    clean_case = copy.deepcopy(case_info)
                    clean_case.pop("result", None)
                    clean_case.pop("evidence", None)
                    uncertain_test_cases[case_id] = clean_case

            if uncertain_test_cases:
                uncertain_cases[task_id] = copy.deepcopy(task_info)
                uncertain_cases[task_id]["test_cases"] = uncertain_test_cases

        return uncertain_cases

    def _merge_results(self, original_result: dict, retry_result: dict) -> dict:
        """Merge retry results into original results"""
        merged_result = copy.deepcopy(original_result)

        for task_id, task_info in retry_result.items():
            if task_id not in merged_result or "test_cases" not in task_info:
                continue

            for case_id, case_info in task_info["test_cases"].items():
                matched_key = self._find_matching_key(
                    case_id, merged_result[task_id]["test_cases"])
                if matched_key is not None:
                    merged_result[task_id]["test_cases"][matched_key]["result"] = case_info.get(
                        "result", "")
                    merged_result[task_id]["test_cases"][matched_key]["evidence"] = case_info.get(
                        "evidence", "")
                    logger.info(
                        f"Updated case {case_id} in task {task_id} with retry result: {case_info.get('result', '')}")

        return merged_result

    def _count_uncertain_cases(self, test_result: dict) -> int:
        """Count total number of uncertain cases"""
        return sum(len(task_info["test_cases"]) for task_info in test_result.values() if "test_cases" in task_info)

    def _should_retry_uncertain(self, uncertain_cases: dict, retry_count: int, max_retry: int, previous_count: int) -> tuple[bool, int]:
        """Determine if uncertain cases should be retried"""
        if not uncertain_cases:
            logger.info("No uncertain cases found, skipping retry")
            return False, 0

        if retry_count >= max_retry:
            return False, 0

        current_count = self._count_uncertain_cases(uncertain_cases)

        if current_count >= previous_count:
            logger.warning(
                f"No improvement in uncertain cases ({current_count} cases still uncertain), " f"stopping retry to avoid redundant testing"
            )
            return False, current_count

        logger.info(
            f"Found {current_count} uncertain cases, starting retry {retry_count + 1}/{max_retry}...")
        return True, current_count

    async def _execute_test_cases(self, test_cases: dict, log_dir_suffix: str = "") -> None:
        """Execute test cases with environment setup and cleanup"""
        for task_id, task_info in test_cases.items():
            if "test_cases" not in task_info:
                continue

            # Handle log directory
            original_log_dir = None
            if log_dir_suffix:
                original_log_dir = self.osagent.log_dirs
                self.osagent.log_dirs = f"{original_log_dir}/{log_dir_suffix}"
                logger.info(
                    f"Setting log_dirs to: {self.osagent.log_dirs} (retry)")

            # Start environment and wait
            is_web = "url" in task_info
            pid = await self._start_environment(url=task_info.get("url"), work_path=task_info.get("work_path"))
            await asyncio.sleep(SLEEP_AFTER_START_APP if not is_web else 0)
            await asyncio.sleep(SLEEP_AFTER_START_WEB)

            # Execute tests - pass only test_cases dict to maintain consistency
            await self.execute_batch_check(task_id, len(task_info["test_cases"]), task_info["test_cases"])

            # Restore and cleanup
            if original_log_dir:
                self.osagent.log_dirs = original_log_dir
            await self._cleanup_environment(is_web, pid)

    async def _run_test_with_retry(
        self,
        task_name: str,
        test_cases: dict,
        start_func: str,
        log_dir: str,
        max_retry_uncertain: int,
        save_to_file: bool = True,
        sequential_mode: bool = False,
        case_name_for_log: Optional[str] = None,
        resume_checkpoint_path: str = "",
        save_checkpoint: bool = False,
        chrome_profile_src: str = "",
    ) -> tuple[dict, bool]:
        """Core test execution logic with retry mechanism

        Args:
            task_name: Task identifier
            test_cases: Dictionary of test cases to execute
            start_func: URL or work path to start the environment
            log_dir: Directory for logs
            max_retry_uncertain: Maximum retries for uncertain cases
            save_to_file: Whether to save results to file
            sequential_mode: If True, execute test cases one by one without browser cleanup between cases,
                           only reset osagent state. If False, execute all test cases at once (default).
            case_name_for_log: If set, log subdirs are {case_name_for_log}0, {case_name_for_log}1, ... (no task_name level).
            resume_checkpoint_path: Path to a resume_checkpoint.json from a previous round.
                When set (resume_mode), the first case navigates to the saved URL and
                restores agent context rather than starting fresh from start_func.
            save_checkpoint: If True, write a resume_checkpoint.json after the last
                sequential case (for use as round-2 input).
            chrome_profile_src: live user_data_dir; copied into checkpoint dir when
                save_checkpoint=True.
        """
        log_base = self.rc.agent_params.get("log_dirs", "work_dirs")
        if case_name_for_log is not None:
            self.osagent.log_dirs = f"{log_base}/{log_dir}"
        else:
            self.osagent.log_dirs = f"{log_base}/{log_dir}/{task_name}"
        is_web = start_func.startswith(
            "http://") or start_func.startswith("https://")

        # Start environment once for batch mode; sequential mode rebuilds fresh session per case.
        if not sequential_mode:
            try:
                if hasattr(self.osagent, "controller") and hasattr(self.osagent.controller, "set_expected_url"):
                    self.osagent.controller.set_expected_url(start_func if is_web else "")
            except Exception as e:
                logger.debug(f"Failed to bind expected URL before start: {e}")
            await self._start_environment(url=start_func if is_web else None, work_path=start_func if not is_web else None)
            if is_web and self.rc.agent_params.get("a11y_mode") == "cdp" and self._remote_debugging_port:
                await _wait_for_cdp(self._remote_debugging_port, max_wait=20.0)
            await asyncio.sleep(SLEEP_BEFORE_EXECUTE)

        def _get_llm_total_usd() -> float:
            # Fallback pricing by model when upstream TOKEN_COSTS misses model entries.
            fallback_pricing = {
                # USD / 1K tokens
                "gemini-3-flash-preview": {"prompt": 0.0005, "completion": 0.003},
            }

            def _cost_to_usd(cost_obj: Any, model_name: str) -> float:
                direct = float(
                    getattr(cost_obj, "total_cost_usd", None)
                    or getattr(cost_obj, "total_cost", None)
                    or getattr(cost_obj, "cost", None)
                    or 0.0
                )
                if direct > 0:
                    return direct
                prompt_tokens = int(getattr(cost_obj, "total_prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(cost_obj, "total_completion_tokens", 0) or 0)
                if prompt_tokens == 0 and completion_tokens == 0:
                    return 0.0
                price = fallback_pricing.get(model_name or "")
                if not price:
                    return 0.0
                return (
                    prompt_tokens * float(price["prompt"])
                    + completion_tokens * float(price["completion"])
                ) / 1000.0

            total = 0.0
            for llm_source in (
                getattr(self.test_generator, "llm", None),
                getattr(self.osagent, "llm", None),
            ):
                if llm_source and hasattr(llm_source, "get_costs"):
                    c = llm_source.get_costs()
                    model_name = getattr(llm_source, "model", "") or ""
                    total += _cost_to_usd(c, model_name)
            return total

        def _to_bool_result(v) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            s = str(v).strip().lower()
            return s in ("pass", "true", "1", "yes", "y")

        def _build_case_item(case_id, case_data, case_name: Optional[str] = None) -> dict:
            return {
                "test_id": f"{case_name}{case_id}" if case_name else str(case_id),
                "case_desc": case_data.get("case_desc", ""),
                "evidence": case_data.get("evidence", ""),
                "result": _to_bool_result(case_data.get("result", "")),
                "cost": case_data.get("cost", ""),
            }

        def _build_retry_context_text(restart_rec: Optional[dict]) -> str:
            """Compose supplemental guidance for round-2 from supervisor recommendation."""
            if not restart_rec:
                return ""
            fail_reason = str(restart_rec.get("fail_reason", "") or "").strip()
            retry_reason = str(restart_rec.get("retry_reason", "") or "").strip()
            restart_from_iter = restart_rec.get("restart_from_iter")
            restart_explanation = str(restart_rec.get("restart_explanation", "") or "").strip()
            parts = []
            if fail_reason:
                parts.append(f"- Fail reason from previous round: {fail_reason}")
            if retry_reason:
                parts.append(f"- Retry reason from previous round: {retry_reason}")
            if restart_from_iter is not None:
                parts.append(f"- Recommended restart_from_iter: {restart_from_iter}")
            if restart_explanation:
                parts.append(f"- Restart guidance: {restart_explanation}")
            if not parts:
                return ""
            return (
                "\\n\\n[ROUND-2 SUPPLEMENTAL GUIDANCE]\\n"
                "Use this as additional context before starting actions in this retry run:\\n"
                + "\\n".join(parts)
            )

        def _save_incremental_case_json(case_id, case_data) -> None:
            """Persist one case result immediately so partial progress survives interruptions."""
            if not save_to_file:
                return
            try:
                log_base = self.rc.agent_params.get("log_dirs", "work_dirs")
                if case_name_for_log is not None:
                    case_dir = Path(log_base) / log_dir / f"{case_name_for_log}{case_id}"
                    case_dir.mkdir(parents=True, exist_ok=True)
                    ts_dirs = [
                        p for p in case_dir.iterdir()
                        if p.is_dir() and p.name.isdigit() and len(p.name) >= 12
                    ]
                    output_dir = sorted(ts_dirs, key=lambda p: p.name)[-1] if ts_dirs else case_dir
                    output_file = output_dir / "test_case.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {"test_cases": [_build_case_item(case_id, case_data, case_name_for_log)]},
                            f,
                            indent=4,
                            ensure_ascii=False,
                        )
                    logger.info(f"Incremental results saved to {output_file}")
            except Exception as e:
                logger.warning(f"Failed to save incremental result for case {case_id}: {e}")

        def _save_retry_plans_json(case_id: str, payload: dict) -> None:
            """Persist candidate and selected retry plans for audit/debug."""
            try:
                output_dir = Path(getattr(self.osagent, "save_img", "") or "")
                if not output_dir.exists():
                    log_base = self.rc.agent_params.get("log_dirs", "work_dirs")
                    if case_name_for_log is not None:
                        output_dir = Path(log_base) / log_dir / f"{case_name_for_log}{case_id}"
                    else:
                        output_dir = Path(log_base) / log_dir / str(case_id)
                    output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "retry_plans.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                logger.info(f"[branching] Saved retry plans JSON: {output_file}")
            except Exception as e:
                logger.warning(f"[branching] Failed to save retry_plans.json for case {case_id}: {e}")

        def _extract_plan_dimension(plan_text: str) -> str:
            s = str(plan_text or "").strip()
            m = re.match(r"^\[Dimension\s+([ABC])\]", s, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
            return ""

        # ── Resume mode: load and validate checkpoint (sequential_mode only) ──
        resume_cp: Optional[dict] = None
        resume_restart_rec: Optional[dict] = None
        resume_cp_for_branching: Optional[dict] = None
        resume_retry_context_text = ""
        force_baseline_from_start = False
        if sequential_mode and resume_checkpoint_path:
            _cp = _load_resume_checkpoint(resume_checkpoint_path)
            if _cp is None:
                logger.warning("resume_checkpoint_path unreadable; falling back to baseline.")
            else:
                # If SupervisorJudge recommended a specific step, load that step's checkpoint for replay state
                rr = _cp.get("restart_recommendation") or {}
                resume_restart_rec = rr
                resume_cp_for_branching = copy.deepcopy(_cp)
                resume_retry_context_text = _build_retry_context_text(rr)
                restart_iter = rr.get("restart_from_iter")
                if restart_iter is not None:
                    if int(restart_iter) == 0:
                        # restart_from_iter=0 means retry from scratch; do baseline startup instead of resume.
                        force_baseline_from_start = True
                        logger.info("[resume] restart_from_iter=0, forcing baseline restart from start_func")
                    else:
                        step_path = Path(resume_checkpoint_path).parent / "checkpoints" / f"step_{int(restart_iter):03d}.json"
                        if step_path.exists():
                            step_cp = _load_resume_checkpoint(str(step_path))
                            if step_cp and step_cp.get("replay_core"):
                                _cp["replay_core"] = step_cp["replay_core"]
                                _cp["decision_payload"] = step_cp.get("decision_payload") or _cp.get("decision_payload")
                                _apply_replay_core_to_checkpoint(_cp)
                                logger.info(f"[resume] Using restart_recommendation: step {restart_iter} → {step_path}")

                _degraded = _cp.get("integrity", {}).get("resume_degraded", False)
                _reason = _cp.get("integrity", {}).get("degrade_reason", "")
                _profile_base = (_cp.get("profile", {}).get("base_path", "") or "").strip()
                _profile_ok = bool(_profile_base and Path(_profile_base).exists())

                if _degraded and not _profile_ok:
                    logger.warning(
                        f"Resume checkpoint degraded AND profile missing → baseline fallback. "
                        f"Reason: {_reason} | profile.base_path={_profile_base!r}"
                    )
                else:
                    if _degraded:
                        logger.warning(
                            f"[resume] Checkpoint partially degraded ({_reason}), "
                            f"but profile exists → proceeding with resume "
                            f"(URL will fall back to start_func if empty)."
                        )
                    if force_baseline_from_start:
                        resume_cp = None
                        logger.info("Baseline mode ENABLED (restart_from_iter=0): first case will start from start_func.")
                    else:
                        resume_cp = _cp
                        logger.info("Resume mode ENABLED: first case will start from checkpoint.")

        if sequential_mode:
            # Sequential mode: execute test cases one by one, record per-case time and cost
            logger.info(
                f"Start executing automated testing in sequential mode ({len(test_cases)} cases)...")
            all_results = {}
            base_log_dir = self.osagent.log_dirs
            cost_before = _get_llm_total_usd()

            for idx, (case_id, case_info) in enumerate(test_cases.items(), 1):
                logger.info(
                    f"Executing test case {idx}/{len(test_cases)}: {case_id}")

                # Determine if this case should use resume mode (first case only)
                use_resume_this_case = (resume_cp is not None and idx == 1 and not force_baseline_from_start)

                t0 = time.perf_counter()
                try:
                    if use_resume_this_case:
                        # ── Resume mode startup (restore_order: copy_profile → start_chrome
                        #    → navigate → wait_stable → set_scroll → confirm_screenshot) ──
                        resume_url = resume_cp.get("resume_from_url", "") or start_func
                        logger.info(f"[resume] Starting from checkpoint URL: {resume_url}")

                        # Step 1: copy_profile — copy base profile into live user_data_dir
                        profile_info = resume_cp.get("profile", {})
                        base_path = profile_info.get("base_path", "")
                        if base_path and Path(base_path).exists() and self._user_data_dir:
                            try:
                                if Path(self._user_data_dir).exists():
                                    shutil.rmtree(self._user_data_dir, ignore_errors=True)
                                shutil.copytree(base_path, self._user_data_dir)
                                logger.info(f"[resume] Profile copied: {base_path} → {self._user_data_dir}")
                            except Exception as pe:
                                logger.warning(f"[resume] Profile copy failed: {pe}")

                        # Step 2: apply guard_policy — set expected_url to resume URL
                        # (prevents domain-drift guard from pulling back to prod_url)
                        guard = resume_cp.get("guard_policy", {})
                        expected_url_mode = guard.get("expected_url_mode", "set_to_resume_from_url")
                        try:
                            if hasattr(self.osagent, "controller") and hasattr(self.osagent.controller, "set_expected_url"):
                                url_for_guard = resume_url if expected_url_mode == "set_to_resume_from_url" else start_func
                                self.osagent.controller.set_expected_url(url_for_guard if is_web else "")
                        except Exception as e:
                            logger.debug(f"[resume] set_expected_url failed: {e}")

                        # Step 3: start_chrome + navigate — launch Chrome with profile and open resume URL
                        await self._cleanup_environment(is_web)
                        await asyncio.sleep(SLEEP_AFTER_CLEANUP)
                        await self._start_environment(
                            url=resume_url if is_web else None,
                            work_path=resume_url if not is_web else None,
                        )
                        # Step 4: wait_stable
                        await asyncio.sleep(SLEEP_BEFORE_EXECUTE)

                        # Step 5: set_scroll (after page is stable)
                        scroll_pos = resume_cp.get("scroll", {})
                        if (scroll_pos.get("y", 0) or scroll_pos.get("x", 0)) and \
                                hasattr(self.osagent, "controller") and \
                                hasattr(self.osagent.controller, "set_scroll"):
                            try:
                                self.osagent.controller.set_scroll(
                                    x=int(scroll_pos.get("x", 0)),
                                    y=int(scroll_pos.get("y", 0)),
                                )
                                logger.info(f"[resume] Scroll restored: {scroll_pos}")
                            except Exception as e:
                                logger.debug(f"[resume] set_scroll failed: {e}")
                    else:
                        # ── Baseline mode: rebuild a clean browser session ──
                        try:
                            if hasattr(self.osagent, "controller") and hasattr(self.osagent.controller, "set_expected_url"):
                                self.osagent.controller.set_expected_url(start_func if is_web else "")
                        except Exception as e:
                            logger.debug(f"Failed to bind expected URL before case {case_id}: {e}")
                        await self._cleanup_environment(is_web)
                        await asyncio.sleep(SLEEP_AFTER_CLEANUP)
                        await self._start_environment(url=start_func if is_web else None, work_path=start_func if not is_web else None)
                        await asyncio.sleep(SLEEP_BEFORE_EXECUTE)

                    # Set case-specific log directory to avoid overwriting
                    if case_name_for_log is not None:
                        self.osagent.log_dirs = f"{base_log_dir}/{case_name_for_log}{case_id}"
                    else:
                        self.osagent.log_dirs = f"{base_log_dir}/{case_id}"
                    case_log_root = self.osagent.log_dirs
                    self.osagent._get_timestamped_paths()
                    # Lock timestamped paths so _reset_state won't regenerate a new timestamp dir
                    self.osagent._lock_timestamped_paths = True

                    # Restore agent history from checkpoint before first case
                    if use_resume_this_case:
                        self.osagent.rc.restore_from_checkpoint(resume_cp)
                        logger.info(
                            f"[resume] Agent context restored: iter={self.osagent.rc.iter}, "
                            f"history_len={len(self.osagent.rc.action_history)}"
                        )

                    # Create single case dict for execution.
                    # In round-2, inject supervisor's fail/retry context so agent can use it before acting.
                    case_info_for_run = copy.deepcopy(case_info)
                    if idx == 1 and resume_retry_context_text:
                        case_info_for_run["case_desc"] = (
                            str(case_info_for_run.get("case_desc", "") or "") + resume_retry_context_text
                        )
                        logger.info("[resume] Injected supplemental fail/retry context into case_desc for round-2 run")

                    # ── Branching retry (round-2 first case only) ──────────────────────────
                    branching_n = int(self.rc.agent_params.get("branching_n_candidates", 0))
                    branching_k = int(self.rc.agent_params.get("branching_k", 1))
                    do_branching = (
                        idx == 1
                        and (bool(resume_checkpoint_path) or bool(resume_retry_context_text))
                        and branching_n >= 2
                    )

                    if do_branching:
                        rr = (resume_restart_rec or {})
                        restart_iter = int(rr.get("restart_from_iter") or 0)
                        task_desc = str(case_info_for_run.get("case_desc", ""))
                        fail_reason = str(rr.get("fail_reason", "") or "")
                        retry_reason = str(rr.get("retry_reason", "") or "")
                        restart_explanation = str(rr.get("restart_explanation", "") or "")
                        cp_for_tail = resume_cp_for_branching or resume_cp or {}
                        traj_tail = _build_trajectory_tail(cp_for_tail, restart_iter) if cp_for_tail else "(no trajectory available)"

                        # Step A: capture current screenshot at retry node
                        screenshot_b64 = ""
                        try:
                            from metagpt.utils.common import encode_image as _enc_img
                            screenshot_b64 = _enc_img(self.osagent.screenshot_file) if Path(self.osagent.screenshot_file).exists() else ""
                        except Exception as _se:
                            logger.debug(f"[branching] Screenshot capture failed: {_se}")

                        # Step B: agent generates N candidate plans
                        logger.info(f"[branching] Generating {branching_n} candidate retry plans...")
                        candidate_plan_items = await self.osagent.generate_retry_plans(
                            n=branching_n,
                            task_desc=task_desc,
                            fail_reason=fail_reason,
                            restart_explanation=restart_explanation,
                            trajectory_tail=traj_tail,
                            screenshot_b64=screenshot_b64,
                        )
                        if not candidate_plan_items:
                            logger.warning("[branching] No plans generated, falling back to single-run mode")
                            do_branching = False

                    if do_branching:
                        plans = [str(p.get("plan", "")).strip() for p in candidate_plan_items if isinstance(p, dict)]
                        candidate_plan_items = [p for p in candidate_plan_items if isinstance(p, dict) and str(p.get("plan", "")).strip()]
                        if not plans:
                            logger.warning("[branching] Candidate plans empty after normalization, fallback to single-run mode")
                            do_branching = False
                    if do_branching:
                        # Step C: SupervisorJudge selects top-K plans
                        config_file = str(getattr(self, "_config_file", "") or "")
                        selected_plan_items = await supervisor_select_plans_async(
                            plans=plans,
                            k=branching_k,
                            context={
                                "task_desc": task_desc,
                                "fail_reason": fail_reason,
                                "retry_reason": retry_reason,
                                "restart_explanation": restart_explanation,
                                "trajectory_tail": traj_tail,
                                "screenshot_b64": screenshot_b64,
                            },
                            config_path=config_file,
                        )
                        if not selected_plan_items:
                            logger.warning("[branching] Judge returned empty plan set, fallback to first generated plan")
                            selected_plan_items = [{"idx": 0, "plan": plans[0], "reason": "fallback: first generated plan"}]
                        selected_plans = [str(it.get("plan", "")) for it in selected_plan_items if isinstance(it, dict)]
                        selected_reasons = [str(it.get("reason", "")) for it in selected_plan_items if isinstance(it, dict)]
                        logger.info(f"[branching] Judge selected {len(selected_plans)}/{len(plans)} plans")
                        if len(selected_plans) < min(branching_k, len(plans)):
                            logger.warning(
                                f"[branching] Selected plans fewer than branching_k: "
                                f"{len(selected_plans)} < {min(branching_k, len(plans))}. "
                                "Will execute all selected plans in order."
                            )
                        selected_indices: List[int] = []
                        _used_indices = set()
                        for sp in selected_plans:
                            _idx = -1
                            for i, p in enumerate(plans):
                                if i not in _used_indices and p == sp:
                                    _idx = i
                                    _used_indices.add(i)
                                    break
                            selected_indices.append(_idx)
                        # Persist early so candidate/selected plans are not lost if execution crashes later.
                        _save_retry_plans_json(
                            case_id=str(case_id),
                            payload={
                                "case_id": str(case_id),
                                "branching_n_candidates": int(branching_n),
                                "branching_k": int(branching_k),
                                "task_desc": task_desc,
                                "failure_context": {
                                    "fail_reason": fail_reason,
                                    "retry_reason": retry_reason,
                                    "restart_explanation": restart_explanation,
                                    "restart_from_iter": restart_iter,
                                },
                                "trajectory_tail": traj_tail,
                                "candidate_plans": [
                                    {
                                        "idx": i,
                                        "dimension": str(candidate_plan_items[i].get("dimension", "")),
                                        "title": str(candidate_plan_items[i].get("title", "")),
                                        "plan": p,
                                        "reason": str(candidate_plan_items[i].get("reason", "")),
                                    }
                                    for i, p in enumerate(plans)
                                ],
                                "selected_plan_indices": selected_indices,
                                "selected_plans": [
                                    {
                                        "order": i,
                                        "dimension": _extract_plan_dimension(p),
                                        "plan": p,
                                        "reason": selected_reasons[i] if i < len(selected_reasons) else "",
                                    }
                                    for i, p in enumerate(selected_plans)
                                ],
                                "branch_execution": [],
                                "stopped_early_on_success": False,
                                "status": "selected_not_executed_yet",
                            },
                        )

                        # Save browser + agent state at retry node for inter-branch restore
                        retry_node_cp = copy.deepcopy(
                            resume_cp
                            or resume_cp_for_branching
                            or {
                                "last_completed_iter": 0,
                                "action_history_prefix": [],
                                "summary_history_prefix": [],
                            }
                        )
                        # IMPORTANT: inter-branch restore must align with supervisor recommendation.
                        # Otherwise branch 1+ may resume from stale last_completed_iter (e.g. 16/20)
                        # even when restart_from_iter is 0.
                        try:
                            restart_iter_int = int(restart_iter or 0)
                        except Exception:
                            restart_iter_int = 0
                        if not isinstance(retry_node_cp, dict):
                            retry_node_cp = {}
                        retry_node_cp["last_completed_iter"] = restart_iter_int
                        # OSAgent.restore_from_checkpoint prefers replay_core.last_completed_iter
                        # over top-level last_completed_iter, so we must align both.
                        core = retry_node_cp.get("replay_core")
                        if isinstance(core, dict):
                            core["last_completed_iter"] = restart_iter_int
                        if restart_iter_int <= 0:
                            retry_node_cp["action_history_prefix"] = []
                            retry_node_cp["summary_history_prefix"] = []
                            payload = retry_node_cp.get("decision_payload")
                            if isinstance(payload, dict):
                                payload["action_tail"] = []
                                payload["summary_tail"] = []
                        else:
                            ah = retry_node_cp.get("action_history_prefix")
                            if isinstance(ah, list):
                                retry_node_cp["action_history_prefix"] = ah[:restart_iter_int]
                            sh = retry_node_cp.get("summary_history_prefix")
                            if isinstance(sh, list):
                                retry_node_cp["summary_history_prefix"] = sh[:restart_iter_int]
                        retry_node_profile_src = ""
                        retry_node_profile_bak = ""
                        if len(selected_plans) > 1 and self._user_data_dir and Path(self._user_data_dir).exists():
                            retry_node_profile_bak = self._user_data_dir + "_branching_bak"
                            try:
                                if Path(retry_node_profile_bak).exists():
                                    shutil.rmtree(retry_node_profile_bak, ignore_errors=True)
                                shutil.copytree(self._user_data_dir, retry_node_profile_bak)
                                retry_node_profile_src = retry_node_profile_bak
                                logger.info(f"[branching] Saved retry-node browser profile → {retry_node_profile_bak}")
                            except Exception as _pe:
                                logger.warning(f"[branching] Profile snapshot failed: {_pe}")
                        if len(selected_plans) > 1 and not retry_node_profile_src:
                            logger.warning(
                                "[branching] Cannot snapshot browser profile for multi-branch restore; "
                                "will still execute remaining selected plans without profile-restore shortcut"
                            )

                        resume_anchor_cp = resume_cp or resume_cp_for_branching or {}
                        retry_node_url = resume_anchor_cp.get("resume_from_url", "") or start_func
                        retry_node_scroll = resume_anchor_cp.get("scroll", {})

                        # Step D: execute each selected plan sequentially, stop on first success
                        result_dict = {}
                        branch_results = []
                        for branch_idx, plan in enumerate(selected_plans):
                            if branch_idx > 0:
                                # Restore browser + agent state to retry node
                                logger.info(f"[branching] Restoring retry-node state for branch {branch_idx}...")
                                await self._cleanup_environment(is_web)
                                await asyncio.sleep(SLEEP_AFTER_CLEANUP)
                                if retry_node_profile_src and Path(retry_node_profile_src).exists() and self._user_data_dir:
                                    try:
                                        if Path(self._user_data_dir).exists():
                                            shutil.rmtree(self._user_data_dir, ignore_errors=True)
                                        shutil.copytree(retry_node_profile_src, self._user_data_dir)
                                        logger.info(f"[branching] Profile restored for branch {branch_idx}")
                                    except Exception as _rpe:
                                        logger.warning(f"[branching] Profile restore failed: {_rpe}")
                                await self._start_environment(
                                    url=retry_node_url if is_web else None,
                                    work_path=retry_node_url if not is_web else None,
                                )
                                await asyncio.sleep(SLEEP_BEFORE_EXECUTE)
                                if retry_node_scroll.get("y") or retry_node_scroll.get("x"):
                                    try:
                                        self.osagent.controller.set_scroll(
                                            x=int(retry_node_scroll.get("x", 0)),
                                            y=int(retry_node_scroll.get("y", 0)),
                                        )
                                    except Exception:
                                        pass
                                # Restore osagent context
                                self.osagent.rc.restore_from_checkpoint(retry_node_cp)
                                logger.info(f"[branching] Agent context restored for branch {branch_idx}")

                            # Inject selected plan into case_desc
                            case_branched = copy.deepcopy(case_info_for_run)
                            case_branched["case_desc"] = (
                                str(case_branched.get("case_desc", "") or "")
                                + f"\n\n[Retry Plan for this attempt]\n{plan}"
                            )
                            branch_case = {case_id: case_branched}
                            # Put each selected plan execution under an explicit branch folder:
                            #   .../<case>/0/<timestamp>  (first selected plan)
                            #   .../<case>/1/<timestamp>  (second selected plan)
                            #   .../<case>/2/<timestamp>  (third selected plan)
                            # This makes per-plan logs easy to inspect.
                            self.osagent.log_dirs = f"{case_log_root}/{branch_idx}"
                            logger.info(f"[branching] Branch {branch_idx} log root: {self.osagent.log_dirs}")
                            # Re-lock timestamped paths for this branch
                            self.osagent._get_timestamped_paths()
                            self.osagent._lock_timestamped_paths = True

                            logger.info(f"[branching] Executing branch {branch_idx + 1}/{len(selected_plans)}...")
                            # Keep soft_reset=True for every branch, otherwise osagent.run() hard-reset
                            # will wipe the restored retry-node context before execution.
                            branch_result = await self.execute_api_check(
                                task_name, 1, branch_case, soft_reset=True
                            )
                            result_dict = branch_result
                            # Keep branch checkpoint storage aligned with restart policy:
                            # - restart_from_iter == 0: remove all per-step checkpoints
                            # - restart_from_iter > 0: keep only the recommended step if present
                            try:
                                _prune_checkpoints_keep_only_restart(
                                    checkpoint_dir_path=str(getattr(self.osagent, "save_img", "") or ""),
                                    restart_from_iter=restart_iter,
                                )
                            except Exception as _prune_exc:
                                logger.warning(f"[branching] Failed to prune branch checkpoints: {_prune_exc}")

                            # Early stop: task succeeded
                            matched = self._find_matching_key(case_id, branch_result)
                            branch_ok = False
                            if matched is not None:
                                matched_payload = branch_result.get(matched)
                                if not isinstance(matched_payload, dict):
                                    matched_payload = {}
                                rv = matched_payload.get("result", "")
                                branch_ok = bool(str(rv).strip().lower() in ("true", "pass", "1", "yes", "y"))
                            branch_results.append(
                                {
                                    "order": branch_idx,
                                    "selected_from_candidate_idx": selected_indices[branch_idx]
                                    if branch_idx < len(selected_indices)
                                    else -1,
                                    "dimension": _extract_plan_dimension(plan),
                                    "plan": plan,
                                    "select_reason": selected_reasons[branch_idx] if branch_idx < len(selected_reasons) else "",
                                    "result": bool(branch_ok),
                                }
                            )
                            # Save per-branch result json to {case_log_root}/{branch_idx}/test_case.json
                            try:
                                branch_dir = Path(case_log_root) / str(branch_idx)
                                branch_dir.mkdir(parents=True, exist_ok=True)
                                _branch_payload = matched_payload if matched is not None else {}
                                _branch_case_data = test_cases.get(case_id, {})
                                _branch_item = {
                                    "test_id": f"{case_name_for_log}{case_id}" if case_name_for_log else str(case_id),
                                    "case_desc": _branch_case_data.get("case_desc", ""),
                                    "evidence": _branch_payload.get("evidence", ""),
                                    "result": bool(branch_ok),
                                    "plan": plan,
                                }
                                _branch_json_path = branch_dir / "test_case.json"
                                with open(_branch_json_path, "w", encoding="utf-8") as _bf:
                                    json.dump({"test_cases": [_branch_item]}, _bf, indent=4, ensure_ascii=False)
                                logger.info(f"[branching] Saved branch {branch_idx} result to {_branch_json_path}")
                            except Exception as _bje:
                                logger.warning(f"[branching] Failed to save branch {branch_idx} result json: {_bje}")
                            logger.info(f"[branching] Branch {branch_idx + 1} result: {'SUCCESS' if branch_ok else 'fail'}")
                            if branch_ok:
                                logger.info(f"[branching] Early stop after branch {branch_idx + 1}")
                                break

                        # Clean up profile backup
                        if retry_node_profile_bak and Path(retry_node_profile_bak).exists():
                            try:
                                shutil.rmtree(retry_node_profile_bak, ignore_errors=True)
                            except Exception:
                                pass
                        _save_retry_plans_json(
                            case_id=str(case_id),
                            payload={
                                "case_id": str(case_id),
                                "branching_n_candidates": int(branching_n),
                                "branching_k": int(branching_k),
                                "task_desc": task_desc,
                                "failure_context": {
                                    "fail_reason": fail_reason,
                                    "retry_reason": retry_reason,
                                    "restart_explanation": restart_explanation,
                                    "restart_from_iter": restart_iter,
                                },
                                "trajectory_tail": traj_tail,
                                "candidate_plans": [
                                    {
                                        "idx": i,
                                        "dimension": str(candidate_plan_items[i].get("dimension", "")),
                                        "title": str(candidate_plan_items[i].get("title", "")),
                                        "plan": p,
                                        "reason": str(candidate_plan_items[i].get("reason", "")),
                                    }
                                    for i, p in enumerate(plans)
                                ],
                                "selected_plan_indices": selected_indices,
                                "selected_plans": [
                                    {
                                        "order": i,
                                        "dimension": _extract_plan_dimension(p),
                                        "plan": p,
                                        "reason": selected_reasons[i] if i < len(selected_reasons) else "",
                                    }
                                    for i, p in enumerate(selected_plans)
                                ],
                                "branch_execution": branch_results,
                                "stopped_early_on_success": any(bool(x.get("result")) for x in branch_results),
                                "status": "execution_finished",
                            },
                        )

                    else:
                        # Normal single-run (no branching or branching disabled)
                        single_case = {case_id: case_info_for_run}
                        result_dict = await self.execute_api_check(task_name, 1, single_case, soft_reset=use_resume_this_case)

                    elapsed = time.perf_counter() - t0
                    cost_after = _get_llm_total_usd()
                    delta_usd = max(0.0, cost_after - cost_before)
                    cost_before = cost_after

                    # Unlock after execution
                    self.osagent._lock_timestamped_paths = False

                    # Merge result and set per-case cost (time=...s, usd=$... for server compatibility)
                    test_cases[case_id]["cost"] = f"time={elapsed:.1f}s, usd=${delta_usd:.6f}"
                    matched_key = self._find_matching_key(case_id, result_dict)
                    if matched_key is not None:
                        matched_payload = result_dict.get(matched_key)
                        if not isinstance(matched_payload, dict):
                            matched_payload = {}
                        all_results[case_id] = matched_payload
                        test_cases[case_id].update(
                            {"result": matched_payload.get("result", ""), "evidence": matched_payload.get("evidence", "")}
                        )
                except Exception as case_err:
                    logger.error(f"Case {case_id} failed with error: {case_err}")
                    self.osagent._lock_timestamped_paths = False
                    elapsed = time.perf_counter() - t0
                    test_cases[case_id]["cost"] = f"time={elapsed:.1f}s, usd=$0.000000"
                    test_cases[case_id]["result"] = False
                    test_cases[case_id]["evidence"] = f"Error: {case_err}"

                # Persist each case immediately so interrupted runs still have partial JSON outputs.
                _save_incremental_case_json(case_id, test_cases[case_id])

                # After last case: optionally write resume checkpoint for the next round.
                if save_checkpoint and idx == len(test_cases):
                    log_base = self.rc.agent_params.get("log_dirs", "work_dirs")
                    checkpoint_dir = str(Path(log_base) / log_dir)
                    _save_resume_checkpoint(
                        osagent=self.osagent,
                        save_dir=checkpoint_dir,
                        chrome_profile_src=chrome_profile_src,
                        run_id=getattr(self, "_run_id", ""),
                        worker_id=str(getattr(self, "_worker_id", "")),
                        remote_debugging_port=self._remote_debugging_port,
                    )
                    self._last_checkpoint_path = str(Path(checkpoint_dir) / "resume_checkpoint.json")
                    # On first-round failure, SupervisorJudge recommends restart step for round-2
                    last_failed = not _to_bool_result(test_cases.get(case_id, {}).get("result", ""))
                    if last_failed and self._last_checkpoint_path:
                        try:
                            with open(self._last_checkpoint_path, "r", encoding="utf-8") as f:
                                cp = json.load(f)
                            config_file = str(getattr(self, "_config_file", "") or "")
                            case_desc = test_cases.get(case_id, {}).get("case_desc", "")
                            rec = await supervisor_analyze_trajectory_async(
                                checkpoint_dict=cp,
                                config_path=config_file,
                                task_desc=case_desc,
                            )
                            if rec:
                                cp["restart_recommendation"] = rec
                                with open(self._last_checkpoint_path, "w", encoding="utf-8") as f:
                                    json.dump(cp, f, indent=2, ensure_ascii=False)
                                logger.info(f"Wrote restart_recommendation into checkpoint: {rec}")
                                save_img = getattr(self.osagent, "save_img", "")
                                if save_img:
                                    ts_path = Path(save_img) / "resume_checkpoint.json"
                                    if ts_path.parent.exists():
                                        with open(ts_path, "w", encoding="utf-8") as f:
                                            json.dump(cp, f, indent=2, ensure_ascii=False)
                                        logger.debug(f"Synced restart_recommendation to {ts_path}")
                                # 只保留大模型推荐的 retry 节点，删除其他 step checkpoint
                                _prune_checkpoints_keep_only_restart(
                                    str(Path(self._last_checkpoint_path).parent),
                                    rec.get("restart_from_iter"),
                                )
                        except Exception as e:
                            logger.warning(f"Failed to add restart_recommendation: {e}")

                # Reset in-memory agent state for next case; browser session is rebuilt at case start.
                if idx < len(test_cases):
                    logger.info("Resetting osagent state for next case...")
                    self.osagent.rc.reset()

            # Restore base log directory
            # If only one case is executed in sequential API mode, keep log root at that case folder
            # so uncertain retry logs go under the same case (e.g. .../3D Showcase6/retry_0/...).
            if case_name_for_log is not None and len(test_cases) == 1:
                only_case_id = next(iter(test_cases.keys()))
                self.osagent.log_dirs = f"{base_log_dir}/{case_name_for_log}{only_case_id}"
            else:
                self.osagent.log_dirs = base_log_dir
        else:
            # Batch mode: execute all test cases at once (original behavior)
            logger.info("Start executing automated testing...")
            result_dict = await self.execute_api_check(task_name, len(test_cases), test_cases)

            # Merge results
            for key, value in result_dict.items():
                matched_key = self._find_matching_key(key, test_cases)
                if matched_key is not None:
                    test_cases[matched_key].update({"result": value.get(
                        "result", ""), "evidence": value.get("evidence", "")})

            # After batch run: save checkpoint and restart_recommendation when save_checkpoint=True
            # (first-round run uses sequential_mode=False, so this is the only path that writes them)
            if save_checkpoint and test_cases:
                log_base = self.rc.agent_params.get("log_dirs", "work_dirs")
                checkpoint_dir = str(Path(log_base) / log_dir)
                _save_resume_checkpoint(
                    osagent=self.osagent,
                    save_dir=checkpoint_dir,
                    chrome_profile_src=chrome_profile_src,
                    run_id=getattr(self, "_run_id", ""),
                    worker_id=str(getattr(self, "_worker_id", "")),
                    remote_debugging_port=self._remote_debugging_port,
                )
                self._last_checkpoint_path = str(Path(checkpoint_dir) / "resume_checkpoint.json")
                last_case_id = next(reversed(test_cases))
                last_failed = not _to_bool_result(test_cases.get(last_case_id, {}).get("result", ""))
                if last_failed and self._last_checkpoint_path:
                    try:
                        with open(self._last_checkpoint_path, "r", encoding="utf-8") as f:
                            cp = json.load(f)
                        config_file = str(getattr(self, "_config_file", "") or "")
                        last_case_desc = test_cases.get(last_case_id, {}).get("case_desc", "")
                        rec = await supervisor_analyze_trajectory_async(
                            checkpoint_dict=cp,
                            config_path=config_file,
                            task_desc=last_case_desc,
                        )
                        if rec:
                            cp["restart_recommendation"] = rec
                            with open(self._last_checkpoint_path, "w", encoding="utf-8") as f:
                                json.dump(cp, f, indent=2, ensure_ascii=False)
                            logger.info(f"Wrote restart_recommendation into checkpoint: {rec}")
                            save_img = getattr(self.osagent, "save_img", "")
                            if save_img:
                                ts_path = Path(save_img) / "resume_checkpoint.json"
                                if ts_path.parent.exists():
                                    with open(ts_path, "w", encoding="utf-8") as f:
                                        json.dump(cp, f, indent=2, ensure_ascii=False)
                                    logger.debug(f"Synced restart_recommendation to {ts_path}")
                            # 只保留大模型推荐的 retry 节点，删除其他 step checkpoint
                            _prune_checkpoints_keep_only_restart(
                                str(Path(self._last_checkpoint_path).parent),
                                rec.get("restart_from_iter"),
                            )
                    except Exception as e:
                        logger.warning(f"Failed to add restart_recommendation: {e}")

        result = {task_name: {"test_cases": test_cases}}

        # Cleanup after initial test
        logger.info("Cleaning up environment after initial test run...")
        await self._cleanup_environment(is_web)
        await asyncio.sleep(SLEEP_AFTER_CLEANUP)

        # Retry uncertain cases
        result = await self._retry_uncertain_cases(result, max_retry_uncertain, task_name=task_name, start_func=start_func)

        final_test_cases = result[task_name]["test_cases"]

        # Compute cost from LLM instances and inject into each case
        try:
            total_cost = 0.0
            prompt_tokens = completion_tokens = 0
            for llm_source in (
                getattr(self.test_generator, "llm", None),
                getattr(self.osagent, "llm", None),
            ):
                if llm_source and hasattr(llm_source, "get_costs"):
                    c = llm_source.get_costs()
                    prompt_tokens += getattr(c, "total_prompt_tokens", 0) or 0
                    completion_tokens += getattr(c, "total_completion_tokens", 0) or 0
                    total_cost += getattr(c, "total_cost", 0) or 0
            cost_str = f"${total_cost:.4f} (prompt:{prompt_tokens}, completion:{completion_tokens})"
            for case_data in final_test_cases.values():
                existing = str(case_data.get("cost", ""))
                # Preserve per-case cost from sequential mode (format: "time=...s, usd=$...")
                if "time=" in existing and "usd=" in existing:
                    continue
                case_data["cost"] = cost_str
        except Exception:
            pass

        # Save to file if needed
        if save_to_file:
            log_base = self.rc.agent_params.get("log_dirs", "work_dirs")
            if case_name_for_log is not None:
                # 每个任务一个目录（3D Showcase0, 3D Showcase1, ...），各写一份 test_case.json
                for case_id, case_data in final_test_cases.items():
                    case_dir = Path(log_base) / log_dir / f"{case_name_for_log}{case_id}"
                    case_dir.mkdir(parents=True, exist_ok=True)
                    # Prefer timestamp subdir so json lands at .../{case}/{YYYYMMDDHHMM}/test_case.json
                    ts_dirs = [
                        p for p in case_dir.iterdir()
                        if p.is_dir() and p.name.isdigit() and len(p.name) >= 12
                    ]
                    output_dir = sorted(ts_dirs, key=lambda p: p.name)[-1] if ts_dirs else case_dir
                    output_file = output_dir / "test_case.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {"test_cases": [_build_case_item(case_id, case_data, case_name_for_log)]},
                            f,
                            indent=4,
                            ensure_ascii=False,
                        )
                    logger.info(f"Results saved to {output_file}")
            else:
                output_dir = Path(log_base) / log_dir / task_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{Path(task_name).name}.json"
                case_items = [
                    _build_case_item(case_id, case_data)
                    for case_id, case_data in final_test_cases.items()
                ]
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump({"test_cases": case_items}, f, indent=4, ensure_ascii=False)
                logger.info(f"Results saved to {output_file}")

        # Execute executability check (skip if no screenshot available, e.g. TextAgent mode)
        image_path = getattr(self.osagent, 'output_image_path', '')
        if image_path and Path(image_path).is_file():
            executability = await self.test_generator.generate_executability(result, image_path)
        else:
            executability = None
            logger.info("Skipping executability check (no screenshot available)")

        return final_test_cases, executability

    async def run_api(
        self,
        task_name: str,
        test_cases: dict,
        start_func: str,
        log_dir: str,
        max_retry_uncertain: int = 1,
        sequential_mode: bool = False,
        case_name_for_log: Optional[str] = None,
        resume_checkpoint_path: str = "",
        save_checkpoint: bool = False,
        chrome_profile_src: str = "",
    ) -> tuple[dict, bool]:
        """Run API testing with retry mechanism for uncertain results

        Args:
            task_name: Task identifier
            test_cases: Dictionary of test cases to execute
            start_func: URL or work path to start the environment
            log_dir: Directory for logs
            max_retry_uncertain: Maximum retries for uncertain cases
            sequential_mode: If True, execute test cases one by one without browser cleanup between cases,
                           only reset osagent state. If False, execute all test cases at once (default).
            case_name_for_log: If set, log subdirs become {case_name_for_log}0, {case_name_for_log}1, ...
            resume_checkpoint_path: Path to a previously saved resume_checkpoint.json.
                If provided, the first case is started from the checkpoint URL / state
                (resume_mode).  Subsequent cases use normal baseline behaviour.
            save_checkpoint: If True, write a resume_checkpoint.json after the last
                sequential case so the caller can resume in a later round.
            chrome_profile_src: live user_data_dir path; used when save_checkpoint=True
                to copy the profile alongside the checkpoint.
        """
        try:
            final_test_cases, executability = await self._run_test_with_retry(
                task_name=task_name,
                test_cases=test_cases,
                start_func=start_func,
                log_dir=log_dir,
                max_retry_uncertain=max_retry_uncertain,
                save_to_file=True,
                sequential_mode=sequential_mode,
                case_name_for_log=case_name_for_log,
                resume_checkpoint_path=resume_checkpoint_path,
                save_checkpoint=save_checkpoint,
                chrome_profile_src=chrome_profile_src,
            )
            logger.info("Test process completed")
            return final_test_cases, executability
        except Exception as e:
            logger.error(f"Error occurred during test execution: {str(e)}")
            await self._cleanup_environment(is_web=True)
            raise

    async def run_single(
        self,
        case_name: str,
        url: str,
        work_path: str,
        user_requirement: str,
        json_path: str = "data/temp.json",
        use_json_only: bool = False,
        max_retry_uncertain: int = 1,
        sequential_mode: bool = False,
    ) -> tuple[dict, bool]:
        """Execute single test case with retry mechanism for uncertain results

        Args:
            case_name: Test case name
            url: Test target URL
            work_path: Work path for local application
            user_requirement: Requirement description
            json_path: Output JSON file path
            use_json_only: Whether to only use JSON files
            max_retry_uncertain: Maximum retries for uncertain cases
            sequential_mode: If True, execute test cases one by one without browser cleanup between cases,
                           only reset osagent state. If False, execute all test cases at once (default).
        """
        # Generate test cases if needed
        if not use_json_only:
            logger.info(
                f"Start generating automated test cases for '{case_name}'...")
            generated_cases = await self.test_generator.generate_test_cases(user_requirement)
            logger.info("Start converting to JSON format...")
            make_json_single(case_name, url, generated_cases,
                             json_path, work_path)

        # Read and validate JSON
        self.rc.json_file = json_path
        test_data = read_json_file(json_path)

        if case_name not in test_data:
            raise ValueError(
                f"Case '{case_name}' not found in JSON file {json_path}")

        task_info = test_data[case_name]
        test_cases = task_info.get("test_cases", {})

        # Determine start function
        start_func = (task_info.get("url") or url or task_info.get(
            "work_path") or work_path or "").strip()
        if not start_func:
            raise ValueError(
                "No valid url or work_path provided for single test execution")

        # Execute tests with retry
        final_test_cases, executability = await self._run_test_with_retry(
            task_name=case_name,
            test_cases=test_cases,
            start_func=start_func,
            log_dir=f"single/{Path(json_path).stem}",
            max_retry_uncertain=max_retry_uncertain,
            save_to_file=False,
            sequential_mode=sequential_mode,
        )

        # Update and save results
        task_info["test_cases"] = final_test_cases
        test_data[case_name] = task_info
        write_json_file(json_path, test_data, indent=4)

        logger.info(f"Test process completed for case '{case_name}'")
        return test_data, executability

    async def run_batch(
        self,
        project_excel_path: str = None,
        case_excel_path: str = None,
        batch_mode: str = "standard",
        generate_case_only: bool = False,
        max_retry_uncertain: int = 1,
        sequential_mode: bool = False,
    ) -> Union[tuple[dict, bool], Any, None]:
        """Run batch testing (unified for both standard and mini modes)

        Complete testing process includes:
        1. Generate test cases from Excel
        2. Convert to JSON format
        3. Execute automated testing
        4. Retry uncertain cases (default enabled)
        5. (Optional) Output results to Excel
        6. (Optional) Only generate test cases

        Args:
            project_excel_path: Project level Excel file path
            case_excel_path: Case level Excel file path (optional)
            batch_mode: Batch mode - "standard" or "mini" (default: "standard")
            generate_case_only: Whether to only generate test cases (only for mini mode)
            max_retry_uncertain: Maximum retry times for uncertain cases (default: 1)
            sequential_mode: If True, execute test cases one by one without browser cleanup between cases,
                           only reset osagent state. If False, execute all test cases at once (default).

        Returns:
            - For standard mode: tuple[dict, bool] (result dict and executability)
            - For mini mode with generate_case_only: Case result
            - For mini mode without generate_case_only: None
        """
        try:
            # Select converters based on mode
            is_mini = batch_mode == "mini"
            operation_type = OperationType.GENERATE_CASES_MINI_BATCH if is_mini else OperationType.GENERATE_CASES
            json_converter = mini_list_to_json if is_mini else list_to_json
            excel_converter = mini_list_to_excel if is_mini else convert_json_to_excel

            # Prepare test cases
            case_result = await self._prepare_batch_test_cases(project_excel_path, operation_type, json_converter)
            
            # Return early if only generating cases (mini mode only)
            if generate_case_only:
                if not is_mini:
                    raise ValueError(
                        "generate_case_only is only supported for mini batch mode.")
                return case_result

            # Execute tests with retry support
            logger.info("Start executing automated testing...")
            test_cases = read_json_file(self.rc.json_file)
            await self._execute_task_batch(test_cases, max_retry_uncertain=max_retry_uncertain, sequential_mode=sequential_mode)
            write_json_file(self.rc.json_file, test_cases, indent=4)

            # Output results to Excel
            if case_excel_path:
                logger.info("Start generating result spreadsheet...")
                excel_converter(self.rc.json_file, case_excel_path)

            # Update project Excel with iteration counts
            logger.info("Updating project Excel with iteration counts...")
            update_project_excel_iters(project_excel_path, self.rc.json_file)

            logger.info("Test process completed")
            return ({}, None) if batch_mode == "standard" else None

        except Exception as e:
            logger.error(f"Error occurred during test execution: {str(e)}")
            await self._cleanup_environment(is_web=True)
            raise

    async def run_mini_batch(
        self,
        project_excel_path: str = None,
        case_excel_path: str = None,
        generate_case_only: bool = False,
        max_retry_uncertain: int = 1,
        sequential_mode: bool = False,
    ) -> Optional[Any]:
        """Deprecated: Use run_batch(batch_mode='mini') instead"""
        return await self.run_batch(
            project_excel_path=project_excel_path,
            case_excel_path=case_excel_path,
            batch_mode="mini",
            generate_case_only=generate_case_only,
            max_retry_uncertain=max_retry_uncertain,
            sequential_mode=sequential_mode,
        )

    async def run(self, **kwargs) -> Union[tuple[dict, bool], dict, Exception]:
        """Run automated testing

        Supports two calling methods:
        1. Batch testing: run(project_excel_path="xxx.xlsx", case_excel_path="xxx.xlsx", use_json_only=False)
        2. Single test: run(case_name="xxx", url="xxx", user_requirement="xxx")

        Args:
            **kwargs: Parameters
                Batch testing:
                    - project_excel_path: Project level Excel file path
                    - case_excel_path: Case level Excel file path (optional)
                    - use_json_only: Whether to only use JSON files (optional)
                    - sequential_mode: If True, execute test cases one by one (optional)
                Single test:
                    - case_name: Test case name
                    - url: Test target URL
                    - user_requirement: Requirement description
                    - json_path: Output JSON file path (optional)
                    - sequential_mode: If True, execute test cases one by one (optional)
        """
        try:
            if kwargs.get("case_name") and kwargs.get("user_requirement"):
                # Single test scenario
                return await self.run_single(
                    case_name=kwargs["case_name"],
                    url=kwargs.get("url", ""),
                    work_path=kwargs.get("work_path", ""),
                    user_requirement=kwargs["user_requirement"],
                    json_path=kwargs.get("json_path", "data/temp.json"),
                    use_json_only=kwargs.get("use_json_only", False),
                    sequential_mode=kwargs.get("sequential_mode", False),
                )
            else:
                # Batch test scenario
                return await self.run_batch(
                    kwargs.get("project_excel_path"),
                    kwargs.get("case_excel_path"),
                    sequential_mode=kwargs.get("sequential_mode", False),
                )
        except Exception as e:
            logger.error(f"Test execution failed: {str(e)}")
            logger.exception("Detailed error information")
            return e
