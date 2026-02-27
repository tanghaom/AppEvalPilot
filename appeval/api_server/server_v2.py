#!/usr/bin/env python3
"""
精简版 API 服务层 —— 只做三件事：
  1. 接收并解析请求
  2. 将请求存为本地 task JSON
  3. 启动 run_test.py 执行 → 读结果 → 回调

执行:
  cd /data/zhijieliu/AppEvalPilot
  conda activate appeval
  no_proxy='*' python -m appeval.api_server.server_v2 --port 8888
"""
import argparse
import asyncio
from datetime import datetime
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, model_validator

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
API_LOG_BASE = PROJECT_DIR / "appeval" / "api_log"
TASK_JSON_DIR = API_LOG_BASE / "tasks"
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
CALLBACK_BASE_URL = "https://test-tool.deepwisdomai.com/api/v1/agent-eval/notice/detail"


# ---------------------------------------------------------------------------
#  Pydantic models (与原 server.py 完全一致)
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    task_id: Union[int, str]
    detail_id: Union[int, str]
    case_name: str
    test_arry: List[str]
    prod_url: str

    @model_validator(mode="before")
    @classmethod
    def accept_test_array(cls, data):
        if isinstance(data, dict) and "test_array" in data and "test_arry" not in data:
            data = {**data, "test_arry": data["test_array"]}
        return data


class BatchRequest(BaseModel):
    tasks: List[StartRequest]
    callback_url: Optional[str] = None


class StartResponse(BaseModel):
    code: int = 0
    message: str = "ok"
    data: dict = {"success": True}


# ---------------------------------------------------------------------------
#  App & global state
# ---------------------------------------------------------------------------

app = FastAPI(title="AppEval API v2", version="2.0")
_task_status: Dict[int, dict] = {}
_executor: Optional[ProcessPoolExecutor] = None
_full_cfg: dict = {}
_config_path: str = ""
_case_counter: Dict[str, int] = {}
_counter_lock = asyncio.Lock()
_worker_slots: Optional[asyncio.Queue] = None


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> tuple:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    workers = int(cfg.get("workers", 5))
    return workers, cfg


# ---------------------------------------------------------------------------
#  Task JSON 持久化
# ---------------------------------------------------------------------------

def _save_task_json(req: StartRequest, start_index: int, run_group_ts: str, callback_url: str = "") -> str:
    """Save API request to a local JSON file. Returns the file path."""
    TASK_JSON_DIR.mkdir(parents=True, exist_ok=True)
    task_data = {
        "task_id": req.task_id,
        "detail_id": req.detail_id,
        "case_name": req.case_name,
        "prod_url": req.prod_url,
        "test_arry": req.test_arry,
        "start_index": start_index,
        "run_group_ts": run_group_ts,
        "callback_url": callback_url,
    }
    filename = f"{req.task_id}_{req.detail_id}_{run_group_ts}.json"
    filepath = TASK_JSON_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)
    logger.info(f"[Server] Task JSON saved → {filepath}")
    return str(filepath)


# ---------------------------------------------------------------------------
#  Callback helpers (从原 server.py 保留)
# ---------------------------------------------------------------------------

def _should_skip_callback() -> bool:
    return bool(_full_cfg.get("skip_callback", False))


def _parse_cost_to_seconds(cost_str: str) -> int:
    if not cost_str or not isinstance(cost_str, str):
        return 0
    m = re.search(r"(?:avg_)?time[=:]?\s*([\d.]+)\s*s", cost_str, re.I)
    if m:
        try:
            return int(float(m.group(1)))
        except (ValueError, TypeError):
            pass
    return 0


def _transform_result_for_callback(result: dict) -> dict:
    out = dict(result)
    out["eval_cost"] = _parse_cost_to_seconds(str(out.get("eval_cost", "")))
    did = result.get("detail_id")
    try:
        out["detail_id"] = int(did) if did is not None and str(did).strip() != "" else did
    except (TypeError, ValueError):
        out["detail_id"] = did
    cases = []
    for i, tc in enumerate(out.get("test_cases", [])):
        c = dict(tc)
        tid = tc.get("test_id", i)
        if isinstance(tid, str):
            nums = re.findall(r"\d+", tid)
            raw = int(nums[0]) if nums else i
        else:
            try:
                raw = int(tid)
            except (TypeError, ValueError):
                raw = i
        c["test_id"] = max(1, raw)
        c["cost"] = _parse_cost_to_seconds(str(tc.get("cost", "")))
        cases.append(c)
    out["test_cases"] = cases
    return out


def _callback_url_with_detail_id(base_url: str, detail_id) -> str:
    base_url = base_url.strip().rstrip("/")
    base_url = re.sub(r"/\d+$", "", base_url)
    return f"{base_url}/{detail_id}"


async def _callback_one(callback_url: str, result: dict):
    detail_id = result.get("detail_id", "")
    if _should_skip_callback():
        logger.info(f"[Callback] detail_id={detail_id} 已跳过（本地测试不回调）")
        return
    retry_times = max(1, int(_full_cfg.get("callback_retry_times", 3) or 3))
    retry_interval_sec = max(1, int(_full_cfg.get("callback_retry_interval_sec", 300) or 300))
    url = _callback_url_with_detail_id(callback_url, detail_id)
    payload = _transform_result_for_callback(result)
    logger.info(f"[Callback] POST {url} | body 前 300 字: {json.dumps(payload, ensure_ascii=False)[:300]}")

    for attempt in range(1, retry_times + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0), trust_env=False) as c:
                resp = await c.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次失败: HTTP {resp.status_code}")
            else:
                callback_ok = False
                try:
                    rj = resp.json()
                    if isinstance(rj, dict):
                        data = rj.get("data", {})
                        callback_ok = data.get("success") is True or rj.get("ok") is True or rj.get("success") is True
                except Exception:
                    pass
                if callback_ok:
                    logger.info(f"[Callback] detail_id={detail_id} 成功: HTTP {resp.status_code}")
                    return
                logger.warning(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次 data.success!=true")
        except Exception as e:
            logger.error(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次异常: {type(e).__name__} {e}")
        if attempt < retry_times:
            await asyncio.sleep(retry_interval_sec)
    logger.error(f"[Callback] detail_id={detail_id} 回调最终失败")


# ---------------------------------------------------------------------------
#  Dispatch: 存 JSON → 调 run_test → 回调
# ---------------------------------------------------------------------------

async def _dispatch(req: StartRequest, start_index: int, run_group_ts: str):
    """Save task JSON, execute via process pool. Callback done inside worker process."""
    _task_status[req.detail_id] = {"status": "running", "case_name": req.case_name}

    cb_url = (_full_cfg.get("callback_base_url") or CALLBACK_BASE_URL or "").strip().rstrip("/")
    task_json_path = _save_task_json(req, start_index, run_group_ts, callback_url=cb_url)

    worker_id = await _worker_slots.get()
    try:
        logger.info(
            f"[Server] 提交任务 detail_id={req.detail_id} worker_id={worker_id} "
            f"start_index={start_index} task_json={task_json_path}"
        )
        loop = asyncio.get_event_loop()
        args_tuple = (task_json_path, _config_path, worker_id)

        from appeval.run_test import run_single_task_wrapper
        result = await loop.run_in_executor(_executor, run_single_task_wrapper, args_tuple)
        _task_status[req.detail_id] = {"status": "done", "result": result}
        logger.info(f"[Server] detail_id={req.detail_id} 完成（回调已在子进程内执行）")
    except Exception as e:
        _task_status[req.detail_id] = {"status": "error", "error": str(e)}
        logger.error(f"[Server] detail_id={req.detail_id} 子进程异常: {e}（回调已在子进程内执行或进程被杀）")
    finally:
        _worker_slots.put_nowait(worker_id)


# ---------------------------------------------------------------------------
#  Batch dispatch
# ---------------------------------------------------------------------------

async def _run_batch(req: BatchRequest, start_index_map: Dict[int, int], callback_url: str, run_group_ts: str):
    """回调在子进程内完成，主进程仅更新状态。"""
    loop = asyncio.get_event_loop()

    async def _run_one(t: StartRequest):
        start_index = start_index_map[t.detail_id]
        _task_status[t.detail_id] = {"status": "running", "case_name": t.case_name}

        task_json_path = _save_task_json(t, start_index, run_group_ts, callback_url=callback_url)

        worker_id = await _worker_slots.get()
        try:
            logger.info(f"[Server] /batch detail_id={t.detail_id} worker_id={worker_id} 开始执行")
            args_tuple = (task_json_path, _config_path, worker_id)
            from appeval.run_test import run_single_task_wrapper
            result = await loop.run_in_executor(_executor, run_single_task_wrapper, args_tuple)
            _task_status[t.detail_id] = {"status": "done", "result": result}
            logger.info(f"[Server] /batch detail_id={t.detail_id} 完成（回调已在子进程内执行）")
        except Exception as e:
            _task_status[t.detail_id] = {"status": "error", "error": str(e)}
            logger.error(f"[Server] /batch detail_id={t.detail_id} 子进程异常: {e}（回调已在子进程内执行或进程被杀）")
        finally:
            _worker_slots.put_nowait(worker_id)

    await asyncio.gather(*[_run_one(t) for t in req.tasks])
    logger.info(f"[Server] /batch 全部 {len(req.tasks)} 个任务已完成")


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

@app.post("/start", response_model=StartResponse)
async def start_task(req: StartRequest):
    logger.info(f"[API] /start task_id={req.task_id} detail_id={req.detail_id} case={req.case_name} tests={len(req.test_arry)}")
    if _task_status.get(req.detail_id, {}).get("status") == "running":
        return StartResponse(code=1, message="任务执行中", data={"success": False})

    counter_key = f"{req.task_id}/{req.case_name}"
    async with _counter_lock:
        start_index = _case_counter.get(counter_key, 0)
        _case_counter[counter_key] = start_index + len(req.test_arry)
    run_group_ts = datetime.now().strftime("%Y%m%d%H%M")

    asyncio.create_task(_dispatch(req, start_index, run_group_ts))
    return StartResponse()


@app.post("/batch")
async def batch_tasks(request: Request):
    """接收一批任务（JSON body / form-data / query），接单后台执行，完成后逐个回调。"""
    req = await _parse_batch_request(request)
    if not req or not req.tasks:
        return {"code": 1, "message": "无法解析请求", "data": {"success": False}}

    running_ids = [t.detail_id for t in req.tasks if _task_status.get(t.detail_id, {}).get("status") == "running"]
    if running_ids:
        return {"code": 1, "message": f"任务执行中: {running_ids}", "data": {"success": False}}

    callback_url = (_full_cfg.get("callback_base_url") or CALLBACK_BASE_URL or req.callback_url or "").strip().rstrip("/")
    if not callback_url:
        return {"code": 1, "message": "未配置 callback_base_url", "data": {"success": False}}

    run_group_ts = datetime.now().strftime("%Y%m%d%H%M")
    start_index_map: Dict[int, int] = {}
    async with _counter_lock:
        for t in req.tasks:
            counter_key = f"{t.task_id}/{t.case_name}"
            si = _case_counter.get(counter_key, 0)
            _case_counter[counter_key] = si + len(t.test_arry)
            start_index_map[t.detail_id] = si

    logger.info(f"[API] /batch 收到 {len(req.tasks)} 个任务，后台执行")
    asyncio.create_task(_run_batch(req, start_index_map, callback_url, run_group_ts))
    return {"code": 0, "message": "ok", "data": {"success": True}}


@app.get("/status")
async def list_status():
    return {"total": len(_task_status), "tasks": _task_status}


@app.get("/status/{detail_id}")
async def get_status(detail_id: int):
    if detail_id not in _task_status:
        raise HTTPException(404, "任务不存在")
    return _task_status[detail_id]


@app.get("/health")
async def health():
    return {"status": "ok", "workers": _executor._max_workers if _executor else 0}


# 本地测试用：自接回调
_callback_results: Dict[str, dict] = {}


@app.post("/api/v1/agent-eval/notice/detail/{detail_id}")
async def receive_callback(detail_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    _callback_results[detail_id] = body
    logger.info(f"[LocalCallback] detail_id={detail_id} 已接收")
    return {"data": {"success": True}, "code": 0, "msg": "ok"}


@app.get("/callback_results")
async def list_callback_results():
    return {"total": len(_callback_results), "results": _callback_results}


# ---------------------------------------------------------------------------
#  Request parsing (支持 JSON body / form-data / query)
# ---------------------------------------------------------------------------

async def _parse_batch_request(request: Request) -> Optional[BatchRequest]:
    """Try multiple formats to parse a BatchRequest."""
    content_type = request.headers.get("content-type", "").lower()

    # 1. multipart/form-data
    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
            form_dict = {}
            for key, value in form.items():
                if hasattr(value, "read"):
                    content = await value.read()
                    form_dict[key] = content.decode("utf-8")
                else:
                    form_dict[key] = value

            # JSON 文件上传
            for fkey in ("file", "json_file", "json", "tasks_file", "tasks_json"):
                raw = form_dict.get(fkey, "")
                if raw and isinstance(raw, str) and raw.strip()[:1] in ("{", "["):
                    try:
                        data = json.loads(raw.strip())
                        if isinstance(data, list):
                            return BatchRequest(tasks=data, callback_url=form_dict.get("callback_url"))
                        return BatchRequest(
                            tasks=data.get("tasks", []),
                            callback_url=data.get("callback_url") or form_dict.get("callback_url"),
                        )
                    except Exception:
                        continue

            # 分散字段
            if all(k in form_dict for k in ("task_id", "case_name", "prod_url")):
                detail_id = form_dict.get("detail_id", form_dict.get("task_id"))
                test_arry: list = []
                for field in ("test_arry", "test_array"):
                    if field in form_dict:
                        try:
                            test_arry = json.loads(form_dict[field])
                        except Exception:
                            test_arry = [form_dict[field]]
                        break
                if not test_arry:
                    test_arry = [v for k, v in form_dict.items() if k.startswith("test_")]
                if not test_arry:
                    test_arry = ["默认测试步骤"]
                task = StartRequest(
                    task_id=form_dict["task_id"],
                    detail_id=detail_id,
                    case_name=form_dict["case_name"],
                    prod_url=form_dict["prod_url"],
                    test_arry=test_arry,
                )
                return BatchRequest(tasks=[task], callback_url=form_dict.get("callback_url"))

            if "tasks" in form_dict:
                return BatchRequest(tasks=json.loads(form_dict["tasks"]), callback_url=form_dict.get("callback_url"))
        except Exception as e:
            logger.exception(f"Form-data 解析失败: {e}")

    # 2. JSON body
    if "application/json" in content_type:
        try:
            body = await request.json()
            return BatchRequest(**body)
        except Exception:
            pass

    # 3. raw body
    try:
        raw = (await request.body()).decode("utf-8").strip()
        if raw:
            return BatchRequest(**json.loads(raw))
    except Exception:
        pass

    # 4. query params
    qp = dict(request.query_params)
    if "tasks" in qp:
        try:
            return BatchRequest(tasks=json.loads(qp["tasks"]), callback_url=qp.get("callback_url"))
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main():
    global _executor, _full_cfg, _config_path, CALLBACK_BASE_URL, _worker_slots

    parser = argparse.ArgumentParser(description="AppEval API Server v2 (精简版)")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--workers", "--worker", type=int, default=None, dest="workers")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG))
    parser.add_argument("--callback-url", type=str, default=None)
    parser.add_argument("--no-callback", dest="no_callback", action="store_true")
    args = parser.parse_args()

    if args.callback_url:
        CALLBACK_BASE_URL = args.callback_url.rstrip("/")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    _config_path = str(config_path)
    workers, _full_cfg = load_config(_config_path)
    if args.workers is not None:
        workers = args.workers
    if args.no_callback:
        _full_cfg["skip_callback"] = True
    if not args.callback_url and _full_cfg.get("callback_base_url"):
        CALLBACK_BASE_URL = _full_cfg["callback_base_url"].strip().rstrip("/")

    _executor = ProcessPoolExecutor(max_workers=workers)
    _worker_slots = asyncio.Queue(maxsize=workers)
    for _i in range(workers):
        _worker_slots.put_nowait(_i)
    API_LOG_BASE.mkdir(parents=True, exist_ok=True)
    TASK_JSON_DIR.mkdir(parents=True, exist_ok=True)

    callback_hint = "已关闭(本地)" if _full_cfg.get("skip_callback") else CALLBACK_BASE_URL
    _lan_ip = "127.0.0.1"
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        _lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    logger.info(f"AppEval API v2 | port={args.port} workers={workers} callback={callback_hint}")
    logger.info(f"  内网: http://{_lan_ip}:{args.port}")
    logger.info(f"  Task JSONs: {TASK_JSON_DIR}")
    logger.info(f"  api_log: {API_LOG_BASE}")
    for r in app.routes:
        if hasattr(r, "methods") and hasattr(r, "path"):
            logger.info(f"  路由: {list(r.methods)} {r.path}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
