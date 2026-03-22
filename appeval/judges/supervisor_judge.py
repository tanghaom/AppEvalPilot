"""
SupervisorJudge: strong-model decision layer for resume and retry.

Responsibilities:
1. After first-round failure: analyze trajectory and output restart_recommendation
   (which step to restart from — typically last stable state before ambiguity).
2. During retry: from N candidate actions (from GUI agent), select and rank K
   for execution (coverage, diversity, low risk, avoid duplicates).

Strong model is used only in these two high-information-density, low-call-count places.
"""
import asyncio
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

# LLM imports (same pattern as TellVerifier / CaseGenerator)
try:
    from metagpt.config2 import Config
    from metagpt.llm import LLM
    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False

# ---------------------------------------------------------------------------
#  Prompts
# ---------------------------------------------------------------------------

TRAJECTORY_ANALYSIS_PROMPT = """\
You are a test replay supervisor. A GUI agent just finished its first-round test run and the result is FAILURE. You are given the compressed trajectory.

Analyze the trajectory and answer three questions in order:
1. **Why did it fail?** — Is it the agent's fault (wrong strategy, wrong clicks) or the environment's fault (app broken, page not loading, UI non-functional)?
2. **Should we retry?** — Will a second attempt with a different strategy likely succeed, or is retrying pointless?
3. **If retry, from which step?** — Which step is the last known-good state to restart from, and what should the agent do differently?

## How to read the trajectory
- **"action"** = the raw pyautogui command executed (click, scroll, key press, etc.).
- **"summary"** = what the agent INTENDED to do. This is NOT confirmation the action succeeded.
- You MUST use the **final result evidence** to retroactively judge earlier steps. If evidence says "all elements were unresponsive", then earlier clicks on those elements also failed — even if summaries sound optimistic.
- A click + optimistic summary ≠ success. Only mark a step as successful if there is corroborating evidence.

## Failure type definitions
- **agent**: The application works (at least partially), but the agent chose wrong actions, clicked wrong elements, used a bad strategy, or gave up too early. Retry with a better approach is likely to help.
- **env**: The application itself is broken — page didn't load, JS not running, all UI elements non-responsive, blank/black screen, or critical infrastructure failure. Retrying the same test is unlikely to help without fixing the environment.
- **ambiguous**: Cannot determine clearly; some elements worked but the failure pattern doesn't clearly point to agent or environment.

## Task under test
{task_desc}

## Trajectory
last_completed_iter: {last_completed_iter}

### Step-by-step timeline (action + summary pairs):
{timeline_text}

### Final result:
{result_history_text}

## Required output
Return ONLY a single JSON object (no markdown fences):

{{
  "failure_type": "<agent | env | ambiguous>",
  "fail_reason": "<2-3 sentences: why did the test fail? What went wrong?>",
  "should_retry": <true | false>,
  "retry_reason": "<1-2 sentences: why retry will/won't help>",
  "restart_from_iter": <int, 0-based step index to restart from; set to 0 if should_retry is false or if no step succeeded>,
  "restart_explanation": "<2-3 sentences: why restart from this step, and what should the agent do differently>"
}}

## Examples

Agent failure (retry worthwhile):
{{
  "failure_type": "agent",
  "fail_reason": "The agent successfully loaded the page and opened the date picker, but then repeatedly clicked the same start-date field without ever selecting an end date. The date range was never completed, so the display info never updated.",
  "should_retry": true,
  "retry_reason": "The application is functional. The agent just needs a better strategy for selecting both start and end dates.",
  "restart_from_iter": 4,
  "restart_explanation": "Step 4 successfully opened the start-date picker (confirmed by the date value changing in step 5). Restart here and immediately select a start date, then switch to the end-date field — do not re-click the start-date field."
}}

Environment failure (retry not worthwhile):
{{
  "failure_type": "env",
  "fail_reason": "No UI element responded to any interaction throughout the entire session. Buttons, dropdowns, and scroll all failed. The page appears to have loaded as a static render without JavaScript execution.",
  "should_retry": false,
  "retry_reason": "The application is completely non-functional. Retrying with the same environment will produce the same result.",
  "restart_from_iter": 0,
  "restart_explanation": "No step produced any verifiable progress. If the environment is fixed, start from scratch."
}}

Ambiguous failure:
{{
  "failure_type": "ambiguous",
  "fail_reason": "The agent managed to interact with some elements (dropdown worked, scroll worked) but the core Upload File button was consistently unresponsive across 6 attempts at different coordinates.",
  "should_retry": true,
  "retry_reason": "Some UI elements work, so the app may be partially functional. A different upload approach (drag-and-drop, keyboard shortcut) might succeed.",
  "restart_from_iter": 2,
  "restart_explanation": "Step 2 confirmed the page is partially interactive (dropdown responded). Restart here and try file upload via keyboard shortcut (Ctrl+O) or drag-and-drop instead of clicking the button."
}}

## Your answer (JSON only):
"""

ACTION_SELECTION_PROMPT = """\
You are a test retry supervisor. The GUI agent has generated {n_candidates} candidate actions for the current retry step.

Your job: select up to {k} actions to execute, ranked by priority.

## Selection criteria
1. **Coverage**: pick actions that test different aspects of the feature.
2. **Diversity**: avoid near-duplicate actions.
3. **Low risk**: prefer actions unlikely to corrupt page state.
4. **No repeats**: do not pick actions already attempted in previous rounds.

## Context
{context_text}

## Candidate actions (indexed 0..{n_minus_1}):
{candidates_text}

## Your answer
Return ONLY a JSON array of selected candidate indices, e.g. [2, 0, 4]. No other text.
"""


# ---------------------------------------------------------------------------
#  LLM helper
# ---------------------------------------------------------------------------

def _get_llm(config_path: str = ""):
    """Create LLM instance from supervisor_judge or tell_verifier config section."""
    if not _HAS_LLM:
        return None

    paths_to_try = []
    if config_path:
        paths_to_try.append(config_path)
    # Worker temp config written by run_test.py
    paths_to_try.append("/tmp/appeval_run_cfg_w0.yaml")
    # API server config
    paths_to_try.append(str(Path(__file__).parent.parent / "api_server" / "config.yaml"))

    for p in paths_to_try:
        if not Path(p).exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # Prefer dedicated supervisor_judge section, fallback to tell_verifier (same strong model)
            cfg = data.get("supervisor_judge") or data.get("tell_verifier")
            if not cfg or not cfg.get("model"):
                continue
            config = Config.from_llm_config(cfg)
            llm = LLM(config.llm)
            logger.debug(f"SupervisorJudge LLM loaded from {p}: model={cfg.get('model')}")
            return llm
        except Exception as e:
            logger.debug(f"SupervisorJudge: failed to load LLM from {p}: {e}")
            continue
    return None


def _extract_json_object(text: str) -> Optional[dict]:
    """Extract the first JSON object {...} from *text*, tolerating surrounding prose."""
    # Strip markdown fences (```json ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned).strip()
    # Try full text first
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Search for first { ... } block
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    # Nested braces fallback
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    start = None
    return None


def _extract_llm_usage(llm: Any) -> Dict[str, Any]:
    """Best-effort usage snapshot from LLM cost manager."""
    usage = {
        "model": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "usd": 0.0,
    }
    try:
        usage["model"] = str(getattr(llm, "model", "") or "")
        if hasattr(llm, "get_costs"):
            c = llm.get_costs()
            usage["prompt_tokens"] = int(getattr(c, "total_prompt_tokens", 0) or 0)
            usage["completion_tokens"] = int(getattr(c, "total_completion_tokens", 0) or 0)
            usage["usd"] = float(
                getattr(c, "total_cost_usd", None)
                or getattr(c, "total_cost", None)
                or getattr(c, "cost", None)
                or 0.0
            )
    except Exception:
        pass
    return usage


# ---------------------------------------------------------------------------
#  Public API — async core, sync wrappers
# ---------------------------------------------------------------------------

async def analyze_trajectory_async(
    checkpoint_dict: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[str] = None,
    config_path: str = "",
    task_desc: str = "",
) -> Dict[str, Any]:
    """Analyze first-round trajectory: why it failed, should we retry, from which step.

    Async version — call directly from async code (e.g. eval_runner).
    Falls back to heuristic if LLM is unavailable.

    Returns:
        {
            "failure_type": "agent" | "env" | "ambiguous",
            "fail_reason": str,
            "should_retry": bool,
            "retry_reason": str,
            "restart_from_iter": int,
            "restart_explanation": str,
            "model_response": str,
        }
    """
    if checkpoint_dict is None and checkpoint_dir:
        path = Path(checkpoint_dir) / "resume_checkpoint.json"
        if not path.exists():
            logger.warning(f"SupervisorJudge: no checkpoint at {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                checkpoint_dict = json.load(f)
        except Exception as e:
            logger.warning(f"SupervisorJudge: failed to load checkpoint: {e}")
            return {}

    if not checkpoint_dict:
        return {}

    core = checkpoint_dict.get("replay_core") or checkpoint_dict
    last_iter = int(core.get("last_completed_iter", 0))
    payload = checkpoint_dict.get("decision_payload") or {}
    action_tail = payload.get("action_tail") or []
    summary_tail = payload.get("summary_tail") or []
    result_history = payload.get("result_history") or []

    # Infer task_desc from result_history if not provided explicitly
    if not task_desc and result_history:
        for rh in result_history:
            action_str = rh.get("action", "")
            m = re.search(r'"case_desc"\s*:\s*"([^"]+)"', action_str)
            if m:
                task_desc = m.group(1)
                break

    # Build paired timeline: action + summary per step
    n_steps = max(len(action_tail), len(summary_tail))
    timeline_lines = []
    for i in range(n_steps):
        act = action_tail[i] if i < len(action_tail) else ""
        summ = summary_tail[i] if i < len(summary_tail) else ""
        line = f"  [step {i}] action: {act or '(none)'}"
        if summ:
            line += f"\n           summary: {summ}"
        timeline_lines.append(line)
    timeline_text = "\n".join(timeline_lines) or "(no steps recorded)"

    # --- Try LLM-based analysis ---
    llm = _get_llm(config_path)
    if llm:
        model_response_raw = ""
        try:
            result_history_text = "\n".join(
                f"  iter={r.get('iter')}: {r.get('action', '')}" for r in result_history
            ) or "(none)"

            prompt = TRAJECTORY_ANALYSIS_PROMPT.format(
                task_desc=task_desc or "(not provided)",
                last_completed_iter=last_iter,
                timeline_text=timeline_text,
                result_history_text=result_history_text,
            )

            model_response_raw = await llm.aask(prompt)
            parsed = _extract_json_object(model_response_raw)
            if parsed is None:
                raise ValueError(f"No JSON object found in model response: {model_response_raw[:300]}")

            failure_type = str(parsed.get("failure_type", "ambiguous") or "ambiguous").strip().lower()
            if failure_type not in ("agent", "env", "ambiguous"):
                failure_type = "ambiguous"
            # Treat ambiguous as agent fail for counting/reporting (e.g. work_dirs stats)
            if failure_type == "ambiguous":
                failure_type = "agent"
            should_retry = bool(parsed.get("should_retry", True))
            restart = int(parsed.get("restart_from_iter", last_iter - 2))
            restart = max(0, min(restart, last_iter - 1))

            out = {
                "failure_type": failure_type,
                "fail_reason": str(parsed.get("fail_reason", "") or "").strip(),
                "should_retry": should_retry,
                "retry_reason": str(parsed.get("retry_reason", "") or "").strip(),
                "restart_from_iter": restart,
                "restart_explanation": str(parsed.get("restart_explanation", "") or "").strip(),
                "model_response": model_response_raw,
                "usage": _extract_llm_usage(llm),
            }
            logger.info(
                f"SupervisorJudge (LLM): failure_type={failure_type}, should_retry={should_retry}, "
                f"restart_from_iter={restart}, fail_reason={out['fail_reason'][:150]} "
                f"(last_completed_iter={last_iter})"
            )
            return out
        except Exception as e:
            logger.warning(
                f"SupervisorJudge LLM call failed, falling back to heuristic: {e}\n"
                f"  model_response_raw={model_response_raw[:500] if model_response_raw else '(empty)'}\n"
                f"  traceback: {traceback.format_exc()}"
            )

    # --- Heuristic fallback ---
    if last_iter <= 1:
        restart_from = 0
    else:
        restart_from = max(0, last_iter - 2)

    out = {
        "failure_type": "agent",  # ambiguous treated as agent fail
        "fail_reason": f"Heuristic fallback (LLM unavailable): last_completed_iter={last_iter}.",
        "should_retry": True,
        "retry_reason": "LLM analysis unavailable; defaulting to retry.",
        "restart_from_iter": restart_from,
        "restart_explanation": f"Heuristic: restart from step {restart_from} (last_completed_iter={last_iter}, skip last 2 steps).",
        "model_response": "",
        "usage": {"model": "", "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0},
    }
    logger.info(f"SupervisorJudge (heuristic): {out} (last_completed_iter={last_iter})")
    return out


def analyze_trajectory(
    checkpoint_dict: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[str] = None,
    config_path: str = "",
    task_desc: str = "",
) -> Dict[str, Any]:
    """Sync wrapper for analyze_trajectory_async (for standalone / non-async callers)."""
    coro = analyze_trajectory_async(checkpoint_dict, checkpoint_dir, config_path, task_desc=task_desc)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result(timeout=120)
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
#  Failure classification: fail_reason → category → dimension weights
# ---------------------------------------------------------------------------

FAILURE_CATEGORY_PROMPT = """\
Classify the agent's failure reason into exactly ONE category.

## Categories

- **insufficient_exploration**: The agent gave up too early or failed to navigate/scroll
  enough. It drew premature conclusions without fully exploring the page or available UI.

- **wrong_strategy**: The agent located the correct elements but used the wrong technique,
  algorithm, or execution approach (wrong key timing, wrong game strategy, wrong sequence).

- **wrong_target**: The agent interacted with the wrong element, wrong coordinates, or
  misidentified what a UI component does. The element itself may be correct but the agent
  pointed at the wrong thing.

- **env_boundary**: The feature may not be implemented, the app has a broken component,
  or the UI is non-responsive regardless of what the agent tries.

- **unknown**: The failure does not clearly fit any of the above categories.

## Examples

[insufficient_exploration]
"The agent only pressed 'pagedown' twice and then concluded no carousel exists. It never took a screenshot to confirm the page state."
"The agent completed the survey but could not find dimension percentage scores on the results page. It did not scroll through the full results page."
"The agent scrolled through the timeline but did not explore all UI controls or buttons that might reveal comparison photos."

[wrong_strategy]
"The agent moved the paddle briefly (0.5 seconds) and then the ball was lost. It never implemented continuous paddle control to keep the ball alive."
"The agent repeatedly anchored on card (0,0) paired with every other card — an invalid memory match strategy since both cards need to be different."
"The agent performed hard drops without attempting to clear lines, so the LINES counter remained at 0."

[wrong_target]
"The agent repeatedly attempted to fill form fields using incorrect coordinates, resulting in validation errors."
"The agent clicked the preset text label instead of the actual radio button circle, so the resize was never applied."
"The agent attempted to drag a locked (correctly-placed) puzzle piece instead of testing an unlocked piece."

[env_boundary]
"Clicking a tag on a note card navigated to the detail page instead of filtering — the tag filtering feature may not be implemented."
"The agent waited 105 seconds but no inactivity prompt appeared — the timeout threshold may be longer than tested or not implemented."
"Every time a shape property value was committed, the selected shape disappeared from the canvas — likely an application bug."

## Failure reason to classify
{fail_reason}

Return JSON only, no markdown:
{{"category": "insufficient_exploration | wrong_strategy | wrong_target | env_boundary | unknown"}}
"""

# Empirical success counts per category×dimension (from 103 branches, 41 cases).
# Source: test2_agent_fail_nologin_round2_ours_gemini-3-flash-preview
_RAW_SUCCESS_COUNTS: Dict[str, Dict[str, int]] = {
    "insufficient_exploration": {"A": 1, "B": 7, "C": 0},
    "wrong_strategy":           {"A": 3, "B": 0, "C": 0},
    "wrong_target":             {"A": 2, "B": 0, "C": 0},
    # Bias env-boundary branch allocation toward A (probe/alternate target) over C.
    "env_boundary":             {"A": 2, "B": 1, "C": 0},
    "unknown":                  {"A": 0, "B": 0, "C": 0},
}

# P(branch fails | confirmed agent_fail) per dimension — used for Bayesian env_fail update.
# Calibrated from confirmed-agent-fail subset of the same dataset.
P_BRANCH_FAIL_GIVEN_AGENT: Dict[str, float] = {
    "A": 0.60,
    "B": 0.50,
    "C": 0.40,
}

# Dim-A generation guidance per category (injected into _RETRY_PLAN_PROMPT).
DIM_A_GUIDANCE: Dict[str, str] = {
    "wrong_target":
        "same goal, but find/target the CORRECT element through alternative means "
        "(try parent/sibling elements, different location on page, hover to confirm before clicking).",
    "wrong_strategy":
        "same goal, but use a fundamentally DIFFERENT execution method or algorithm "
        "(different key sequence, different timing, different logical approach).",
    "insufficient_exploration":
        "same goal, but be more thorough — scroll systematically, take screenshots at each step, "
        "do not conclude until the full page has been examined.",
    "env_boundary":
        "same goal, but first probe whether the feature responds at all before committing to the full interaction.",
    "unknown":
        "same goal, different trigger forms (single-click vs double-click, hotkey fallback, etc.).",
}

_LAPLACE_ALPHA = 1.0  # smoothing to avoid weight collapse


def _compute_weights_from_counts(
    counts: Dict[str, int], alpha: float = _LAPLACE_ALPHA
) -> Dict[str, float]:
    """Laplace-smoothed weights from raw success counts."""
    dims = ["A", "B", "C"]
    smoothed = {d: counts.get(d, 0) + alpha for d in dims}
    total = sum(smoothed.values())
    return {d: smoothed[d] / total for d in dims}


# Pre-compute the weight table once at import time.
CATEGORY_TO_WEIGHTS: Dict[str, Dict[str, float]] = {
    cat: _compute_weights_from_counts(counts)
    for cat, counts in _RAW_SUCCESS_COUNTS.items()
}


def compute_dimension_weights(failure_category: str) -> Dict[str, float]:
    """Return A/B/C branch-allocation weights for a given failure category."""
    return CATEGORY_TO_WEIGHTS.get(failure_category, CATEGORY_TO_WEIGHTS["unknown"])


def update_env_belief(p_env: float, failed_dim: str) -> float:
    """Bayesian update of P(env_fail) after a branch on `failed_dim` fails.

    P(env | fail) ∝ P(env) * 1          (env always causes failure)
    P(agent | fail) ∝ P(agent) * P(fail | agent)
    """
    p_agent = 1.0 - p_env
    lk = P_BRANCH_FAIL_GIVEN_AGENT.get(failed_dim.upper(), 0.5)
    denom = p_env + p_agent * lk
    if denom <= 0:
        return p_env
    return p_env / denom


async def classify_failure_async(
    fail_reason: str,
    config_path: str = "",
) -> dict:
    """Classify a fail_reason string into one of the known failure categories.

    Returns dict with:
        "category": "insufficient_exploration" | "wrong_strategy" | "wrong_target"
                    | "env_boundary" | "unknown"
        "usage": {prompt_tokens, completion_tokens, ...}

    Falls back to "unknown" if LLM is unavailable or parsing fails.
    """
    if not fail_reason or not fail_reason.strip():
        return {"category": "unknown", "usage": {}}

    llm = _get_llm(config_path)
    if llm:
        try:
            prompt = FAILURE_CATEGORY_PROMPT.format(fail_reason=fail_reason.strip())
            raw = await llm.aask(prompt)
            usage = _extract_llm_usage(llm)
            parsed = _extract_json_object(raw)
            if parsed:
                cat = str(parsed.get("category", "") or "").strip().lower()
                valid = {"insufficient_exploration", "wrong_strategy", "wrong_target",
                         "env_boundary", "unknown"}
                if cat in valid:
                    logger.info(f"[classify_failure] category={cat!r} for: {fail_reason[:100]}")
                    return {"category": cat, "usage": usage}
        except Exception as e:
            logger.warning(f"[classify_failure] LLM call failed: {e}")

    logger.info("[classify_failure] Falling back to 'unknown'")
    return {"category": "unknown", "usage": {}}


async def select_actions_async(
    candidates: List[Any],
    K: int,
    context: Optional[Dict[str, Any]] = None,
    config_path: str = "",
) -> List[Any]:
    """From N candidate actions, select and rank K for execution (async).

    Uses strong model to rank by coverage, diversity, low risk, no duplicates.
    Falls back to first-K if LLM unavailable.
    """
    if not candidates or K <= 0:
        return []
    if len(candidates) <= K:
        return list(candidates)

    llm = _get_llm(config_path)
    if llm:
        try:
            candidates_text = "\n".join(
                f"  [{i}] {json.dumps(c, ensure_ascii=False) if isinstance(c, dict) else str(c)}"
                for i, c in enumerate(candidates)
            )
            context_text = ""
            if context:
                dp = context.get("decision_payload") or context
                action_tail = dp.get("action_tail") or dp.get("action_history_prefix") or []
                if action_tail:
                    context_text = "Previously attempted actions:\n" + "\n".join(
                        f"  - {a}" for a in action_tail[-5:]
                    )

            prompt = ACTION_SELECTION_PROMPT.format(
                n_candidates=len(candidates),
                k=K,
                n_minus_1=len(candidates) - 1,
                candidates_text=candidates_text,
                context_text=context_text or "(no prior context)",
            )

            raw = await llm.aask(prompt)
            raw = raw.strip()
            raw = re.sub(r"```(?:json)?\s*", "", raw)
            raw = re.sub(r"```", "", raw).strip()
            indices = json.loads(raw)
            if isinstance(indices, list):
                selected = []
                seen = set()
                for idx in indices:
                    idx = int(idx)
                    if 0 <= idx < len(candidates) and idx not in seen:
                        selected.append(candidates[idx])
                        seen.add(idx)
                    if len(selected) >= K:
                        break
                if selected:
                    logger.info(f"SupervisorJudge (LLM): selected {len(selected)}/{len(candidates)} actions")
                    return selected
        except Exception as e:
            logger.warning(f"SupervisorJudge select_actions LLM failed, falling back: {e}")

    return list(candidates[:K])


def select_actions(
    candidates: List[Any],
    K: int,
    context: Optional[Dict[str, Any]] = None,
    config_path: str = "",
) -> List[Any]:
    """Sync wrapper for select_actions_async."""
    coro = select_actions_async(candidates, K, context, config_path)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result(timeout=120)
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
#  Plan selection (branching retry)
# ---------------------------------------------------------------------------

PLAN_SELECTION_PROMPT = """\
You are a test retry supervisor. A GUI agent failed its first-round test and has now proposed \
{n_candidates} candidate retry plans. Your job: select the top {k} most promising plans to execute.

## Selection criteria (in priority order)
1. **Root-cause fit** — Does the plan directly fix the documented failure reason?
2. **Feasibility** — Can it be executed from what is visible in the current screenshot?
3. **Non-repetition** — Does it avoid the actions that already failed (see action history)?
4. **Specificity** — A concrete, step-by-step plan beats a vague strategy.

## Task description
{task_desc}

## Previous round failure analysis
- Failure reason: {fail_reason}
- Retry rationale: {retry_reason}
- Supervisor advice: {restart_explanation}

## Action history near the retry point (last steps before failure)
{trajectory_tail}

## Current screen at retry node
[See attached screenshot]

## Candidate plans (indexed 0..{n_minus_1}):
{candidates_text}

## Your answer
Return ONLY a JSON object in this format:
{{
  "selected": [
    {{"idx": 2, "reason": "why this plan is selected"}},
    {{"idx": 0, "reason": "why this plan is selected"}}
  ]
}}
Rules:
1) Keep the order as priority ranking (best first).
2) Select at most {k} items.
3) reason should be concise and specific.
4) No markdown, no extra text.
"""


async def select_plans_async(
    plans: List[str],
    k: int,
    context: Optional[Dict[str, Any]] = None,
    config_path: str = "",
) -> List[Dict[str, Any]]:
    """From N candidate retry plans, select and rank the top-k using a strong model.

    context keys (all optional):
        task_desc, fail_reason, retry_reason, restart_explanation,
        trajectory_tail, screenshot_b64
    Returns list of selected plan objects:
      [{"idx": int, "plan": str, "reason": str}, ...]
    Falls back to first-k plans with generic reasons if LLM unavailable/unparseable.
    """
    if not plans or k <= 0:
        return []
    if len(plans) <= k:
        return [{"idx": i, "plan": p, "reason": "auto-selected: candidates <= k"} for i, p in enumerate(plans)]

    ctx = context or {}
    llm = _get_llm(config_path)
    if llm:
        try:
            candidates_text = "\n".join(
                f"[{i}]\n{p}" for i, p in enumerate(plans)
            )
            prompt = PLAN_SELECTION_PROMPT.format(
                n_candidates=len(plans),
                k=k,
                n_minus_1=len(plans) - 1,
                task_desc=ctx.get("task_desc", "(not provided)"),
                fail_reason=ctx.get("fail_reason", "(not provided)"),
                retry_reason=ctx.get("retry_reason", "(not provided)"),
                restart_explanation=ctx.get("restart_explanation", "(not provided)"),
                trajectory_tail=ctx.get("trajectory_tail", "(not provided)"),
                candidates_text=candidates_text,
            )
            screenshot_b64 = ctx.get("screenshot_b64")
            images = [screenshot_b64] if screenshot_b64 else []
            raw = await llm.aask(prompt, images=images)
            raw = raw.strip()
            raw = re.sub(r"```(?:json)?\s*", "", raw)
            raw = re.sub(r"```", "", raw).strip()
            payload = json.loads(raw)
            selected_items = payload.get("selected", []) if isinstance(payload, dict) else []
            if isinstance(selected_items, list):
                selected: List[Dict[str, Any]] = []
                seen: set = set()
                for item in selected_items:
                    if not isinstance(item, dict):
                        continue
                    idx = int(item.get("idx", -1))
                    reason = str(item.get("reason", "")).strip()
                    if 0 <= idx < len(plans) and idx not in seen:
                        selected.append({"idx": idx, "plan": plans[idx], "reason": reason})
                        seen.add(idx)
                    if len(selected) >= k:
                        break
                if selected:
                    logger.info(f"SupervisorJudge (select_plans): selected {len(selected)}/{len(plans)} plans")
                    for item in selected:
                        item["usage"] = _extract_llm_usage(llm)
                    return selected
        except Exception as e:
            logger.warning(f"SupervisorJudge select_plans LLM failed, falling back: {e}")

    return [{"idx": i, "plan": p, "reason": "fallback: first-k selection", "usage": {}} for i, p in enumerate(plans[:k])]


def select_plans(
    plans: List[str],
    k: int,
    context: Optional[Dict[str, Any]] = None,
    config_path: str = "",
) -> List[Dict[str, Any]]:
    """Sync wrapper for select_plans_async."""
    coro = select_plans_async(plans, k, context, config_path)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result(timeout=120)
    return asyncio.run(coro)
