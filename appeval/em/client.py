"""
EM Client - HTTP client for OSAgent to interact with EM FastAPI Service

This client provides the same interface as EMManager but uses HTTP calls
to communicate with the FastAPI service.

Usage in OSAgent:
```python
from appeval.em import EMClient

# Replace EMManager with EMClient
self.em_client = EMClient(base_url="http://localhost:8000")

# Add evidence (same interface)
await self.em_client.add_evidence(
    gui_evidence=1,
    code_evidence=0,
    test_case_id="web_01",
    iter_num=1,
)

# Get prediction (same interface)
prediction = await self.em_client.predict(case_id="web_01")

# Correct judgment
result = await self.em_client.correct_judgment(
    agent_original=0,
    case_id="web_01",
)
```
"""

import asyncio
from typing import Any, Dict, List, Optional

import httpx
from metagpt.logs import logger


class EMClientError(Exception):
    """Exception raised by EMClient operations."""
    pass


class EMClient:
    """
    HTTP Client for EM FastAPI Service.

    Provides the same interface as EMManager for seamless integration
    with OSAgent.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize EM Client.

        Args:
            base_url: Base URL of the EM FastAPI service
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # API endpoints
        self.api_prefix = "/api/v1"

        # Store correction result for compatibility
        self._correction_result: Optional[Dict] = None

        logger.info(f"EMClient initialized with base_url: {self.base_url}")

    def _get_url(self, endpoint: str) -> str:
        """Get full URL for an endpoint."""
        return f"{self.base_url}{self.api_prefix}{endpoint}"

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            json: JSON body data
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            EMClientError: If request fails after all retries
        """
        url = self._get_url(endpoint)
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        json=json,
                        params=params,
                    )

                    if response.status_code >= 400:
                        error_detail = response.json().get("detail", response.text)
                        raise EMClientError(
                            f"API error ({response.status_code}): {error_detail}")

                    return response.json()

            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
            except EMClientError:
                raise
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)

        raise EMClientError(
            f"Request failed after {self.retry_attempts} attempts: {last_error}")

    # ==================== Health Check ====================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status dictionary
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/health")
            return response.json()

    async def is_healthy(self) -> bool:
        """
        Check if service is healthy.

        Returns:
            True if service is healthy
        """
        try:
            health = await self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False

    # ==================== Model Management ====================

    async def load_params(self, params_path: str) -> bool:
        """
        Load model parameters from file.

        Args:
            params_path: Path to parameters file

        Returns:
            True if successful
        """
        result = await self._request(
            "POST",
            "/model/load",
            json={"params_path": params_path}
        )
        return result.get("success", False)

    async def save_params(self, params_path: str) -> str:
        """
        Save model parameters to file.

        Args:
            params_path: Path to save parameters

        Returns:
            Path where parameters were saved
        """
        result = await self._request(
            "POST",
            "/model/save",
            json={"params_path": params_path}
        )
        return result.get("message", "")

    async def get_params(self) -> Dict[str, Any]:
        """
        Get current model parameters.

        Returns:
            Model parameters dictionary
        """
        return await self._request("GET", "/model/params")

    async def reset(self) -> None:
        """Reset model state (clear evidences)."""
        await self._request("POST", "/model/reset")
        self._correction_result = None

    async def set_case_id(self, case_id: str) -> None:
        """
        Set current test case ID.

        Args:
            case_id: Test case ID
        """
        await self._request(
            "POST",
            "/model/set-case-id",
            json={"case_id": case_id}
        )

    async def train(
        self,
        data: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the EM model with provided data.

        Args:
            data: List of training data rows
            save_path: Optional path to save trained parameters

        Returns:
            Training result dictionary
        """
        return await self._request(
            "POST",
            "/model/train",
            json={
                "data": data,
                "save_path": save_path,
            }
        )

    # ==================== Evidence Management ====================

    async def add_evidence(
        self,
        gui_evidence: Optional[int] = None,
        code_evidence: Optional[int] = None,
        agent_noresp: Optional[int] = None,
        agent_score: Optional[float] = None,
        test_case_id: str = "default",
        step_id: Optional[str] = None,
        iter_num: int = 0,
        weight: float = 1.0,
    ) -> None:
        """
        Add a single evidence item.

        Args:
            gui_evidence: GUI evidence (1=hit, 0=miss, None=invalid)
            code_evidence: Code evidence (1=implemented, 0=not implemented)
            agent_noresp: Agent no-response evidence (1=problem, 0=normal)
            agent_score: Agent score (0-1)
            test_case_id: Test case ID
            step_id: Step ID (auto-generated if not provided)
            iter_num: Iteration number
            weight: Sample weight
        """
        await self._request(
            "POST",
            "/evidence/add",
            json={
                "gui_evidence": gui_evidence,
                "code_evidence": code_evidence,
                "agent_noresp": agent_noresp,
                "agent_score": agent_score,
                "test_case_id": test_case_id,
                "step_id": step_id,
                "iter_num": iter_num,
                "weight": weight,
            }
        )

    async def add_evidences_batch(self, evidences: List[Dict[str, Any]]) -> int:
        """
        Add multiple evidences at once.

        Args:
            evidences: List of evidence dictionaries

        Returns:
            Number of evidences added
        """
        result = await self._request(
            "POST",
            "/evidence/batch",
            json={"evidences": evidences}
        )
        # Parse "Added X evidences" message
        message = result.get("message", "")
        try:
            return int(message.split()[1])
        except (IndexError, ValueError):
            return 0

    async def get_evidences(self) -> List[Dict[str, Any]]:
        """
        Get all stored evidences.

        Returns:
            List of evidence dictionaries
        """
        result = await self._request("GET", "/evidence/list")
        return result.get("evidences", [])

    async def get_evidence_count(self) -> int:
        """
        Get number of stored evidences.

        Returns:
            Evidence count
        """
        result = await self._request("GET", "/evidence/count")
        return result.get("count", 0)

    async def clear_evidences(self) -> None:
        """Clear all stored evidences."""
        await self._request("DELETE", "/evidence/clear")

    async def save_evidences(self, output_path: Optional[str] = None) -> str:
        """
        Save evidences to file.

        Args:
            output_path: Optional output path

        Returns:
            Path where evidences were saved
        """
        result = await self._request(
            "POST",
            "/evidence/save",
            json={"output_path": output_path} if output_path else None
        )
        return result.get("message", "")

    # ==================== Prediction ====================

    async def predict(self, case_id: str) -> Dict[str, float]:
        """
        Predict root cause probabilities.

        Args:
            case_id: Test case ID

        Returns:
            Dictionary with probabilities:
            - P_EnvFail
            - P_AgentRetryFail
            - P_AgentReasoningFail
            - P_AgentFail
        """
        return await self._request(
            "POST",
            "/predict/proba",
            json={"case_id": case_id}
        )

    async def should_retry(
        self,
        case_id: str,
        tau_retry: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Check if retry is recommended.

        Args:
            case_id: Test case ID
            tau_retry: Retry threshold

        Returns:
            Dictionary with:
            - should_retry: bool
            - P_AgentRetryFail: float
            - P_AgentReasoningFail: float
            - P_EnvFail: float
            - P_AgentFail: float
            - reason: str
        """
        return await self._request(
            "POST",
            "/predict/should-retry",
            json={
                "case_id": case_id,
                "tau_retry": tau_retry,
            }
        )

    async def correct_judgment(
        self,
        agent_original: int,
        case_id: str,
        tau_agentfail: Optional[float] = None,
        tau_envfail: Optional[float] = None,
        margin: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Correct agent's judgment.

        Args:
            agent_original: Agent's original judgment (0=FAIL, 1=PASS)
            case_id: Test case ID
            tau_agentfail: AgentFail threshold
            tau_envfail: EnvFail threshold
            margin: Flip margin threshold

        Returns:
            Dictionary with correction result
        """
        result = await self._request(
            "POST",
            "/predict/correct",
            json={
                "agent_original": agent_original,
                "case_id": case_id,
                "tau_agentfail": tau_agentfail,
                "tau_envfail": tau_envfail,
                "margin": margin,
            }
        )
        # Store for get_em_correction_result compatibility
        self._correction_result = result
        return result

    # ==================== Convenience Methods ====================

    async def get_em_prediction(self, case_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get EM prediction (alias for predict).

        This method provides compatibility with EMManager interface.

        Args:
            case_id: Test case ID (required)

        Returns:
            Prediction dictionary or None
        """
        if case_id is None:
            return None
        try:
            return await self.predict(case_id)
        except EMClientError:
            return None

    def get_em_correction_result(self) -> Optional[Dict]:
        """
        Get the latest EM correction result.

        Returns:
            Dict containing correction result, or None if not available
        """
        return self._correction_result


# Synchronous wrapper for non-async code
class EMClientSync:
    """
    Synchronous wrapper for EMClient.

    Use this when you need to call EMClient from synchronous code.
    """

    def __init__(self, *args, **kwargs):
        self._async_client = EMClient(*args, **kwargs)

    def _run(self, coro):
        """Run a coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def health_check(self) -> Dict[str, Any]:
        return self._run(self._async_client.health_check())

    def is_healthy(self) -> bool:
        return self._run(self._async_client.is_healthy())

    def load_params(self, params_path: str) -> bool:
        return self._run(self._async_client.load_params(params_path))

    def save_params(self, params_path: str) -> str:
        return self._run(self._async_client.save_params(params_path))

    def get_params(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_params())

    def reset(self) -> None:
        return self._run(self._async_client.reset())

    def set_case_id(self, case_id: str) -> None:
        return self._run(self._async_client.set_case_id(case_id))

    def add_evidence(self, **kwargs) -> None:
        return self._run(self._async_client.add_evidence(**kwargs))

    def get_evidences(self) -> List[Dict[str, Any]]:
        return self._run(self._async_client.get_evidences())

    def clear_evidences(self) -> None:
        return self._run(self._async_client.clear_evidences())

    def predict(self, case_id: str) -> Dict[str, float]:
        return self._run(self._async_client.predict(case_id))

    def should_retry(self, case_id: str, tau_retry: float = 0.4) -> Dict[str, Any]:
        return self._run(self._async_client.should_retry(case_id, tau_retry))

    def correct_judgment(self, agent_original: int, case_id: str, **kwargs) -> Dict[str, Any]:
        return self._run(self._async_client.correct_judgment(agent_original, case_id, **kwargs))

    def get_em_correction_result(self) -> Optional[Dict]:
        return self._async_client.get_em_correction_result()
