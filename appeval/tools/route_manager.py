#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/02
@Author  : luruyuan
@File    : route_manager.py
@Desc    : Route manager for loading, saving, and maintaining route health status
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

from metagpt.logs import logger


class RouteManager:
    """
    Route Manager

    Responsibilities:
    - Load/save routes from/to JSON files
    - Maintain route health status
    - Provide route information for LLM prompts
    - Process agent's route save decisions

    Note: Routes are stored with full URLs as keys (not path + base_url)
    """

    # Delete route after 3 consecutive failures
    FAILURE_THRESHOLD = 3

    def __init__(self, routes_dir: str = "routes"):
        """
        Initialize route manager.

        Args:
            routes_dir: Directory to store route files
        """
        self.routes_dir = Path(routes_dir)
        self.routes: Dict = {}  # {full_url: route_info}
        self.metadata = {"created_at": "", "last_updated": "", "total_routes": 0}
        self._current_project = ""

        # Ensure directory exists
        self.routes_dir.mkdir(parents=True, exist_ok=True)

    # ==================== Route Loading ====================

    def load_routes(self, project_name: str) -> bool:
        """
        Load routes from project file.

        Args:
            project_name: Project name

        Returns:
            bool: True if routes loaded successfully, False otherwise
        """
        self._current_project = project_name

        file_path = self._get_file_path(project_name)

        if not file_path.exists():
            logger.info(f"[RouteManager] No route file found: {file_path}")
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.metadata = data.get("metadata", {})
            self.routes = data.get("routes", {})

            logger.info(f"[RouteManager] Loaded {len(self.routes)} routes from {file_path}")
            return len(self.routes) > 0

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"[RouteManager] Error loading routes: {e}")
            return False

    # ==================== Route Saving ====================

    def save_routes(self, project_name: str = None):
        """
        Save routes to file.

        Args:
            project_name: Project name (optional, uses loaded project if not provided)
        """
        project_name = project_name or self._current_project

        if not project_name:
            logger.warning("[RouteManager] No project name specified, skip saving")
            return

        file_path = self._get_file_path(project_name)

        # Update metadata
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.metadata.get("created_at"):
            self.metadata["created_at"] = now
        self.metadata["last_updated"] = now
        self.metadata["total_routes"] = len(self.routes)

        data = {"metadata": self.metadata, "routes": self.routes}

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"[RouteManager] Saved {len(self.routes)} routes to {file_path}")

    def _get_file_path(self, project_name: str) -> Path:
        """Get route file path."""
        return self.routes_dir / f"{project_name}.json"

    # ==================== Route CRUD ====================

    def add_route(self, url: str, name: str) -> bool:
        """
        Add a new route with full URL.

        Args:
            url: Full URL (e.g., "https://www.ctrip.com/flights")
            name: Route name (e.g., "flights_page")
        Returns:
            bool: True if added successfully, False if route already exists
        """
        if url in self.routes:
            logger.debug(f"[RouteManager] Route already exists: {url}")
            return False

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.routes[url] = {"identity": {"url": url, "name": name}, "health": {"last_verified": now, "success_count": 1, "failure_count": 0}}

        logger.info(f"[RouteManager] Added new route: {url} ({name})")
        return True

    def remove_route(self, url: str) -> bool:
        """
        Remove a route.

        Args:
            url: Full URL

        Returns:
            bool: True if removed successfully
        """
        if url in self.routes:
            del self.routes[url]
            logger.info(f"[RouteManager] Removed route: {url}")
            return True
        return False

    # ==================== Health Status Management ====================

    def record_access(self, url: str, success: bool) -> dict:
        """
        Record route access result and update health status.

        Health rules:
        - Success: Reset failure_count to 0, increment success_count
        - Failure: Reset success_count to 0, increment failure_count
        - Delete route after FAILURE_THRESHOLD consecutive failures

        Args:
            url: Full URL
            success: Whether the access was successful

        Returns:
            dict: Result with keys 'url', 'removed', 'message'
        """
        if url not in self.routes:
            return {"error": "Route not found", "removed": False}

        route = self.routes[url]
        health = route["health"]

        # Update last verified time
        health["last_verified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result = {"url": url, "removed": False}

        if success:
            health["failure_count"] = 0
            health["success_count"] += 1
            result["message"] = f"Access success, success_count={health['success_count']}"
        else:
            health["success_count"] = 0
            health["failure_count"] += 1

            if health["failure_count"] >= self.FAILURE_THRESHOLD:
                del self.routes[url]
                result["removed"] = True
                result["message"] = f"Route removed after {self.FAILURE_THRESHOLD} consecutive failures"
            else:
                result["message"] = f"Access failed, failure_count={health['failure_count']}/{self.FAILURE_THRESHOLD}"

        return result

    # ==================== Prompt Generation ====================

    def get_routes_for_prompt(self) -> str:
        """
        Generate route information summary for LLM prompt.

        Includes: full URL + name

        Returns:
            str: Formatted route list for prompt
        """
        if not self.routes:
            return "No historical routes available."

        lines = ["Available Routes:"]

        for url, route in self.routes.items():
            name = route["identity"]["name"]

            lines.append(f"  - url: {url}\n" f"    name: {name}")

        return "\n".join(lines)

    # ==================== Process Agent Decisions ====================

    def process_agent_save_decision(self, save_decision: str, route_url: str = None, route_name: str = None, access_success: bool = True) -> dict:
        """
        Process agent's route save decision.

        Args:
            save_decision: "save" or "skip"
            route_url: Full URL to save
            route_name: Route name
            access_success: Whether the current access was successful

        Returns:
            dict: Result with keys 'action', 'url', 'message'
        """
        result = {"action": None, "url": route_url, "message": ""}

        if save_decision != "save":
            result["action"] = "skipped"
            result["message"] = "Agent decided not to save this route"
            return result

        if not route_url:
            result["action"] = "error"
            result["message"] = "No route URL provided"
            return result

        if route_url in self.routes:
            # Route exists: update health
            self.record_access(route_url, access_success)
            result["action"] = "updated"
            result["message"] = f"Updated existing route: {route_url}"
        else:
            # Route doesn't exist: add new route
            # Generate name from URL path if not provided
            if not route_name:
                path = self.extract_path_from_url(route_url)
                route_name = path.strip("/").replace("/", "_") or "home"

            self.add_route(
                url=route_url,
                name=route_name,
            )
            result["action"] = "added"
            result["message"] = f"Added new route: {route_url}"

        return result

    # ==================== Query Methods ====================

    def route_exists(self, url: str) -> bool:
        """Check if route exists by full URL."""
        return url in self.routes

    def get_route(self, url: str) -> Optional[dict]:
        """Get route information by full URL."""
        return self.routes.get(url)

    def find_route_by_path(self, path: str) -> Optional[str]:
        """
        Find a route URL that contains the given path.

        Args:
            path: Path to search for (e.g., "/flights")

        Returns:
            str: Full URL if found, None otherwise
        """
        for url in self.routes:
            if path in url:
                return url
        return None

    # ==================== URL Utility Methods ====================

    @staticmethod
    def extract_path_from_url(url: str) -> str:
        """
        Extract path from URL (used for auto-generating route names).

        Args:
            url: Full URL (e.g., "https://www.taobao.com/cart?id=1")

        Returns:
            str: Path component (e.g., "/cart")
        """
        if not url:
            return ""
        parsed = urlparse(url)
        return parsed.path or "/"
