from __future__ import annotations

from typing import Any
from copy import deepcopy
import difflib
from hashlib import sha256
import json
import os
import subprocess
import threading
import time
import urllib.parse
try:
    import tkinter as tk
except Exception:  # pragma: no cover - optional GUI clipboard support
    tk = None

import pyautogui

from copilot.adapters.browser import BrowserAdapter
from copilot.adapters.windows import WindowsAdapter
from copilot.memory.store import MemoryStore
from copilot.perception.legacy_agent import create_legacy_vlm_agent
from copilot.perception.parse_worker import ResidentParseWorker
from copilot.planner.legacy import create_legacy_task_planner
from copilot.schemas import ObservationGraph, ObservationNode
from copilot.state.dom_identity import DOMIdentityTracker


class BrowserDOMUnavailable(RuntimeError):
    pass


class VisionRuntimeBridge:
    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store
        self._agent = None
        self._agent_lock = threading.Lock()
        self.legacy_planner = create_legacy_task_planner()
        self.windows_adapter = WindowsAdapter()
        self.browser_adapter = BrowserAdapter()
        self.dom_identity = DOMIdentityTracker()
        self._last_typed_text = ""
        self._last_explorer_location = ""
        self._parse_cache: dict[str, list[dict[str, Any]]] = {}
        self.last_parse_health: dict[str, Any] = {}
        self._parse_worker = ResidentParseWorker(
            self._parse_with_agent,
            self._agent_parse_health,
            warmup_fn=self._warm_agent,
        )

    @property
    def agent(self):
        if self._agent is None:
            with self._agent_lock:
                if self._agent is None:
                    self._agent = create_legacy_vlm_agent()
                    self._agent.managed_semantic_memory = True
                    self._agent.semantic_memory_path = self.memory_store.semantic_path
                    self._agent.semantic_memory = self.memory_store.semantic_memory
        return self._agent

    def sync_agent_memory(self) -> None:
        if self._agent is not None:
            self._agent.semantic_memory = self.memory_store.semantic_memory

    def observe_environment(self) -> dict[str, Any]:
        return {
            "windows": self.windows_adapter.observe(),
            "browser": self.browser_adapter.observe(),
        }

    def observe_state_probe(self) -> dict[str, Any]:
        active_window = {}
        active_app = ""
        try:
            active_window = self.windows_adapter.get_active_window()
            active_app = self.windows_adapter._guess_app(str(active_window.get("title", "")))
        except Exception:
            active_window = {}
            active_app = ""
        browser = self.browser_adapter.observe() if active_app in {"chrome", "edge"} else {}
        return {
            "windows": {
                "active_window": active_window,
                "active_app_guess": active_app,
                "adapter_mode": "state_probe",
            },
            "browser": browser,
        }

    def read_browser_state_hash(self) -> str:
        if not self.browser_dom_available():
            return ""
        snapshot = self.browser_adapter.snapshot_dom()
        if not isinstance(snapshot, dict):
            return ""
        if snapshot.get("stable_hash"):
            return str(snapshot.get("stable_hash", ""))
        return sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()[:16]

    def read_uia_state_hash(self) -> str:
        try:
            windows = self.windows_adapter.observe()
        except Exception:
            return ""
        payload = {
            "active_window": windows.get("active_window", {}) if isinstance(windows, dict) else {},
            "uia": [
                item.get("stable_hash") or {
                    "name": item.get("name", ""),
                    "automation_id": item.get("automation_id", ""),
                    "control_type": item.get("control_type", ""),
                    "rectangle": item.get("rectangle", {}),
                }
                for item in (windows.get("uia_elements", []) if isinstance(windows, dict) else [])
                if isinstance(item, dict)
            ],
        }
        return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()[:16]

    def read_focused_element(self) -> str:
        if self.browser_dom_available():
            snapshot = self.browser_adapter.snapshot_dom()
            if isinstance(snapshot, dict):
                active_selector = str(snapshot.get("active_selector", "") or "")
                if active_selector:
                    return active_selector
                for item in snapshot.get("items", []):
                    if isinstance(item, dict) and item.get("focused"):
                        return str(item.get("selector") or item.get("id") or item.get("aria_label") or item.get("text") or item.get("tag") or "")
        return ""

    def parse_ui(self, output_filename: str = "ui_parsed_map.png") -> ObservationGraph:
        started = time.time()
        environment = self.observe_environment()
        structured_graph, structured_mode = self._structured_graph_if_sufficient(environment)
        if structured_graph is not None:
            self.last_parse_health = {
                "parse_mode": structured_mode,
                "cache_hit": False,
                "screen_hash_cache_hit": False,
                "screen_hash_elapsed_seconds": 0.0,
                "ocr_skipped_by_dom": structured_mode == "browser_dom",
                "ocr_skipped_by_uia": structured_mode == "windows_uia",
                "ocr_calls": 0,
                "ocr_cache_hits": 0,
                "ocr_timeouts": 0,
                "ocr_errors": 0,
                "ocr_elapsed_seconds": 0.0,
                "worker_used": False,
                "worker_queue_wait_seconds": 0.0,
                "worker_exec_seconds": 0.0,
                "worker_boot_seconds": 0.0,
                "parse_elapsed_seconds": round(time.time() - started, 6),
                "element_count": len(structured_graph),
            }
            metadata = {
                "output_filename": output_filename,
                "environment": environment,
                "parse_health": self.last_parse_health,
            }
            return ObservationGraph.from_raw(structured_graph, metadata=metadata)
        if structured_mode == "browser_dom_required_failed":
            self.last_parse_health = {
                "parse_mode": "browser_dom_required_failed",
                "cache_hit": False,
                "screen_hash_cache_hit": False,
                "screen_hash_elapsed_seconds": 0.0,
                "ocr_skipped_by_dom": True,
                "ocr_skipped_by_uia": False,
                "ocr_calls": 0,
                "ocr_cache_hits": 0,
                "ocr_timeouts": 0,
                "ocr_errors": 0,
                "ocr_elapsed_seconds": 0.0,
                "worker_used": False,
                "worker_queue_wait_seconds": 0.0,
                "worker_exec_seconds": 0.0,
                "worker_boot_seconds": 0.0,
                "parse_elapsed_seconds": round(time.time() - started, 6),
                "element_count": 0,
                "failure_reason": "CDP_UNAVAILABLE",
            }
            metadata = {
                "output_filename": output_filename,
                "environment": environment,
                "parse_health": self.last_parse_health,
                "app_id": environment.get("windows", {}).get("active_app_guess", ""),
                "scene": {"app_id": environment.get("windows", {}).get("active_app_guess", ""), "summary": "Browser DOM unavailable."},
                "scene_summary": "Browser DOM unavailable.",
            }
            return ObservationGraph.from_raw(
                [
                    {
                        "id": "browser_dom_unavailable",
                        "label": "Browser DOM unavailable",
                        "type": "blocked",
                        "semantic_role": "safe_failure",
                        "region": "browser_dom",
                        "state_tags": ["blocked", "cdp_unavailable"],
                        "affordances": [],
                        "box": {"x": 0, "y": 0, "width": 1, "height": 1},
                        "center": {"x": 0, "y": 0},
                    }
                ],
                metadata=metadata,
            )

        screen_hash_started = time.time()
        screen_hash = self._current_screen_hash()
        screen_hash_elapsed = max(0.0, time.time() - screen_hash_started)
        if screen_hash and screen_hash in self._parse_cache:
            raw_graph = deepcopy(self._parse_cache[screen_hash])
            self.last_parse_health = {
                "parse_mode": "vision_cache",
                "screen_hash": screen_hash,
                "cache_hit": True,
                "screen_hash_cache_hit": True,
                "screen_hash_elapsed_seconds": round(screen_hash_elapsed, 6),
                "ocr_calls": 0,
                "ocr_cache_hits": 0,
                "ocr_timeouts": 0,
                "ocr_errors": 0,
                "ocr_elapsed_seconds": 0.0,
                "worker_used": False,
                "worker_queue_wait_seconds": 0.0,
                "worker_exec_seconds": 0.0,
                "worker_boot_seconds": 0.0,
                "parse_elapsed_seconds": round(time.time() - started, 6),
                "element_count": len(raw_graph),
            }
            metadata = {
                "output_filename": output_filename,
                "environment": environment,
                "parse_health": self.last_parse_health,
            }
            return ObservationGraph.from_raw(raw_graph, metadata=metadata)

        worker_result = self._parse_worker.parse(output_filename=output_filename)
        raw_graph = worker_result.raw_graph
        parse_health = dict(getattr(self.agent, "last_parse_health", {}) or {})
        parse_health.update(worker_result.parse_health)
        parse_health.update(worker_result.worker_metrics)
        if screen_hash:
            parse_health["screen_hash"] = screen_hash
        parse_health.setdefault("screen_hash_cache_hit", False)
        parse_health["screen_hash_elapsed_seconds"] = round(screen_hash_elapsed, 6)
        if parse_health.get("screen_hash"):
            self._parse_cache[str(parse_health["screen_hash"])] = deepcopy(raw_graph)
        parse_health.setdefault("parse_mode", "vision_worker")
        parse_health.setdefault("cache_hit", False)
        parse_health.setdefault("ocr_skipped_by_dom", False)
        parse_health.setdefault("ocr_skipped_by_uia", False)
        parse_health.setdefault("worker_used", True)
        parse_health.setdefault("worker_queue_wait_seconds", 0.0)
        parse_health.setdefault("worker_exec_seconds", 0.0)
        parse_health.setdefault("worker_boot_seconds", 0.0)
        parse_health["parse_elapsed_seconds"] = round(time.time() - started, 6)
        metadata = {
            "output_filename": output_filename,
            "environment": environment,
            "parse_health": parse_health,
        }
        self.last_parse_health = parse_health
        return ObservationGraph.from_raw(raw_graph, metadata=metadata)

    def _current_screen_hash(self) -> str:
        try:
            active = self.windows_adapter.get_active_window()
            region = None
            if active:
                left = int(active.get("left", 0) or 0)
                top = int(active.get("top", 0) or 0)
                width = int(active.get("width", 0) or 0)
                height = int(active.get("height", 0) or 0)
                if width > 20 and height > 20 and left > -30000 and top > -30000:
                    region = (left, top, width, height)
            shot = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()
            shot = shot.resize((96, 54))
            return sha256(shot.tobytes()).hexdigest()[:16]
        except Exception:
            return ""

    def _structured_graph_if_sufficient(self, environment: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str]:
        windows = environment.get("windows", {}) if isinstance(environment, dict) else {}
        if windows.get("active_app_guess") in {"chrome", "edge"}:
            dom_graph = self._dom_graph_if_sufficient(environment)
            if dom_graph is not None:
                return dom_graph, "browser_dom"
            self._ensure_chrome_dom_available()
            environment = self.observe_environment()
            dom_graph = self._dom_graph_if_sufficient(environment)
            if dom_graph is not None:
                return dom_graph, "browser_dom"
            if self._browser_vision_fallback_enabled():
                return None, ""
            raise BrowserDOMUnavailable("Browser DOM is required for Chrome/Edge perception but is unavailable.")
        uia_graph = self._uia_graph_if_sufficient(environment)
        if uia_graph is not None:
            return uia_graph, "windows_uia"
        dom_graph = self._dom_graph_if_sufficient(environment)
        if dom_graph is not None:
            return dom_graph, "browser_dom"
        return None, ""

    def _uia_graph_if_sufficient(self, environment: dict[str, Any]) -> list[dict[str, Any]] | None:
        windows = environment.get("windows", {}) if isinstance(environment, dict) else {}
        items = [item for item in windows.get("uia_elements", []) if isinstance(item, dict)]
        if len(items) < 2:
            return None
        children = []
        active_window = windows.get("active_window", {}) if isinstance(windows, dict) else {}
        for idx, item in enumerate(items[:120]):
            label = (
                item.get("name")
                or item.get("automation_id")
                or item.get("control_type")
                or "UIA element"
            )
            role = str(item.get("control_type") or "uia")
            box = dict(item.get("rectangle", {}) or {})
            point = dict(item.get("click_point", item.get("clickable_point", {})) or {})
            state_tags = []
            if item.get("enabled", True):
                state_tags.append("enabled")
            else:
                state_tags.append("disabled")
            if item.get("visible", True):
                state_tags.append("visible")
            else:
                state_tags.append("hidden")
            children.append(
                {
                    "id": f"uia_{idx}",
                    "label": str(label),
                    "type": str(item.get("control_type") or "uia"),
                    "semantic_role": role,
                    "region": "windows_uia",
                    "state_tags": state_tags,
                    "affordances": ["uia", "click"] if item.get("enabled", True) and item.get("visible", True) else ["uia"],
                    "box": {
                        "x": int(box.get("x", 0) or 0),
                        "y": int(box.get("y", 0) or 0),
                        "width": int(box.get("width", 1) or 1),
                        "height": int(box.get("height", 1) or 1),
                    },
                    "center": {
                        "x": int(point.get("x", box.get("x", 0)) or 0),
                        "y": int(point.get("y", box.get("y", 0)) or 0),
                    },
                    "visual_id": str(item.get("stable_hash", "")),
                    "source_frame_id": str(item.get("parent_window", "")),
                }
            )
        root_box = {
            "x": int(active_window.get("left", 0) or 0),
            "y": int(active_window.get("top", 0) or 0),
            "width": int(active_window.get("width", 1) or 1),
            "height": int(active_window.get("height", 1) or 1),
        }
        return [
            {
                "id": "windows_uia",
                "label": str(windows.get("active_window", {}).get("title", "") or "Windows UIA"),
                "type": "container",
                "semantic_role": "windows_uia",
                "region": "windows_uia",
                "children": children,
                "box": root_box,
                "center": {"x": root_box["x"] + root_box["width"] // 2, "y": root_box["y"] + root_box["height"] // 2},
            }
        ]

    def _dom_graph_if_sufficient(self, environment: dict[str, Any]) -> list[dict[str, Any]] | None:
        browser = environment.get("browser", {}) if isinstance(environment, dict) else {}
        windows = environment.get("windows", {}) if isinstance(environment, dict) else {}
        if windows.get("active_app_guess") not in {"chrome", "edge"}:
            return None
        if not browser.get("cdp_available"):
            return None
        snapshot = self.browser_adapter.snapshot_dom()
        if not isinstance(snapshot, dict):
            snapshot = {}
        snapshot.setdefault("title", browser.get("active_title", ""))
        snapshot.setdefault("url", browser.get("active_url", ""))
        items = [item for item in snapshot.get("items", []) if isinstance(item, dict)]
        if not items:
            return None

        children = []
        for idx, item in enumerate(items[:120]):
            tag = str(item.get("tag", "dom") or "dom")
            role = str(item.get("role", "") or tag)
            text = str(item.get("text") or item.get("value") or "")
            selector = str(item.get("selector") or (item.get("selectors") or [""])[0] or "")
            frame_path = str(item.get("frame_path", ""))
            shadow_path = str(item.get("shadow_path", ""))
            stable_hash = self._dom_stable_hash(tag=tag, role=role, text=text, selector=selector, frame_path=frame_path, shadow_path=shadow_path)
            box = dict(item.get("box", {}) or {})
            rect = {
                "x": int(box.get("x", 0) or 0),
                "y": int(box.get("y", 0) or 0),
                "w": int(box.get("width", box.get("w", 1)) or 1),
                "h": int(box.get("height", box.get("h", 1)) or 1),
            }
            node_id = self.dom_identity.node_id_for_hash(stable_hash) or f"dom_{idx}"
            identity_record = self.dom_identity.track(node_id, stable_hash, rect)
            label = (
                text
                or item.get("aria_label")
                or item.get("placeholder")
                or item.get("name")
                or item.get("id")
                or tag
                or "DOM element"
            )
            state_tags = []
            if item.get("visible", True):
                state_tags.append("visible")
            else:
                state_tags.append("hidden")
            if item.get("enabled", True):
                state_tags.append("enabled")
            else:
                state_tags.append("disabled")
            if item.get("focused"):
                state_tags.append("focused")
            children.append(
                {
                    "id": identity_record.node_id,
                    "label": str(label),
                    "type": tag,
                    "semantic_role": role,
                    "region": "browser_dom",
                    "state_tags": state_tags,
                    "affordances": ["dom", "click"] if item.get("enabled", True) and item.get("visible", True) else ["dom"],
                    "box": {
                        "x": rect["x"],
                        "y": rect["y"],
                        "width": rect["w"],
                        "height": rect["h"],
                    },
                    "center": {
                        "x": rect["x"] + rect["w"] // 2,
                        "y": rect["y"] + rect["h"] // 2,
                    },
                    "visual_id": stable_hash,
                    "source_frame_id": selector,
                    "tag": tag,
                    "role": role,
                    "text": text,
                    "aria_label": str(item.get("aria_label", "")),
                    "accessible_name": str(item.get("accessible_name") or item.get("aria_label") or text or item.get("placeholder", "")),
                    "placeholder": str(item.get("placeholder", "")),
                    "selector": selector,
                    "frame_path": frame_path,
                    "shadow_path": shadow_path,
                    "rect": rect,
                    "visible": bool(item.get("visible", True)),
                    "enabled": bool(item.get("enabled", True)),
                    "stable_hash": stable_hash,
                    "dom_identity": identity_record.to_dict(),
                }
            )

        return [
            {
                "id": "browser_dom",
                "label": str(snapshot.get("title") or browser.get("active_title") or "Browser DOM"),
                "type": "container",
                "semantic_role": "browser_dom",
                "region": "browser_dom",
                "children": children,
                "box": {"x": 0, "y": 0, "width": 1, "height": 1},
                "center": {"x": 0, "y": 0},
                "dom_identity_records": self.dom_identity.records(),
            }
        ]

    def _browser_vision_fallback_enabled(self) -> bool:
        return str(os.environ.get("COPILOT_ALLOW_BROWSER_VISION_FALLBACK", "")).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _dom_stable_hash(*, tag: str, role: str, text: str, selector: str, frame_path: str = "", shadow_path: str = "") -> str:
        payload = {
            "tag": str(tag or ""),
            "role": str(role or ""),
            "text": str(text or ""),
            "selector": str(selector or ""),
            "frame_path": str(frame_path or ""),
            "shadow_path": str(shadow_path or ""),
        }
        return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()[:16]

    def _warm_agent(self) -> None:
        _ = self.agent

    def _parse_with_agent(self, output_filename: str) -> list[dict[str, Any]]:
        return self.agent.parse_interface(output_filename=output_filename)

    def _agent_parse_health(self) -> dict[str, Any]:
        return dict(getattr(self.agent, "last_parse_health", {}) or {})

    def execute_legacy_command(self, command_text: str, debug: bool = False) -> bool:
        tuple_plan = self.legacy_planner.compile_plan([command_text])
        if not tuple_plan:
            return False
        ok = bool(self.agent.execute_plan(tuple_plan, debug=debug))
        self.memory_store.semantic_memory = self.agent.semantic_memory
        self.memory_store._save_json(self.memory_store.semantic_path, self.memory_store.semantic_memory)
        return ok

    def route_window(self, app_id: str, window_title: str = "") -> dict[str, Any]:
        result = self.windows_adapter.route_to_application(
            app_id=app_id,
            window_title=window_title,
            launch_callback=lambda: self.execute_legacy_command(f"Open {app_id}", debug=False),
        )
        if result.get("ok") and app_id.lower().strip() == "chrome":
            self._ensure_chrome_dom_available()
        if result.get("ok") and self._agent is not None:
            self.agent.focus_window(window_title or app_id)
        return result

    def _ensure_chrome_dom_available(self) -> None:
        if self.browser_dom_available():
            return

        user_data_dir = self._chrome_user_data_dir()
        profile_dir = self._choose_chrome_profile_dir(
            os.environ.get("COPILOT_CHROME_PROFILE", "") or os.environ.get("COPILOT_CHROME_PROFILE_HINT", ""),
            user_data_dir,
        )
        original_debug_url = self.browser_adapter.debug_url
        for port in (9222, 9223, 9224, 9225):
            self.browser_adapter.debug_url = f"http://127.0.0.1:{port}/json"
            command = [
                "cmd.exe",
                "/c",
                "start",
                "",
                "chrome",
                f"--remote-debugging-port={port}",
                "--remote-allow-origins=*",
                f"--user-data-dir={user_data_dir}",
                f"--profile-directory={profile_dir}",
                "about:blank",
            ]
            try:
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                continue
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if self.browser_dom_available():
                    self.windows_adapter.focus_window(app_id="chrome", timeout=0.8)
                    return
                time.sleep(0.2)
        self.browser_adapter.debug_url = original_debug_url

    def _chrome_user_data_dir(self) -> str:
        explicit = os.environ.get("COPILOT_CHROME_USER_DATA_DIR", "").strip()
        if explicit:
            return explicit
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return os.path.join(local_app_data, "Google", "Chrome", "User Data")
        return os.path.join(os.path.expanduser("~"), "AppData", "Local", "Google", "Chrome", "User Data")

    def _choose_chrome_profile_dir(self, profile_hint: str, user_data_dir: str) -> str:
        profiles = self._chrome_profiles(user_data_dir)
        if not profiles:
            return "Default"
        normalized_hint = " ".join(str(profile_hint or "").strip().lower().split())
        if not normalized_hint:
            for profile in profiles:
                if profile.get("dir") == "Default":
                    return "Default"
            return str(profiles[0].get("dir") or "Default")

        def profile_score(profile: dict[str, str]) -> float:
            names = [profile.get("dir", ""), profile.get("name", "")]
            return max(
                difflib.SequenceMatcher(None, normalized_hint, " ".join(str(name).lower().split())).ratio()
                for name in names
                if name
            )

        best = max(profiles, key=profile_score)
        return str(best.get("dir") or "Default")

    def _chrome_profiles(self, user_data_dir: str) -> list[dict[str, str]]:
        local_state_path = os.path.join(user_data_dir, "Local State")
        profiles: list[dict[str, str]] = []
        try:
            with open(local_state_path, "r", encoding="utf-8") as handle:
                local_state = json.load(handle)
            info_cache = local_state.get("profile", {}).get("info_cache", {})
            if isinstance(info_cache, dict):
                for directory, details in info_cache.items():
                    if not isinstance(details, dict):
                        continue
                    profiles.append({"dir": str(directory), "name": str(details.get("name", directory))})
        except (OSError, ValueError):
            pass
        if not profiles:
            for directory in ("Default", "Profile 1", "Profile 2", "Profile 3"):
                if os.path.isdir(os.path.join(user_data_dir, directory)):
                    profiles.append({"dir": directory, "name": directory})
        return profiles

    def confirm_focus(self, expected: str) -> bool:
        return self.windows_adapter.confirm_focus(title_contains=expected, app_id=expected)

    def clear_focus(self) -> None:
        if self._agent is not None:
            self.agent.clear_app_context()

    def _absolute_point(self, x: int, y: int) -> tuple[int, int]:
        agent = self._agent
        if agent is not None and getattr(agent, "focused_window_bbox", None):
            wx, wy, _, _ = agent.focused_window_bbox
            x += max(0, int(wx))
            y += max(0, int(wy))
        return self.windows_adapter.clamp_point_to_active_window(int(x), int(y))

    def browser_dom_available(self) -> bool:
        return bool(self.browser_adapter.observe().get("cdp_available"))

    def click_point(self, x: int, y: int, clicks: int = 1) -> bool:
        abs_x, abs_y = self._absolute_point(x, y)
        pyautogui.moveTo(abs_x, abs_y, duration=0.2)
        pyautogui.click(clicks=max(1, int(clicks)), interval=0.12)
        return True

    def click_node(self, node: ObservationNode, clicks: int = 1) -> bool:
        return self.click_point(node.center.get("x", 0), node.center.get("y", 0), clicks=clicks)

    def click_node_with_modifiers(self, node: ObservationNode, modifiers: list[str], clicks: int = 1) -> bool:
        clean_modifiers = [str(modifier).lower() for modifier in modifiers if str(modifier).strip()]
        abs_x, abs_y = self._absolute_point(node.center.get("x", 0), node.center.get("y", 0))
        try:
            for modifier in clean_modifiers:
                pyautogui.keyDown(modifier)
            pyautogui.moveTo(abs_x, abs_y, duration=0.2)
            pyautogui.click(clicks=max(1, int(clicks)), interval=0.12)
            return True
        finally:
            for modifier in reversed(clean_modifiers):
                try:
                    pyautogui.keyUp(modifier)
                except Exception:
                    pass

    def open_explorer_location(self, location: str) -> bool:
        normalized = " ".join(str(location or "").strip().lower().split())
        profile = os.environ.get("USERPROFILE", "")
        path_targets = {
            "downloads": os.path.join(profile, "Downloads") if profile else "shell:Downloads",
            "desktop": os.path.join(profile, "Desktop") if profile else "shell:Desktop",
            "documents": os.path.join(profile, "Documents") if profile else "shell:Personal",
        }
        target = path_targets.get(normalized)
        if not target:
            return False
        self.windows_adapter.focus_window(app_id="explorer", timeout=1.0)
        if self.windows_adapter.confirm_focus(app_id="explorer"):
            try:
                pyautogui.hotkey("alt", "d")
                time.sleep(0.05)
                pyautogui.write(target, interval=0.01)
                pyautogui.press("enter")
                time.sleep(0.5)
                self._last_explorer_location = normalized
                return self.windows_adapter.confirm_focus(app_id="explorer")
            except Exception:
                pass
        try:
            subprocess.Popen(["explorer.exe", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return False
        deadline = time.time() + 3.0
        while time.time() < deadline:
            active = self.windows_adapter.get_active_window()
            if normalized in str(active.get("title", "")).lower() and self.windows_adapter.confirm_focus(app_id="explorer"):
                return True
            self.windows_adapter.focus_window(title_contains=location, app_id="explorer", timeout=0.5)
            if self.windows_adapter.confirm_focus(app_id="explorer"):
                active = self.windows_adapter.get_active_window()
                if normalized in str(active.get("title", "")).lower():
                    return True
            self.windows_adapter.focus_window(app_id="explorer", timeout=0.5)
            time.sleep(0.1)
        active = self.windows_adapter.get_active_window()
        ok = normalized in str(active.get("title", "")).lower() and self.windows_adapter.confirm_focus(app_id="explorer")
        if ok:
            self._last_explorer_location = normalized
        return ok

    def current_explorer_location(self) -> str:
        return self._last_explorer_location

    def hover_point(self, x: int, y: int) -> bool:
        abs_x, abs_y = self._absolute_point(x, y)
        pyautogui.moveTo(abs_x, abs_y, duration=0.2)
        return True

    def hover_node(self, node: ObservationNode) -> bool:
        return self.hover_point(node.center.get("x", 0), node.center.get("y", 0))

    def click_selector(self, selector: str) -> bool:
        if self.browser_dom_available():
            return bool(self.browser_adapter.click_selector(selector))
        return False

    def click_first_selector(self, selectors: list[str]) -> str:
        for selector in selectors:
            if selector and self.click_selector(selector):
                return selector
        return ""

    def focused_element_info(self, selector: str = "") -> dict[str, Any]:
        if not self.browser_dom_available():
            return {}
        return self.browser_adapter.focused_element_info(selector=selector)

    def type_text(self, text: str, selector: str = "", clear_first: bool = False) -> bool:
        if selector and self.browser_dom_available():
            if self.browser_adapter.type_text(text, selector=selector, clear_first=clear_first):
                return True
        if clear_first:
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")
        pyautogui.write(text, interval=0.03)
        self._last_typed_text = str(text or "")
        return True

    def read_focused_text(self, *, select_all: bool = False) -> str:
        try:
            if tk is None:
                return ""
            if select_all:
                pyautogui.hotkey("ctrl", "a")
                time.sleep(0.05)
            pyautogui.hotkey("ctrl", "c")
            time.sleep(0.05)
            root = tk.Tk()
            root.withdraw()
            try:
                value = root.clipboard_get()
            finally:
                root.destroy()
            return str(value or "")
        except Exception:
            return ""

    def press_key(self, keys: list[str], hotkey: bool = False) -> bool:
        if not keys:
            return False
        normalized = [str(key).lower() for key in keys]
        if (
            len(normalized) == 1
            and normalized[0] == "enter"
            and self.browser_dom_available()
            and self.windows_adapter.confirm_focus(app_id="chrome")
            and self._last_typed_text.strip()
        ):
            query = urllib.parse.quote_plus(self._last_typed_text.strip())
            if self.browser_adapter.navigate(f"https://www.google.com/search?q={query}"):
                self._last_typed_text = ""
                return True
        if len(keys) == 1 and keys[0].lower() == "enter" and self.browser_dom_available() and not hotkey:
            if self.browser_adapter.press_enter():
                return True
        if hotkey or len(keys) > 1:
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(keys[0])
        return True

    def wait_for(self, seconds: float = 0.0, expected_focus: str = "", timeout: float = 0.0, cancel_callback=None) -> bool:
        cancel_callback = cancel_callback or (lambda: False)
        if seconds > 0:
            deadline = time.time() + seconds
            while time.time() < deadline:
                if cancel_callback():
                    return False
                time.sleep(min(0.1, max(0.0, deadline - time.time())))
        if expected_focus:
            deadline = time.time() + max(timeout, 0.1)
            while time.time() < deadline:
                if cancel_callback():
                    return False
                if self.confirm_focus(expected_focus):
                    return True
                time.sleep(0.1)
            return False
        return True

    def read_browser_dom(self) -> dict[str, Any]:
        return self.browser_adapter.snapshot_dom()

    def run_random_exploration(self, rounds: int = 4, initial_wait: float = 3.0, settle_wait: float = 2.0) -> bool:
        ok = bool(self.agent.run_random_ui_exploration(rounds=rounds, initial_wait=initial_wait, settle_wait=settle_wait))
        self.memory_store.semantic_memory = self.agent.semantic_memory
        self.memory_store._save_json(self.memory_store.semantic_path, self.memory_store.semantic_memory)
        return ok
