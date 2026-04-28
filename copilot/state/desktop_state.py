from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any
import json
import os
import time

from copilot.schemas import ObservationGraph


def stable_hash(payload: Any) -> str:
    try:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        data = str(payload)
    return sha256(data.encode("utf-8")).hexdigest()[:16]


@dataclass
class DesktopState:
    active_app: str = ""
    active_window: dict[str, Any] = field(default_factory=dict)
    dom_snapshot: dict[str, Any] = field(default_factory=dict)
    uia_snapshot: dict[str, Any] = field(default_factory=dict)
    last_action: str = ""
    last_verified_change: str = ""
    state_hash: str = ""
    state_id: str = ""
    timestamp: float = field(default_factory=time.time)
    focused_element: str = ""
    screen_hash: str = ""
    ui_tree_hash: str = ""
    dom_hash: str = ""
    uia_hash: str = ""
    state_signature: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DesktopState":
        return cls(
            active_app=str(payload.get("active_app", "")),
            active_window=dict(payload.get("active_window", {}) or {}),
            dom_snapshot=dict(payload.get("dom_snapshot", {}) or {}),
            uia_snapshot=dict(payload.get("uia_snapshot", {}) or {}),
            last_action=str(payload.get("last_action", "")),
            last_verified_change=str(payload.get("last_verified_change", "")),
            state_hash=str(payload.get("state_hash", "") or payload.get("state_signature", "")),
            state_id=str(payload.get("state_id", "")),
            timestamp=float(payload.get("timestamp", 0.0) or 0.0),
            focused_element=str(payload.get("focused_element", "")),
            screen_hash=str(payload.get("screen_hash", "")),
            ui_tree_hash=str(payload.get("ui_tree_hash", "")),
            dom_hash=str(payload.get("dom_hash", "")),
            uia_hash=str(payload.get("uia_hash", "")),
            state_signature=str(payload.get("state_signature", "") or payload.get("state_hash", "")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
        )


class DesktopStateStore:
    def __init__(self, path: str = "") -> None:
        self.path = path

    def load(self) -> DesktopState | None:
        if not self.path or not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        return DesktopState.from_dict(payload)

    def save(self, state: DesktopState) -> None:
        if not self.path:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2, ensure_ascii=False)


class DesktopStateManager:
    def __init__(self, bridge, state_path: str = "") -> None:
        self.bridge = bridge
        self.store = DesktopStateStore(state_path)
        self.current_state: DesktopState | None = self.store.load()

    def observe(
        self,
        *,
        graph: ObservationGraph | None = None,
        last_action: str = "",
        last_verified_change: str = "",
    ) -> DesktopState:
        environment = self.bridge.observe_environment() if hasattr(self.bridge, "observe_environment") else {}
        windows = environment.get("windows", {}) if isinstance(environment, dict) else {}
        active_window = windows.get("active_window", {}) if isinstance(windows, dict) else {}
        if not isinstance(active_window, dict):
            active_window = {"title": str(active_window)}
        active_app = str(windows.get("active_app_guess", "") if isinstance(windows, dict) else "")

        dom_snapshot = self._read_dom()
        uia_snapshot = self._uia_snapshot(windows)
        dom_hash = stable_hash(dom_snapshot)
        uia_hash = self._uia_hash(windows, uia_snapshot)
        graph_payload = graph.to_dict() if graph else {}
        screen_artifact = str(graph.metadata.get("output_filename", "")) if graph else ""
        focused_element = self._focused_element(graph, dom_snapshot)
        ui_tree_hash = stable_hash(graph_payload)
        state_hash = self._state_hash(
            active_app=active_app,
            active_window=active_window,
            focused_element=focused_element,
            dom_hash=dom_hash,
            uia_hash=uia_hash,
        )
        state = DesktopState(
            active_app=active_app,
            active_window=active_window,
            dom_snapshot=dom_snapshot,
            uia_snapshot=uia_snapshot,
            last_action=last_action,
            last_verified_change=last_verified_change,
            state_hash=state_hash,
            state_id=stable_hash(
                {
                    "active_app": active_app,
                    "active_window": active_window,
                    "focused_element": focused_element,
                    "screen": screen_artifact,
                    "ui": graph_payload,
                    "dom": dom_snapshot,
                    "uia": uia_snapshot,
                    "last_action": last_action,
                }
            ),
            timestamp=time.time(),
            focused_element=focused_element,
            screen_hash=stable_hash(screen_artifact or graph_payload),
            ui_tree_hash=ui_tree_hash,
            dom_hash=dom_hash,
            uia_hash=uia_hash,
            state_signature=state_hash,
            confidence=self._confidence(active_window, focused_element, graph, dom_snapshot, uia_snapshot),
        )
        return self._remember(state)

    def probe(self, *, last_action: str = "", last_verified_change: str = "") -> DesktopState:
        environment = self._probe_environment()
        windows = environment.get("windows", {}) if isinstance(environment, dict) else {}
        browser = environment.get("browser", {}) if isinstance(environment, dict) else {}
        active_window = windows.get("active_window", {}) if isinstance(windows, dict) else {}
        if not isinstance(active_window, dict):
            active_window = {"title": str(active_window)}
        active_app = str(windows.get("active_app_guess", "") if isinstance(windows, dict) else "")

        dom_snapshot: dict[str, Any] = {}
        uia_snapshot: dict[str, Any] = {}
        dom_hash = ""
        uia_hash = ""
        focused_element = self._bridge_focused_element()
        if active_app in {"chrome", "edge"} and bool(browser.get("cdp_available")):
            dom_hash = self._bridge_hash("read_browser_state_hash")
            if not dom_hash:
                dom_snapshot = self._read_dom()
                dom_hash = stable_hash(dom_snapshot)
        elif active_app not in {"chrome", "edge"}:
            uia_snapshot = self._uia_snapshot(windows)
            uia_hash = self._bridge_hash("read_uia_state_hash") or stable_hash(uia_snapshot)

        if active_app in {"chrome", "edge"} and not dom_hash:
            dom_snapshot = self._read_dom()
            dom_hash = stable_hash(dom_snapshot)
        if active_app not in {"chrome", "edge"} and not uia_hash:
            uia_snapshot = self._uia_snapshot(windows)
            uia_hash = stable_hash(uia_snapshot)
        if not focused_element:
            focused_element = self._focused_element(None, dom_snapshot)

        state_hash = self._state_hash(
            active_app=active_app,
            active_window=active_window,
            focused_element=focused_element,
            dom_hash=dom_hash,
            uia_hash=uia_hash,
        )
        state = DesktopState(
            active_app=active_app,
            active_window=active_window,
            dom_snapshot=dom_snapshot,
            uia_snapshot=uia_snapshot,
            last_action=last_action,
            last_verified_change=last_verified_change,
            state_hash=state_hash,
            state_id=stable_hash(
                {
                    "active_app": active_app,
                    "active_window": active_window,
                    "focused_element": focused_element,
                    "dom_hash": dom_hash,
                    "uia_hash": uia_hash,
                    "last_action": last_action,
                }
            ),
            timestamp=time.time(),
            focused_element=focused_element,
            dom_hash=dom_hash,
            uia_hash=uia_hash,
            state_signature=state_hash,
            confidence=self._probe_confidence(active_window, focused_element, dom_hash, uia_hash),
        )
        return self._remember(state)

    def has_changed(self, before: DesktopState | None, after: DesktopState | None) -> bool:
        if not before or not after:
            return True
        return before.state_hash != after.state_hash

    def observe_before_action(self, action_type: str, graph: ObservationGraph | None = None) -> DesktopState:
        return self.observe(graph=graph, last_action=f"before:{action_type}")

    def observe_after_action(
        self,
        action_type: str,
        graph: ObservationGraph | None = None,
        verified_change: str = "",
    ) -> DesktopState:
        return self.observe(graph=graph, last_action=f"after:{action_type}", last_verified_change=verified_change)

    def state_diff(self, before: DesktopState | None, after: DesktopState | None) -> dict[str, Any]:
        if not before or not after:
            return {"changed": False, "missing_state": True}
        changes = {
            "active_window_changed": before.active_window != after.active_window,
            "active_app_changed": before.active_app != after.active_app,
            "focused_element_changed": before.focused_element != after.focused_element,
            "screen_changed": before.screen_hash != after.screen_hash,
            "ui_tree_changed": before.ui_tree_hash != after.ui_tree_hash,
            "dom_changed": before.dom_hash != after.dom_hash,
            "uia_changed": before.uia_hash != after.uia_hash,
            "state_signature_changed": before.state_signature != after.state_signature,
            "state_hash_changed": before.state_hash != after.state_hash,
        }
        return {
            **changes,
            "changed": any(changes.values()),
            "before_state_id": before.state_id,
            "after_state_id": after.state_id,
            "before_state_hash": before.state_hash,
            "after_state_hash": after.state_hash,
            "confidence_delta": round(after.confidence - before.confidence, 3),
        }

    def is_state_stale(self, state: DesktopState | None, *, max_age_seconds: float = 5.0) -> bool:
        if not state:
            return True
        if time.time() - state.timestamp > max_age_seconds:
            return True
        current = self.probe(last_action="stale_check")
        return self.has_changed(state, current)

    def _remember(self, state: DesktopState) -> DesktopState:
        self.current_state = state
        self.store.save(state)
        return state

    def _probe_environment(self) -> dict[str, Any]:
        if hasattr(self.bridge, "observe_state_probe"):
            try:
                return self.bridge.observe_state_probe() or {}
            except Exception:
                return {}
        if hasattr(self.bridge, "observe_environment"):
            try:
                return self.bridge.observe_environment() or {}
            except Exception:
                return {}
        return {}

    def _read_dom(self) -> dict[str, Any]:
        if not hasattr(self.bridge, "read_browser_dom"):
            return {}
        try:
            dom = self.bridge.read_browser_dom() or {}
            return dom if isinstance(dom, dict) else {}
        except Exception:
            return {}

    def _uia_snapshot(self, windows: dict[str, Any]) -> dict[str, Any]:
        elements = windows.get("uia_elements", []) if isinstance(windows, dict) else []
        return {"elements": elements if isinstance(elements, list) else []}

    def _uia_hash(self, windows: dict[str, Any], uia_snapshot: dict[str, Any]) -> str:
        explicit = self._bridge_hash("read_uia_state_hash")
        return explicit or stable_hash(uia_snapshot or self._uia_snapshot(windows))

    def _bridge_hash(self, method_name: str) -> str:
        if not hasattr(self.bridge, method_name):
            return ""
        try:
            return str(getattr(self.bridge, method_name)() or "")
        except Exception:
            return ""

    def _bridge_focused_element(self) -> str:
        if not hasattr(self.bridge, "read_focused_element"):
            return ""
        try:
            return str(self.bridge.read_focused_element() or "")
        except Exception:
            return ""

    def _focused_element(self, graph: ObservationGraph | None, dom: dict[str, Any]) -> str:
        for item in dom.get("items", []) if isinstance(dom, dict) else []:
            if isinstance(item, dict) and (item.get("focused") or item.get("active")):
                return str(item.get("id") or item.get("aria_label") or item.get("text") or item.get("tag") or "")
        if graph:
            for node in graph.flatten():
                tags = {str(tag).lower() for tag in node.state_tags}
                if "focused" in tags or "active" in tags:
                    return node.display_label() or node.node_id
        return ""

    def _state_hash(
        self,
        *,
        active_app: str,
        active_window: dict[str, Any],
        focused_element: str,
        dom_hash: str,
        uia_hash: str,
    ) -> str:
        return stable_hash(
            {
                "active_app": active_app,
                "active_window": active_window,
                "focused_element": focused_element,
                "dom_hash": dom_hash,
                "uia_hash": uia_hash,
            }
        )

    def _confidence(
        self,
        active_window: dict[str, Any],
        focused_element: str,
        graph: ObservationGraph | None,
        dom: dict[str, Any],
        uia: dict[str, Any],
    ) -> float:
        score = 0.0
        if active_window.get("title"):
            score += 0.25
        if focused_element:
            score += 0.2
        if graph and graph.flatten():
            score += 0.2
        if dom:
            score += 0.2
        if uia.get("elements"):
            score += 0.15
        return round(min(1.0, score), 3)

    def _probe_confidence(
        self,
        active_window: dict[str, Any],
        focused_element: str,
        dom_hash: str,
        uia_hash: str,
    ) -> float:
        score = 0.0
        if active_window.get("title"):
            score += 0.35
        if focused_element:
            score += 0.25
        if dom_hash:
            score += 0.25
        if uia_hash:
            score += 0.25
        return round(min(1.0, score), 3)
