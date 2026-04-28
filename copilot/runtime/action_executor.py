from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from copilot.runtime.desktop_state import DesktopState
from copilot.runtime.action_contract import (
    ActionContract,
    build_click_contract,
    build_dom_click_contract,
    build_click_point_contract,
    build_press_key_contract,
    build_type_text_contract,
    build_wait_contract,
    verify_click_contract,
    verify_dom_click_contract,
    finalize_contract,
    verify_point_contract,
    verify_press_key_contract,
    verify_type_text_contract,
    verify_wait_contract,
)
from copilot.schemas import ObservationGraph, ObservationNode, TaskSpec


ParseCallback = Callable[[str, TaskSpec], ObservationGraph]
CancelCallback = Callable[[], bool]


@dataclass
class ExecutionResult:
    ok: bool
    mode: str
    target_node: ObservationNode | None
    notes: str
    contract: ActionContract
    after: ObservationGraph | None = None
    before_state: DesktopState | None = None
    after_state: DesktopState | None = None


class ActionExecutor:
    def __init__(self, bridge, parse_callback: ParseCallback, cancel_callback: CancelCallback | None = None) -> None:
        self.bridge = bridge
        self.parse_callback = parse_callback
        self.cancel_callback = cancel_callback or (lambda: False)

    def _read_dom(self) -> dict[str, Any]:
        if not hasattr(self.bridge, "read_browser_dom"):
            return {}
        try:
            dom = self.bridge.read_browser_dom() or {}
        except Exception:
            return {}
        if not isinstance(dom, dict):
            return {}
        if not dom.get("active_selector") and hasattr(self.bridge, "read_focused_element"):
            try:
                focused = str(self.bridge.read_focused_element() or "")
            except Exception:
                focused = ""
            if focused:
                dom = dict(dom)
                dom["active_selector"] = focused
        return dom

    def _active_window_title(self) -> str:
        if not hasattr(self.bridge, "observe_environment"):
            return ""
        try:
            windows = (self.bridge.observe_environment() or {}).get("windows", {})
            active = windows.get("active_window", {})
            if isinstance(active, dict):
                return str(active.get("title", "") or windows.get("active_app_guess", ""))
            return str(windows.get("active_app_guess", ""))
        except Exception:
            return ""

    def _environment_signature(self) -> dict[str, Any]:
        if not hasattr(self.bridge, "observe_environment"):
            return {}
        try:
            environment = self.bridge.observe_environment() or {}
        except Exception:
            return {}
        windows = environment.get("windows", {}) if isinstance(environment, dict) else {}
        return {
            "active_window": windows.get("active_window", {}) if isinstance(windows, dict) else {},
            "active_app": windows.get("active_app_guess", "") if isinstance(windows, dict) else "",
            "uia": windows.get("uia_elements", []) if isinstance(windows, dict) else [],
        }

    def _confirm_focus(self, expected_app: str) -> bool:
        if not expected_app:
            return True
        if not hasattr(self.bridge, "confirm_focus"):
            return False
        try:
            return bool(self.bridge.confirm_focus(expected_app))
        except Exception:
            return False

    def _active_focus_editable(self, selector: str = "") -> bool | None:
        if not hasattr(self.bridge, "focused_element_info"):
            return None
        try:
            info = self.bridge.focused_element_info(selector=selector) or {}
        except Exception:
            return None
        if not isinstance(info, dict) or not info:
            return None
        return bool(info.get("editable"))

    def execute_click_node(
        self,
        *,
        step_id: str,
        intent: str,
        node: ObservationNode,
        graph: ObservationGraph,
        task: TaskSpec,
        click_count: int = 1,
        settle_wait: float = 0.6,
        extra_evidence: list[str] | None = None,
        target_ranking: dict[str, Any] | None = None,
        modifiers: list[str] | None = None,
    ) -> ExecutionResult:
        contract = build_click_contract(step_id, intent, node, graph, click_count, target_ranking=target_ranking)
        clean_modifiers = [str(modifier).lower() for modifier in (modifiers or []) if str(modifier).strip()]
        if extra_evidence:
            contract.evidence.extend(extra_evidence)
            contract = finalize_contract(contract)
        if clean_modifiers:
            contract.evidence.extend(f"modifier:{modifier}" for modifier in clean_modifiers)
            contract = finalize_contract(contract)
        if contract.status == "blocked":
            return ExecutionResult(False, "hybrid", node, "; ".join(contract.notes), contract, graph)

        direct_uia = node.region == "windows_uia"
        env_before = self._environment_signature() if direct_uia else {}
        if clean_modifiers and hasattr(self.bridge, "click_node_with_modifiers"):
            ok = bool(self.bridge.click_node_with_modifiers(node, modifiers=clean_modifiers, clicks=click_count))
        else:
            ok = bool(self.bridge.click_node(node, clicks=click_count))
        if direct_uia:
            if ok:
                self.bridge.wait_for(seconds=settle_wait)
            env_after = self._environment_signature()
            window_changed = env_before.get("active_window") != env_after.get("active_window")
            app_changed = env_before.get("active_app") != env_after.get("active_app")
            uia_changed = env_before.get("uia") != env_after.get("uia")
            focus_candidate = node.semantic_role == "text_field" or node.entity_type in {"search_field", "omnibox"}
            verified = bool(ok and (window_changed or app_changed or uia_changed or focus_candidate))
            contract.during_checks["executor_accepted"] = bool(ok)
            contract.verification = {
                "verified": verified,
                "direct_verification": "windows_uia",
                "window_changed": window_changed,
                "active_app_changed": app_changed,
                "uia_changed": uia_changed,
                "focus_candidate": focus_candidate,
            }
            contract.status = "verified" if verified else "failed"
            if not verified:
                contract.failure_reason = "NO_STATE_CHANGE"
                contract.notes.append("UIA click did not produce a verified focus, control, or window delta.")
            contract = finalize_contract(contract)
            if not contract.verified:
                return ExecutionResult(False, "native", node, "; ".join(contract.notes), contract, graph)
            return ExecutionResult(True, "native", node, f"Verified UIA click on '{node.display_label()}'.", contract, graph)
        after = graph
        if ok:
            self.bridge.wait_for(seconds=settle_wait)
            after = self.parse_callback(f"{step_id}_after.png", task)
        contract = verify_click_contract(contract, graph, after, ok, node)
        if not contract.verified:
            return ExecutionResult(False, "hybrid", node, "; ".join(contract.notes), contract, after)
        return ExecutionResult(True, "hybrid", node, f"Verified click on '{node.display_label()}'.", contract, after)

    def execute_dom_click(
        self,
        *,
        step_id: str,
        intent: str,
        selectors: list[str],
        task: TaskSpec,
        settle_wait: float = 0.6,
    ) -> ExecutionResult:
        dom_available = bool(hasattr(self.bridge, "browser_dom_available") and self.bridge.browser_dom_available())
        contract = build_dom_click_contract(step_id, intent, selectors, dom_available=dom_available)
        if contract.status == "blocked":
            return ExecutionResult(False, "browser_dom", None, "; ".join(contract.notes), contract, None)
        dom_before = self._read_dom()
        chosen_selector = self.bridge.click_first_selector(selectors)
        ok = bool(chosen_selector)
        after = None
        if ok:
            self.bridge.wait_for(seconds=settle_wait)
            after = None
        dom_after = self._read_dom()
        contract = verify_dom_click_contract(contract, chosen_selector, ok, dom_before=dom_before, dom_after=dom_after, after=after)
        if not contract.verified:
            return ExecutionResult(False, "browser_dom", None, "; ".join(contract.notes), contract, after)
        return ExecutionResult(True, "browser_dom", None, f"Verified DOM click selector '{chosen_selector}'.", contract, after)

    def execute_click_point(
        self,
        *,
        step_id: str,
        intent: str,
        x: int,
        y: int,
        task: TaskSpec,
        before: ObservationGraph | None,
        evidence: list[str],
        click_count: int = 1,
        settle_wait: float = 0.6,
    ) -> ExecutionResult:
        contract = build_click_point_contract(step_id, intent, x, y, evidence, click_count=click_count)
        if contract.status == "blocked":
            return ExecutionResult(False, "native", None, "; ".join(contract.notes), contract, before)
        ok = bool(self.bridge.click_point(x, y, clicks=click_count))
        after = before
        if ok:
            self.bridge.wait_for(seconds=settle_wait)
            after = self.parse_callback(f"{step_id}_after.png", task)
        contract = verify_point_contract(contract, before, after, ok)
        if not contract.verified:
            return ExecutionResult(False, "native", None, "; ".join(contract.notes), contract, after)
        return ExecutionResult(True, "native", None, f"Verified point click at ({x}, {y}).", contract, after)

    def execute_type_text(
        self,
        *,
        step_id: str,
        intent: str,
        text: str,
        task: TaskSpec,
        selector: str = "",
        clear_first: bool = False,
        settle_wait: float = 0.1,
        focused_target: str = "",
        expected_app: str = "",
        deterministic_focus: str = "",
    ) -> ExecutionResult:
        dom_available = bool(hasattr(self.bridge, "browser_dom_available") and self.bridge.browser_dom_available())
        focus_confirmed = self._confirm_focus(expected_app)
        active_focus_editable = self._active_focus_editable(selector=selector)
        contract = build_type_text_contract(
            step_id,
            intent,
            text,
            selector=selector,
            focused_target=focused_target,
            dom_available=dom_available,
            active_window=self._active_window_title(),
            expected_app=expected_app,
            focus_confirmed=focus_confirmed,
            active_focus_editable=active_focus_editable,
            deterministic_focus=deterministic_focus,
        )
        if contract.status == "blocked":
            return ExecutionResult(False, "browser_dom" if selector and dom_available else "native", None, "; ".join(contract.notes), contract, None)
        dom_before = self._read_dom()
        ok = bool(self.bridge.type_text(text, selector=selector, clear_first=clear_first))
        after = None
        if ok:
            self.bridge.wait_for(seconds=settle_wait)
            if not selector:
                after = self.parse_callback(f"{step_id}_after.png", task)
        dom_after = self._read_dom()
        if ok and deterministic_focus and hasattr(self.bridge, "read_focused_text"):
            try:
                focused_text = self.bridge.read_focused_text(select_all=True)
            except Exception:
                focused_text = ""
            if focused_text:
                dom_after = dict(dom_after or {})
                dom_after["focused_text"] = focused_text
        contract = verify_type_text_contract(contract, text, ok, dom_before=dom_before, dom_after=dom_after, after=after)
        mode = "browser_dom" if selector and dom_available else "native"
        if not contract.verified:
            return ExecutionResult(False, mode, None, "; ".join(contract.notes), contract, after)
        return ExecutionResult(True, mode, None, f"Verified typed {len(text)} characters.", contract, after)

    def execute_press_key(
        self,
        *,
        step_id: str,
        intent: str,
        keys: list[str],
        task: TaskSpec,
        hotkey: bool = False,
        settle_wait: float = 0.1,
        before: ObservationGraph | None = None,
        expected_change: str = "",
    ) -> ExecutionResult:
        contract = build_press_key_contract(
            step_id,
            intent,
            keys,
            hotkey=hotkey,
            active_window=self._active_window_title(),
            expected_change=expected_change,
        )
        if contract.status == "blocked":
            return ExecutionResult(False, "native", None, "; ".join(contract.notes), contract, before)
        dom_before = self._read_dom()
        ok = bool(self.bridge.press_key([str(key) for key in keys], hotkey=hotkey))
        after = before
        if ok:
            self.bridge.wait_for(seconds=settle_wait)
            browser_direct = bool(dom_before)
            if expected_change and not browser_direct:
                after = self.parse_callback(f"{step_id}_after.png", task)
        dom_after = self._read_dom()
        contract = verify_press_key_contract(contract, ok, dom_before=dom_before, dom_after=dom_after, before=before, after=after)
        if not contract.verified:
            return ExecutionResult(False, "native", None, "; ".join(contract.notes), contract, after)
        return ExecutionResult(True, "native", None, f"Verified keys: {keys}", contract, after)

    def execute_wait(
        self,
        *,
        step_id: str,
        intent: str,
        seconds: float = 0.0,
        expected_focus: str = "",
        timeout: float = 0.0,
    ) -> ExecutionResult:
        contract = build_wait_contract(step_id, intent, seconds, expected_focus=expected_focus, timeout=timeout)
        if contract.status == "blocked":
            return ExecutionResult(False, "native", None, "; ".join(contract.notes), contract, None)
        if self.cancel_callback():
            contract = verify_wait_contract(contract, False)
            contract.failure_reason = "CANCELLED"
            contract.notes.append("Wait cancelled by operator.")
            return ExecutionResult(False, "native", None, "Wait cancelled by operator.", contract, None)
        if hasattr(self.bridge, "wait_for"):
            try:
                ok = bool(self.bridge.wait_for(seconds=seconds, expected_focus=expected_focus, timeout=timeout, cancel_callback=self.cancel_callback))
            except TypeError:
                ok = bool(self.bridge.wait_for(seconds=seconds, expected_focus=expected_focus, timeout=timeout))
        else:
            ok = False
        contract = verify_wait_contract(contract, ok)
        if self.cancel_callback() and not ok:
            contract.failure_reason = "CANCELLED"
            contract.notes.append("Wait cancelled by operator.")
        notes = "Wait condition satisfied." if ok else "; ".join(contract.notes)
        return ExecutionResult(ok, "native", None, notes, contract, None)
