from __future__ import annotations

from collections import Counter
from typing import Any

from copilot.adapters import BrowserAdapter
from copilot.memory.store import MemoryStore
from copilot.profiles import AppProfileRegistry, ChromeProfile, ExplorerProfile
from copilot.reasoner import HybridLocalReasoner
from copilot.schemas import (
    ActionIntent,
    ActionTarget,
    ControlMode,
    ExecutionPlan,
    PlanStep,
    RiskLevel,
    TaskSpec,
)


def _normalize(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


class PromptCompiler:
    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store
        self.profiles = AppProfileRegistry([ExplorerProfile(), ChromeProfile()])
        self.reasoner = HybridLocalReasoner(self.profiles)
        self.browser = BrowserAdapter()

    def _make_step(
        self,
        step_id: int,
        title: str,
        action_type: str,
        target: ActionTarget | None = None,
        intent: ActionIntent | None = None,
        parameters: dict[str, Any] | None = None,
        risk: RiskLevel = RiskLevel.LOW,
        requires_approval: bool = False,
        success_criteria: str = "",
        fallback_hint: str = "",
        confidence: float = 0.7,
        modes: list[ControlMode] | None = None,
    ) -> PlanStep:
        return PlanStep(
            step_id=f"step_{step_id}",
            title=title,
            action_type=action_type,
            target=target,
            intent=intent,
            parameters=parameters or {},
            risk_level=risk,
            requires_approval=requires_approval,
            success_criteria=success_criteria,
            fallback_hint=fallback_hint,
            confidence=confidence,
            control_modes=modes or [ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
        )

    def _risk_for_action(self, verb: str, parameters: dict[str, Any]) -> RiskLevel:
        if bool(parameters.get("destructive")):
            return RiskLevel.HIGH
        if bool(parameters.get("requires_approval")):
            return RiskLevel.MEDIUM
        if verb in {"recover", "explore_safe", "interaction_learning"}:
            return RiskLevel.MEDIUM
        if verb in {"modified_click_node"}:
            return RiskLevel.MEDIUM
        if verb in {"click_point"}:
            return RiskLevel.HIGH
        if verb in {"type_text", "press_key"} and any(
            token in str(parameters).lower() for token in ("delete", "rm ", "checkout", "payment", "submit", "account")
        ):
            return RiskLevel.HIGH
        return RiskLevel.LOW

    def _modes_for_action(self, verb: str) -> list[ControlMode]:
        mapping = {
            "route_window": [ControlMode.NATIVE, ControlMode.LEGACY],
            "open_explorer_location": [ControlMode.NATIVE],
            "confirm_focus": [ControlMode.NATIVE],
            "parse_ui": [ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
            "click_node": [ControlMode.BROWSER_DOM, ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
            "modified_click_node": [ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
            "click_point": [ControlMode.NATIVE],
            "type_text": [ControlMode.BROWSER_DOM, ControlMode.NATIVE, ControlMode.LEGACY],
            "press_key": [ControlMode.BROWSER_DOM, ControlMode.NATIVE, ControlMode.LEGACY],
            "wait_for": [ControlMode.NATIVE],
            "ocr_read": [ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
            "hover_probe": [ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
            "learning_session": [ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
            "interaction_learning": [ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
            "replay_interaction": [ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
            "verify_scene": [ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
            "scene_diff": [ControlMode.HYBRID],
            "checkpoint": [ControlMode.HYBRID],
            "recover": [ControlMode.NATIVE, ControlMode.HYBRID],
            "explore_safe": [ControlMode.HYBRID, ControlMode.VISION_FALLBACK],
        }
        return mapping.get(verb, [ControlMode.HYBRID, ControlMode.VISION_FALLBACK])

    def _low_action(
        self,
        verb: str,
        title: str,
        target_kind: str,
        target: str,
        parameters: dict[str, Any] | None = None,
        success: str = "",
        description: str = "",
        fallback_hint: str = "",
        confidence: float = 0.82,
        risk_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "verb": verb,
            "title": title,
            "target_kind": target_kind,
            "target": target,
            "parameters": parameters or {},
            "success": success,
            "description": description or title,
            "fallback_hint": fallback_hint,
            "confidence": confidence,
            "risk_tags": risk_tags or [],
        }

    def _compile_runtime_actions(self, runtime_actions: list[dict[str, Any]], required_apps: list[str]) -> list[PlanStep]:
        steps: list[PlanStep] = []
        step_id = 1
        for action in runtime_actions:
            if not isinstance(action, dict):
                continue
            verb = str(action.get("verb", "")).strip()
            if not verb:
                continue
            parameters = dict(action.get("parameters", {}))
            target_value = str(action.get("target", ""))
            target_kind = str(action.get("target_kind", "semantic_target"))
            title = str(action.get("title", verb.replace("_", " ").title()))
            success = str(action.get("success", ""))
            description = str(action.get("description", title))
            risk = self._risk_for_action(verb, parameters)
            requires_approval = verb in {"explore_safe", "click_point"} or bool(parameters.get("requires_approval"))
            steps.append(
                self._make_step(
                    step_id=step_id,
                    title=title,
                    action_type=verb,
                    target=ActionTarget(kind=target_kind, value=target_value, filters=parameters.get("filters", {})),
                    intent=ActionIntent(
                        verb=verb,
                        semantic_target=target_value,
                        description=description,
                        preferred_mode=self._modes_for_action(verb)[0],
                        fallback_modes=self._modes_for_action(verb)[1:],
                        risk_tags=list(action.get("risk_tags", [])),
                    ),
                    parameters=parameters,
                    risk=risk,
                    requires_approval=requires_approval,
                    success_criteria=success,
                    fallback_hint=str(action.get("fallback_hint", "")),
                    confidence=float(action.get("confidence", 0.82)),
                    modes=self._modes_for_action(verb),
                )
            )
            step_id += 1
        return steps

    def _workflow_matches_context(
        self,
        workflow: dict[str, Any],
        observation: Any | None,
        environment: dict[str, Any],
    ) -> bool:
        if workflow.get("promotion_state") != "trusted":
            return False
        if not observation:
            return False

        scene = observation.metadata.get("scene", {})
        active_app = _normalize(environment.get("windows", {}).get("active_app_guess", ""))
        required_apps = [_normalize(item) for item in workflow.get("required_apps", []) if item]
        if required_apps and active_app and active_app not in required_apps:
            return False

        for raw_step in workflow.get("steps", []):
            if raw_step.get("action_type") == "click_node":
                parameters = raw_step.get("parameters", {}) or {}
                selector_candidates = parameters.get("selector_candidates", []) or []
                filters = parameters.get("filters", {}) or ((raw_step.get("target") or {}).get("filters") or {})
                if selector_candidates and environment.get("browser", {}).get("cdp_available"):
                    continue
                if not filters:
                    return False
                if not self.reasoner.choose_action_target(filters, observation, scene):
                    return False
            if raw_step.get("action_type") == "verify_scene":
                expected_app = _normalize((raw_step.get("parameters", {}) or {}).get("expected_app", ""))
                if expected_app and expected_app != _normalize(scene.get("app_id", "")):
                    return False
        return True

    def _browser_snapshot(self, environment: dict[str, Any]) -> dict[str, Any]:
        if not environment.get("browser", {}).get("cdp_available"):
            return {}
        snapshot = self.browser.snapshot_dom()
        return snapshot if isinstance(snapshot, dict) else {}

    def _resolve_runtime_app(self, app_id: str, environment: dict[str, Any]) -> str:
        app_id = str(app_id or "").strip()
        if app_id and app_id != "current_window":
            return app_id
        active_guess = str(environment.get("windows", {}).get("active_app_guess", "")).strip()
        return active_guess or app_id or "current_window"

    def _memory_bias(self, app_id: str, target_hint: str = "") -> tuple[list[str], list[str], list[str]]:
        normalized_app = _normalize(app_id)
        normalized_hint = _normalize(target_hint)
        preferred = Counter()
        if normalized_hint:
            for label in self.memory_store.preferred_interaction_labels(app_id=app_id, min_reward=0.2):
                if normalized_hint in _normalize(label):
                    preferred[label] += 8
            for control in self.memory_store.semantic_memory.get("controls", {}).values():
                if not isinstance(control, dict):
                    continue
                if normalized_app and _normalize(control.get("app_id", "")) != normalized_app:
                    continue
                label = str(control.get("display_label", "")).strip()
                normalized_label = _normalize(label)
                if not label or (label.startswith("[") and label.endswith("]")):
                    continue
                if normalized_hint not in normalized_label:
                    continue
                preferred[label] += int(control.get("seen_count", 0) or 0) + len(control.get("affordances", {}))

            for outcome in reversed(self.memory_store.episodic_memory.get("action_outcomes", [])):
                if not isinstance(outcome, dict) or not outcome.get("ok"):
                    continue
                label = str(outcome.get("target_label", "")).strip()
                normalized_label = _normalize(label)
                if not label or (label.startswith("[") and label.endswith("]")):
                    continue
                if normalized_hint not in normalized_label:
                    continue
                preferred[label] += 3

        avoid_labels: list[str] = []
        avoid_visual_ids: list[str] = []
        for item in reversed(self.memory_store.semantic_memory.get("negative_examples", [])):
            if not isinstance(item, dict):
                continue
            if normalized_app and _normalize(item.get("app_id", "")) not in {"", normalized_app}:
                continue
            label = str(item.get("label", "")).strip()
            if label and label not in avoid_labels:
                avoid_labels.append(label)
            visual_id = str(item.get("visual_id", "")).strip()
            if visual_id and visual_id not in avoid_visual_ids:
                avoid_visual_ids.append(visual_id)
            if len(avoid_labels) >= 8 and len(avoid_visual_ids) >= 8:
                break

        preferred_labels = [label for label, _ in preferred.most_common(5)]
        return preferred_labels, avoid_labels[:8], avoid_visual_ids[:8]

    def _apply_memory_bias(self, filters: dict[str, Any], app_id: str, target_hint: str = "") -> dict[str, Any]:
        enriched = dict(filters)
        if app_id and not enriched.get("app_id"):
            enriched["app_id"] = app_id
        enriched["exclude_destructive"] = True
        preferred_labels, avoid_labels, avoid_visual_ids = self._memory_bias(app_id, target_hint=target_hint)
        if preferred_labels:
            enriched["preferred_labels"] = preferred_labels
        if avoid_labels:
            enriched["avoid_labels"] = avoid_labels
        if avoid_visual_ids:
            enriched["avoid_visual_ids"] = avoid_visual_ids
        return enriched

    def _query_expectations(self, query: str, extras: list[str] | None = None) -> list[str]:
        values: list[str] = []
        query = str(query or "").strip()
        if query:
            values.append(query)
            for token in query.split():
                if len(token) >= 3:
                    values.append(token)
        for extra in extras or []:
            extra = str(extra or "").strip()
            if extra:
                values.append(extra)
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = _normalize(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(value)
        return deduped[:5]

    def _click_chain(
        self,
        title: str,
        app_id: str,
        target_label: str,
        filters: dict[str, Any],
        expected_change: str,
        expected_labels: list[str] | None = None,
        selector_candidates: list[str] | None = None,
        click_count: int = 1,
        verify_filters: dict[str, Any] | None = None,
        settle_wait: float = 0.7,
        target_kind: str = "semantic_target",
        fallback_hint: str = "",
    ) -> list[dict[str, Any]]:
        expected_labels = expected_labels or []
        selector_candidates = [selector for selector in (selector_candidates or []) if selector]
        return [
            self._low_action(
                "click_node",
                title,
                target_kind,
                target_label,
                {
                    "filters": filters,
                    "selector_candidates": selector_candidates,
                    "expected_change": expected_change,
                    "expected_app": app_id,
                    "settle_wait": settle_wait,
                    "click_count": max(1, int(click_count)),
                },
                success=expected_change or f"{target_label} should be activated.",
                fallback_hint=fallback_hint,
            ),
            self._low_action(
                "wait_for",
                f"Wait for {app_id} to settle",
                "application",
                app_id,
                {"seconds": 0.2, "expected_focus": app_id, "timeout": 1.0},
                success=f"{app_id} should remain focused.",
            ),
            self._low_action(
                "verify_scene",
                f"Verify scene after {title.lower()}",
                "ui_surface",
                app_id,
                {
                    "expected_app": app_id,
                    "filters": verify_filters or {},
                    "expected_labels": expected_labels,
                },
                success=expected_change or f"The scene should reflect '{target_label}'.",
                fallback_hint=fallback_hint,
            ),
            self._low_action(
                "scene_diff",
                f"Summarize scene change after {title.lower()}",
                "ui_surface",
                app_id,
                {
                    "expected_scene": expected_change,
                    "recovery_hint": fallback_hint or "Reparse the current app surface before retrying.",
                },
                success="Scene delta should be recorded.",
            ),
        ]

    def _ensure_app_actions(self, app_id: str, window_title: str = "") -> list[dict[str, Any]]:
        actions = [
            self._low_action(
                "route_window",
                f"Route to {app_id}",
                "application",
                app_id,
                {"app_id": app_id, "window_title": window_title},
                success=f"{app_id} should be visible and focused.",
            ),
            self._low_action(
                "confirm_focus",
                f"Confirm {app_id} focus",
                "application",
                app_id,
                {"expected": app_id},
                success=f"{app_id} should remain focused.",
            ),
        ]
        if _normalize(app_id) == "explorer":
            actions.append(
                self._low_action(
                    "press_key",
                    "Set Explorer stable details view",
                    "keyboard",
                    "ctrl+shift+6",
                    {
                        "keys": ["ctrl", "shift", "6"],
                        "hotkey": True,
                        "expected_app": "explorer",
                        "shortcut_id": "explorer_stable_view",
                        "expected_change": "Explorer should use the stable details view.",
                        "settle_wait": 0.2,
                    },
                    success="Explorer should use the stable details view.",
                    fallback_hint="Refocus Explorer and retry Ctrl+Shift+6.",
                    risk_tags=["shortcut", "view_normalization"],
                )
            )
        return actions

    def _compile_learned_replay(
        self,
        task: TaskSpec,
        environment: dict[str, Any],
        observation: Any | None = None,
    ) -> ExecutionPlan | None:
        active_app = self._resolve_runtime_app("current_window", environment)
        start_scene = self.memory_store._scene_signature(observation) if observation else ""
        path = self.memory_store.find_interaction_path(
            task.prompt,
            app_id=active_app if active_app != "current_window" else "",
            start_scene=start_scene,
            max_steps=4,
        )
        if len(path) >= 2:
            actions: list[dict[str, Any]] = []
            required_apps: set[str] = set()
            labels: list[str] = []
            for idx, edge in enumerate(path, start=1):
                control = edge.get("control", {}) if isinstance(edge.get("control", {}), dict) else {}
                label = str(edge.get("target_label") or control.get("label") or "").strip()
                app_id = str(control.get("app_id") or active_app or "current_window").strip()
                if app_id and app_id != "current_window":
                    required_apps.add(app_id)
                labels.append(label)
                filters = self._apply_memory_bias(
                    {
                        "label_contains": label,
                        "app_id": app_id,
                        "min_score": 0.50,
                    },
                    app_id=app_id,
                    target_hint=label,
                )
                if control.get("entity_type"):
                    filters["entity_type"] = control["entity_type"]
                if control.get("semantic_role"):
                    filters["semantic_role"] = control["semantic_role"]
                if control.get("region"):
                    filters["region"] = control["region"]

                actions.extend(
                    [
                        self._low_action(
                            "replay_interaction",
                            f"Replay learned step {idx}: {label}",
                            "learned_control",
                            label,
                            {
                                "app_id": app_id,
                                "filters": filters,
                                "control_signature": edge.get("control_signature", ""),
                                "settle_wait": 0.8,
                                "expected_reward": edge.get("reward_avg", 0.0),
                            },
                            success=f"Learned step {idx} should reproduce its rewarded transition.",
                            fallback_hint="If this learned step fails, stop and refresh the interaction graph.",
                        ),
                        self._low_action(
                            "wait_for",
                            f"Wait after learned step {idx}",
                            "time",
                            "0.5",
                            {"seconds": 0.5, "expected_focus": app_id if app_id != "current_window" else ""},
                            success="The UI should settle after the learned step.",
                        ),
                    ]
                )
            final_app = next(iter(required_apps), active_app)
            actions.append(
                self._low_action(
                    "scene_diff",
                    "Summarize multi-step learned result",
                    "ui_surface",
                    final_app,
                    {
                        "expected_scene": f"Learned path should complete: {' -> '.join(labels)}.",
                        "recovery_hint": "Run interaction learning again if the learned path no longer matches the UI.",
                    },
                    success="Multi-step learned path scene delta should be recorded.",
                )
            )
            steps = self._compile_runtime_actions(actions, sorted(required_apps))
            return ExecutionPlan(
                task=task,
                steps=steps,
                source="interaction_graph_path",
                summary=f"Replayed learned path: {' -> '.join(labels)}.",
                required_apps=sorted(required_apps),
                success_conditions=[step.success_criteria for step in steps if step.success_criteria],
            )

        replay = self.memory_store.find_interaction_replay(task.prompt, app_id=active_app if active_app != "current_window" else "")
        if not replay:
            return None

        app_id = str(replay.get("app_id") or active_app or "current_window")
        label = str(replay.get("label", "")).strip()
        if not label:
            return None

        actions: list[dict[str, Any]] = []
        if app_id and app_id != "current_window" and app_id != active_app:
            actions.extend(
                [
                    self._low_action(
                        "route_window",
                        f"Route to learned app {app_id}",
                        "application",
                        app_id,
                        {"app_id": app_id, "window_title": app_id},
                        success=f"{app_id} should be visible and focused.",
                    ),
                    self._low_action(
                        "confirm_focus",
                        f"Confirm {app_id} focus",
                        "application",
                        app_id,
                        {"expected": app_id},
                        success=f"{app_id} should remain focused.",
                    ),
                ]
            )

        filters = self._apply_memory_bias(
            {
                "label_contains": label,
                "app_id": app_id,
                "min_score": 0.50,
            },
            app_id=app_id,
            target_hint=label,
        )
        if replay.get("entity_type"):
            filters["entity_type"] = replay["entity_type"]
        if replay.get("semantic_role"):
            filters["semantic_role"] = replay["semantic_role"]
        if replay.get("region"):
            filters["region"] = replay["region"]

        actions.extend(
            [
                self._low_action(
                    "replay_interaction",
                    f"Replay learned action for {label}",
                    "learned_control",
                    label,
                    {
                        "app_id": app_id,
                        "filters": filters,
                        "control_signature": replay.get("control_signature", ""),
                        "settle_wait": 0.8,
                        "expected_reward": replay.get("reward_avg", 0.0),
                    },
                    success=f"Learned interaction for {label} should reproduce its rewarded outcome.",
                    fallback_hint="If the learned target is missing, re-map the current UI and run interaction learning again.",
                ),
                self._low_action(
                    "wait_for",
                    f"Wait after replaying {label}",
                    "time",
                    "0.8",
                    {"seconds": 0.8, "expected_focus": app_id if app_id != "current_window" else ""},
                    success="The UI should settle after replay.",
                ),
                self._low_action(
                    "scene_diff",
                    f"Summarize replay result for {label}",
                    "ui_surface",
                    app_id,
                    {
                        "expected_scene": f"Learned interaction for {label} should reproduce its rewarded outcome.",
                        "recovery_hint": "Run interaction learning again if the scene no longer matches learned memory.",
                    },
                    success="Replay scene delta should be recorded.",
                ),
            ]
        )

        steps = self._compile_runtime_actions(actions, [app_id] if app_id and app_id != "current_window" else [])
        return ExecutionPlan(
            task=task,
            steps=steps,
            source="interaction_graph_replay",
            summary=f"Replayed learned interaction for {label}.",
            required_apps=sorted({app_id} - {"current_window", ""}),
            success_conditions=[step.success_criteria for step in steps if step.success_criteria],
        )

    def _expand_reasoned_actions(
        self,
        reasoned: dict[str, Any],
        environment: dict[str, Any],
        observation: Any | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        runtime_actions: list[dict[str, Any]] = []
        required_apps = list(reasoned.get("required_apps", []))
        dom_snapshot = self._browser_snapshot(environment)

        for action in reasoned.get("actions", []):
            if not isinstance(action, dict):
                continue
            verb = str(action.get("verb", "")).strip()
            parameters = dict(action.get("parameters", {}))
            target = str(action.get("target", ""))
            title = str(action.get("title", verb.replace("_", " ").title()))
            success = str(action.get("success", ""))
            fallback_hint = str(action.get("fallback_hint", ""))

            if verb == "ensure_app":
                app_id = str(parameters.get("app_id") or target).strip()
                window_title = str(parameters.get("window_title", "")).strip()
                runtime_actions.extend(
                    self._ensure_app_actions(app_id=app_id, window_title=window_title)
                )
                continue

            if verb == "observe":
                output_filename = str(parameters.get("output_filename", "panel_parse.png"))
                runtime_actions.append(
                    self._low_action(
                        "parse_ui",
                        title,
                        "ui_surface",
                        str(parameters.get("app_id") or target or "current_window"),
                        {"output_filename": output_filename},
                        success=success or "A current UI observation should be captured.",
                    )
                )
                continue

            if verb == "navigate_sidebar":
                location = str(parameters.get("location") or target).strip()
                runtime_actions.extend(
                    [
                        self._low_action(
                            "open_explorer_location",
                            title,
                            "location",
                            location,
                            {
                                "app_id": "explorer",
                                "location": location,
                                "ranking": {
                                    "candidate_count": 1,
                                    "top_candidate_score": 1.0,
                                    "runner_up_score": 0.0,
                                    "score_gap": 1.0,
                                    "duplicate_disambiguation_used": True,
                                    "ambiguous": False,
                                    "ambiguity_reason": "",
                                    "selected_label": f"Explorer {location}",
                                },
                            },
                            success=success or f"Explorer should navigate to {location}.",
                            fallback_hint=fallback_hint or f"Retry Explorer deterministic navigation to {location}.",
                        ),
                        self._low_action(
                            "wait_for",
                            "Wait for explorer to settle",
                            "application",
                            "explorer",
                            {"seconds": 0.2, "expected_focus": "explorer", "timeout": 1.0},
                            success="explorer should remain focused.",
                        ),
                        self._low_action(
                            "verify_scene",
                            f"Verify scene after {title.lower()}",
                            "ui_surface",
                            "explorer",
                            {
                                "expected_app": "explorer",
                                "expected_labels": [location],
                            },
                            success=success or f"Explorer should navigate to {location}.",
                            fallback_hint=fallback_hint or f"Retry Explorer deterministic navigation to {location}.",
                        ),
                        self._low_action(
                            "scene_diff",
                            f"Summarize scene change after {title.lower()}",
                            "ui_surface",
                            "explorer",
                            {
                                "expected_scene": success or f"Explorer should navigate to {location}.",
                                "recovery_hint": fallback_hint or f"Retry Explorer deterministic navigation to {location}.",
                            },
                            success="Scene delta should be recorded.",
                        ),
                    ]
                )
                continue

            if verb == "focus_search":
                runtime_actions.append(
                    self._low_action(
                        "press_key",
                        title,
                        "keyboard",
                        "f3",
                        {
                            "keys": ["f3"],
                            "hotkey": False,
                            "expected_app": "explorer",
                            "shortcut_id": "search",
                            "expected_change": success or "Explorer search should be focused.",
                        },
                        success=success or "Explorer search should be focused.",
                        fallback_hint=fallback_hint or "Refocus Explorer and retry F3.",
                    )
                )
                continue

            if verb == "press_shortcut":
                keys = [str(key).lower() for key in parameters.get("keys", []) if str(key).strip()]
                app_id = self._resolve_runtime_app(str(parameters.get("app_id") or target).strip(), environment)
                runtime_actions.append(
                    self._low_action(
                        "press_key",
                        title,
                        "keyboard",
                        "+".join(keys),
                        {
                            "keys": keys,
                            "hotkey": bool(parameters.get("hotkey", len(keys) > 1)),
                            "expected_app": app_id,
                            "shortcut_id": str(parameters.get("shortcut_id", "")),
                            "expected_change": success or str(parameters.get("expected_change", "")),
                            "destructive": bool(parameters.get("destructive", False)),
                            "requires_approval": bool(parameters.get("requires_approval", False)),
                            "allow_destructive_shortcut": bool(parameters.get("allow_destructive_shortcut", False)),
                            "settle_wait": float(parameters.get("settle_wait", 0.2)),
                        },
                        success=success or str(parameters.get("expected_change", "")) or f"{'+'.join(keys)} should complete.",
                        fallback_hint=fallback_hint or f"Refocus {app_id} and retry {'+'.join(keys)}.",
                        risk_tags=["shortcut"] + (["destructive"] if bool(parameters.get("destructive", False)) else []),
                    )
                )
                continue

            if verb == "type_query":
                query = str(parameters.get("query") or target).strip()
                runtime_actions.extend(
                    [
                        self._low_action(
                            "type_text",
                            title,
                            "text",
                            query,
                            {
                                "text": query,
                                "clear_first": True,
                                "selector": "",
                                "focused_target": "Explorer search",
                                "expected_app": str(parameters.get("app_id") or "explorer"),
                            },
                            success=success or "The query text should be entered.",
                        ),
                        self._low_action(
                            "verify_scene",
                            f"Verify query text for {query}",
                            "ui_surface",
                            str(parameters.get("app_id") or "explorer"),
                            {
                                "expected_app": str(parameters.get("app_id") or "explorer"),
                                "expected_labels": self._query_expectations(query),
                            },
                            success=f"The query '{query}' should be visible.",
                            fallback_hint=fallback_hint,
                        ),
                    ]
                )
                continue

            if verb == "submit_query":
                query = str(parameters.get("query", "")).strip()
                app_id = str(parameters.get("app_id") or "explorer")
                runtime_actions.extend(
                    [
                        self._low_action(
                            "press_key",
                            title,
                            "keyboard",
                            "enter",
                            {"keys": ["enter"]},
                            success=success or "The query should be submitted.",
                        ),
                        self._low_action(
                            "wait_for",
                            f"Wait for {app_id} query results",
                            "application",
                            app_id,
                            {"seconds": 0.35, "expected_focus": app_id, "timeout": 1.0},
                            success=f"{app_id} should remain focused after the query.",
                        ),
                        self._low_action(
                            "verify_scene",
                            f"Verify results after submitting {query}",
                            "ui_surface",
                            app_id,
                            {
                                "expected_app": app_id,
                                "expected_labels": self._query_expectations(query),
                            },
                            success=f"The scene should reflect the submitted query '{query}'.",
                            fallback_hint=fallback_hint,
                        ),
                        self._low_action(
                            "scene_diff",
                            f"Summarize scene change after submitting {query}",
                            "ui_surface",
                            app_id,
                            {
                                "expected_scene": success or f"{app_id} should react to the submitted query.",
                                "recovery_hint": fallback_hint or f"Retry the query inside {app_id} after refocusing the window.",
                            },
                            success="The query-triggered scene change should be recorded.",
                        ),
                    ]
                )
                continue

            if verb == "open_visible_target":
                label = str(parameters.get("label") or target).strip()
                filters = self._apply_memory_bias(
                    {
                        "region": "main_page",
                        "label_contains": label,
                        "affordance": "open",
                        "min_score": 0.55,
                    },
                    app_id="explorer",
                    target_hint=label,
                )
                runtime_actions.extend(
                    self._click_chain(
                        title=title,
                        app_id="explorer",
                        target_label=label,
                        filters=filters,
                        expected_change=success or f"{label} should open from Explorer.",
                        expected_labels=[label],
                        click_count=2,
                        target_kind="label",
                        fallback_hint=fallback_hint or f"Reparse Explorer and retry opening {label}.",
                    )
                )
                continue

            if verb == "classify_visible_items":
                entity_type = str(parameters.get("entity_type") or target or "file").strip()
                runtime_actions.extend(
                    [
                        self._low_action(
                            "parse_ui",
                            f"Parse Explorer before classifying {entity_type}",
                            "ui_surface",
                            "explorer",
                            {"output_filename": f"classify_{entity_type}.png"},
                            success="Explorer UI should be parsed before classification.",
                        ),
                        self._low_action(
                            "checkpoint",
                            title,
                            "ui_surface",
                            "explorer",
                            {
                                "expected_scene": success or f"Explorer should classify visible items as {entity_type}.",
                                "filters": {
                                    "app_id": "explorer",
                                    "region": "main_page",
                                    "entity_type": entity_type,
                                    "min_score": 0.45,
                                },
                                "expected_labels": [entity_type],
                                "recovery_hint": fallback_hint or "Teach ambiguous Explorer rows if classification remains unclear.",
                            },
                            success=success or f"At least one visible Explorer item should classify as {entity_type}.",
                            fallback_hint=fallback_hint,
                        ),
                    ]
                )
                continue

            if verb == "summarize_folder":
                runtime_actions.extend(
                    [
                        self._low_action(
                            "parse_ui",
                            "Parse Explorer for folder summary",
                            "ui_surface",
                            "explorer",
                            {"output_filename": "explorer_summary.png"},
                            success="Explorer UI should be parsed for summarization.",
                        ),
                        self._low_action(
                            "checkpoint",
                            title,
                            "ui_surface",
                            "explorer",
                            {
                                "expected_scene": success or "The current Explorer folder should be summarized.",
                                "filters": {
                                    "app_id": "explorer",
                                    "region": "main_page",
                                    "min_score": 0.35,
                                },
                                "recovery_hint": fallback_hint or "Reparse Explorer and retry summarization.",
                            },
                            success=success or "Explorer folder contents should be summarized.",
                            fallback_hint=fallback_hint,
                        ),
                    ]
                )
                continue

            if verb == "browser_search":
                query = str(parameters.get("query") or target).strip()
                scope = str(parameters.get("scope", "web")).strip() or "web"
                typed_query = f"youtube {query}".strip() if scope == "youtube" else query
                runtime_actions.extend(
                    [
                        self._low_action(
                            "press_key",
                            "Focus Chrome address bar",
                            "keyboard",
                            "ctrl+l",
                            {
                                "keys": ["ctrl", "l"],
                                "hotkey": True,
                                "expected_app": "chrome",
                                "ranking": {
                                    "candidate_count": 1,
                                    "top_candidate_score": 1.0,
                                    "runner_up_score": 0.0,
                                    "score_gap": 1.0,
                                    "duplicate_disambiguation_used": True,
                                    "ambiguous": False,
                                    "ambiguity_reason": "",
                                    "selected_label": "Chrome address bar",
                                },
                            },
                            success="Chrome address bar should be focused by deterministic shortcut.",
                            fallback_hint="Refocus Chrome and retry Ctrl+L; do not click page UI for the address bar.",
                        ),
                        self._low_action(
                            "type_text",
                            f"Type Chrome query: {typed_query}",
                            "text",
                            typed_query,
                            {
                                "text": typed_query,
                                "selector": "",
                                "clear_first": False,
                                "focused_target": "Chrome address bar",
                                "expected_app": "chrome",
                                "deterministic_focus": "ctrl+l",
                            },
                            success=f"The Chrome query '{typed_query}' should be entered.",
                            fallback_hint=fallback_hint,
                        ),
                        self._low_action(
                            "verify_scene",
                            f"Verify Chrome query text for {typed_query}",
                            "ui_surface",
                            "chrome",
                            {
                                "expected_app": "chrome",
                                "expected_labels": self._query_expectations(typed_query, extras=["chrome"]),
                            },
                            success=f"The query '{typed_query}' should be visible in Chrome.",
                            fallback_hint=fallback_hint,
                        ),
                        self._low_action(
                            "press_key",
                            "Submit Chrome query",
                            "keyboard",
                            "enter",
                            {"keys": ["enter"]},
                            success="Chrome should submit the typed query.",
                            fallback_hint=fallback_hint,
                        ),
                        self._low_action(
                            "wait_for",
                            "Wait for Chrome results page",
                            "application",
                            "chrome",
                            {"seconds": 0.4, "expected_focus": "chrome", "timeout": 1.2},
                            success="Chrome should remain focused while results load.",
                        ),
                        self._low_action(
                            "verify_scene",
                            "Verify Chrome results page",
                            "ui_surface",
                            "chrome",
                            {
                                "expected_app": "chrome",
                                "expected_labels": self._query_expectations(query, extras=["youtube" if scope == "youtube" else "search"]),
                            },
                            success=success or "Chrome should show a relevant results page.",
                            fallback_hint=fallback_hint,
                        ),
                        self._low_action(
                            "scene_diff",
                            "Summarize Chrome results page change",
                            "ui_surface",
                            "chrome",
                            {
                                "expected_scene": success or "Chrome should move from query entry to a results page.",
                                "recovery_hint": fallback_hint or "Retry the Chrome query after refocusing the omnibox.",
                            },
                            success="The Chrome search scene delta should be recorded.",
                        ),
                    ]
                )
                continue

            if verb == "verify_results_page":
                runtime_actions.append(
                    self._low_action(
                        "verify_scene",
                        title,
                        "ui_surface",
                        "chrome",
                        {
                            "expected_app": "chrome",
                            "expected_labels": ["search", "results", "google", "youtube"],
                        },
                        success=success or "Chrome should show a search results page.",
                        fallback_hint=fallback_hint,
                    )
                )
                continue

            if verb == "open_safe_result":
                query = str(parameters.get("query") or target).strip()
                blocked_terms = ["close", "delete", "submit", "payment", "account", "sign out"]
                selector_candidates = self.browser.rank_selector_candidates(
                    dom_snapshot,
                    purpose="safe_result",
                    query=query,
                    blocked_terms=blocked_terms,
                )
                filters = self._apply_memory_bias(
                    {
                        "app_id": "chrome",
                        "region": "main_page",
                        "affordance": "open",
                        "semantic_role": "clickable_container",
                        "label_contains": query,
                        "min_score": 0.46,
                    },
                    app_id="chrome",
                    target_hint=query,
                )
                runtime_actions.extend(
                    self._click_chain(
                        title=title,
                        app_id="chrome",
                        target_label=query or "Chrome result",
                        filters=filters,
                        selector_candidates=selector_candidates,
                        expected_change=success or f"Chrome should open a safe result matching {query}.",
                        expected_labels=self._query_expectations(query, extras=["youtube", "result"]),
                        target_kind="page_content",
                        fallback_hint=fallback_hint or "Retry with a safer, query-matching browser result target.",
                    )
                )
                continue

            if verb == "dismiss_modal":
                selector_candidates = self.browser.rank_selector_candidates(dom_snapshot, purpose="modal_dismiss", blocked_terms=["delete", "payment", "account"])
                filters = self._apply_memory_bias(
                    {
                        "app_id": "chrome",
                        "entity_type": "modal_dialog",
                        "semantic_role": "clickable_container",
                        "min_score": 0.45,
                    },
                    app_id="chrome",
                    target_hint="dismiss",
                )
                runtime_actions.extend(
                    self._click_chain(
                        title=title,
                        app_id="chrome",
                        target_label="Chrome modal dismiss",
                        filters=filters,
                        selector_candidates=selector_candidates,
                        expected_change=success or "A simple Chrome modal should be dismissed.",
                        fallback_hint=fallback_hint or "Retry with a safe modal dismiss target only.",
                    )
                )
                continue

            if verb == "explore_safe":
                app_id = self._resolve_runtime_app(str(parameters.get("app_id") or target).strip(), environment)
                runtime_actions.append(
                    self._low_action(
                        "explore_safe",
                        title,
                        "application",
                        app_id,
                        {"rounds": int(parameters.get("rounds", 4)), "app_id": app_id},
                        success=success,
                        fallback_hint=fallback_hint,
                    )
                )
                continue

            if verb == "hover_probe":
                app_id = self._resolve_runtime_app(str(parameters.get("app_id") or target).strip(), environment)
                runtime_actions.append(
                    self._low_action(
                        "hover_probe",
                        title,
                        "application",
                        app_id,
                        {
                            "app_id": app_id,
                            "max_nodes": int(parameters.get("max_nodes", 8)),
                            "settle_wait": float(parameters.get("settle_wait", 0.45)),
                            "filters": self._apply_memory_bias(
                                dict(parameters.get("filters", {})),
                                app_id=app_id,
                                target_hint=str(parameters.get("target_hint", "")),
                            ),
                        },
                        success=success or "Hover feedback should be harvested without clicking.",
                        fallback_hint=fallback_hint or "Parse again and reduce the hover target count if feedback is unstable.",
                    )
                )
                continue

            if verb == "learning_session":
                app_id = self._resolve_runtime_app(str(parameters.get("app_id") or target).strip(), environment)
                runtime_actions.append(
                    self._low_action(
                        "learning_session",
                        title,
                        "application",
                        app_id,
                        {
                            "app_id": app_id,
                            "max_nodes": int(parameters.get("max_nodes", 8)),
                            "settle_wait": float(parameters.get("settle_wait", 0.45)),
                            "filters": self._apply_memory_bias(
                                {
                                    "exclude_destructive": True,
                                    **dict(parameters.get("filters", {})),
                                },
                                app_id=app_id,
                                target_hint=str(parameters.get("target_hint", "learn")),
                            ),
                        },
                        success=success or "A safe learning session should record hover feedback and review items.",
                        fallback_hint=fallback_hint or "Parse again and reduce target count if feedback is unstable.",
                    )
                )
                continue

            if verb == "interaction_learning":
                app_id = self._resolve_runtime_app(str(parameters.get("app_id") or target).strip(), environment)
                runtime_actions.append(
                    self._low_action(
                        "interaction_learning",
                        title,
                        "application",
                        app_id,
                        {
                            "app_id": app_id,
                            "max_actions": int(parameters.get("max_actions", 5)),
                            "settle_wait": float(parameters.get("settle_wait", 0.8)),
                            "recover_after_each": bool(parameters.get("recover_after_each", True)),
                            "filters": self._apply_memory_bias(
                                {
                                    "exclude_destructive": True,
                                    **dict(parameters.get("filters", {})),
                                },
                                app_id=app_id,
                                target_hint=str(parameters.get("target_hint", "open")),
                            ),
                        },
                        success=success or "Safe clicks should produce rewarded or punished interaction edges.",
                        fallback_hint=fallback_hint or "Stop, recover focus, and reduce action count if scene changes are unstable.",
                    )
                )
                continue

            if verb in {"route_window", "confirm_focus", "parse_ui", "click_node", "modified_click_node", "type_text", "press_key", "wait_for", "ocr_read", "verify_scene", "scene_diff", "checkpoint", "recover", "hover_probe", "learning_session", "interaction_learning", "replay_interaction"}:
                runtime_actions.append(
                    self._low_action(
                        verb,
                        title,
                        str(action.get("target_kind", "semantic_target")),
                        str(action.get("target", "")),
                        parameters,
                        success=success,
                        fallback_hint=fallback_hint,
                        description=str(action.get("description", title)),
                        confidence=float(action.get("confidence", 0.82)),
                        risk_tags=list(action.get("risk_tags", [])),
                    )
                )

        return runtime_actions, required_apps

    def compile(
        self,
        task: TaskSpec,
        observation: Any | None = None,
        environment: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        environment = environment or {}
        workflow = self.memory_store.find_workflow(task.prompt)
        if workflow and self._workflow_matches_context(workflow, observation, environment):
            steps = []
            for idx, raw_step in enumerate(workflow.get("steps", []), start=1):
                target = raw_step.get("target") or None
                target_obj = ActionTarget(**target) if target else None
                intent = raw_step.get("intent") or None
                intent_obj = ActionIntent(**intent) if intent else None
                steps.append(
                    PlanStep(
                        step_id=raw_step.get("step_id", f"wf_step_{idx}"),
                        title=raw_step.get("title", f"Workflow step {idx}"),
                        action_type=raw_step.get("action_type", "parse_ui"),
                        target=target_obj,
                        intent=intent_obj,
                        parameters=raw_step.get("parameters", {}),
                        confidence=float(raw_step.get("confidence", 0.9)),
                        risk_level=RiskLevel(raw_step.get("risk_level", RiskLevel.LOW.value)),
                        requires_approval=bool(raw_step.get("requires_approval", False)),
                        success_criteria=raw_step.get("success_criteria", ""),
                        fallback_hint=raw_step.get("fallback_hint", ""),
                        control_modes=[ControlMode(mode) for mode in raw_step.get("control_modes", [ControlMode.HYBRID.value])],
                    )
                )
            return ExecutionPlan(
                task=task,
                steps=steps,
                source="trusted_workflow_memory",
                summary=f"Reused trusted workflow for: {task.goal}",
                required_apps=workflow.get("required_apps", []),
                success_conditions=["Execute the trusted workflow successfully in the current scene."],
            )

        learned_replay = self._compile_learned_replay(task, environment, observation=observation)
        if learned_replay:
            return learned_replay

        reasoned = self.reasoner.decompose_task(
            task,
            observation=observation,
            environment=environment,
        )
        runtime_actions, required_apps = self._expand_reasoned_actions(reasoned, environment, observation)
        steps = self._compile_runtime_actions(runtime_actions, required_apps)

        if not steps:
            steps = self._compile_runtime_actions(
                [
                    self._low_action(
                        "parse_ui",
                        "Parse current UI for planning context",
                        "ui_surface",
                        "current_window",
                        {"output_filename": "panel_parse.png"},
                        success="The system should capture the current UI before taking action.",
                        fallback_hint="Ask the user to teach the target if the UI remains ambiguous.",
                    )
                ],
                [],
            )

        scene = reasoned.get("scene", {}) if isinstance(reasoned, dict) else {}
        success_conditions = [step.success_criteria for step in steps if step.success_criteria]
        summary = scene.get("summary") or scene.get("subgoal") or f"Plan for: {task.goal}"
        return ExecutionPlan(
            task=task,
            steps=steps,
            source="reasoner_compiler_v2",
            summary=summary,
            required_apps=sorted(set(required_apps)),
            success_conditions=success_conditions,
        )
