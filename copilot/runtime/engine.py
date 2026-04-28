from __future__ import annotations

from typing import Any, Callable
from copy import deepcopy
import json
import os
import time

from copilot.memory.store import MemoryStore
from copilot.perception.bridge import VisionRuntimeBridge
from copilot.perception.target_ranking import result_to_contract_metrics
from copilot.planner.compiler import PromptCompiler
from copilot.runtime.action_executor import ActionExecutor
from copilot.runtime.desktop_state import DesktopState, DesktopStateManager
from copilot.runtime.policy import PolicyEngine
from copilot.runtime.repair_planner import RepairPlanner
from copilot.runtime.replanner import Replanner
from copilot.runtime.recovery import RecoveryPlanner
from copilot.runtime.run_control import CancelLevel, CancelState
from copilot.runtime.task_state import TaskState, TaskStateManager
from copilot.runtime.voice_narrator import VoiceNarrator
from copilot.schemas import (
    ActionOutcome,
    ExecutionPlan,
    MissionCheckpoint,
    MissionState,
    MissionStatus,
    ObservationGraph,
    ObservationNode,
    RunStatus,
    RunTrace,
    SceneDelta,
    TaskSpec,
    TrustMode,
)


ApprovalCallback = Callable[[str, dict[str, Any]], bool]
TraceCallback = Callable[[dict[str, Any]], None]


class StateObservationCache:
    def __init__(self, ttl_seconds: float = 2.0) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self._entry: dict[str, Any] | None = None
        self.hit_count = 0
        self.miss_count = 0

    def get(self, state: DesktopState) -> tuple[ObservationGraph | None, dict[str, Any]]:
        entry = self._entry
        now = time.time()
        if not entry or not state.state_signature:
            self.miss_count += 1
            return None, {"state_cache_hit": False, "state_cache_miss": True, "state_cache_reason": "empty"}
        if entry.get("state_signature") != state.state_signature:
            self.miss_count += 1
            return None, {"state_cache_hit": False, "state_cache_miss": True, "state_cache_reason": "signature_changed"}
        age = max(0.0, now - float(entry.get("timestamp", 0.0) or 0.0))
        if age > float(entry.get("ttl_seconds", self.ttl_seconds) or self.ttl_seconds):
            self.miss_count += 1
            return None, {"state_cache_hit": False, "state_cache_miss": True, "state_cache_reason": "expired", "state_cache_age_seconds": round(age, 6)}
        parse_mode = str(entry.get("parse_mode", ""))
        if parse_mode not in {"browser_dom", "windows_uia", "browser_dom_required_failed"}:
            self.miss_count += 1
            return None, {"state_cache_hit": False, "state_cache_miss": True, "state_cache_reason": "uncacheable_parse_mode", "state_cache_age_seconds": round(age, 6)}
        graph = entry.get("graph")
        if not isinstance(graph, ObservationGraph):
            self.miss_count += 1
            return None, {"state_cache_hit": False, "state_cache_miss": True, "state_cache_reason": "missing_graph"}
        self.hit_count += 1
        return deepcopy(graph), {
            "state_cache_hit": True,
            "state_cache_miss": False,
            "state_cache_age_seconds": round(age, 6),
            "state_signature": state.state_signature,
            "state_cache_parse_mode": parse_mode,
        }

    def put(self, state: DesktopState, graph: ObservationGraph, parse_health: dict[str, Any]) -> None:
        parse_mode = str(parse_health.get("parse_mode", "") or graph.metadata.get("parse_health", {}).get("parse_mode", ""))
        if parse_mode not in {"browser_dom", "windows_uia", "browser_dom_required_failed"}:
            return
        if not state.state_signature:
            return
        self._entry = {
            "state_signature": state.state_signature,
            "timestamp": time.time(),
            "ttl_seconds": self.ttl_seconds,
            "parse_mode": parse_mode,
            "graph": deepcopy(graph),
            "parse_health": dict(parse_health),
        }

    def invalidate(self, reason: str = "") -> None:
        self._entry = None


class CopilotEngine:
    def __init__(self) -> None:
        self.memory = MemoryStore()
        self.bridge = VisionRuntimeBridge(self.memory)
        self.executor = ActionExecutor(self.bridge, self._parse_for_executor, cancel_callback=self._cancel_requested)
        self.state_manager = DesktopStateManager(self.bridge, state_path=os.path.join(self.memory.base_dir, "desktop_state.json"))
        self.task_state_manager = TaskStateManager()
        self.planner = PromptCompiler(self.memory)
        self.replanner = Replanner(self.planner)
        self.repair_planner = RepairPlanner()
        self.policy = PolicyEngine(self.memory)
        self.recovery = RecoveryPlanner()
        self.voice_narrator = VoiceNarrator.from_env()
        self.stop_requested = False
        self.cancel_state = CancelState()
        self.last_observation: ObservationGraph | None = None
        self.last_plan: ExecutionPlan | None = None
        self.last_state: DesktopState | None = None
        self.current_task_state: TaskState | None = None
        self.current_trace: RunTrace | None = None
        self.current_voice_narrator: VoiceNarrator | None = None
        self.state_observation_cache = StateObservationCache()

    @property
    def profiles(self):
        return self.planner.profiles

    @property
    def reasoner(self):
        return self.planner.reasoner

    def request_stop(self) -> None:
        self.request_cancel("soft")

    def request_cancel(self, level: str = "soft") -> None:
        try:
            cancel_level = CancelLevel(level)
        except ValueError:
            cancel_level = CancelLevel.SOFT
        self.stop_requested = True
        if not hasattr(self, "cancel_state"):
            self.cancel_state = CancelState()
        self.cancel_state.level = cancel_level
        self.cancel_state.requested_at = time.time()

    def reset_stop(self) -> None:
        self.stop_requested = False
        self.cancel_state = CancelState()

    def _cancel_requested(self) -> bool:
        stop_requested = bool(getattr(self, "stop_requested", False))
        cancel_state = getattr(self, "cancel_state", None)
        cancel_requested = bool(cancel_state.requested()) if cancel_state is not None else False
        return bool(stop_requested or cancel_requested)

    def _record_cancel_effective(self, trace: RunTrace, step_id: str, *, forced: bool = False) -> None:
        if not self.cancel_state.requested():
            self.cancel_state.level = CancelLevel.SOFT
            self.cancel_state.requested_at = self.cancel_state.requested_at or time.time()
        self.cancel_state.effective_at = time.time()
        self.cancel_state.cancelled_step_id = step_id
        self.cancel_state.forced_cancel = forced
        trace.outputs["cancel_level"] = self.cancel_state.level.value
        trace.outputs["cancel_requested_at"] = self.cancel_state.requested_at
        trace.outputs["cancel_effective_at"] = self.cancel_state.effective_at
        trace.outputs["cancelled_step_id"] = step_id
        trace.outputs["forced_cancel"] = bool(forced)

    def _sync_bridge_memory(self) -> None:
        if hasattr(self.bridge, "sync_agent_memory"):
            self.bridge.sync_agent_memory()
        elif hasattr(self.bridge, "agent"):
            self.bridge.agent.semantic_memory = self.memory.semantic_memory

    def build_task(self, prompt: str, trust_mode: TrustMode = TrustMode.PLAN_AND_RISK_GATES) -> TaskSpec:
        goal = prompt.strip() or "Observe the current UI"
        return TaskSpec(prompt=prompt, goal=goal, trust_mode=trust_mode)

    def _emit(self, callback: TraceCallback | None, phase: str, message: str, **metadata: Any) -> None:
        if callback:
            callback({"phase": phase, "message": message, "metadata": metadata, "timestamp": time.time()})

    def _voice_for_run(self, mode_override: str | None = None) -> VoiceNarrator:
        if mode_override is not None:
            return VoiceNarrator.from_env(mode_override=mode_override)
        if not hasattr(self, "voice_narrator") or self.voice_narrator is None:
            self.voice_narrator = VoiceNarrator.from_env()
        return self.voice_narrator

    def _narrate(
        self,
        trace: RunTrace | None,
        line: str,
        *,
        event_type: str = "runtime",
        throttle_key: str = "",
        **metadata: Any,
    ) -> None:
        narrator = getattr(self, "current_voice_narrator", None)
        if narrator is None:
            narrator = self._voice_for_run()
        narrator.speak(line, trace=trace, event_type=event_type, throttle_key=throttle_key, metadata=metadata)

    def _narrate_phase(
        self,
        trace: RunTrace | None,
        phase: str,
        *,
        step: Any | None = None,
        throttle_key: str = "",
        **metadata: Any,
    ) -> None:
        narrator = getattr(self, "current_voice_narrator", None)
        if narrator is None:
            narrator = self._voice_for_run()
        narrator.speak_phase(phase, trace=trace, step=step, throttle_key=throttle_key, metadata=metadata)

    def _annotate_graph(self, graph: ObservationGraph, environment: dict[str, Any], task: TaskSpec | None = None) -> ObservationGraph:
        graph, profile = self.profiles.annotate(
            graph,
            environment,
            app_id=environment.get("windows", {}).get("active_app_guess", ""),
        )
        scene = self.reasoner.interpret_scene(task or self.build_task("Observe current UI"), graph, environment)
        if profile and not scene.get("app_id"):
            scene["app_id"] = profile.app_id
            scene["app_name"] = profile.display_name
        graph.metadata["scene"] = scene
        graph.metadata["scene_summary"] = scene.get("summary", "")
        graph.metadata["app_id"] = scene.get("app_id", graph.metadata.get("app_id", ""))
        return graph

    def plan_prompt(self, prompt: str, trust_mode: TrustMode = TrustMode.PLAN_AND_RISK_GATES) -> ExecutionPlan:
        task = self.build_task(prompt, trust_mode=trust_mode)
        if getattr(self, "last_state", None) and self._state_manager().is_state_stale(self.last_state):
            self.last_observation = None
        environment = self.observe_environment()
        observation = self.last_observation
        self.last_state = self._state_manager().observe(graph=observation, last_action="plan_prompt")
        self.last_plan = self.planner.compile(task, observation=observation, environment=environment)
        return self.last_plan

    def parse_current_ui(self, output_filename: str = "ui_panel_parse.png", task: TaskSpec | None = None) -> ObservationGraph:
        probe_started = time.time()
        probe_state = self._state_manager().probe(last_action="parse_current_ui")
        probe_elapsed = round(time.time() - probe_started, 6)
        last_observation = getattr(self, "last_observation", None)
        last_state = getattr(self, "last_state", None)
        if last_observation is not None and last_state is not None and not self._state_manager().has_changed(last_state, probe_state):
            environment = self.observe_environment()
            cached = deepcopy(last_observation)
            cached.metadata["output_filename"] = output_filename
            cached.metadata["environment"] = environment
            parse_health = dict(cached.metadata.get("parse_health", {}) or {})
            parse_health.update(
                {
                    "cache_hit": True,
                    "state_cache_hit": True,
                    "state_cache_miss": False,
                    "state_cache_reason": "unchanged_persistent_state",
                    "state_probe_elapsed_seconds": probe_elapsed,
                    "state_hash": probe_state.state_hash,
                    "state_signature": probe_state.state_signature,
                    "parse_elapsed_seconds": 0.0,
                }
            )
            cached.metadata["parse_health"] = parse_health
            cached = self._annotate_graph(cached, environment, task=task)
            self._record_parse_health(cached)
            self.last_observation = cached
            self.last_state = self._state_manager().observe(graph=cached, last_action="parse_current_ui")
            return self.last_observation

        cached, cache_meta = self._state_cache().get(probe_state)
        if cached is not None:
            environment = self.observe_environment()
            cached.metadata["output_filename"] = output_filename
            cached.metadata["environment"] = environment
            parse_health = dict(cached.metadata.get("parse_health", {}) or {})
            parse_health.update(cache_meta)
            parse_health["cache_hit"] = True
            parse_health["state_probe_elapsed_seconds"] = probe_elapsed
            parse_health["parse_elapsed_seconds"] = 0.0
            cached.metadata["parse_health"] = parse_health
            cached = self._annotate_graph(cached, environment, task=task)
            self._record_parse_health(cached)
            self.last_observation = cached
            self.last_state = self._state_manager().observe(graph=cached, last_action="parse_current_ui")
            return self.last_observation

        environment = self.observe_environment()
        graph = self.bridge.parse_ui(output_filename=output_filename)
        graph = self._annotate_graph(graph, environment, task=task)
        parse_health = dict(graph.metadata.get("parse_health", {}) or {})
        parse_health.update(cache_meta)
        parse_health.setdefault("state_cache_hit", False)
        parse_health.setdefault("state_cache_miss", True)
        parse_health["state_probe_elapsed_seconds"] = probe_elapsed
        parse_health["state_hash"] = probe_state.state_hash
        parse_health["state_signature"] = probe_state.state_signature
        graph.metadata["parse_health"] = parse_health
        self._record_parse_health(graph)
        self.last_observation = graph
        self.last_state = self._state_manager().observe(graph=graph, last_action="parse_current_ui")
        self._state_cache().put(probe_state, graph, parse_health)
        self.memory.remember_observation_graph(graph)
        self._sync_bridge_memory()
        return self.last_observation

    def _record_parse_health(self, graph: ObservationGraph) -> None:
        trace = getattr(self, "current_trace", None)
        if trace is None:
            return
        parse_health = dict(graph.metadata.get("parse_health", {}) or {})
        if not parse_health:
            parse_health = dict(getattr(self.bridge, "last_parse_health", {}) or {})
        if parse_health:
            trace.outputs.setdefault("parse_health", []).append(parse_health)

    def _parse_for_executor(self, output_filename: str, task: TaskSpec) -> ObservationGraph:
        return self.parse_current_ui(output_filename=output_filename, task=task)

    def _action_executor(self) -> ActionExecutor:
        if not hasattr(self, "executor") or self.executor is None:
            self.executor = ActionExecutor(self.bridge, self._parse_for_executor, cancel_callback=self._cancel_requested)
        return self.executor

    def _state_manager(self) -> DesktopStateManager:
        if not hasattr(self, "state_manager") or self.state_manager is None:
            memory = getattr(self, "memory", None)
            base_dir = getattr(memory, "base_dir", "") if memory is not None else ""
            state_path = os.path.join(base_dir, "desktop_state.json") if base_dir else ""
            self.state_manager = DesktopStateManager(self.bridge, state_path=state_path)
        return self.state_manager

    def _state_cache(self) -> StateObservationCache:
        if not hasattr(self, "state_observation_cache") or self.state_observation_cache is None:
            self.state_observation_cache = StateObservationCache()
        return self.state_observation_cache

    def _invalidate_state_cache(self, reason: str = "") -> None:
        self._state_cache().invalidate(reason)

    def _action_invalidates_state_cache(self, action_type: str) -> bool:
        return action_type in {
            "legacy_command",
            "route_window",
            "open_explorer_location",
            "click_node",
            "modified_click_node",
            "click_point",
            "type_text",
            "press_key",
            "hotkey",
            "hover_probe",
            "learning_session",
            "interaction_learning",
            "replay_interaction",
            "explore_safe",
            "recover",
        }

    def _task_state_manager(self) -> TaskStateManager:
        if not hasattr(self, "task_state_manager") or self.task_state_manager is None:
            self.task_state_manager = TaskStateManager()
        return self.task_state_manager

    def _replanner(self) -> Replanner:
        if not hasattr(self, "replanner") or self.replanner is None:
            self.replanner = Replanner(self.planner)
        return self.replanner

    def _repair_planner(self) -> RepairPlanner:
        if not hasattr(self, "repair_planner") or self.repair_planner is None:
            self.repair_planner = RepairPlanner()
        return self.repair_planner

    def _recovery_planner(self) -> RecoveryPlanner:
        if not hasattr(self, "recovery") or self.recovery is None:
            self.recovery = RecoveryPlanner()
        return self.recovery

    def teach_node(
        self,
        node: ObservationNode,
        label: str,
        concepts: list[str],
        app_identity: str = "",
        risk_level: str = "",
        entity_type: str = "",
        affordances: list[str] | None = None,
        outcome_correct: bool = True,
    ) -> None:
        self.memory.teach_node(
            node=node,
            label=label,
            concepts=concepts,
            app_identity=app_identity,
            risk_level=risk_level,
            entity_type=entity_type,
            affordances=affordances or [],
            outcome_correct=outcome_correct,
        )
        self._sync_bridge_memory()

    def export_memory(self, target_path: str) -> str:
        return self.memory.export_pack(target_path)

    def import_memory(self, source_path: str) -> None:
        self.memory.import_pack(source_path)
        self._sync_bridge_memory()

    def get_memory_summary(self) -> dict[str, Any]:
        return self.memory.summary()

    def get_learning_dashboard(self) -> dict[str, Any]:
        return self.memory.interaction_dashboard()

    def get_operator_status(self) -> dict[str, Any]:
        return self.memory.operator_status()

    def get_recent_runs(self, limit: int = 12) -> list[dict[str, Any]]:
        return self.memory.recent_runs(limit=limit)

    def get_workflows(self) -> list[dict[str, Any]]:
        return self.memory.list_workflows()

    def get_skill_capsules(self) -> list[dict[str, Any]]:
        return self.memory.list_skill_capsules()

    def get_review_items(self, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
        return self.memory.list_review_items(status=status, limit=limit)

    def resolve_review_item(
        self,
        review_id: str,
        status: str,
        label: str = "",
        concepts: list[str] | None = None,
        entity_type: str = "",
        affordances: list[str] | None = None,
        app_identity: str = "",
        note: str = "",
    ) -> bool:
        ok = self.memory.resolve_review_item(
            review_id=review_id,
            status=status,
            label=label,
            concepts=concepts,
            entity_type=entity_type,
            affordances=affordances,
            app_identity=app_identity,
            note=note,
        )
        self._sync_bridge_memory()
        return ok

    def save_current_plan_as_skill(self, name: str = "") -> dict[str, Any] | None:
        if not self.last_plan:
            return None
        return self.memory.save_plan_as_workflow(
            task=self.last_plan.task,
            plan=self.last_plan,
            name=name,
            promotion_state="draft",
        )

    def promote_workflow(self, workflow_id: str, promotion_state: str = "trusted") -> bool:
        return self.memory.promote_workflow(workflow_id, promotion_state=promotion_state)

    def approve_workflow(self, workflow_id: str) -> bool:
        return self.memory.approve_workflow(workflow_id)

    def allow_high_risk_for_app(self, app_id: str) -> bool:
        return self.memory.allow_high_risk_for_app(app_id)

    def revoke_high_risk_for_app(self, app_id: str) -> bool:
        return self.memory.revoke_high_risk_for_app(app_id)

    def record_skill_replay(
        self,
        workflow_id: str,
        *,
        success: bool,
        variant_count: int = 1,
        latency_seconds: float | None = None,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return self.memory.record_skill_replay(
            workflow_id,
            success=success,
            variant_count=variant_count,
            latency_seconds=latency_seconds,
            trace=trace,
        )

    def build_skill_manifest(self, workflow_ids: list[str] | None = None) -> dict[str, Any]:
        return self.memory.build_skill_manifest(workflow_ids)

    def import_skill_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        return self.memory.import_skill_manifest(manifest)

    def observe_environment(self) -> dict[str, Any]:
        return self.bridge.observe_environment()

    def _default_approval(self, prompt: str, payload: dict[str, Any]) -> bool:
        print(f"\nApproval required: {prompt}")
        for key, value in payload.items():
            print(f"  - {key}: {value}")
        answer = input("Approve? [y/N]: ").strip().lower()
        return answer in {"y", "yes"}

    def _build_mission(self, plan: ExecutionPlan) -> MissionState:
        checkpoints = [
            MissionCheckpoint(
                checkpoint_id=step.step_id,
                title=step.title,
                expected_scene=step.success_criteria,
                completion_rule=step.success_criteria,
                recovery_rule=step.fallback_hint or "Refocus, reparse, and retry once.",
            )
            for step in plan.steps
        ]
        current_id = checkpoints[0].checkpoint_id if checkpoints else ""
        return MissionState(
            mission_id=f"mission_{plan.task.task_id}",
            goal=plan.task.goal,
            subgoal=plan.summary,
            app_id=plan.required_apps[0] if plan.required_apps else "",
            status=MissionStatus.PLANNED,
            current_checkpoint_id=current_id,
            checkpoints=checkpoints,
        )

    def _set_checkpoint_status(self, mission: MissionState | None, step_id: str, status: str, note: str = "") -> None:
        if not mission:
            return
        for checkpoint in mission.checkpoints:
            if checkpoint.checkpoint_id == step_id:
                checkpoint.status = status
                if note:
                    checkpoint.notes.append(note)
                mission.current_checkpoint_id = step_id
                break

    def _ensure_observation(self, task: TaskSpec, output_filename: str) -> ObservationGraph:
        if self.last_observation:
            return self.last_observation
        return self.parse_current_ui(output_filename=output_filename, task=task)

    def _scene_contains_expected_labels(self, graph: ObservationGraph | None, expected_labels: list[str]) -> bool:
        if not expected_labels:
            return True

        graph_labels = []
        if graph:
            graph_labels.extend(node.display_label() for node in graph.flatten() if node.display_label())
            graph_labels.append(str(graph.metadata.get("scene_summary", "")))

        dom_snapshot = {}
        if hasattr(self.bridge, "read_browser_dom"):
            try:
                dom_snapshot = self.bridge.read_browser_dom() or {}
            except Exception:
                dom_snapshot = {}

        dom_texts = [str(dom_snapshot.get("title", "")), str(dom_snapshot.get("url", ""))]
        for item in dom_snapshot.get("items", []):
            if isinstance(item, dict):
                dom_texts.append(str(item.get("text", "")))
                dom_texts.append(str(item.get("aria_label", "")))
                dom_texts.append(str(item.get("placeholder", "")))

        try:
            environment = self.observe_environment() or {}
            active = environment.get("windows", {}).get("active_window", {})
            if isinstance(active, dict):
                dom_texts.append(str(active.get("title", "")))
            browser = environment.get("browser", {})
            if isinstance(browser, dict):
                dom_texts.append(str(browser.get("active_title", "")))
                dom_texts.append(str(browser.get("active_url", "")))
        except Exception:
            pass
        if hasattr(self.bridge, "current_explorer_location"):
            try:
                dom_texts.append(str(self.bridge.current_explorer_location()))
            except Exception:
                pass

        blob = " ".join(part.strip().lower() for part in (graph_labels + dom_texts) if str(part).strip())
        return any(label.strip().lower() in blob for label in expected_labels if str(label).strip())

    def _graph_label_set(self, graph: ObservationGraph | None) -> set[str]:
        if not graph:
            return set()
        labels = set()
        for node in graph.flatten():
            label = str(node.display_label() or "").strip()
            if not label or (label.startswith("[") and label.endswith("]")):
                continue
            labels.add(label)
        return labels

    def _hover_probe_candidates(
        self,
        graph: ObservationGraph,
        filters: dict[str, Any],
        app_id: str,
        max_nodes: int,
    ) -> list[ObservationNode]:
        scene = graph.metadata.get("scene", {})
        profile = self.profiles.get(app_id or graph.metadata.get("app_id", ""))
        candidates: list[ObservationNode] = []

        if filters:
            for node in graph.flatten():
                if self.reasoner.choose_action_target({**filters, "label_contains": node.display_label()}, graph, scene) == node:
                    candidates.append(node)

        if not candidates and profile:
            candidates = profile.safe_nodes(graph)

        if not candidates:
            candidates = [
                node
                for node in graph.flatten()
                if node.semantic_role in {"menu_item", "clickable_container", "list_row"}
                and "destructive" not in node.state_tags
            ]

        deduped = []
        seen: set[str] = set()
        for node in candidates:
            signature = "|".join(
                [
                    node.node_id,
                    node.display_label(),
                    node.semantic_role,
                    node.region,
                    str(node.center.get("x", "")),
                    str(node.center.get("y", "")),
                ]
            )
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(node)
            if len(deduped) >= max_nodes:
                break
        return deduped

    def _is_safe_learning_target(self, node: ObservationNode, app_id: str) -> tuple[bool, str]:
        label = (node.display_label() or node.label or "").lower()
        text_blob = " ".join([label, node.semantic_role, node.entity_type, node.region, " ".join(node.state_tags)]).lower()
        blocked_terms = {
            "close",
            "delete",
            "remove",
            "submit",
            "payment",
            "purchase",
            "account",
            "security",
            "password",
            "sign in",
            "logout",
            "log out",
        }
        if app_id and node.app_id and node.app_id != app_id:
            return False, "app_boundary"
        if node.region == "top_menu" and any(term in text_blob for term in {"close", "minimize", "maximize"}):
            return False, "titlebar_control"
        if any(term in text_blob for term in blocked_terms):
            return False, "risky_label"
        if "destructive" in node.state_tags:
            return False, "destructive_state"
        return True, ""

    def _learning_confidence(self, node: ObservationNode, feedback_labels: list[str]) -> float:
        score = 0.0
        label = node.display_label() or node.label
        if label and not (label.startswith("[") and label.endswith("]")):
            score += 0.35
        if node.entity_type:
            score += 0.2
        if node.semantic_role:
            score += 0.15
        if node.visual_id or getattr(node, "visual_ids", []):
            score += 0.15
        if feedback_labels:
            score += 0.15
        return min(1.0, score)

    def _interaction_allowed(self, node: ObservationNode, app_id: str) -> tuple[bool, str]:
        scoped_app = "" if app_id == "current_window" else app_id
        safe, reason = self._is_safe_learning_target(node, scoped_app)
        if not safe:
            return False, reason
        role = node.semantic_role
        entity = node.entity_type
        label = (node.display_label() or node.label or "").strip()
        if app_id == "explorer":
            if entity in {"navigation_item", "folder"}:
                return True, ""
            if role == "menu_item" and node.region == "left_menu":
                return True, ""
            return False, "explorer_interaction_learning_only_clicks_navigation_and_folders"
        if app_id == "chrome":
            if entity in {"tab", "search_field", "omnibox"}:
                return True, ""
            return False, "chrome_interaction_learning_avoids_page_clicks"
        if not label or (label.startswith("[") and label.endswith("]")):
            return False, "unlabeled_target"
        return role in {"menu_item", "clickable_container", "list_row"}, ""

    def _interaction_click_count(self, node: ObservationNode, app_id: str) -> int:
        if app_id == "explorer" and node.entity_type == "folder":
            return 2
        return 1

    def _scene_label_fingerprint(self, graph: ObservationGraph | None) -> set[str]:
        if not graph:
            return set()
        labels = set()
        for node in graph.flatten():
            label = self.memory.normalize_prompt(node.display_label() or node.label)
            if label and not (label.startswith("[") and label.endswith("]")):
                labels.add(label)
        return labels

    def _score_interaction_reward(
        self,
        before: ObservationGraph | None,
        after: ObservationGraph | None,
        app_id: str,
    ) -> tuple[float, str]:
        scoped_app = "" if app_id == "current_window" else app_id
        if not after:
            return -1.0, "no_after_scene"
        before_app = str((before.metadata.get("app_id", "") if before else "") or "").lower()
        after_app = str(after.metadata.get("app_id", "") or after.metadata.get("scene", {}).get("app_id", "")).lower()
        if scoped_app and after_app and after_app != scoped_app:
            return -1.0, "left_trusted_app"
        before_labels = self._scene_label_fingerprint(before)
        after_labels = self._scene_label_fingerprint(after)
        added = after_labels - before_labels
        removed = before_labels - after_labels
        before_summary = str(before.metadata.get("scene_summary", "") if before else "")
        after_summary = str(after.metadata.get("scene_summary", ""))
        changed = bool(added or removed or before_summary != after_summary or (before_app and before_app != after_app))
        if not changed:
            return -0.25, "no_visible_change"
        reward = 0.75
        if added:
            reward += min(0.4, len(added) * 0.05)
        if scoped_app and after_app == scoped_app:
            reward += 0.25
        return min(1.5, reward), "opened_or_navigated"

    def _recover_after_interaction(self, app_id: str, session_id: str, task: TaskSpec) -> str:
        try:
            if app_id == "explorer":
                self.bridge.press_key(["alt", "left"], hotkey=True)
            else:
                self.bridge.press_key(["esc"])
            self.bridge.wait_for(seconds=0.35)
            self.parse_current_ui(output_filename=f"{session_id}_recover.png", task=task)
            return "back" if app_id == "explorer" else "escape"
        except Exception:
            return "recovery_failed"

    def _execute_interaction_learning(
        self,
        step,
        trace: RunTrace,
        trace_callback: TraceCallback | None,
    ) -> tuple[bool, str, ObservationNode | None, str]:
        app_id = str(step.parameters.get("app_id") or (step.target.value if step.target else "")).strip()
        max_actions = max(1, min(int(step.parameters.get("max_actions", 5)), 12))
        settle_wait = float(step.parameters.get("settle_wait", 0.8))
        recover_after_each = bool(step.parameters.get("recover_after_each", True))
        filters = {"exclude_destructive": True, **dict(step.parameters.get("filters", {}))}
        session_id = f"interact_{int(time.time())}"
        memory_before = self.memory.summary()

        baseline = self.parse_current_ui(output_filename=f"{session_id}_baseline.png", task=trace.task)
        app_id = app_id or baseline.metadata.get("app_id", "")
        raw_candidates = self._hover_probe_candidates(baseline, filters, app_id, max_actions * 3)
        candidates: list[ObservationNode] = []
        skipped = []
        for candidate in raw_candidates:
            allowed, reason = self._interaction_allowed(candidate, app_id)
            if allowed:
                candidates.append(candidate)
            else:
                skipped.append({"label": candidate.display_label(), "node_id": candidate.node_id, "reason": reason})
            if len(candidates) >= max_actions:
                break

        trials = []
        current = baseline
        positive = 0
        negative = 0
        for idx, candidate in enumerate(candidates, start=1):
            if self.stop_requested:
                break
            before = current
            click_count = self._interaction_click_count(candidate, app_id)
            ok = self.bridge.click_node(candidate, clicks=click_count)
            if not ok:
                reward, outcome = -1.0, "click_failed"
                after = before
            else:
                self.bridge.wait_for(seconds=settle_wait)
                after = self.parse_current_ui(output_filename=f"{session_id}_after_{idx}.png", task=trace.task)
                reward, outcome = self._score_interaction_reward(before, after, app_id)

            recovery = ""
            edge = self.memory.record_interaction_outcome(
                before=before,
                after=after,
                node=candidate,
                action_type=f"click_{click_count}",
                reward=reward,
                outcome=outcome,
                app_id=app_id,
                recovery=recovery,
            )
            if reward > 0:
                positive += 1
            else:
                negative += 1

            if reward <= -0.75:
                self.memory.remember_negative_example(candidate, note=f"Interaction learning punished target: {outcome}")

            if recover_after_each:
                recovery = self._recover_after_interaction(app_id, session_id, trace.task)
                edge["recovery"] = recovery
                current = self.last_observation or after
            else:
                current = after

            trials.append(
                {
                    "target": candidate.display_label(),
                    "node_id": candidate.node_id,
                    "entity_type": candidate.entity_type,
                    "semantic_role": candidate.semantic_role,
                    "click_count": click_count,
                    "reward": reward,
                    "outcome": outcome,
                    "edge_id": edge.get("edge_id"),
                    "recovery": recovery,
                }
            )

        memory_after = self.memory.summary()
        session = {
            "session_id": session_id,
            "timestamp": time.time(),
            "app_id": app_id,
            "status": "cancelled" if self.stop_requested else "completed",
            "candidate_count": len(raw_candidates),
            "trial_count": len(trials),
            "positive_rewards": positive,
            "negative_rewards": negative,
            "skipped_count": len(skipped),
            "skipped": skipped,
            "trials": trials,
            "memory_before": memory_before,
            "memory_after": memory_after,
        }
        self.memory.record_learning_session(session)

        report_dir = os.path.join("debug_steps", "learning_sessions")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"{session_id}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)

        trace.outputs.setdefault("interaction_learning", []).append({**session, "report_path": report_path})
        trace.memory_hits["memory_summary"] = memory_after
        self._emit(
            trace_callback,
            "interaction_learning",
            "Interaction learning completed.",
            report_path=report_path,
            trials=len(trials),
            positive=positive,
            negative=negative,
            skipped=len(skipped),
        )
        return True, "hybrid", None, f"Interaction learning ran {len(trials)} trials; rewards +{positive}/-{negative}."

    def _execute_replay_interaction(
        self,
        step,
        trace: RunTrace,
        trace_callback: TraceCallback | None,
    ) -> tuple[bool, str, ObservationNode | None, str]:
        app_id = str(step.parameters.get("app_id") or (step.target.value if step.target else "")).strip()
        filters = dict(step.parameters.get("filters", {}))
        settle_wait = float(step.parameters.get("settle_wait", 0.8))
        before = self._ensure_observation(trace.task, output_filename=f"{step.step_id}_replay_seed.png")
        scene = before.metadata.get("scene", {})
        target_node = self.reasoner.choose_action_target(filters, before, scene)
        if not target_node:
            return False, "hybrid", None, "Learned replay target was not visible in the current scene."

        safe, reason = self._is_safe_learning_target(target_node, app_id if app_id != "current_window" else "")
        if not safe:
            return False, "hybrid", target_node, f"Learned replay target is no longer safe: {reason}."

        click_count = self._interaction_click_count(target_node, app_id)
        ok = self.bridge.click_node(target_node, clicks=click_count)
        if not ok:
            self.memory.record_interaction_outcome(
                before=before,
                after=before,
                node=target_node,
                action_type=f"replay_click_{click_count}",
                reward=-1.0,
                outcome="replay_click_failed",
                app_id=app_id,
            )
            return False, "hybrid", target_node, "Replay click failed."

        self.bridge.wait_for(seconds=settle_wait)
        after = self.parse_current_ui(output_filename=f"{step.step_id}_replay_after.png", task=trace.task)
        reward, outcome = self._score_interaction_reward(before, after, app_id if app_id != "current_window" else after.metadata.get("app_id", ""))
        self.memory.record_interaction_outcome(
            before=before,
            after=after,
            node=target_node,
            action_type=f"replay_click_{click_count}",
            reward=reward,
            outcome=f"replay_{outcome}",
            app_id=app_id,
        )
        if reward <= 0:
            self.memory.remember_negative_example(target_node, note=f"Replay punished learned target: {outcome}")
            return False, "hybrid", target_node, f"Replay did not reproduce a useful change: {outcome}."

        self._emit(
            trace_callback,
            "replay",
            "Learned interaction replayed.",
            target=target_node.display_label(),
            reward=reward,
            outcome=outcome,
        )
        return True, "hybrid", target_node, f"Replayed learned interaction '{target_node.display_label()}' with reward {reward:.2f}."

    def _execute_learning_session(
        self,
        step,
        trace: RunTrace,
        trace_callback: TraceCallback | None,
    ) -> tuple[bool, str, ObservationNode | None, str]:
        app_id = str(step.parameters.get("app_id") or (step.target.value if step.target else "")).strip()
        max_nodes = max(1, min(int(step.parameters.get("max_nodes", 8)), 24))
        settle_wait = float(step.parameters.get("settle_wait", 0.45))
        filters = {"exclude_destructive": True, **dict(step.parameters.get("filters", {}))}
        memory_before = self.memory.summary()
        session_id = f"learn_{int(time.time())}"

        baseline = self.parse_current_ui(output_filename=f"{session_id}_baseline.png", task=trace.task)
        app_id = app_id or baseline.metadata.get("app_id", "")
        candidates = self._hover_probe_candidates(baseline, filters, app_id, max_nodes * 2)
        safe_candidates: list[ObservationNode] = []
        skipped = []
        for candidate in candidates:
            safe, reason = self._is_safe_learning_target(candidate, app_id)
            if safe:
                safe_candidates.append(candidate)
            else:
                skipped.append({"label": candidate.display_label(), "node_id": candidate.node_id, "reason": reason})
            if len(safe_candidates) >= max_nodes:
                break

        hover_results = []
        review_items = []
        learned_count = 0
        before_labels = self._graph_label_set(baseline)
        graph = baseline

        for idx, candidate in enumerate(safe_candidates, start=1):
            if self.stop_requested:
                break
            if not hasattr(self.bridge, "hover_node") or not self.bridge.hover_node(candidate):
                hover_results.append({"target": candidate.display_label(), "ok": False, "feedback_labels": []})
                continue
            self.bridge.wait_for(seconds=settle_wait)
            after = self.parse_current_ui(output_filename=f"{session_id}_hover_{idx}.png", task=trace.task)
            after_labels = self._graph_label_set(after)
            added_labels = sorted(after_labels - before_labels)[:12]
            stable_feedback = [
                label
                for label in added_labels
                if label.lower() not in {candidate.display_label().lower(), "main page", "top menu", "left menu", "bottom menu"}
            ]
            if stable_feedback:
                self.memory.remember_hover_feedback(candidate, stable_feedback, app_id=app_id)
                learned_count += 1

            confidence = self._learning_confidence(candidate, stable_feedback)
            if confidence < 0.72 or not candidate.entity_type or not stable_feedback:
                review_items.append(
                    self.memory.enqueue_review_item(
                        candidate,
                        reason="low_confidence_learning_session",
                        feedback_labels=stable_feedback,
                        confidence=confidence,
                        app_id=app_id,
                    )
                )

            hover_results.append(
                {
                    "target": candidate.display_label(),
                    "node_id": candidate.node_id,
                    "entity_type": candidate.entity_type,
                    "confidence": confidence,
                    "feedback_labels": stable_feedback,
                }
            )
            graph = after
            before_labels = after_labels

        memory_after = self.memory.summary()
        session = {
            "session_id": session_id,
            "timestamp": time.time(),
            "app_id": app_id,
            "status": "cancelled" if self.stop_requested else "completed",
            "baseline_node_count": len(baseline.flatten()),
            "candidate_count": len(candidates),
            "hovered_count": len(hover_results),
            "learned_count": learned_count,
            "uncertain_count": len(review_items),
            "skipped_risky_count": len(skipped),
            "skipped": skipped,
            "hover_results": hover_results,
            "review_ids": [item.get("review_id") for item in review_items],
            "memory_before": memory_before,
            "memory_after": memory_after,
        }
        self.memory.record_learning_session(session)

        report_dir = os.path.join("debug_steps", "learning_sessions")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"{session_id}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)

        trace.outputs.setdefault("learning_sessions", []).append({**session, "report_path": report_path})
        trace.memory_hits["memory_summary"] = memory_after
        self.last_observation = graph
        self._emit(
            trace_callback,
            "learning",
            "Learning session completed.",
            report_path=report_path,
            hovered=len(hover_results),
            learned=learned_count,
            uncertain=len(review_items),
            skipped=len(skipped),
        )
        return True, "hybrid", None, f"Learning session hovered {len(hover_results)} safe targets; queued {len(review_items)} reviews."

    def _record_scene_delta(
        self,
        trace: RunTrace,
        before: ObservationGraph | None,
        after: ObservationGraph | None,
        step_id: str,
        expected_change: str = "",
    ) -> SceneDelta:
        delta = self.reasoner.summarize_scene_change(before, after, step_id=step_id, expected_change=expected_change)
        trace.scene_deltas.append(delta)
        self.memory.record_scene_delta(delta)
        trace.outputs.setdefault("scene_deltas", []).append(delta.to_dict())
        return delta

    def _record_action_outcome(
        self,
        trace: RunTrace,
        step,
        ok: bool,
        selected_mode: str,
        focus_confirmed: bool = False,
        target_node: ObservationNode | None = None,
        notes: str = "",
        scene_delta: SceneDelta | None = None,
    ) -> ActionOutcome:
        outcome = ActionOutcome(
            step_id=step.step_id,
            action_type=step.action_type,
            ok=ok,
            control_mode=selected_mode,
            target_label=target_node.display_label() if target_node else "",
            target_node_id=target_node.node_id if target_node else "",
            focus_confirmed=focus_confirmed,
            notes=notes,
            scene_delta=scene_delta,
        )
        trace.action_outcomes.append(outcome)
        self.memory.record_action_outcome(outcome)
        return outcome

    def _record_action_contract(self, trace: RunTrace, contract, trace_callback: TraceCallback | None = None) -> None:
        trace.outputs.setdefault("action_contracts", []).append(contract.to_dict())
        self._emit(
            trace_callback,
            "contract",
            f"{contract.action_type} contract {contract.status}.",
            step_id=contract.step_id,
            target=contract.target,
        )

    def _record_target_ranking(self, trace: RunTrace, step, rank_result: Any) -> dict[str, Any]:
        payload = rank_result.to_dict() if hasattr(rank_result, "to_dict") else dict(rank_result or {})
        payload.setdefault("candidate_count", 0)
        payload.setdefault("top_candidate_score", 0.0)
        payload.setdefault("runner_up_score", 0.0)
        payload.setdefault("score_gap", 0.0)
        payload.setdefault("duplicate_disambiguation_used", False)
        payload.setdefault("ambiguous", False)
        record = {
            "step_id": step.step_id,
            "action_type": step.action_type,
            **payload,
        }
        trace.outputs.setdefault("target_rankings", []).append(record)
        return record

    def _last_target_ranking_for_step(self, trace: RunTrace, step_id: str) -> dict[str, Any]:
        for ranking in reversed(trace.outputs.get("target_rankings", [])):
            if isinstance(ranking, dict) and ranking.get("step_id") == step_id:
                return ranking
        return {}

    def _record_perception_quality(
        self,
        trace: RunTrace,
        step,
        after_state: DesktopState | None,
        focus_confirmed: bool,
    ) -> None:
        ranking = self._last_target_ranking_for_step(trace, step.step_id)
        recovery_depth = sum(
            1
            for item in trace.outputs.get("recovery_attempts", [])
            if isinstance(item, dict) and item.get("step_id") == step.step_id
        )
        trace.outputs.setdefault("perception_quality", []).append(
            {
                "step_id": step.step_id,
                "action_type": step.action_type,
                "candidate_count": int(ranking.get("candidate_count", 0) or 0),
                "top_candidate_score": float(ranking.get("top_candidate_score", 0.0) or 0.0),
                "runner_up_score": float(ranking.get("runner_up_score", 0.0) or 0.0),
                "score_gap": float(ranking.get("score_gap", 0.0) or 0.0),
                "focus_confidence": float(getattr(after_state, "confidence", 0.0) or 0.0),
                "focus_confirmed": bool(focus_confirmed),
                "duplicate_disambiguation_used": bool(ranking.get("duplicate_disambiguation_used", False)),
                "target_ambiguity": bool(ranking.get("ambiguous", False)),
                "recovery_depth_per_step": recovery_depth,
            }
        )

    def _record_state_snapshot(self, trace: RunTrace, step_id: str, phase: str, state: DesktopState) -> None:
        trace.outputs.setdefault("state_snapshots", []).append(
            {"step_id": step_id, "phase": phase, **state.to_dict()}
        )

    def _record_state_diff(
        self,
        trace: RunTrace,
        step_id: str,
        before_state: DesktopState | None,
        after_state: DesktopState | None,
    ) -> dict[str, Any]:
        diff = self._state_manager().state_diff(before_state, after_state)
        trace.outputs.setdefault("state_diffs", []).append({"step_id": step_id, **diff})
        return diff

    def _record_task_state(self, trace: RunTrace, phase: str, task_state: TaskState) -> None:
        trace.outputs.setdefault("task_state_timeline", []).append({"phase": phase, **task_state.to_dict()})

    def _next_expected_state(self, plan: ExecutionPlan, task_state: TaskState) -> str:
        pending = set(task_state.pending_steps)
        for step in plan.steps:
            if step.step_id in pending:
                return step.success_criteria or ""
        return ""

    def _attempt_replan(
        self,
        trace: RunTrace,
        current_plan: ExecutionPlan,
        reason: str,
        step_id: str,
    ) -> list[Any] | None:
        self._invalidate_state_cache(f"attempt_replan:{reason}")
        if not getattr(self, "current_task_state", None) or not getattr(self, "last_state", None):
            return None
        failed_contract = self._last_contract_for_step(trace, step_id) or {}
        failed_step = None
        for candidate in current_plan.steps:
            if candidate.step_id == step_id:
                failed_step = candidate
                break
        if failed_step is None:
            return None
        repair = self._repair_planner().repair(
            failed_step=failed_step,
            failure_reason=reason,
            desktop_state=self.last_state,
            task_state=self.current_task_state,
            available_targets=[dict(failed_contract.get("target_identity", {}))] if failed_contract.get("target_identity") else [],
        )
        if repair:
            self._narrate_phase(
                trace,
                "repair",
                step=failed_step,
                throttle_key=f"repair:{step_id}:{reason}",
                action_type=failed_step.action_type,
            )
            trace.outputs.setdefault("plan_replacements", []).append(
                {
                    "step_id": step_id,
                    "old_plan_step_ids": [step.step_id for step in current_plan.steps],
                    "new_plan_step_ids": [step.step_id for step in repair.fragment],
                    "pending_preserved": list(self.current_task_state.pending_steps),
                    "reason": repair.reason,
                    "planner_type": repair.planner_type,
                    "full_restart_required": repair.stop_required,
                }
            )
            return list(repair.fragment)
        result = self._replanner().replan(
            user_goal=trace.task.goal,
            desktop_state=self.last_state,
            task_state=self.current_task_state,
            failure_reason=reason,
            original_plan=current_plan,
            observation=self.last_observation,
            environment=self.observe_environment(),
        )
        if not result:
            return None
        trace.outputs.setdefault("plan_replacements", []).append(
            {
                "step_id": step_id,
                **result.to_dict(),
                "planner_type": "replan",
            }
        )
        return list(result.fragment)

    def _last_contract_for_step(self, trace: RunTrace, step_id: str) -> dict[str, Any] | None:
        for contract in reversed(trace.outputs.get("action_contracts", [])):
            if contract.get("step_id") == step_id:
                return contract
        return None

    def _record_failure_recovery(self, trace: RunTrace, step, reason: str, contract: dict[str, Any] | None) -> dict[str, Any]:
        self._invalidate_state_cache(f"failure_recovery:{reason}")
        plan = self._recovery_planner().plan(reason, step, contract)
        record = {"step_id": step.step_id, "action_type": step.action_type, **plan.to_dict()}
        if contract:
            record["contract_target"] = contract.get("target", "")
            record["evidence_score"] = contract.get("evidence_score", 0.0)
            record["evidence_grade"] = contract.get("evidence_grade", "")
        trace.outputs.setdefault("failure_recovery", []).append(record)
        return record

    def _retry_contract(
        self,
        trace: RunTrace,
        step,
        recovery_record: dict[str, Any],
        failed_contract: dict[str, Any] | None,
        trace_callback: TraceCallback | None,
    ) -> tuple[bool, str]:
        if self._cancel_requested():
            return False, "Recovery retry skipped because cancellation was requested."
        if recovery_record.get("stop_required"):
            return False, recovery_record.get("note", "Recovery requires operator review.")

        attempts = [
            item for item in trace.outputs.get("recovery_attempts", [])
            if item.get("step_id") == step.step_id
        ]
        if len(attempts) >= self._recovery_planner().max_retries_per_action:
            return False, "Retry limit reached for this action."

        attempt_index = len(attempts) + 1
        self._invalidate_state_cache(f"retry:{step.action_type}")
        graph = self.parse_current_ui(output_filename=f"{step.step_id}_recovery_{attempt_index}.png", task=trace.task)
        retry_before_state = self._state_manager().observe_before_action(f"retry:{step.action_type}", graph=graph)
        self._record_state_snapshot(trace, step.step_id, f"retry_{attempt_index}_before", retry_before_state)
        attempt_record = {
            "step_id": step.step_id,
            "failure_reason": recovery_record.get("failure_reason", ""),
            "strategy_attempted": list(recovery_record.get("strategy", [])),
            "resolver_used": "",
            "new_evidence_score": 0.0,
            "recovery_depth_per_step": attempt_index,
            "target_switch_after_recovery": False,
            "retry_success": False,
        }

        def finish_attempt(result, resolver_used: str) -> tuple[bool, str]:
            self._record_action_contract(trace, result.contract, trace_callback)
            retry_after_state = self._state_manager().observe_after_action(
                f"retry:{step.action_type}",
                graph=result.after or self.last_observation,
                verified_change=result.notes if result.ok else "",
            )
            self._record_state_snapshot(trace, step.step_id, f"retry_{attempt_index}_after", retry_after_state)
            self._record_state_diff(trace, step.step_id, retry_before_state, retry_after_state)
            attempt_record["resolver_used"] = resolver_used
            attempt_record["new_evidence_score"] = result.contract.evidence_score
            attempt_record["retry_success"] = bool(result.ok)
            attempt_record["target"] = result.contract.target
            old_target = str((failed_contract or {}).get("target", "") or "")
            old_identity = (failed_contract or {}).get("target_identity", {}) if isinstance(failed_contract, dict) else {}
            new_identity = result.contract.target_identity or {}
            old_signature = str(old_identity.get("stable_signature", "") if isinstance(old_identity, dict) else "")
            new_signature = str(new_identity.get("stable_signature", "") if isinstance(new_identity, dict) else "")
            attempt_record["target_switch_after_recovery"] = bool(
                (old_signature and new_signature and old_signature != new_signature)
                or (old_target and result.contract.target and old_target != result.contract.target)
            )
            trace.outputs.setdefault("recovery_attempts", []).append(attempt_record)
            for quality in reversed(trace.outputs.get("perception_quality", [])):
                if isinstance(quality, dict) and quality.get("step_id") == step.step_id:
                    quality["recovery_depth_per_step"] = attempt_index
                    break
            self._emit(
                trace_callback,
                "recover",
                "Executable recovery retry completed.",
                step_id=step.step_id,
                resolver_used=resolver_used,
                retry_success=result.ok,
            )
            if result.ok:
                return True, f"Recovered via {resolver_used} retry."
            return False, result.notes

        if step.action_type == "type_text":
            text = str(step.parameters.get("text") or (step.target.value if step.target else ""))
            selector = str(step.parameters.get("selector", ""))
            dom_snapshot = self.bridge.read_browser_dom() if hasattr(self.bridge, "read_browser_dom") else {}
            dom_blob = str(dom_snapshot).lower()
            if text and text.lower() in dom_blob:
                attempt_record["resolver_used"] = "field_check"
                attempt_record["new_evidence_score"] = float((failed_contract or {}).get("evidence_score", 0.0) or 0.0)
                attempt_record["retry_success"] = True
                attempt_record["target"] = selector or "active_focus"
                attempt_record["note"] = "Typed value was already present; no duplicate typing performed."
                trace.outputs.setdefault("recovery_attempts", []).append(attempt_record)
                return True, "Recovered by field content check; typed value already present."
            result = self._action_executor().execute_type_text(
                step_id=f"{step.step_id}_retry_{attempt_index}",
                intent=f"Recovery retry for {step.title}",
                text=text,
                task=trace.task,
                selector=selector,
                clear_first=bool(step.parameters.get("clear_first", False)),
                settle_wait=float(step.parameters.get("settle_wait", 0.1)),
                focused_target=str(step.parameters.get("focused_target", "")),
                expected_app=str(step.parameters.get("expected_app", "")),
                deterministic_focus=str(step.parameters.get("deterministic_focus", "")),
            )
            return finish_attempt(result, "field_check")

        if step.action_type == "press_key":
            keys = step.parameters.get("keys")
            if isinstance(keys, str):
                keys = [keys]
            elif not isinstance(keys, list):
                keys = [step.target.value] if step.target and step.target.value else []
            normalized = [str(key).lower() for key in keys]
            destructive = tuple(normalized) in {
                ("alt", "f4"),
                ("ctrl", "w"),
                ("ctrl", "shift", "w"),
                ("ctrl", "q"),
                ("ctrl", "a"),
                ("delete",),
                ("shift", "delete"),
                ("backspace",),
            }
            expected_change = step.success_criteria or str(step.parameters.get("expected_change", ""))
            if destructive and not expected_change:
                attempt_record["resolver_used"] = "safety_gate"
                attempt_record["note"] = "Destructive hotkey retry blocked without expected change."
                trace.outputs.setdefault("recovery_attempts", []).append(attempt_record)
                return False, "Destructive hotkey retry blocked without expected change."
            result = self._action_executor().execute_press_key(
                step_id=f"{step.step_id}_retry_{attempt_index}",
                intent=f"Recovery retry for {step.title}",
                keys=[str(key) for key in keys],
                task=trace.task,
                hotkey=bool(step.parameters.get("hotkey", False)),
                settle_wait=float(step.parameters.get("settle_wait", 0.1)),
                before=graph,
                expected_change=expected_change,
            )
            return finish_attempt(result, "focus_key_retry")

        if step.action_type == "wait_for":
            seconds = min(30.0, max(float(step.parameters.get("seconds", 0.0)), 0.1))
            timeout = min(120.0, max(float(step.parameters.get("timeout", 0.0)), seconds))
            result = self._action_executor().execute_wait(
                step_id=f"{step.step_id}_retry_{attempt_index}",
                intent=f"Recovery retry for {step.title}",
                seconds=seconds,
                expected_focus=str(step.parameters.get("expected_focus", "")),
                timeout=timeout,
            )
            return finish_attempt(result, "bounded_wait_retry")

        if step.action_type == "click_point":
            recovered_target = self._recovery_planner().recover_target(
                strategy=list(recovery_record.get("strategy", [])),
                step=step,
                graph=graph,
                reasoner=self.reasoner,
                scene=graph.metadata.get("scene", {}),
                failed_contract=failed_contract,
            )
            if not recovered_target or not recovered_target.node:
                attempt_record["resolver_used"] = "coordinate_upgrade"
                attempt_record["note"] = "Raw coordinate was not retried; no resolved target was available."
                trace.outputs.setdefault("recovery_attempts", []).append(attempt_record)
                return False, "Raw coordinate retry blocked; no resolved target was available."
            result = self._action_executor().execute_click_node(
                step_id=f"{step.step_id}_retry_{attempt_index}",
                intent=f"Recovery retry for {step.title}",
                node=recovered_target.node,
                graph=graph,
                task=trace.task,
                click_count=1,
                settle_wait=float(step.parameters.get("settle_wait", 0.6)),
                extra_evidence=list(recovered_target.evidence or []) + ["reason:coordinate_upgraded_to_resolved_target"],
            )
            return finish_attempt(result, recovered_target.resolver_used or "coordinate_upgrade")

        if step.action_type != "click_node":
            return False, "Executable recovery is not implemented for this action type yet."

        recovered_target = self._recovery_planner().recover_target(
            strategy=list(recovery_record.get("strategy", [])),
            step=step,
            graph=graph,
            reasoner=self.reasoner,
            scene=graph.metadata.get("scene", {}),
            failed_contract=failed_contract,
        )
        if not recovered_target:
            attempt_record["note"] = "No alternate target resolved."
            trace.outputs.setdefault("recovery_attempts", []).append(attempt_record)
            return False, "Recovery could not resolve an alternate target."
        attempt_record["recovered_target"] = recovered_target.to_dict()

        if recovered_target.selectors:
            result = self._action_executor().execute_dom_click(
                step_id=f"{step.step_id}_retry_{attempt_index}",
                intent=f"Recovery retry for {step.title}",
                selectors=recovered_target.selectors,
                task=trace.task,
                settle_wait=float(step.parameters.get("settle_wait", 0.6)),
            )
        else:
            result = self._action_executor().execute_click_node(
                step_id=f"{step.step_id}_retry_{attempt_index}",
                intent=f"Recovery retry for {step.title}",
                node=recovered_target.node,
                graph=graph,
                task=trace.task,
                click_count=int(step.parameters.get("click_count", 1)),
                settle_wait=float(step.parameters.get("settle_wait", 0.6)),
                extra_evidence=list(recovered_target.evidence or []),
            )
        return finish_attempt(result, recovered_target.resolver_used)

    def _classify_failure_reason(self, notes: str, contract: dict[str, Any] | None = None) -> str:
        if contract and contract.get("failure_reason"):
            return str(contract.get("failure_reason"))
        lowered = notes.lower()
        if "policy" in lowered:
            return "POLICY_BLOCKED"
        if "could not route" in lowered or "focus mismatch" in lowered:
            return "FOCUS_NOT_CONFIRMED"
        if "no node matched" in lowered or "not found" in lowered or "no safe" in lowered:
            return "TARGET_NOT_FOUND"
        if "ambiguous" in lowered or "could not choose" in lowered:
            return "TARGET_AMBIGUOUS"
        if "expected labels were not visible" in lowered:
            return "NO_STATE_CHANGE"
        if "focus" in lowered:
            return "FOCUS_NOT_CONFIRMED"
        if "timed out" in lowered or "timeout" in lowered:
            return "TIMEOUT"
        if "coordinate" in lowered or "point clicks require" in lowered:
            return "UNSAFE_COORDINATE"
        return "NO_STATE_CHANGE"

    def _recover_step(
        self,
        task: TaskSpec,
        step,
        trace: RunTrace,
        trace_callback: TraceCallback | None,
        recovery_record: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        if self._cancel_requested():
            return False, "Recovery skipped because cancellation was requested."
        if recovery_record:
            self._emit(
                trace_callback,
                "recover",
                f"Recovery plan: {', '.join(recovery_record.get('strategy', []))}",
                step_id=step.step_id,
                failure_reason=recovery_record.get("failure_reason", ""),
            )
            if recovery_record.get("stop_required"):
                return False, recovery_record.get("note", "Recovery requires operator review.")
        app_id = str(step.parameters.get("app_id") or (step.target.value if step.target and step.target.kind == "application" else "")).strip()
        if app_id:
            self._emit(trace_callback, "recover", f"Attempting to refocus {app_id}.", step_id=step.step_id)
            route = self.bridge.route_window(app_id=app_id, window_title=str(step.parameters.get("window_title", "")))
            if route.get("ok"):
                self.bridge.wait_for(seconds=0.35)
                self.parse_current_ui(output_filename=f"{step.step_id}_recover.png", task=task)
                return True, f"Recovered by refocusing {app_id}."

        try:
            self.bridge.press_key(["esc"])
            self.bridge.wait_for(seconds=0.2)
            self.parse_current_ui(output_filename=f"{step.step_id}_recover.png", task=task)
            return True, "Recovered by sending Escape and reparsing."
        except Exception:
            return False, "Recovery failed."

    def execute_prompt(
        self,
        prompt: str,
        trust_mode: TrustMode = TrustMode.PLAN_AND_RISK_GATES,
        approval_callback: ApprovalCallback | None = None,
        trace_callback: TraceCallback | None = None,
        dry_run: bool = False,
        max_runtime_seconds: float = 0.0,
        voice_mode: str | None = None,
        narration_context: str = "task",
    ) -> RunTrace:
        self.reset_stop()
        self.current_voice_narrator = self._voice_for_run(mode_override=voice_mode)
        run_started_at = time.time()
        self._narrate_phase(None, "planning")
        plan = self.plan_prompt(prompt, trust_mode=trust_mode)
        mission = self._build_mission(plan)
        trace = RunTrace(run_id=f"run_{int(time.time())}", task=plan.task, plan=plan, status=RunStatus.PLANNED, mission=mission)
        self.current_trace = trace
        if narration_context == "benchmark":
            self._narrate_phase(trace, "benchmark")
        else:
            self._narrate_phase(trace, "runtime")
        trace.memory_hits["memory_summary"] = self.memory.summary()
        trace.outputs["environment"] = self.observe_environment()
        self.current_task_state = self._task_state_manager().initialize(plan)
        self._record_task_state(trace, "initialized", self.current_task_state)
        mission.status = MissionStatus.PLANNED
        trace.add_event("plan", "Execution plan created.", source=plan.source, step_count=len(plan.steps))
        self._emit(trace_callback, "plan", "Execution plan created.", source=plan.source, step_count=len(plan.steps))

        if dry_run:
            trace.status = RunStatus.SUCCESS
            mission.status = MissionStatus.SUCCESS
            trace.finished_at = time.time()
            trace.outputs["dry_run"] = True
            self._narrate_phase(trace, "done")
            self.current_voice_narrator.flush(timeout_seconds=max(2.0, self.current_voice_narrator.config.timeout))
            trace_path = self.memory.record_trace(trace)
            trace.outputs["trace_path"] = trace_path
            self.current_trace = None
            self.current_voice_narrator = None
            return trace

        approval_callback = approval_callback or self._default_approval
        trace.status = RunStatus.RUNNING
        mission.status = MissionStatus.RUNNING

        working_plan = ExecutionPlan(
            task=plan.task,
            steps=list(plan.steps),
            source=plan.source,
            summary=plan.summary,
            required_apps=list(plan.required_apps),
            success_conditions=list(plan.success_conditions),
            generated_at=plan.generated_at,
        )
        idx = 0
        while idx < len(working_plan.steps):
            step = working_plan.steps[idx]
            if max_runtime_seconds > 0 and time.time() - run_started_at > max_runtime_seconds:
                self._narrate(trace, "This mission timed out safely.", event_type="failure")
                trace.status = RunStatus.FAILED
                mission.status = MissionStatus.FAILED
                trace.outputs.setdefault("runtime_limits", []).append(
                    {
                        "limit": "max_runtime_seconds",
                        "max_runtime_seconds": max_runtime_seconds,
                        "elapsed_seconds": round(time.time() - run_started_at, 6),
                        "step_id": step.step_id,
                    }
                )
                trace.outputs.setdefault("failure_recovery", []).append(
                    {
                        "step_id": step.step_id,
                        "failure_reason": "MISSION_TIMEOUT",
                        "notes": "Mission exceeded benchmark runtime budget.",
                    }
                )
                trace.add_event("failure", "Mission runtime budget exceeded.", step_id=step.step_id)
                break
            if self._cancel_requested():
                self._narrate(trace, "Execution was stopped by the operator.", event_type="failure")
                trace.status = RunStatus.CANCELLED
                mission.status = MissionStatus.FAILED
                self._record_cancel_effective(trace, step.step_id)
                trace.add_event("runtime", "Execution stopped by operator.", step_id=step.step_id)
                break

            if self.current_task_state and self._task_state_manager().prevent_repeating_completed_step(self.current_task_state, step.step_id):
                trace.add_event("task_state", "Skipped completed step.", step_id=step.step_id)
                self._emit(trace_callback, "task_state", "Skipped completed step.", step_id=step.step_id)
                idx += 1
                continue

            decision = self.policy.evaluate_step(plan.task, step)
            trace.add_event("policy", decision.reason, step_id=step.step_id, risk_level=decision.risk_level.value)
            self._emit(trace_callback, "policy", decision.reason, step_id=step.step_id, risk_level=decision.risk_level.value)

            if not decision.allowed:
                self._narrate(trace, "This step failed safely because policy blocked it.", event_type="failure")
                trace.status = RunStatus.BLOCKED
                mission.status = MissionStatus.BLOCKED
                trace.add_event("blocked", "Policy blocked execution.", step_id=step.step_id)
                self._record_failure_recovery(trace, step, "POLICY_BLOCKED", None)
                self._set_checkpoint_status(mission, step.step_id, "blocked", decision.reason)
                break

            if decision.requires_approval:
                approval_app = (
                    str(step.parameters.get("app_id") or step.parameters.get("expected_app") or "").strip()
                    or (step.target.value if step.target and step.target.kind in {"application", "app"} else "")
                )
                approved = approval_callback(
                    f"{step.title} requires approval",
                    {
                        "risk_level": decision.risk_level.value,
                        "reason": decision.reason,
                        "action_type": step.action_type,
                        "app_id": approval_app,
                        "target": step.target.value if step.target else "",
                        "choices": ["allow_once", "always_allow_app", "cancel"],
                    },
                )
                trace.approvals.append({"step_id": step.step_id, "approved": approved, "timestamp": time.time()})
                if not approved:
                    self._narrate(trace, "This step was cancelled because approval was not granted.", event_type="failure")
                    trace.status = RunStatus.CANCELLED
                    mission.status = MissionStatus.FAILED
                    trace.add_event("approval", "Operator rejected the step.", step_id=step.step_id)
                    self._set_checkpoint_status(mission, step.step_id, "rejected", "Operator rejected the step.")
                    break

            step_meta = {
                "step_id": step.step_id,
                "action_type": step.action_type,
                "step_index": idx + 1,
                "step_total": len(working_plan.steps),
                "goal": plan.task.goal,
                "target": step.target.value if step.target else "",
                "risk_level": step.risk_level.value,
                "requires_approval": bool(step.requires_approval),
            }
            trace.add_event("step", f"Executing: {step.title}", **step_meta)
            self._emit(trace_callback, "step", f"Executing: {step.title}", **step_meta)
            self._narrate_phase(
                trace,
                step.action_type,
                step=step,
                throttle_key=f"{step.action_type}:{step.step_id}",
                action_type=step.action_type,
                app_id=step.parameters.get("app_id") or (step.target.value if step.target and step.target.kind == "application" else ""),
                target=step.target.value if step.target else "",
            )

            before_state = self._state_manager().observe_before_action(step.action_type, graph=self.last_observation)
            self._record_state_snapshot(trace, step.step_id, "before", before_state)
            before = self.last_observation
            if self._action_invalidates_state_cache(step.action_type):
                self._invalidate_state_cache(f"before:{step.action_type}")
            ok, selected_mode, target_node, notes = self._execute_step(step, trace, trace_callback)
            if self.cancel_state.level == CancelLevel.HARD and self._cancel_requested():
                ok = False
                notes = "Hard cancel requested by operator."
            after = self.last_observation
            after_state = self._state_manager().observe_after_action(
                step.action_type,
                graph=after,
                verified_change=notes if ok else "",
            )
            self._record_state_snapshot(trace, step.step_id, "after", after_state)
            state_diff = self._record_state_diff(trace, step.step_id, before_state, after_state)
            delta = self._record_scene_delta(trace, before, after, step.step_id, expected_change=step.success_criteria)
            focus_confirmed = True
            if step.action_type in {"route_window", "confirm_focus", "explore_safe", "interaction_learning"} and (step.target or step.parameters.get("app_id")):
                expected = str(step.parameters.get("app_id") or step.parameters.get("expected") or (step.target.value if step.target else ""))
                focus_confirmed = self.bridge.confirm_focus(expected)

            outcome = self._record_action_outcome(
                trace=trace,
                step=step,
                ok=ok,
                selected_mode=selected_mode,
                focus_confirmed=focus_confirmed,
                target_node=target_node,
                notes=notes,
                scene_delta=delta,
            )
            self._record_perception_quality(trace, step, after_state, focus_confirmed)
            if self.current_task_state:
                self.current_task_state = self._task_state_manager().update_task_state_after_step(
                    self.current_task_state,
                    step,
                    ok=ok,
                    notes=notes,
                )
                self.current_task_state.expected_next_state = self._next_expected_state(working_plan, self.current_task_state)
                self._record_task_state(trace, f"after_{step.step_id}", self.current_task_state)
            trace.add_event(
                "step_result",
                f"Completed using {selected_mode}.",
                step_id=step.step_id,
                action_type=step.action_type,
                control_mode=selected_mode,
                ok=ok,
                notes=notes,
            )
            self._emit(
                trace_callback,
                "step_result",
                f"Completed using {selected_mode}.",
                step_id=step.step_id,
                action_type=step.action_type,
                control_mode=selected_mode,
                ok=ok,
                notes=notes,
            )

            if ok:
                self._narrate_phase(
                    trace,
                    "step_result",
                    step=step,
                    throttle_key=f"step_result:{step.step_id}",
                    action_type=step.action_type,
                )
                if self.current_task_state and self._task_state_manager().detect_plan_drift(self.current_task_state, after_state):
                    self._narrate_phase(
                        trace,
                        "replan",
                        step=step,
                        throttle_key=f"replan:{step.step_id}",
                        action_type=step.action_type,
                    )
                    replan = self._task_state_manager().replan_from_current_state(working_plan, self.current_task_state)
                    trace.outputs.setdefault("task_replans", []).append(
                        {
                            "step_id": step.step_id,
                            "reason": "PLAN_DRIFT",
                            "state_diff": state_diff,
                            **replan,
                        }
                    )
                    fragment = self._attempt_replan(trace, working_plan, "PLAN_DRIFT", step.step_id)
                    if fragment:
                        working_plan.steps = working_plan.steps[: idx + 1] + fragment
                self._set_checkpoint_status(mission, step.step_id, "completed", outcome.notes)
                idx += 1
                continue

            self._invalidate_state_cache(f"failed:{step.action_type}")
            if self._cancel_requested():
                self._narrate(trace, "Execution was stopped by the operator.", event_type="failure")
                trace.status = RunStatus.CANCELLED
                mission.status = MissionStatus.FAILED
                self._record_cancel_effective(trace, step.step_id)
                trace.add_event("runtime", "Execution cancelled after current OS action.", step_id=step.step_id)
                break

            failed_contract = self._last_contract_for_step(trace, step.step_id)
            failure_reason = self._classify_failure_reason(notes, failed_contract)
            if failure_reason == "TARGET_AMBIGUOUS":
                self._narrate(
                    trace,
                    "This step failed safely because the target was ambiguous.",
                    event_type="recovery",
                    throttle_key=f"{step.action_type}:ambiguous",
                    step_id=step.step_id,
                )
            elif failure_reason == "NO_STATE_CHANGE" and step.action_type in {"click_node", "click_point"}:
                self._narrate(
                    trace,
                    "The click did not change the screen. I am trying recovery.",
                    event_type="recovery",
                    throttle_key=f"{step.action_type}:no_state_change",
                    step_id=step.step_id,
                )
            else:
                self._narrate(
                    trace,
                    "The step did not verify. I am trying recovery.",
                    event_type="recovery",
                    throttle_key=f"{step.action_type}:{failure_reason}",
                    step_id=step.step_id,
                    failure_reason=failure_reason,
                )
            recovery_record = self._record_failure_recovery(trace, step, failure_reason, failed_contract)
            if self.current_task_state:
                replan = self._task_state_manager().replan_from_current_state(working_plan, self.current_task_state)
                trace.outputs.setdefault("task_replans", []).append(
                    {
                        "step_id": step.step_id,
                        "reason": failure_reason,
                        **replan,
                    }
                )
            retry_ok, retry_note = self._retry_contract(trace, step, recovery_record, failed_contract, trace_callback)
            if retry_ok:
                self._set_checkpoint_status(mission, step.step_id, "recovered", retry_note)
                trace.add_event("recover", retry_note, step_id=step.step_id)
                idx += 1
                continue
            recovered, recovery_note = self._recover_step(plan.task, step, trace, trace_callback, recovery_record=recovery_record)
            if recovered:
                self._set_checkpoint_status(mission, step.step_id, "recovered", recovery_note)
                trace.add_event("recover", recovery_note, step_id=step.step_id)
                idx += 1
                continue

            fragment = self._attempt_replan(trace, working_plan, failure_reason, step.step_id)
            if fragment:
                self._invalidate_state_cache(f"replan:{failure_reason}")
                self._narrate_phase(
                    trace,
                    "replan",
                    step=step,
                    throttle_key=f"replan:{step.step_id}:{failure_reason}",
                    action_type=step.action_type,
                )
                working_plan.steps = working_plan.steps[:idx] + fragment + working_plan.steps[idx + 1 :]
                continue

            trace.status = RunStatus.FAILED
            mission.status = MissionStatus.FAILED
            trace.add_event("failure", f"Step failed: {step.title}", step_id=step.step_id, reason=recovery_note)
            self._set_checkpoint_status(mission, step.step_id, "failed", recovery_note)
            break
        else:
            trace.status = RunStatus.SUCCESS
            mission.status = MissionStatus.SUCCESS

        trace.finished_at = time.time()
        if trace.status == RunStatus.SUCCESS:
            if narration_context == "benchmark":
                self._narrate_phase(trace, "done")
            else:
                self._narrate(trace, "Task completed successfully.", event_type="done")
        elif trace.status in {RunStatus.FAILED, RunStatus.BLOCKED, RunStatus.CANCELLED}:
            self._narrate(trace, f"Run completed with status {trace.status.value}.", event_type="done")
        if self._cancel_requested() and self.current_voice_narrator:
            self.current_voice_narrator.cancel()
        self.current_voice_narrator.flush(timeout_seconds=max(2.0, self.current_voice_narrator.config.timeout))
        self._sync_bridge_memory()
        self.memory.record_workflow_trace(plan.task, plan, trace)
        trace_path = self.memory.record_trace(trace)
        trace.outputs["trace_path"] = trace_path
        self._emit(trace_callback, "done", f"Run completed with status: {trace.status.value}", trace_path=trace_path)
        self.current_trace = None
        self.current_voice_narrator = None
        return trace

    def _execute_step(self, step, trace: RunTrace, trace_callback: TraceCallback | None) -> tuple[bool, str, ObservationNode | None, str]:
        target_node: ObservationNode | None = None
        notes = ""

        if step.action_type == "legacy_command":
            command_text = step.parameters.get("command_text") or (step.target.value if step.target else "")
            self._invalidate_state_cache("legacy_command")
            ok = self.bridge.execute_legacy_command(command_text, debug=False)
            if ok:
                self.last_observation = None
            return bool(ok), "legacy", None, "Legacy fallback executed."

        if step.action_type == "parse_ui":
            output_filename = step.parameters.get("output_filename", "panel_parse.png")
            self._narrate_phase(trace, "observe", step=step, throttle_key=f"observe:{step.step_id}")
            graph = self.parse_current_ui(output_filename=output_filename, task=trace.task)
            trace.outputs["last_observation"] = graph.to_dict()
            trace.memory_hits["memory_summary"] = self.memory.summary()
            self._emit(trace_callback, "observation", "UI parsed successfully.", output_filename=output_filename)
            return True, "hybrid", None, graph.metadata.get("scene_summary", "")

        if step.action_type == "route_window":
            app_id = str(step.parameters.get("app_id") or (step.target.value if step.target else "")).strip()
            self._narrate_phase(trace, "focus", step=step, throttle_key=f"focus:{app_id}", app_id=app_id)
            self._invalidate_state_cache("route_window")
            result = self.bridge.route_window(app_id=app_id, window_title=str(step.parameters.get("window_title", "")))
            ok = bool(result.get("ok"))
            if ok:
                self.last_observation = None
                if bool(step.parameters.get("parse_after_route", False)):
                    self.parse_current_ui(output_filename=f"{step.step_id}_route.png", task=trace.task)
            return ok, "native", None, "Window routed." if ok else "Could not route the requested window."

        if step.action_type == "open_explorer_location":
            self._invalidate_state_cache("open_explorer_location")
            location = str(step.parameters.get("location") or (step.target.value if step.target else "")).strip()
            self._record_target_ranking(trace, step, dict(step.parameters.get("ranking", {}) or {}))
            if not hasattr(self.bridge, "open_explorer_location"):
                return False, "native", None, "Explorer deterministic location routing is unavailable."
            ok = bool(self.bridge.open_explorer_location(location))
            if ok:
                self.last_observation = None
            return ok, "native", None, f"Explorer routed to {location}." if ok else f"Could not route Explorer to {location}."

        if step.action_type == "confirm_focus":
            expected = str(step.parameters.get("expected") or (step.target.value if step.target else "")).strip()
            self._narrate_phase(trace, "focus", step=step, throttle_key=f"focus:{expected}", app_id=expected)
            ok = self.bridge.confirm_focus(expected)
            return ok, "native", None, f"Focus confirmed for {expected}." if ok else f"Focus mismatch for {expected}."

        if step.action_type in {"click_node", "modified_click_node"}:
            selector_candidates = [str(item) for item in step.parameters.get("selector_candidates", []) if str(item).strip()]
            click_count = int(step.parameters.get("click_count", 1))
            modifiers = [str(item).lower() for item in step.parameters.get("modifiers", []) if str(item).strip()]
            rank_result = None
            if not selector_candidates and not (step.parameters.get("filters") or (step.target and step.target.filters)):
                return False, "unsupported", None, "click_node requires filters or selector_candidates."

            if not modifiers and selector_candidates and hasattr(self.bridge, "browser_dom_available") and self.bridge.browser_dom_available():
                self._record_target_ranking(
                    trace,
                    step,
                    {
                        "selected_node_id": "",
                        "selected_label": step.target.value if step.target else "dom_selector",
                        "candidate_count": len(selector_candidates),
                        "top_candidate_score": 1.0,
                        "runner_up_score": 0.85 if len(selector_candidates) > 1 else 0.0,
                        "score_gap": 0.15 if len(selector_candidates) > 1 else 1.0,
                        "duplicate_disambiguation_used": len(selector_candidates) > 1,
                        "ambiguous": False,
                        "ambiguity_reason": "",
                        "candidates": [{"selector": selector, "score": 1.0 if idx == 0 else 0.85} for idx, selector in enumerate(selector_candidates[:8])],
                    },
                )
                self._narrate_phase(trace, "target_found", step=step, throttle_key=f"target:{step.step_id}:dom", target=step.target.value if step.target else "")
                self._narrate_phase(trace, "click", step=step, throttle_key=f"click:{step.step_id}:dom")
                self._invalidate_state_cache("dom_click")
                result = self._action_executor().execute_dom_click(
                    step_id=step.step_id,
                    intent=step.intent.description if step.intent else step.title,
                    selectors=selector_candidates,
                    task=trace.task,
                    settle_wait=float(step.parameters.get("settle_wait", 0.6)),
                )
                self._record_action_contract(trace, result.contract, trace_callback)
                if result.ok:
                    return result.ok, result.mode, result.target_node, result.notes

            observation = self._ensure_observation(trace.task, output_filename=f"{step.step_id}_target.png")
            scene = observation.metadata.get("scene", {})
            filters = dict(step.target.filters if step.target else {})
            filters.update(step.parameters.get("filters", {}))
            if not filters and step.target and step.target.value:
                filters["label_contains"] = step.target.value
            if hasattr(self.reasoner, "rank_action_targets"):
                rank_result = self.reasoner.rank_action_targets(filters, observation, scene)
                ranking_payload = self._record_target_ranking(trace, step, rank_result)
                target_node = rank_result.selected_node
                if rank_result.ambiguous:
                    return False, "hybrid", None, "Target ambiguous: ranking score gap is too small for a safe click."
            else:
                target_node = self.reasoner.choose_action_target(filters, observation, scene)
                ranking_payload = {}
            if not target_node:
                return False, "hybrid", None, "No node matched the requested target filters."
            self._narrate_phase(
                trace,
                "target_found",
                step=step,
                throttle_key=f"target:{step.step_id}:{target_node.display_label()}",
                target=target_node.display_label(),
                action_type=step.action_type,
            )
            self._narrate_phase(trace, "click", step=step, throttle_key=f"click:{step.step_id}")
            self._invalidate_state_cache("click_node")
            result = self._action_executor().execute_click_node(
                step_id=step.step_id,
                intent=step.intent.description if step.intent else step.title,
                node=target_node,
                graph=observation,
                task=trace.task,
                click_count=click_count,
                settle_wait=float(step.parameters.get("settle_wait", 0.6)),
                target_ranking=result_to_contract_metrics(rank_result) if rank_result is not None else ranking_payload,
                modifiers=modifiers,
            )
            self._record_action_contract(trace, result.contract, trace_callback)
            return result.ok, result.mode, result.target_node, result.notes

        if step.action_type == "click_point":
            x = int(step.parameters.get("x", 0))
            y = int(step.parameters.get("y", 0))
            evidence = [str(item) for item in step.parameters.get("evidence", []) if str(item).strip()]
            if step.parameters.get("screenshot"):
                evidence.append(f"screenshot:{step.parameters.get('screenshot')}")
            if step.parameters.get("region"):
                evidence.append(f"region:{step.parameters.get('region')}")
            if step.parameters.get("reason"):
                evidence.append(f"reason:{step.parameters.get('reason')}")
            before = self._ensure_observation(trace.task, output_filename=f"{step.step_id}_before.png")
            self._invalidate_state_cache("click_point")
            result = self._action_executor().execute_click_point(
                step_id=step.step_id,
                intent=step.intent.description if step.intent else step.title,
                x=x,
                y=y,
                task=trace.task,
                before=before,
                evidence=evidence,
                click_count=int(step.parameters.get("click_count", 1)),
                settle_wait=float(step.parameters.get("settle_wait", 0.6)),
            )
            self._record_action_contract(trace, result.contract, trace_callback)
            return result.ok, result.mode, result.target_node, result.notes

        if step.action_type == "type_text":
            text = str(step.parameters.get("text") or (step.target.value if step.target else ""))
            selector = str(step.parameters.get("selector", ""))
            clear_first = bool(step.parameters.get("clear_first", False))
            self._narrate_phase(trace, "type_text", step=step, throttle_key=f"type_text:{step.step_id}", action_type=step.action_type)
            self._invalidate_state_cache("type_text")
            result = self._action_executor().execute_type_text(
                step_id=step.step_id,
                intent=step.intent.description if step.intent else step.title,
                text=text,
                task=trace.task,
                selector=selector,
                clear_first=clear_first,
                settle_wait=float(step.parameters.get("settle_wait", 0.1)),
                focused_target=str(step.parameters.get("focused_target", "")),
                expected_app=str(step.parameters.get("expected_app", "")),
                deterministic_focus=str(step.parameters.get("deterministic_focus", "")),
            )
            self._record_action_contract(trace, result.contract, trace_callback)
            return result.ok, result.mode, result.target_node, result.notes

        if step.action_type == "press_key":
            keys = step.parameters.get("keys")
            if isinstance(keys, str):
                keys = [keys]
            elif not isinstance(keys, list):
                keys = [step.target.value] if step.target and step.target.value else []
            if step.parameters.get("ranking"):
                self._record_target_ranking(trace, step, dict(step.parameters.get("ranking", {}) or {}))
            self._narrate_phase(trace, "press_key", step=step, throttle_key=f"press_key:{step.step_id}", action_type=step.action_type)
            self._invalidate_state_cache("press_key")
            result = self._action_executor().execute_press_key(
                step_id=step.step_id,
                intent=step.intent.description if step.intent else step.title,
                keys=[str(key) for key in keys],
                task=trace.task,
                hotkey=bool(step.parameters.get("hotkey", False)),
                settle_wait=float(step.parameters.get("settle_wait", 0.1)),
                before=self.last_observation,
                expected_change=step.success_criteria or str(step.parameters.get("expected_change", "")),
            )
            self._record_action_contract(trace, result.contract, trace_callback)
            return result.ok, result.mode, result.target_node, result.notes

        if step.action_type == "wait_for":
            seconds = float(step.parameters.get("seconds", 0.0))
            expected_focus = str(step.parameters.get("expected_focus", ""))
            timeout = float(step.parameters.get("timeout", 0.0))
            self._narrate_phase(trace, "wait_for", step=step, throttle_key=f"wait_for:{step.step_id}", action_type=step.action_type)
            result = self._action_executor().execute_wait(
                step_id=step.step_id,
                intent=step.intent.description if step.intent else step.title,
                seconds=seconds,
                expected_focus=expected_focus,
                timeout=timeout,
            )
            self._record_action_contract(trace, result.contract, trace_callback)
            return result.ok, result.mode, result.target_node, result.notes

        if step.action_type == "ocr_read":
            graph = self.parse_current_ui(output_filename=f"{step.step_id}_ocr.png", task=trace.task)
            labels = [node.display_label() for node in graph.flatten() if node.display_label()]
            trace.outputs.setdefault("ocr_reads", {})[step.step_id] = labels[:30]
            return True, "hybrid", None, f"Read {len(labels)} OCR labels."

        if step.action_type == "hover_probe":
            app_id = str(step.parameters.get("app_id") or (step.target.value if step.target else "")).strip()
            max_nodes = max(1, min(int(step.parameters.get("max_nodes", 8)), 24))
            settle_wait = float(step.parameters.get("settle_wait", 0.45))
            filters = dict(step.parameters.get("filters", {}))
            graph = self._ensure_observation(trace.task, output_filename=f"{step.step_id}_hover_seed.png")
            app_id = app_id or graph.metadata.get("app_id", "")
            candidates = self._hover_probe_candidates(graph, filters, app_id, max_nodes)
            if not candidates:
                return False, "hybrid", None, "No safe hover targets were available."

            hover_results = []
            before_labels = self._graph_label_set(graph)
            for idx, candidate in enumerate(candidates, start=1):
                if not hasattr(self.bridge, "hover_node") or not self.bridge.hover_node(candidate):
                    hover_results.append({"target": candidate.display_label(), "ok": False, "feedback_labels": []})
                    continue
                self.bridge.wait_for(seconds=settle_wait)
                after = self.parse_current_ui(output_filename=f"{step.step_id}_hover_{idx}.png", task=trace.task)
                after_labels = self._graph_label_set(after)
                added_labels = sorted(after_labels - before_labels)[:12]
                stable_feedback = [
                    label
                    for label in added_labels
                    if label.lower() not in {candidate.display_label().lower(), "main page", "top menu", "left menu", "bottom menu"}
                ]
                if stable_feedback:
                    self.memory.remember_hover_feedback(candidate, stable_feedback, app_id=app_id)
                hover_results.append(
                    {
                        "target": candidate.display_label(),
                        "node_id": candidate.node_id,
                        "entity_type": candidate.entity_type,
                        "feedback_labels": stable_feedback,
                    }
                )
                graph = after
                before_labels = after_labels

            trace.outputs.setdefault("hover_feedback", {})[step.step_id] = hover_results
            self.last_observation = graph
            feedback_count = sum(1 for item in hover_results if item.get("feedback_labels"))
            return True, "hybrid", None, f"Hover probed {len(hover_results)} targets; {feedback_count} produced new feedback labels."

        if step.action_type == "learning_session":
            return self._execute_learning_session(step, trace, trace_callback)

        if step.action_type == "interaction_learning":
            return self._execute_interaction_learning(step, trace, trace_callback)

        if step.action_type == "replay_interaction":
            return self._execute_replay_interaction(step, trace, trace_callback)

        if step.action_type == "verify_scene":
            self._narrate_phase(trace, "verify_scene", step=step, throttle_key=f"verify_scene:{step.step_id}", action_type=step.action_type)
            graph = self.parse_current_ui(output_filename=f"{step.step_id}_verify.png", task=trace.task)
            scene = graph.metadata.get("scene", {})
            filters = dict(step.parameters.get("filters", {}))
            expected_app = str(step.parameters.get("expected_app", ""))
            expected_labels = [str(item) for item in step.parameters.get("expected_labels", []) if str(item).strip()]
            if expected_app and scene.get("app_id") != expected_app:
                return False, "hybrid", None, f"Expected app '{expected_app}' but saw '{scene.get('app_id', '')}'."
            if filters and not self.reasoner.choose_action_target(filters, graph, scene):
                return False, "hybrid", None, "Verification target was not found."
            if expected_labels and not self._scene_contains_expected_labels(graph, expected_labels):
                return False, "hybrid", None, f"Expected labels were not visible: {expected_labels}"
            return True, "hybrid", None, scene.get("summary", "Scene verified.")

        if step.action_type == "scene_diff":
            self._narrate_phase(trace, "scene_diff", step=step, throttle_key=f"scene_diff:{step.step_id}", action_type=step.action_type)
            before = self.last_observation
            after = self.parse_current_ui(output_filename=f"{step.step_id}_diff.png", task=trace.task)
            delta = self.reasoner.summarize_scene_change(before, after, step_id=step.step_id, expected_change=step.success_criteria)
            trace.outputs.setdefault("manual_scene_diffs", []).append(delta.to_dict())
            return True, "hybrid", None, delta.actual_change

        if step.action_type == "checkpoint":
            self._narrate_phase(trace, "checkpoint", step=step, throttle_key=f"checkpoint:{step.step_id}", action_type=step.action_type)
            graph = self.parse_current_ui(output_filename=f"{step.step_id}_checkpoint.png", task=trace.task)
            scene = graph.metadata.get("scene", {})
            expected_scene = str(step.parameters.get("expected_scene", ""))
            filters = dict(step.parameters.get("filters", {}))
            expected_labels = [str(item) for item in step.parameters.get("expected_labels", []) if str(item).strip()]
            if filters and not self.reasoner.choose_action_target(filters, graph, scene):
                return False, "hybrid", None, "Checkpoint filters did not match the current scene."
            if expected_labels and not self._scene_contains_expected_labels(graph, expected_labels):
                return False, "hybrid", None, f"Checkpoint labels were not visible: {expected_labels}"
            if expected_scene:
                notes = scene.get("summary", "")
                return bool(notes), "hybrid", None, notes
            return True, "hybrid", None, scene.get("summary", "Checkpoint captured.")

        if step.action_type == "recover":
            ok, note = self._recover_step(trace.task, step, trace, trace_callback)
            return ok, "hybrid", None, note

        if step.action_type == "explore_safe":
            app_id = str(step.parameters.get("app_id") or (step.target.value if step.target else "")).strip()
            profile = self.profiles.get(app_id)
            if not profile:
                return False, "hybrid", None, f"No app profile is registered for '{app_id}'."
            if not profile.trusted_for_exploration:
                return False, "hybrid", None, f"'{app_id}' is not trusted for autonomous exploration."

            rounds = int(step.parameters.get("rounds", 4))
            visited_ids: set[str] = set()
            graph = self._ensure_observation(trace.task, output_filename=f"{step.step_id}_seed.png")

            for round_idx in range(1, rounds + 1):
                if graph.metadata.get("app_id") and graph.metadata.get("app_id") != app_id:
                    return False, "hybrid", None, f"Exploration left the '{app_id}' surface."

                safe_nodes = [node for node in profile.safe_nodes(graph) if node.node_id not in visited_ids]
                if not safe_nodes:
                    safe_nodes = profile.safe_nodes(graph)
                if not safe_nodes:
                    return False, "hybrid", None, "No safe nodes were available for exploration."

                chosen = self.reasoner.resolve_ambiguity(safe_nodes[:6], trace.task.prompt, graph.metadata.get("scene", {}))
                if not chosen:
                    return False, "hybrid", None, "The reasoner could not choose a safe exploration target."

                visited_ids.add(chosen.node_id)
                before = graph
                self.bridge.click_node(chosen)
                self.bridge.wait_for(seconds=float(step.parameters.get("settle_wait", 0.7)))
                graph = self.parse_current_ui(output_filename=f"{step.step_id}_round_{round_idx}.png", task=trace.task)
                round_delta = self._record_scene_delta(trace, before, graph, f"{step.step_id}_round_{round_idx}", expected_change="A safe scene change should occur.")
                self._record_action_outcome(
                    trace=trace,
                    step=step,
                    ok=True,
                    selected_mode="hybrid",
                    focus_confirmed=self.bridge.confirm_focus(app_id),
                    target_node=chosen,
                    notes=f"Exploration round {round_idx}: {round_delta.actual_change}",
                    scene_delta=round_delta,
                )
                if graph.metadata.get("app_id") and graph.metadata.get("app_id") != app_id:
                    return False, "hybrid", chosen, f"Exploration left '{app_id}' after clicking '{chosen.display_label()}'."

            self.last_observation = graph
            return True, "hybrid", None, f"Completed {rounds} safe exploration rounds in {app_id}."

        trace.add_event("warning", f"Unsupported action type: {step.action_type}", step_id=step.step_id)
        return False, "unsupported", None, f"Unsupported action type: {step.action_type}"
