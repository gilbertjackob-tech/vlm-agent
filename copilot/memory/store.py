from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from copilot.schemas import ActionOutcome, ExecutionPlan, ObservationGraph, ObservationNode, RunTrace, SceneDelta, TaskSpec


class MemoryStore:
    def __init__(self, base_dir: str = "memory") -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.semantic_path = os.path.join(self.base_dir, "semantic_memory.json")
        self.episodic_path = os.path.join(self.base_dir, "episodic_memory.json")
        self.workflow_path = os.path.join(self.base_dir, "workflow_memory.json")
        self.policy_path = os.path.join(self.base_dir, "policy_memory.json")

        self.semantic_memory = self._load_json(self.semantic_path, self._default_semantic())
        self.episodic_memory = self._load_json(self.episodic_path, self._default_episodic())
        self.workflow_memory = self._load_json(self.workflow_path, self._default_workflows())
        self.policy_memory = self._load_json(self.policy_path, self._default_policy())

    def _load_json(self, path: str, default: dict[str, Any]) -> dict[str, Any]:
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = dict(default)
            if isinstance(data, dict):
                merged.update(data)
            return merged
        except Exception:
            return default

    def _save_json(self, path: str, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _default_semantic(self) -> dict[str, Any]:
        return {
            "version": 3,
            "labels": {},
            "visuals": {},
            "concepts": {},
            "controls": {},
            "entities": {},
            "transitions": [],
            "negative_examples": [],
            "hover_feedback": [],
            "review_queue": [],
            "learning_sessions": [],
            "interaction_graph": {
                "version": 1,
                "scene_nodes": {},
                "control_nodes": {},
                "edges": {},
                "action_values": {},
            },
        }

    def _default_episodic(self) -> dict[str, Any]:
        return {
            "version": 2,
            "runs": [],
            "scene_deltas": [],
            "action_outcomes": [],
        }

    def _default_workflows(self) -> dict[str, Any]:
        return {
            "version": 1,
            "workflows": [],
        }

    def _default_policy(self) -> dict[str, Any]:
        return {
            "version": 1,
            "trusted_apps": ["explorer", "chrome"],
            "high_risk_allowed_apps": [],
            "blocked_apps": [],
            "blocked_concepts": [],
            "allowlisted_paths": [],
            "blocked_paths": [],
        }

    def save_all(self) -> None:
        self._save_json(self.semantic_path, self.semantic_memory)
        self._save_json(self.episodic_path, self.episodic_memory)
        self._save_json(self.workflow_path, self.workflow_memory)
        self._save_json(self.policy_path, self.policy_memory)

    def summary(self) -> dict[str, Any]:
        workflows = self.workflow_memory.get("workflows", [])
        return {
            "known_labels": len(self.semantic_memory.get("labels", {})),
            "known_visuals": len(self.semantic_memory.get("visuals", {})),
            "known_concepts": len(self.semantic_memory.get("concepts", {})),
            "known_controls": len(self.semantic_memory.get("controls", {})),
            "known_entities": len(self.semantic_memory.get("entities", {})),
            "transition_count": len(self.semantic_memory.get("transitions", [])),
            "negative_examples": len(self.semantic_memory.get("negative_examples", [])),
            "review_queue": len([item for item in self.semantic_memory.get("review_queue", []) if item.get("status") == "pending"]),
            "learning_sessions": len(self.semantic_memory.get("learning_sessions", [])),
            "interaction_edges": len(self.semantic_memory.get("interaction_graph", {}).get("edges", {})),
            "action_values": len(self.semantic_memory.get("interaction_graph", {}).get("action_values", {})),
            "workflow_count": len(workflows),
            "trusted_workflows": len([w for w in workflows if w.get("promotion_state") == "trusted"]),
            "trusted_apps": len(self.policy_memory.get("trusted_apps", [])),
            "blocked_apps": len(self.policy_memory.get("blocked_apps", [])),
            "blocked_concepts": len(self.policy_memory.get("blocked_concepts", [])),
            "recent_runs": len(self.episodic_memory.get("runs", [])),
            "scene_deltas": len(self.episodic_memory.get("scene_deltas", [])),
            "action_outcomes": len(self.episodic_memory.get("action_outcomes", [])),
        }

    def normalize_prompt(self, prompt: str) -> str:
        return re.sub(r"\s+", " ", prompt.strip().lower())

    def _is_placeholder_label(self, label: str) -> bool:
        text = str(label or "").strip()
        return not text or (text.startswith("[") and text.endswith("]"))

    def _node_visual_ids(self, node: ObservationNode) -> list[str]:
        visual_ids = []
        seen: set[str] = set()
        for visual_id in [node.visual_id, *getattr(node, "visual_ids", [])]:
            visual_id = str(visual_id or "").strip()
            if visual_id and visual_id not in seen:
                seen.add(visual_id)
                visual_ids.append(visual_id)
        return visual_ids

    def _ensure_counter_fields(self, record: dict[str, Any], fields: list[str]) -> None:
        for field in fields:
            if not isinstance(record.get(field), dict):
                record[field] = {}

    def record_trace(self, trace: RunTrace) -> str:
        traces_dir = os.path.join("debug_steps", "traces")
        os.makedirs(traces_dir, exist_ok=True)
        trace_path = os.path.join(traces_dir, f"{trace.run_id}.json")

        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)

        summary = {
            "run_id": trace.run_id,
            "prompt": trace.task.prompt,
            "goal": trace.task.goal,
            "status": trace.status.value,
            "started_at": trace.started_at,
            "finished_at": trace.finished_at,
            "trace_path": trace_path,
            "step_count": len(trace.plan.steps),
            "mission_status": trace.mission.status.value if trace.mission else "",
        }
        runs = self.episodic_memory.setdefault("runs", [])
        runs.append(summary)
        self.episodic_memory["runs"] = runs[-120:]
        self._save_json(self.episodic_path, self.episodic_memory)
        return trace_path

    def recent_runs(self, limit: int = 12) -> list[dict[str, Any]]:
        return list(reversed(self.episodic_memory.get("runs", [])[-limit:]))

    def record_scene_delta(self, delta: SceneDelta) -> None:
        deltas = self.episodic_memory.setdefault("scene_deltas", [])
        deltas.append(delta.to_dict())
        self.episodic_memory["scene_deltas"] = deltas[-400:]
        self._save_json(self.episodic_path, self.episodic_memory)

    def record_action_outcome(self, outcome: ActionOutcome) -> None:
        outcomes = self.episodic_memory.setdefault("action_outcomes", [])
        outcomes.append(outcome.to_dict())
        self.episodic_memory["action_outcomes"] = outcomes[-400:]
        self._save_json(self.episodic_path, self.episodic_memory)

    def _plan_signature(self, plan: ExecutionPlan) -> str:
        return "|".join(f"{step.action_type}:{step.target.value if step.target else ''}" for step in plan.steps)

    def _workflow_state(self, record: dict[str, Any]) -> str:
        if record.get("promotion_state") == "trusted":
            return "trusted"
        success_count = int(record.get("success_count", 0) or 0)
        failure_count = int(record.get("failure_count", 0) or 0)
        if success_count >= 5 and failure_count == 0:
            return "verified"
        if success_count >= 1:
            return "draft"
        return "draft"

    def _workflow_step_count(self, record: dict[str, Any]) -> int:
        steps = record.get("steps", [])
        return len(steps) if isinstance(steps, list) else 0

    def _workflow_selection_key(self, record: dict[str, Any]) -> tuple[Any, ...]:
        promotion_rank = {"trusted": 0, "verified": 1, "draft": 2}.get(str(record.get("promotion_state", "draft")), 3)
        avg_latency = float(record.get("avg_latency_seconds", 9999.0) or 9999.0)
        step_count = int(record.get("step_count", self._workflow_step_count(record)) or 0)
        failure_count = int(record.get("failure_count", 0) or 0)
        success_count = int(record.get("success_count", 0) or 0)
        workflow_type_rank = {"shortcut": 0, "fragment": 1, "workflow": 2}.get(str(record.get("workflow_type", "workflow")), 3)
        return (
            promotion_rank,
            avg_latency,
            step_count,
            workflow_type_rank,
            failure_count,
            -success_count,
            -float(record.get("last_run", 0.0) or 0.0),
        )

    def _workflow_success_rate(self, record: dict[str, Any]) -> float:
        success_count = int(record.get("success_count", 0) or 0)
        failure_count = int(record.get("failure_count", 0) or 0)
        total = success_count + failure_count
        return round(success_count / total, 4) if total else 0.0

    def _workflow_approval_status(self, record: dict[str, Any]) -> str:
        if record.get("promotion_state") == "trusted":
            return "approved"
        if record.get("user_approved"):
            return "human_approved"
        return "pending"

    def _workflow_app(self, record: dict[str, Any]) -> str:
        required_apps = [str(item).strip() for item in record.get("required_apps", []) if str(item).strip()]
        if required_apps:
            return required_apps[0]
        for step in record.get("steps", []):
            if not isinstance(step, dict):
                continue
            parameters = step.get("parameters", {}) if isinstance(step.get("parameters", {}), dict) else {}
            app_id = str(parameters.get("app_id") or parameters.get("expected_app") or "").strip()
            if app_id:
                return app_id
        return ""

    def _extract_step_target(self, step: dict[str, Any]) -> dict[str, Any]:
        parameters = step.get("parameters", {}) if isinstance(step.get("parameters", {}), dict) else {}
        target = step.get("target", {}) if isinstance(step.get("target", {}), dict) else {}
        filters = parameters.get("filters", {}) if isinstance(parameters.get("filters", {}), dict) else {}
        target_filters = target.get("filters", {}) if isinstance(target.get("filters", {}), dict) else {}
        merged_filters = {**target_filters, **filters}
        selector_candidates = [
            str(selector).strip()
            for selector in parameters.get("selector_candidates", [])
            if str(selector).strip()
        ] if isinstance(parameters.get("selector_candidates", []), list) else []
        identity = parameters.get("target_identity", {}) if isinstance(parameters.get("target_identity", {}), dict) else {}
        return {
            "step_id": str(step.get("step_id", "")),
            "action_type": str(step.get("action_type", "")),
            "target_kind": str(target.get("kind", "")),
            "target": str(target.get("value") or parameters.get("label") or parameters.get("target") or ""),
            "selectors": selector_candidates,
            "uia": {
                "automation_id": str(merged_filters.get("automation_id") or identity.get("automation_id") or ""),
                "name": str(merged_filters.get("name") or identity.get("name") or merged_filters.get("label") or ""),
                "control_type": str(merged_filters.get("control_type") or identity.get("control_type") or ""),
            },
            "filters": merged_filters,
            "success_criteria": str(step.get("success_criteria", "")),
        }

    def _extract_workflow_targets(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        targets = []
        for step in record.get("steps", []):
            if isinstance(step, dict):
                target = self._extract_step_target(step)
                if target["target"] or target["selectors"] or any(target["uia"].values()) or target["filters"]:
                    targets.append(target)
        return targets

    def _ensure_skill_capsule_fields(self, record: dict[str, Any]) -> None:
        record.setdefault("skill_name", record.get("name", ""))
        record.setdefault("trigger_phrase", record.get("prompt_pattern") or record.get("normalized_prompt", ""))
        record.setdefault("app", self._workflow_app(record))
        record["targets"] = self._extract_workflow_targets(record)
        record["success_rate"] = self._workflow_success_rate(record)
        record["approval_status"] = self._workflow_approval_status(record)
        if record.get("last_verified") is None and record.get("last_run") and record["success_rate"] > 0:
            record["last_verified"] = record.get("last_run")

    def build_skill_capsule(self, workflow: dict[str, Any]) -> dict[str, Any]:
        record = dict(workflow)
        self._ensure_skill_capsule_fields(record)
        return {
            "capsule_type": "skill_capsule",
            "workflow_id": record.get("workflow_id", ""),
            "skill_name": record.get("skill_name") or record.get("name", ""),
            "name": record.get("name", ""),
            "trigger_phrase": record.get("trigger_phrase", ""),
            "normalized_prompt": record.get("normalized_prompt", ""),
            "prompt_pattern": record.get("prompt_pattern", ""),
            "plan_signature": record.get("plan_signature", ""),
            "app": record.get("app", ""),
            "required_apps": list(record.get("required_apps", [])),
            "steps": list(record.get("steps", [])),
            "selectors_uia_targets": list(record.get("targets", [])),
            "targets": list(record.get("targets", [])),
            "success_rate": record.get("success_rate", 0.0),
            "success_count": int(record.get("success_count", 0) or 0),
            "failure_count": int(record.get("failure_count", 0) or 0),
            "wrong_click_count": int(record.get("wrong_click_count", 0) or 0),
            "last_verified": record.get("last_verified"),
            "last_run": record.get("last_run"),
            "approval_status": record.get("approval_status", "pending"),
            "user_approved": bool(record.get("user_approved", False)),
            "promotion_state": record.get("promotion_state", "draft"),
            "promotion_blockers": list(record.get("promotion_blockers", [])),
            "replay_score": float(record.get("replay_score", 0.0) or 0.0),
            "replay_count": int(record.get("replay_count", 0) or 0),
            "variant_count": int(record.get("variant_count", 0) or 0),
            "workflow_type": record.get("workflow_type", "workflow"),
            "objective_key": record.get("objective_key", ""),
            "avg_latency_seconds": record.get("avg_latency_seconds", 9999.0),
        }

    def _build_workflow_record(
        self,
        *,
        task: TaskSpec,
        plan: ExecutionPlan,
        normalized_prompt: str,
        signature: str,
        name: str,
        promotion_state: str,
        workflow_type: str,
        objective_key: str,
    ) -> dict[str, Any]:
        return {
            "workflow_id": f"wf_{abs(hash((normalized_prompt, signature, workflow_type, objective_key))) % 10_000_000}",
            "name": name or task.goal[:80],
            "normalized_prompt": normalized_prompt,
            "prompt_pattern": task.prompt,
            "plan_signature": signature,
            "steps": [step.to_dict() for step in plan.steps],
            "required_apps": list(plan.required_apps),
            "step_count": len(plan.steps),
            "success_count": 0,
            "failure_count": 0,
            "wrong_click_count": 0,
            "user_approved": False,
            "promotion_state": promotion_state or "draft",
            "last_run": None,
            "workflow_type": workflow_type or "workflow",
            "objective_key": objective_key or normalized_prompt,
            "total_latency_seconds": 0.0,
            "avg_latency_seconds": 9999.0,
            "skill_name": name or task.goal[:80],
            "trigger_phrase": task.prompt,
            "app": list(plan.required_apps)[0] if plan.required_apps else "",
            "targets": [],
            "success_rate": 0.0,
            "last_verified": None,
            "approval_status": "pending",
            "replay_score": 0.0,
            "replay_count": 0,
            "variant_count": 0,
        }

    def _upsert_workflow_record(
        self,
        *,
        task: TaskSpec,
        plan: ExecutionPlan,
        name: str = "",
        promotion_state: str = "draft",
        workflow_type: str = "workflow",
        objective_key: str = "",
    ) -> dict[str, Any]:
        normalized_prompt = self.normalize_prompt(task.prompt)
        signature = self._plan_signature(plan)
        workflows = self.workflow_memory.setdefault("workflows", [])
        record = None
        for existing in workflows:
            if (
                existing.get("normalized_prompt") == normalized_prompt
                and existing.get("plan_signature") == signature
                and str(existing.get("workflow_type", "workflow")) == (workflow_type or "workflow")
                and str(existing.get("objective_key", normalized_prompt)) == (objective_key or normalized_prompt)
            ):
                record = existing
                break

        if record is None:
            record = self._build_workflow_record(
                task=task,
                plan=plan,
                normalized_prompt=normalized_prompt,
                signature=signature,
                name=name or task.goal[:80],
                promotion_state=promotion_state,
                workflow_type=workflow_type,
                objective_key=objective_key or normalized_prompt,
            )
            workflows.append(record)
        else:
            if name:
                record["name"] = name
                record["skill_name"] = name
            record["prompt_pattern"] = task.prompt
            record["trigger_phrase"] = task.prompt
            record["steps"] = [step.to_dict() for step in plan.steps]
            record["required_apps"] = list(plan.required_apps)
            record["step_count"] = len(plan.steps)
            record["workflow_type"] = workflow_type or record.get("workflow_type", "workflow")
            record["objective_key"] = objective_key or record.get("objective_key", normalized_prompt)
            if promotion_state:
                record["promotion_state"] = promotion_state if record.get("promotion_state") != "trusted" else "trusted"
        record.setdefault("total_latency_seconds", 0.0)
        record.setdefault("avg_latency_seconds", 9999.0)
        record.setdefault("skill_name", record.get("name", ""))
        record.setdefault("trigger_phrase", record.get("prompt_pattern") or normalized_prompt)
        record["app"] = self._workflow_app(record)
        self._ensure_skill_capsule_fields(record)
        return record

    def record_workflow_run(
        self,
        task: TaskSpec,
        plan: ExecutionPlan,
        success: bool,
        *,
        latency_seconds: float | None = None,
        workflow_type: str = "workflow",
        objective_key: str = "",
        name: str = "",
    ) -> dict[str, Any]:
        record = self._upsert_workflow_record(
            task=task,
            plan=plan,
            name=name,
            promotion_state="draft",
            workflow_type=workflow_type,
            objective_key=objective_key,
        )

        if success:
            record["success_count"] += 1
        else:
            record["failure_count"] += 1

        if latency_seconds is not None:
            record["total_latency_seconds"] = float(record.get("total_latency_seconds", 0.0) or 0.0) + float(latency_seconds)
            run_count = int(record.get("success_count", 0) or 0) + int(record.get("failure_count", 0) or 0)
            record["avg_latency_seconds"] = round(record["total_latency_seconds"] / max(1, run_count), 6)

        record["last_run"] = time.time()
        record.setdefault("wrong_click_count", 0)
        record.setdefault("user_approved", False)
        record["promotion_state"] = self._workflow_state(record)
        record["success_rate"] = self._workflow_success_rate(record)
        if success:
            record["last_verified"] = record["last_run"]
        self._ensure_skill_capsule_fields(record)
        self._save_json(self.workflow_path, self.workflow_memory)
        return record

    def record_workflow_trace(self, task: TaskSpec, plan: ExecutionPlan, trace: Any) -> dict[str, Any]:
        status = getattr(getattr(trace, "status", None), "value", str(getattr(trace, "status", "")))
        started_at = float(getattr(trace, "started_at", 0.0) or 0.0)
        finished_at = float(getattr(trace, "finished_at", 0.0) or 0.0)
        latency_seconds = max(0.0, finished_at - started_at) if started_at and finished_at else None
        record = self.record_workflow_run(task, plan, status == "success", latency_seconds=latency_seconds)
        outputs = getattr(trace, "outputs", {}) if trace else {}
        if isinstance(outputs, dict):
            wrong_clicks = int(outputs.get("wrong_click_count", 0) or 0)
            for outcome in outputs.get("action_outcomes", []):
                if isinstance(outcome, dict) and outcome.get("wrong_click"):
                    wrong_clicks += 1
            action_contracts = outputs.get("action_contracts", [])
            if isinstance(action_contracts, list):
                record["action_contracts"] = [contract for contract in action_contracts if isinstance(contract, dict)]
            if wrong_clicks:
                record["wrong_click_count"] = int(record.get("wrong_click_count", 0) or 0) + wrong_clicks
                if record.get("promotion_state") == "trusted":
                    record["promotion_state"] = "verified"
                self._save_json(self.workflow_path, self.workflow_memory)
        if status == "success":
            self._save_shortcut_and_fragment_skills(task, plan, latency_seconds=latency_seconds)
        self._ensure_skill_capsule_fields(record)
        self._save_json(self.workflow_path, self.workflow_memory)
        return record

    def find_workflow(self, prompt: str) -> dict[str, Any] | None:
        normalized_prompt = self.normalize_prompt(prompt)
        candidates = []
        for workflow in self.workflow_memory.get("workflows", []):
            if workflow.get("normalized_prompt") == normalized_prompt or workflow.get("objective_key") == normalized_prompt:
                candidates.append(workflow)
        if not candidates:
            return None
        candidates.sort(key=self._workflow_selection_key)
        return candidates[0]

    def list_workflows(self) -> list[dict[str, Any]]:
        workflows = list(self.workflow_memory.get("workflows", []))
        for workflow in workflows:
            self._ensure_skill_capsule_fields(workflow)
        workflows.sort(key=self._workflow_selection_key)
        return workflows

    def list_skill_capsules(self) -> list[dict[str, Any]]:
        return [self.build_skill_capsule(workflow) for workflow in self.list_workflows()]

    def save_plan_as_workflow(
        self,
        task: TaskSpec,
        plan: ExecutionPlan,
        name: str = "",
        promotion_state: str = "draft",
        workflow_type: str = "workflow",
        objective_key: str = "",
    ) -> dict[str, Any]:
        record = self._upsert_workflow_record(
            task=task,
            plan=plan,
            name=name or task.goal[:80],
            promotion_state=promotion_state,
            workflow_type=workflow_type,
            objective_key=objective_key,
        )
        record["last_run"] = time.time()
        self._ensure_skill_capsule_fields(record)
        self._save_json(self.workflow_path, self.workflow_memory)
        return record

    def _save_shortcut_and_fragment_skills(
        self,
        task: TaskSpec,
        plan: ExecutionPlan,
        *,
        latency_seconds: float | None = None,
    ) -> None:
        meaningful_actions = {"press_key", "open_explorer_location", "click_node", "modified_click_node", "type_text"}
        for prefix_length in range(1, min(3, len(plan.steps)) + 1):
            prefix_steps = plan.steps[:prefix_length]
            prefix_title = prefix_steps[-1].title.strip() or f"{task.goal} part {prefix_length}"
            prefix_task = TaskSpec(prompt=prefix_title, goal=prefix_title, trust_mode=task.trust_mode)
            prefix_plan = ExecutionPlan(
                task=prefix_task,
                steps=prefix_steps,
                source="success_prefix_memory",
                summary=prefix_title,
                required_apps=list({app for app in plan.required_apps if app}),
                success_conditions=[prefix_steps[-1].success_criteria] if prefix_steps[-1].success_criteria else [],
            )
            self.record_workflow_run(
                prefix_task,
                prefix_plan,
                True,
                latency_seconds=latency_seconds,
                workflow_type="fragment",
                objective_key=self.normalize_prompt(prefix_title),
                name=f"{task.goal} / part {prefix_length}",
            )
        for index, step in enumerate(plan.steps):
            if step.action_type not in meaningful_actions:
                continue
            fragment_start = max(0, index - 2)
            fragment_steps = plan.steps[fragment_start : index + 1]
            if not fragment_steps:
                continue
            fragment_prompt = step.title.strip() or step.success_criteria.strip() or task.goal.strip()
            if not fragment_prompt:
                continue
            fragment_task = TaskSpec(prompt=fragment_prompt, goal=fragment_prompt, trust_mode=task.trust_mode)
            fragment_plan = ExecutionPlan(
                task=fragment_task,
                steps=fragment_steps,
                source="success_fragment_memory",
                summary=fragment_prompt,
                required_apps=list({app for app in plan.required_apps if app}),
                success_conditions=[step.success_criteria] if step.success_criteria else [],
            )
            workflow_type = "shortcut" if step.action_type == "press_key" and step.parameters.get("shortcut_id") else "fragment"
            objective_key = self.normalize_prompt(str(step.parameters.get("shortcut_id") or fragment_prompt))
            self.record_workflow_run(
                fragment_task,
                fragment_plan,
                True,
                latency_seconds=latency_seconds,
                workflow_type=workflow_type,
                objective_key=objective_key,
                name=fragment_prompt,
            )

    def workflow_promotion_eligibility(self, workflow: dict[str, Any]) -> dict[str, Any]:
        success_count = int(workflow.get("success_count", 0) or 0)
        failure_count = int(workflow.get("failure_count", 0) or 0)
        wrong_click_count = int(workflow.get("wrong_click_count", 0) or 0)
        user_approved = bool(workflow.get("user_approved", False))
        blockers = []
        if not user_approved:
            blockers.append("user_approval_required")
        if success_count < 5:
            blockers.append("needs_5_successful_runs")
        if failure_count != 0:
            blockers.append("failure_count_must_be_zero")
        if wrong_click_count != 0:
            blockers.append("wrong_click_count_must_be_zero")
        return {
            "eligible": not blockers,
            "blockers": blockers,
            "success_count": success_count,
            "failure_count": failure_count,
            "wrong_click_count": wrong_click_count,
            "user_approved": user_approved,
        }

    def approve_workflow(self, workflow_id: str) -> bool:
        for workflow in self.workflow_memory.get("workflows", []):
            if workflow.get("workflow_id") == workflow_id:
                workflow["user_approved"] = True
                workflow["last_run"] = time.time()
                workflow["promotion_state"] = self._workflow_state(workflow)
                self._ensure_skill_capsule_fields(workflow)
                self._save_json(self.workflow_path, self.workflow_memory)
                return True
        return False

    def promote_workflow(self, workflow_id: str, promotion_state: str = "trusted") -> bool:
        for workflow in self.workflow_memory.get("workflows", []):
            if workflow.get("workflow_id") == workflow_id:
                if promotion_state == "trusted":
                    eligibility = self.workflow_promotion_eligibility(workflow)
                    if not eligibility["eligible"]:
                        workflow["promotion_blockers"] = eligibility["blockers"]
                        self._save_json(self.workflow_path, self.workflow_memory)
                        return False
                workflow["promotion_state"] = promotion_state
                workflow["last_run"] = time.time()
                self._ensure_skill_capsule_fields(workflow)
                self._save_json(self.workflow_path, self.workflow_memory)
                return True
        return False

    def record_skill_replay(
        self,
        workflow_id: str,
        *,
        success: bool,
        variant_count: int = 1,
        latency_seconds: float | None = None,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        for workflow in self.workflow_memory.get("workflows", []):
            if workflow.get("workflow_id") != workflow_id:
                continue
            if success:
                workflow["success_count"] = int(workflow.get("success_count", 0) or 0) + 1
                workflow["last_verified"] = time.time()
            else:
                workflow["failure_count"] = int(workflow.get("failure_count", 0) or 0) + 1
                if workflow.get("promotion_state") == "trusted":
                    workflow["promotion_state"] = "verified"
            workflow["replay_count"] = int(workflow.get("replay_count", 0) or 0) + 1
            workflow["variant_count"] = int(workflow.get("variant_count", 0) or 0) + max(1, int(variant_count or 1))
            workflow["last_run"] = time.time()
            workflow["success_rate"] = self._workflow_success_rate(workflow)
            workflow["replay_score"] = workflow["success_rate"]
            if latency_seconds is not None:
                workflow["total_latency_seconds"] = float(workflow.get("total_latency_seconds", 0.0) or 0.0) + float(latency_seconds)
                run_count = int(workflow.get("success_count", 0) or 0) + int(workflow.get("failure_count", 0) or 0)
                workflow["avg_latency_seconds"] = round(workflow["total_latency_seconds"] / max(1, run_count), 6)
            if trace:
                replays = workflow.setdefault("replay_traces", [])
                if isinstance(replays, list):
                    replays.append(dict(trace))
                    workflow["replay_traces"] = replays[-20:]
            workflow["promotion_state"] = self._workflow_state(workflow)
            self._ensure_skill_capsule_fields(workflow)
            self._save_json(self.workflow_path, self.workflow_memory)
            return self.build_skill_capsule(workflow)
        return None

    def build_skill_manifest(self, workflow_ids: list[str] | None = None) -> dict[str, Any]:
        selected_ids = set(workflow_ids or [])
        workflows = [
            self.build_skill_capsule(workflow)
            for workflow in self.workflow_memory.get("workflows", [])
            if not selected_ids or workflow.get("workflow_id") in selected_ids
        ]
        return {
            "manifest_version": 2,
            "exported_at": time.time(),
            "skills": workflows,
            "skill_count": len(workflows),
        }

    def import_skill_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        skills = manifest.get("skills", [])
        if not isinstance(skills, list):
            return {"imported": 0, "updated": 0, "skipped": 0}
        workflows = self.workflow_memory.setdefault("workflows", [])
        by_id = {workflow.get("workflow_id"): workflow for workflow in workflows}
        imported = 0
        updated = 0
        skipped = 0
        for skill in skills:
            if not isinstance(skill, dict) or not skill.get("workflow_id"):
                skipped += 1
                continue
            record = dict(skill)
            if record.get("capsule_type") == "skill_capsule":
                record["name"] = record.get("name") or record.get("skill_name", "")
                record["prompt_pattern"] = record.get("prompt_pattern") or record.get("trigger_phrase", "")
                record["required_apps"] = record.get("required_apps", []) or ([record.get("app")] if record.get("app") else [])
            record.setdefault("promotion_state", "draft")
            record.setdefault("wrong_click_count", 0)
            record.setdefault("user_approved", False)
            self._ensure_skill_capsule_fields(record)
            existing = by_id.get(record["workflow_id"])
            if existing:
                existing.update(record)
                updated += 1
            else:
                workflows.append(record)
                by_id[record["workflow_id"]] = record
                imported += 1
        self._save_json(self.workflow_path, self.workflow_memory)
        return {"imported": imported, "updated": updated, "skipped": skipped}

    def allow_high_risk_for_app(self, app_id: str) -> bool:
        normalized_app = self.normalize_prompt(app_id)
        if not normalized_app:
            return False
        allowed = self.policy_memory.setdefault("high_risk_allowed_apps", [])
        if normalized_app not in allowed:
            allowed.append(normalized_app)
        trusted = self.policy_memory.setdefault("trusted_apps", [])
        if normalized_app not in trusted:
            trusted.append(normalized_app)
        self._save_json(self.policy_path, self.policy_memory)
        return True

    def revoke_high_risk_for_app(self, app_id: str) -> bool:
        normalized_app = self.normalize_prompt(app_id)
        if not normalized_app:
            return False
        self.policy_memory["high_risk_allowed_apps"] = [
            item for item in self.policy_memory.get("high_risk_allowed_apps", []) if item != normalized_app
        ]
        self._save_json(self.policy_path, self.policy_memory)
        return True

    def _remember_control(
        self,
        node: ObservationNode,
        label: str,
        concepts: list[str],
        entity_type: str = "",
        affordances: list[str] | None = None,
        app_id: str = "",
        risk_level: str = "",
    ) -> None:
        control_key = "|".join(
            [
                node.semantic_role or node.node_type or "unknown",
                node.region or "",
                self.normalize_prompt(label),
                (self._node_visual_ids(node) or [""])[0],
                app_id or node.app_id or "",
            ]
        )
        control_record = self.semantic_memory.setdefault("controls", {}).setdefault(
            control_key,
            {
                "display_label": label,
                "labels": {},
                "concepts": {},
                "affordances": {},
                "semantic_role": node.semantic_role,
                "node_type": node.node_type,
                "entity_type": entity_type or node.entity_type,
                "region": node.region,
                "app_id": app_id or node.app_id,
                "visual_ids": {},
                "seen_count": 0,
                "risk_level": "",
                "last_seen": None,
            },
        )
        self._ensure_counter_fields(control_record, ["labels", "concepts", "affordances", "visual_ids"])
        control_record["display_label"] = label or control_record.get("display_label", "")
        control_record["semantic_role"] = node.semantic_role or control_record.get("semantic_role", "")
        control_record["node_type"] = node.node_type or control_record.get("node_type", "")
        control_record["entity_type"] = entity_type or node.entity_type or control_record.get("entity_type", "")
        control_record["region"] = node.region or control_record.get("region", "")
        control_record["app_id"] = app_id or node.app_id or control_record.get("app_id", "")
        control_record["seen_count"] += 1
        control_record["last_seen"] = time.time()
        if risk_level:
            control_record["risk_level"] = risk_level
        if label:
            control_record["labels"][label] = control_record["labels"].get(label, 0) + 1
        for visual_id in self._node_visual_ids(node):
            control_record["visual_ids"][visual_id] = control_record["visual_ids"].get(visual_id, 0) + 1
        for concept in concepts:
            control_record["concepts"][concept] = control_record["concepts"].get(concept, 0) + 1
        for affordance in affordances or []:
            control_record["affordances"][affordance] = control_record["affordances"].get(affordance, 0) + 1

    def remember_observation_graph(self, graph: ObservationGraph) -> None:
        for node in graph.flatten():
            label = node.display_label() or node.label
            concepts = list(node.learned_concepts or [])
            affordances = list(node.affordances or [])
            entity_type = node.entity_type or ""
            app_id = node.app_id or graph.metadata.get("app_id", "")
            if entity_type and entity_type not in concepts:
                concepts.append(entity_type)
            node_visual_ids = self._node_visual_ids(node)
            if not label and not concepts and not node_visual_ids:
                continue
            self._remember_control(
                node,
                label=label,
                concepts=concepts,
                entity_type=entity_type,
                affordances=affordances,
                app_id=app_id,
            )

            normalized_label = self.normalize_prompt(label)
            if normalized_label and not normalized_label.startswith("["):
                label_record = self.semantic_memory.setdefault("labels", {}).setdefault(
                    normalized_label,
                    {
                        "display_label": label,
                        "seen_count": 0,
                        "concept_counts": {},
                        "visual_ids": {},
                        "entity_types": {},
                        "last_seen": None,
                    },
                )
                self._ensure_counter_fields(label_record, ["concept_counts", "visual_ids", "entity_types"])
                label_record["display_label"] = label
                label_record["seen_count"] += 1
                label_record["last_seen"] = time.time()
                for concept in concepts:
                    label_record["concept_counts"][concept] = label_record["concept_counts"].get(concept, 0) + 1
                if entity_type:
                    label_record["entity_types"][entity_type] = label_record["entity_types"].get(entity_type, 0) + 1
                for visual_id in node_visual_ids:
                    label_record["visual_ids"][visual_id] = label_record["visual_ids"].get(visual_id, 0) + 1

            for visual_id in node_visual_ids:
                visual_record = self.semantic_memory.setdefault("visuals", {}).setdefault(
                    visual_id,
                    {
                        "seen_count": 0,
                        "concept_counts": {},
                        "label_counts": {},
                        "entity_types": {},
                        "last_seen": None,
                    },
                )
                self._ensure_counter_fields(visual_record, ["concept_counts", "label_counts", "entity_types"])
                visual_record["seen_count"] += 1
                visual_record["last_seen"] = time.time()
                if normalized_label and not normalized_label.startswith("["):
                    visual_record["label_counts"][label] = visual_record["label_counts"].get(label, 0) + 1
                for concept in concepts:
                    visual_record["concept_counts"][concept] = visual_record["concept_counts"].get(concept, 0) + 1
                if entity_type:
                    visual_record["entity_types"][entity_type] = visual_record["entity_types"].get(entity_type, 0) + 1

            for concept in concepts:
                concept_record = self.semantic_memory.setdefault("concepts", {}).setdefault(
                    concept,
                    {"seen_count": 0, "labels": {}, "last_seen": None},
                )
                self._ensure_counter_fields(concept_record, ["labels"])
                concept_record["seen_count"] += 1
                concept_record["last_seen"] = time.time()
                if normalized_label and not normalized_label.startswith("["):
                    concept_record["labels"][label] = concept_record["labels"].get(label, 0) + 1

            if app_id and label:
                entity_record = self.semantic_memory.setdefault("entities", {}).setdefault(
                    self.normalize_prompt(app_id),
                    {
                        "display_name": app_id,
                        "labels": {},
                        "visual_ids": {},
                        "window_titles": [],
                        "entity_types": {},
                        "last_seen": None,
                    },
                )
                self._ensure_counter_fields(entity_record, ["labels", "visual_ids", "entity_types"])
                if not isinstance(entity_record.get("window_titles"), list):
                    entity_record["window_titles"] = []
                entity_record["display_name"] = app_id
                entity_record["labels"][label] = entity_record["labels"].get(label, 0) + 1
                if entity_type:
                    entity_record["entity_types"][entity_type] = entity_record["entity_types"].get(entity_type, 0) + 1
                for visual_id in node_visual_ids:
                    entity_record["visual_ids"][visual_id] = entity_record["visual_ids"].get(visual_id, 0) + 1
                entity_record["last_seen"] = time.time()

        self._save_json(self.semantic_path, self.semantic_memory)

    def remember_transition(
        self,
        action_type: str,
        source_visual_id: str = "",
        target_visual_id: str = "",
        target_label: str = "",
        outcome: str = "",
    ) -> None:
        transitions = self.semantic_memory.setdefault("transitions", [])
        transitions.append(
            {
                "timestamp": time.time(),
                "action_type": action_type,
                "source_visual_id": source_visual_id,
                "target_visual_id": target_visual_id,
                "target_label": target_label,
                "outcome": outcome,
            }
        )
        self.semantic_memory["transitions"] = transitions[-400:]
        self._save_json(self.semantic_path, self.semantic_memory)

    def remember_negative_example(self, node: ObservationNode, note: str = "") -> None:
        negatives = self.semantic_memory.setdefault("negative_examples", [])
        negatives.append(
            {
                "timestamp": time.time(),
                "label": node.display_label(),
                "visual_id": node.visual_id,
                "visual_ids": self._node_visual_ids(node),
                "entity_type": node.entity_type,
                "app_id": node.app_id,
                "note": note,
            }
        )
        self.semantic_memory["negative_examples"] = negatives[-300:]
        self._save_json(self.semantic_path, self.semantic_memory)

    def enqueue_review_item(
        self,
        node: ObservationNode,
        reason: str,
        feedback_labels: list[str] | None = None,
        confidence: float = 0.0,
        app_id: str = "",
    ) -> dict[str, Any]:
        review_queue = self.semantic_memory.setdefault("review_queue", [])
        visual_ids = self._node_visual_ids(node)
        item = {
            "review_id": f"review_{int(time.time() * 1000)}_{len(review_queue) + 1}",
            "status": "pending",
            "timestamp": time.time(),
            "reason": reason,
            "confidence": confidence,
            "label": node.display_label() or node.label,
            "node_id": node.node_id,
            "node_type": node.node_type,
            "semantic_role": node.semantic_role,
            "entity_type": node.entity_type,
            "app_id": app_id or node.app_id,
            "region": node.region,
            "affordances": list(node.affordances),
            "state_tags": list(node.state_tags),
            "box": dict(node.box),
            "center": dict(node.center),
            "visual_id": node.visual_id,
            "visual_ids": visual_ids,
            "feedback_labels": list(feedback_labels or []),
        }
        review_queue.append(item)
        self.semantic_memory["review_queue"] = review_queue[-500:]
        self._save_json(self.semantic_path, self.semantic_memory)
        return item

    def list_review_items(self, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
        items = self.semantic_memory.get("review_queue", [])
        if status:
            items = [item for item in items if item.get("status") == status]
        return list(reversed(items[-limit:]))

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
        for item in self.semantic_memory.get("review_queue", []):
            if item.get("review_id") != review_id:
                continue
            item["status"] = status
            item["resolved_at"] = time.time()
            item["resolution_note"] = note

            node = ObservationNode.from_raw(
                {
                    "id": item.get("node_id", ""),
                    "label": item.get("label", ""),
                    "type": item.get("node_type", "unknown"),
                    "semantic_role": item.get("semantic_role", ""),
                    "entity_type": entity_type or item.get("entity_type", ""),
                    "app_id": app_identity or item.get("app_id", ""),
                    "region": item.get("region", ""),
                    "affordances": affordances or item.get("affordances", []),
                    "state_tags": item.get("state_tags", []),
                    "box": item.get("box", {}),
                    "center": item.get("center", {}),
                    "visual_id": item.get("visual_id", ""),
                    "visual_ids": item.get("visual_ids", []),
                }
            )
            if status in {"accepted", "corrected"}:
                final_label = label or item.get("label", "")
                final_concepts = concepts if concepts is not None else ["reviewed"]
                if status == "corrected" and "corrected" not in final_concepts:
                    final_concepts.append("corrected")
                self.teach_node(
                    node=node,
                    label=final_label,
                    concepts=final_concepts,
                    app_identity=app_identity or item.get("app_id", ""),
                    entity_type=entity_type or item.get("entity_type", ""),
                    affordances=affordances or item.get("affordances", []),
                    outcome_correct=status == "accepted",
                )
            elif status == "unsafe":
                self.remember_negative_example(node, note=note or "Marked unsafe from review queue")

            self._save_json(self.semantic_path, self.semantic_memory)
            return True
        return False

    def record_learning_session(self, session: dict[str, Any]) -> None:
        sessions = self.semantic_memory.setdefault("learning_sessions", [])
        sessions.append(session)
        self.semantic_memory["learning_sessions"] = sessions[-120:]
        self._save_json(self.semantic_path, self.semantic_memory)

    def _scene_signature(self, graph: ObservationGraph | None) -> str:
        if not graph:
            return "unknown_scene"
        app_id = str(graph.metadata.get("app_id", "") or graph.metadata.get("scene", {}).get("app_id", "")).strip().lower()
        labels = []
        for node in graph.flatten():
            label = self.normalize_prompt(node.display_label() or node.label)
            if label and not self._is_placeholder_label(label):
                labels.append(label)
        labels = sorted(set(labels))[:24]
        return "|".join([app_id or "unknown_app", *labels])[:600]

    def _control_signature(self, node: ObservationNode, app_id: str = "") -> str:
        visual_ids = self._node_visual_ids(node)
        label = self.normalize_prompt(node.display_label() or node.label)
        parts = [
            app_id or node.app_id or "",
            node.semantic_role or node.node_type or "unknown",
            node.entity_type or "",
            node.region or "",
            label,
            visual_ids[0] if visual_ids else "",
        ]
        return "|".join(str(part).strip().lower() for part in parts)[:420]

    def record_interaction_outcome(
        self,
        before: ObservationGraph | None,
        after: ObservationGraph | None,
        node: ObservationNode,
        action_type: str,
        reward: float,
        outcome: str,
        app_id: str = "",
        recovery: str = "",
    ) -> dict[str, Any]:
        graph = self.semantic_memory.setdefault("interaction_graph", {
            "version": 1,
            "scene_nodes": {},
            "control_nodes": {},
            "edges": {},
            "action_values": {},
        })
        scene_nodes = graph.setdefault("scene_nodes", {})
        control_nodes = graph.setdefault("control_nodes", {})
        edges = graph.setdefault("edges", {})
        action_values = graph.setdefault("action_values", {})

        before_sig = self._scene_signature(before)
        after_sig = self._scene_signature(after)
        control_sig = self._control_signature(node, app_id=app_id)
        label = node.display_label() or node.label
        visual_ids = self._node_visual_ids(node)
        now = time.time()

        scene_nodes.setdefault(before_sig, {"signature": before_sig, "seen_count": 0, "last_seen": None})
        scene_nodes[before_sig]["seen_count"] += 1
        scene_nodes[before_sig]["last_seen"] = now
        scene_nodes.setdefault(after_sig, {"signature": after_sig, "seen_count": 0, "last_seen": None})
        scene_nodes[after_sig]["seen_count"] += 1
        scene_nodes[after_sig]["last_seen"] = now

        control_record = control_nodes.setdefault(
            control_sig,
            {
                "signature": control_sig,
                "label": label,
                "app_id": app_id or node.app_id,
                "semantic_role": node.semantic_role,
                "entity_type": node.entity_type,
                "region": node.region,
                "visual_ids": {},
                "seen_count": 0,
                "last_seen": None,
            },
        )
        control_record["label"] = label or control_record.get("label", "")
        control_record["seen_count"] += 1
        control_record["last_seen"] = now
        for visual_id in visual_ids:
            control_record["visual_ids"][visual_id] = control_record["visual_ids"].get(visual_id, 0) + 1

        edge_key = "|".join([before_sig, action_type, control_sig, after_sig])
        edge = edges.setdefault(
            edge_key,
            {
                "edge_id": f"edge_{len(edges) + 1}",
                "before_scene": before_sig,
                "after_scene": after_sig,
                "control_signature": control_sig,
                "action_type": action_type,
                "target_label": label,
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "reward_total": 0.0,
                "reward_avg": 0.0,
                "outcomes": {},
                "recovery": "",
                "last_seen": None,
            },
        )
        edge["attempts"] += 1
        if reward > 0:
            edge["successes"] += 1
        else:
            edge["failures"] += 1
        edge["reward_total"] += reward
        edge["reward_avg"] = edge["reward_total"] / max(1, edge["attempts"])
        edge["outcomes"][outcome] = edge["outcomes"].get(outcome, 0) + 1
        edge["recovery"] = recovery or edge.get("recovery", "")
        edge["last_seen"] = now

        value_key = control_sig
        value = action_values.setdefault(
            value_key,
            {
                "control_signature": control_sig,
                "label": label,
                "app_id": app_id or node.app_id,
                "entity_type": node.entity_type,
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "reward_total": 0.0,
                "reward_avg": 0.0,
                "last_outcome": "",
                "last_seen": None,
            },
        )
        value["label"] = label or value.get("label", "")
        value["attempts"] += 1
        if reward > 0:
            value["successes"] += 1
        else:
            value["failures"] += 1
        value["reward_total"] += reward
        value["reward_avg"] = value["reward_total"] / max(1, value["attempts"])
        value["last_outcome"] = outcome
        value["last_seen"] = now

        transition = {
            "timestamp": now,
            "action_type": action_type,
            "source_visual_id": visual_ids[0] if visual_ids else "",
            "target_visual_id": "",
            "target_label": label,
            "outcome": outcome,
            "reward": reward,
            "before_scene": before_sig,
            "after_scene": after_sig,
            "control_signature": control_sig,
        }
        transitions = self.semantic_memory.setdefault("transitions", [])
        transitions.append(transition)
        self.semantic_memory["transitions"] = transitions[-800:]
        self._save_json(self.semantic_path, self.semantic_memory)
        return edge

    def preferred_interaction_labels(self, app_id: str = "", min_reward: float = 0.2, limit: int = 8) -> list[str]:
        normalized_app = self.normalize_prompt(app_id)
        values = []
        for value in self.semantic_memory.get("interaction_graph", {}).get("action_values", {}).values():
            if not isinstance(value, dict):
                continue
            if normalized_app and self.normalize_prompt(value.get("app_id", "")) != normalized_app:
                continue
            if float(value.get("reward_avg", 0.0) or 0.0) < min_reward:
                continue
            label = str(value.get("label", "")).strip()
            if not label or self._is_placeholder_label(label):
                continue
            values.append((float(value.get("reward_avg", 0.0) or 0.0), int(value.get("successes", 0) or 0), label))
        values.sort(reverse=True)
        deduped = []
        seen: set[str] = set()
        for _reward, _successes, label in values:
            normalized = self.normalize_prompt(label)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(label)
            if len(deduped) >= limit:
                break
        return deduped

    def find_interaction_replay(
        self,
        prompt: str,
        app_id: str = "",
        min_reward: float = 0.45,
    ) -> dict[str, Any] | None:
        normalized_prompt = self.normalize_prompt(prompt)
        normalized_app = self.normalize_prompt(app_id)
        if not normalized_prompt:
            return None

        candidates = []
        controls = self.semantic_memory.get("interaction_graph", {}).get("control_nodes", {})
        for signature, value in self.semantic_memory.get("interaction_graph", {}).get("action_values", {}).items():
            if not isinstance(value, dict):
                continue
            label = str(value.get("label", "")).strip()
            normalized_label = self.normalize_prompt(label)
            if not label or self._is_placeholder_label(label):
                continue
            if normalized_label not in normalized_prompt:
                continue
            if normalized_app and self.normalize_prompt(value.get("app_id", "")) not in {"", normalized_app}:
                continue
            reward_avg = float(value.get("reward_avg", 0.0) or 0.0)
            successes = int(value.get("successes", 0) or 0)
            failures = int(value.get("failures", 0) or 0)
            if reward_avg < min_reward or successes <= failures:
                continue
            control = controls.get(signature, {}) if isinstance(controls, dict) else {}
            candidates.append(
                (
                    reward_avg,
                    successes,
                    {
                        "label": label,
                        "app_id": value.get("app_id", "") or control.get("app_id", ""),
                        "entity_type": value.get("entity_type", "") or control.get("entity_type", ""),
                        "semantic_role": control.get("semantic_role", ""),
                        "region": control.get("region", ""),
                        "control_signature": signature,
                        "reward_avg": reward_avg,
                        "successes": successes,
                        "failures": failures,
                        "last_outcome": value.get("last_outcome", ""),
                    },
                )
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    def _prompt_terms(self, prompt: str) -> list[str]:
        normalized = self.normalize_prompt(prompt)
        stop_words = {
            "a",
            "an",
            "and",
            "app",
            "click",
            "current",
            "go",
            "in",
            "into",
            "open",
            "please",
            "show",
            "the",
            "to",
            "ui",
            "window",
        }
        return [term for term in re.findall(r"[a-z0-9_.-]+", normalized) if len(term) >= 3 and term not in stop_words]

    def find_interaction_path(
        self,
        prompt: str,
        app_id: str = "",
        start_scene: str = "",
        min_reward: float = 0.45,
        max_steps: int = 4,
    ) -> list[dict[str, Any]]:
        terms = self._prompt_terms(prompt)
        if not terms:
            return []
        normalized_app = self.normalize_prompt(app_id)
        graph = self.semantic_memory.get("interaction_graph", {})
        edges = graph.get("edges", {})
        controls = graph.get("control_nodes", {})

        viable_edges: list[dict[str, Any]] = []
        for edge in edges.values():
            if not isinstance(edge, dict):
                continue
            reward_avg = float(edge.get("reward_avg", 0.0) or 0.0)
            if reward_avg < min_reward:
                continue
            if int(edge.get("successes", 0) or 0) <= int(edge.get("failures", 0) or 0):
                continue
            control = controls.get(edge.get("control_signature", ""), {}) if isinstance(controls, dict) else {}
            edge_app = self.normalize_prompt(control.get("app_id", ""))
            if normalized_app and edge_app and edge_app != normalized_app:
                continue
            enriched = dict(edge)
            enriched["control"] = control
            viable_edges.append(enriched)

        if not viable_edges:
            return []

        def matches_goal(edge: dict[str, Any]) -> bool:
            label = self.normalize_prompt(edge.get("target_label", ""))
            after_scene = self.normalize_prompt(edge.get("after_scene", ""))
            return any(term in label or term in after_scene for term in terms)

        adjacency: dict[str, list[dict[str, Any]]] = {}
        for edge in viable_edges:
            adjacency.setdefault(edge.get("before_scene", ""), []).append(edge)
        for candidates in adjacency.values():
            candidates.sort(key=lambda item: float(item.get("reward_avg", 0.0) or 0.0), reverse=True)

        start_scenes = [start_scene] if start_scene and start_scene in adjacency else []
        if not start_scenes:
            start_scenes = sorted(adjacency, key=lambda scene: 0 if normalized_app and scene.startswith(normalized_app) else 1)

        best_path: list[dict[str, Any]] = []
        best_score = -999.0
        queue: list[tuple[str, list[dict[str, Any]], set[str]]] = [(scene, [], {scene}) for scene in start_scenes[:12]]
        while queue:
            scene, path, visited = queue.pop(0)
            if len(path) >= max_steps:
                continue
            for edge in adjacency.get(scene, [])[:8]:
                after_scene = edge.get("after_scene", "")
                if after_scene in visited:
                    continue
                next_path = [*path, edge]
                reward_score = sum(float(item.get("reward_avg", 0.0) or 0.0) for item in next_path)
                goal_bonus = 10.0 if matches_goal(edge) else 0.0
                shorter_bonus = max(0.0, 2.0 - len(next_path) * 0.25)
                score = reward_score + goal_bonus + shorter_bonus
                if goal_bonus and score > best_score:
                    best_score = score
                    best_path = next_path
                queue.append((after_scene, next_path, {*visited, after_scene}))

        return best_path

    def interaction_dashboard(self, limit: int = 10) -> dict[str, Any]:
        graph = self.semantic_memory.get("interaction_graph", {})
        edges = [edge for edge in graph.get("edges", {}).values() if isinstance(edge, dict)]
        values = [value for value in graph.get("action_values", {}).values() if isinstance(value, dict)]
        positive_edges = [edge for edge in edges if float(edge.get("reward_avg", 0.0) or 0.0) > 0]
        negative_edges = [edge for edge in edges if float(edge.get("reward_avg", 0.0) or 0.0) <= 0]
        attempts = sum(int(value.get("attempts", 0) or 0) for value in values)
        successes = sum(int(value.get("successes", 0) or 0) for value in values)
        failures = sum(int(value.get("failures", 0) or 0) for value in values)

        def compact_value(value: dict[str, Any]) -> dict[str, Any]:
            return {
                "label": value.get("label", ""),
                "app_id": value.get("app_id", ""),
                "entity_type": value.get("entity_type", ""),
                "attempts": value.get("attempts", 0),
                "successes": value.get("successes", 0),
                "failures": value.get("failures", 0),
                "reward_avg": round(float(value.get("reward_avg", 0.0) or 0.0), 3),
                "last_outcome": value.get("last_outcome", ""),
            }

        top_actions = sorted(values, key=lambda item: (float(item.get("reward_avg", 0.0) or 0.0), int(item.get("successes", 0) or 0)), reverse=True)
        weak_actions = sorted(values, key=lambda item: (float(item.get("reward_avg", 0.0) or 0.0), -int(item.get("failures", 0) or 0)))
        return {
            "scene_nodes": len(graph.get("scene_nodes", {})),
            "control_nodes": len(graph.get("control_nodes", {})),
            "edges": len(edges),
            "positive_edges": len(positive_edges),
            "negative_edges": len(negative_edges),
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
            "success_rate": round(successes / attempts, 3) if attempts else 0.0,
            "ready_for_replay": bool(positive_edges),
            "ready_for_multistep": len(positive_edges) >= 2,
            "top_actions": [compact_value(item) for item in top_actions[:limit]],
            "weak_actions": [compact_value(item) for item in weak_actions[:limit]],
        }

    def operator_status(self) -> dict[str, Any]:
        summary = self.summary()
        dashboard = self.interaction_dashboard(limit=5)
        pending_reviews = int(summary.get("review_queue", 0) or 0)
        known_controls = int(summary.get("known_controls", 0) or 0)
        known_visuals = int(summary.get("known_visuals", 0) or 0)
        positive_edges = int(dashboard.get("positive_edges", 0) or 0)
        attempts = int(dashboard.get("attempts", 0) or 0)
        success_rate = float(dashboard.get("success_rate", 0.0) or 0.0)
        trusted_workflows = int(summary.get("trusted_workflows", 0) or 0)

        score = 0
        if known_visuals >= 20:
            score += 20
        elif known_visuals >= 5:
            score += 10
        if known_controls >= 10:
            score += 20
        elif known_controls >= 3:
            score += 10
        if positive_edges >= 3:
            score += 25
        elif positive_edges >= 1:
            score += 15
        if attempts >= 5 and success_rate >= 0.6:
            score += 20
        elif attempts >= 1:
            score += 8
        if trusted_workflows:
            score += 15
        if pending_reviews >= 10:
            score -= 15
        elif pending_reviews >= 1:
            score -= 5
        score = max(0, min(100, score))

        if score >= 75:
            level = "production-ready"
            next_step = "Run known tasks or promote stable workflows."
        elif score >= 45:
            level = "supervised"
            next_step = "Resolve review items and repeat successful learning runs."
        elif known_controls or known_visuals:
            level = "learning"
            next_step = "Scan the current app, teach weak labels, then run safe hover learning."
        else:
            level = "cold-start"
            next_step = "Start with Scan Current App so the agent can build a UI map."

        blockers = []
        if known_controls < 3:
            blockers.append("few_known_controls")
        if positive_edges < 1:
            blockers.append("no_rewarded_clicks")
        if pending_reviews >= 10:
            blockers.append("review_queue_backlog")
        if attempts >= 3 and success_rate < 0.4:
            blockers.append("low_interaction_success_rate")

        return {
            "level": level,
            "readiness_score": score,
            "next_step": next_step,
            "blockers": blockers,
            "safe_to_replay": bool(dashboard.get("ready_for_replay")) and score >= 45,
            "safe_for_multistep": bool(dashboard.get("ready_for_multistep")) and score >= 60,
            "memory": summary,
            "learning": dashboard,
        }

    def remember_hover_feedback(
        self,
        node: ObservationNode,
        feedback_labels: list[str],
        app_id: str = "",
    ) -> None:
        clean_feedback = []
        seen: set[str] = set()
        for label in feedback_labels:
            label = str(label or "").strip()
            normalized = self.normalize_prompt(label)
            if not normalized or normalized in seen or self._is_placeholder_label(label):
                continue
            seen.add(normalized)
            clean_feedback.append(label)
        if not clean_feedback:
            return

        app_identity = app_id or node.app_id
        node_visual_ids = self._node_visual_ids(node)
        target_label = node.display_label() or node.label
        learned_label = target_label if not self._is_placeholder_label(target_label) else clean_feedback[0]
        concepts = ["hover_feedback", "tooltip"]
        if node.entity_type:
            concepts.append(node.entity_type)

        self._remember_control(
            node=node,
            label=learned_label,
            concepts=concepts,
            entity_type=node.entity_type,
            affordances=sorted(set(list(node.affordances) + ["hover"])),
            app_id=app_identity,
        )

        feedback_history = self.semantic_memory.setdefault("hover_feedback", [])
        feedback_history.append(
            {
                "timestamp": time.time(),
                "app_id": app_identity,
                "target_label": target_label,
                "target_node_id": node.node_id,
                "target_role": node.semantic_role,
                "target_region": node.region,
                "target_entity_type": node.entity_type,
                "target_visual_id": node.visual_id,
                "target_visual_ids": node_visual_ids,
                "feedback_labels": clean_feedback[:12],
            }
        )
        self.semantic_memory["hover_feedback"] = feedback_history[-500:]

        for label in clean_feedback:
            normalized_label = self.normalize_prompt(label)
            label_record = self.semantic_memory.setdefault("labels", {}).setdefault(
                normalized_label,
                {
                    "display_label": label,
                    "seen_count": 0,
                    "concept_counts": {},
                    "visual_ids": {},
                    "entity_types": {},
                    "last_seen": None,
                },
            )
            self._ensure_counter_fields(label_record, ["concept_counts", "visual_ids", "entity_types"])
            label_record["display_label"] = label
            label_record["seen_count"] += 2
            label_record["last_seen"] = time.time()
            for concept in concepts:
                label_record["concept_counts"][concept] = label_record["concept_counts"].get(concept, 0) + 2
            if node.entity_type:
                label_record["entity_types"][node.entity_type] = label_record["entity_types"].get(node.entity_type, 0) + 1
            for visual_id in node_visual_ids:
                label_record["visual_ids"][visual_id] = label_record["visual_ids"].get(visual_id, 0) + 2

        for concept in concepts:
            concept_record = self.semantic_memory.setdefault("concepts", {}).setdefault(
                concept,
                {"seen_count": 0, "labels": {}, "last_seen": None},
            )
            self._ensure_counter_fields(concept_record, ["labels"])
            concept_record["seen_count"] += 1
            concept_record["last_seen"] = time.time()
            for label in clean_feedback:
                concept_record["labels"][label] = concept_record["labels"].get(label, 0) + 2

        for visual_id in node_visual_ids:
            visual_record = self.semantic_memory.setdefault("visuals", {}).setdefault(
                visual_id,
                {
                    "seen_count": 0,
                    "concept_counts": {},
                    "label_counts": {},
                    "entity_types": {},
                    "last_seen": None,
                },
            )
            self._ensure_counter_fields(visual_record, ["concept_counts", "label_counts", "entity_types"])
            visual_record["seen_count"] += 1
            visual_record["last_seen"] = time.time()
            for label in clean_feedback:
                visual_record["label_counts"][label] = visual_record["label_counts"].get(label, 0) + 2
            for concept in concepts:
                visual_record["concept_counts"][concept] = visual_record["concept_counts"].get(concept, 0) + 1
            if node.entity_type:
                visual_record["entity_types"][node.entity_type] = visual_record["entity_types"].get(node.entity_type, 0) + 1

        self._save_json(self.semantic_path, self.semantic_memory)

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
        observed_label = node.display_label() or node.label
        node_visual_ids = self._node_visual_ids(node)
        normalized_label = self.normalize_prompt(label)
        if normalized_label:
            label_record = self.semantic_memory.setdefault("labels", {}).setdefault(
                normalized_label,
                {
                    "display_label": label,
                    "seen_count": 0,
                    "concept_counts": {},
                    "visual_ids": {},
                    "entity_types": {},
                    "last_seen": None,
                },
            )
            self._ensure_counter_fields(label_record, ["concept_counts", "visual_ids", "entity_types"])
            label_record["display_label"] = label
            label_record["seen_count"] += 1
            label_record["last_seen"] = time.time()
            for concept in concepts:
                label_record["concept_counts"][concept] = label_record["concept_counts"].get(concept, 0) + 3
            if entity_type:
                label_record["entity_types"][entity_type] = label_record["entity_types"].get(entity_type, 0) + 3
            for visual_id in node_visual_ids:
                label_record["visual_ids"][visual_id] = label_record["visual_ids"].get(visual_id, 0) + 3

        for visual_id in node_visual_ids:
            visual_record = self.semantic_memory.setdefault("visuals", {}).setdefault(
                visual_id,
                {
                    "seen_count": 0,
                    "concept_counts": {},
                    "label_counts": {},
                    "entity_types": {},
                    "last_seen": None,
                },
            )
            self._ensure_counter_fields(visual_record, ["concept_counts", "label_counts", "entity_types"])
            visual_record["seen_count"] += 1
            visual_record["last_seen"] = time.time()
            visual_record["label_counts"][label] = visual_record["label_counts"].get(label, 0) + 3
            if entity_type:
                visual_record["entity_types"][entity_type] = visual_record["entity_types"].get(entity_type, 0) + 3
            for concept in concepts:
                visual_record["concept_counts"][concept] = visual_record["concept_counts"].get(concept, 0) + 3

        for concept in concepts:
            concept_record = self.semantic_memory.setdefault("concepts", {}).setdefault(
                concept,
                {"seen_count": 0, "labels": {}, "last_seen": None},
            )
            self._ensure_counter_fields(concept_record, ["labels"])
            concept_record["seen_count"] += 1
            concept_record["last_seen"] = time.time()
            concept_record["labels"][label] = concept_record["labels"].get(label, 0) + 3

        if app_identity:
            entity_record = self.semantic_memory.setdefault("entities", {}).setdefault(
                self.normalize_prompt(app_identity),
                {
                    "display_name": app_identity,
                    "labels": {},
                    "visual_ids": {},
                    "window_titles": [],
                    "entity_types": {},
                    "last_seen": None,
                },
            )
            self._ensure_counter_fields(entity_record, ["labels", "visual_ids", "entity_types"])
            if not isinstance(entity_record.get("window_titles"), list):
                entity_record["window_titles"] = []
            entity_record["display_name"] = app_identity
            entity_record["last_seen"] = time.time()
            entity_record["labels"][label] = entity_record["labels"].get(label, 0) + 2
            if entity_type:
                entity_record["entity_types"][entity_type] = entity_record["entity_types"].get(entity_type, 0) + 2
            for visual_id in node_visual_ids:
                entity_record["visual_ids"][visual_id] = entity_record["visual_ids"].get(visual_id, 0) + 2
            if risk_level:
                entity_record["risk_level"] = risk_level
                normalized_risk = self.normalize_prompt(risk_level)
                trusted = self.policy_memory.setdefault("trusted_apps", [])
                blocked = self.policy_memory.setdefault("blocked_apps", [])
                normalized_app = self.normalize_prompt(app_identity)
                if normalized_risk in {"trusted", "allow", "safe"}:
                    if normalized_app and normalized_app not in trusted:
                        trusted.append(normalized_app)
                    self.policy_memory["blocked_apps"] = [item for item in blocked if item != normalized_app]
                    self._save_json(self.policy_path, self.policy_memory)
                elif normalized_risk in {"blocked", "deny", "risky", "unsafe"}:
                    if normalized_app and normalized_app not in blocked:
                        blocked.append(normalized_app)
                    self.policy_memory["trusted_apps"] = [item for item in trusted if item != normalized_app]
                    self._save_json(self.policy_path, self.policy_memory)

        self._remember_control(
            node=node,
            label=label,
            concepts=concepts,
            entity_type=entity_type,
            affordances=affordances or [],
            app_id=app_identity or node.app_id,
            risk_level=risk_level,
        )

        if not outcome_correct:
            self.remember_negative_example(node, note=f"Taught correction from '{observed_label}' to '{label}'")

        self._save_json(self.semantic_path, self.semantic_memory)

    def export_pack(self, target_path: str) -> str:
        pack = {
            "semantic_memory": self.semantic_memory,
            "episodic_memory": self.episodic_memory,
            "workflow_memory": self.workflow_memory,
            "policy_memory": self.policy_memory,
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(pack, f, indent=2, ensure_ascii=False)
        return target_path

    def import_pack(self, source_path: str) -> None:
        with open(source_path, "r", encoding="utf-8") as f:
            pack = json.load(f)

        for key, current in (
            ("semantic_memory", self.semantic_memory),
            ("episodic_memory", self.episodic_memory),
            ("workflow_memory", self.workflow_memory),
            ("policy_memory", self.policy_memory),
        ):
            incoming = pack.get(key, {})
            if isinstance(incoming, dict):
                current.update(incoming)

        self.save_all()
