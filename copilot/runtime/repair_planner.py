from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from copilot.schemas import ActionIntent, ActionTarget, PlanStep, RiskLevel


@dataclass
class RepairPlan:
    reason: str
    planner_type: str
    fragment: list[PlanStep]
    stop_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "planner_type": self.planner_type,
            "step_ids": [step.step_id for step in self.fragment],
            "stop_required": self.stop_required,
        }


class RepairPlanner:
    def repair(
        self,
        *,
        failed_step: PlanStep,
        failure_reason: str,
        desktop_state,
        task_state,
        available_targets: list[dict[str, Any]] | None = None,
    ) -> RepairPlan | None:
        if not desktop_state:
            return None
        if failure_reason == "POLICY_BLOCKED":
            return RepairPlan(reason=failure_reason, planner_type="repair", fragment=[], stop_required=True)

        app_hint = ""
        if failed_step.target and failed_step.target.kind == "application":
            app_hint = failed_step.target.value
        if not app_hint:
            app_hint = str(failed_step.parameters.get("app_id", "") or desktop_state.active_window.get("title", ""))

        if failure_reason == "TARGET_NOT_FOUND":
            return RepairPlan(
                reason=failure_reason,
                planner_type="repair",
                fragment=[
                    self._step(
                        failed_step,
                        suffix="repair_parse",
                        title="Refresh UI state for target search",
                        action_type="parse_ui",
                        parameters={"output_filename": f"{failed_step.step_id}_repair_parse.png"},
                        success="Current UI should be reparsed before target resolution.",
                    ),
                    self._clone_step(
                        failed_step,
                        suffix="repair_resolve",
                        title=f"Resolve and retry {failed_step.title}",
                        parameter_patch={"filters": {**dict(failed_step.parameters.get("filters", {})), "min_score": 0.4}},
                    ),
                ],
            )

        if failure_reason == "TARGET_AMBIGUOUS":
            identity = dict((available_targets or [{}])[0]) if available_targets else {}
            refined_filters = {
                **dict(failed_step.parameters.get("filters", {})),
                "label_contains": identity.get("name", "") or (failed_step.target.value if failed_step.target else ""),
                "semantic_role": identity.get("role", "") or dict(failed_step.parameters.get("filters", {})).get("semantic_role", ""),
                "region": identity.get("region", "") or dict(failed_step.parameters.get("filters", {})).get("region", ""),
                "min_score": 0.75,
            }
            return RepairPlan(
                reason=failure_reason,
                planner_type="repair",
                fragment=[
                    self._clone_step(
                        failed_step,
                        suffix="repair_disambiguate",
                        title=f"Disambiguate target for {failed_step.title}",
                        parameter_patch={"filters": refined_filters},
                    )
                ],
            )

        if failure_reason == "FOCUS_NOT_CONFIRMED":
            steps = []
            if app_hint:
                steps.append(
                    self._step(
                        failed_step,
                        suffix="repair_focus",
                        title=f"Repair focus for {app_hint}",
                        action_type="confirm_focus",
                        target=ActionTarget(kind="application", value=app_hint),
                        parameters={"expected": app_hint},
                        success=f"Focus should be confirmed for {app_hint}.",
                    )
                )
            steps.append(self._clone_step(failed_step, suffix="repair_retry", title=f"Retry {failed_step.title} after focus repair"))
            return RepairPlan(reason=failure_reason, planner_type="repair", fragment=steps)

        if failure_reason == "NO_STATE_CHANGE":
            if failed_step.action_type == "route_window":
                if "repair_route" in failed_step.step_id:
                    return RepairPlan(reason=failure_reason, planner_type="repair", fragment=[], stop_required=True)
                return RepairPlan(
                    reason=failure_reason,
                    planner_type="repair",
                    fragment=[
                        self._clone_step(
                            failed_step,
                            suffix="repair_route",
                            title=f"Retry routing to {app_hint or failed_step.title}",
                        ),
                        self._step(
                            failed_step,
                            suffix="repair_focus",
                            title=f"Confirm focus after routing to {app_hint or failed_step.title}",
                            action_type="confirm_focus",
                            target=ActionTarget(kind="application", value=app_hint),
                            parameters={"expected": app_hint},
                            success=f"Focus should be confirmed for {app_hint}." if app_hint else "Focus should be confirmed after routing.",
                        ),
                    ],
                )
            if failed_step.action_type == "click_node":
                parameters = dict(failed_step.parameters)
                selector_candidates = [str(item) for item in parameters.get("selector_candidates", []) if str(item).strip()]
                if selector_candidates:
                    return RepairPlan(
                        reason=failure_reason,
                        planner_type="repair",
                        fragment=[
                            self._clone_step(
                                failed_step,
                                suffix="repair_dom",
                                title=f"Retry {failed_step.title} with DOM evidence",
                                parameter_patch={"selector_candidates": selector_candidates},
                            )
                        ],
                    )
            if failed_step.action_type == "type_text":
                return RepairPlan(
                    reason=failure_reason,
                    planner_type="repair",
                    fragment=[
                        self._step(
                            failed_step,
                            suffix="repair_verify",
                            title="Check field state before retyping",
                            action_type="verify_scene",
                            parameters={"expected_labels": [failed_step.target.value if failed_step.target else ""]},
                            success="Field content should be verified before retrying text entry.",
                        ),
                        self._clone_step(
                            failed_step,
                            suffix="repair_retry",
                            title=f"Retry {failed_step.title} with field check",
                            parameter_patch={"clear_first": True},
                        ),
                    ],
                )
            return RepairPlan(
                reason=failure_reason,
                planner_type="repair",
                fragment=[
                    self._step(
                        failed_step,
                        suffix="repair_verify",
                        title="Verify scene before alternate action",
                        action_type="verify_scene",
                        parameters={"expected_labels": [failed_step.target.value if failed_step.target else ""]},
                        success=failed_step.success_criteria or "Scene should reveal whether the previous action already succeeded.",
                    )
                ],
            )

        if failure_reason == "TIMEOUT":
            return RepairPlan(
                reason=failure_reason,
                planner_type="repair",
                fragment=[
                    self._step(
                        failed_step,
                        suffix="repair_wait",
                        title=f"Retry bounded wait for {failed_step.title}",
                        action_type="wait_for",
                        parameters={
                            "seconds": min(1.0, max(float(failed_step.parameters.get("seconds", 0.2)), 0.1)),
                            "expected_focus": str(failed_step.parameters.get("expected_focus", "")),
                            "timeout": min(1.5, max(float(failed_step.parameters.get("timeout", 0.5)), 0.2)),
                        },
                        success="A bounded retry wait should either confirm focus or fail quickly.",
                    ),
                    self._step(
                        failed_step,
                        suffix="repair_check",
                        title=f"Check state after timeout for {failed_step.title}",
                        action_type="verify_scene",
                        parameters={"expected_labels": [failed_step.target.value if failed_step.target else ""]},
                        success=failed_step.success_criteria or "The scene should confirm whether the wait condition resolved.",
                    ),
                ],
            )
        return None

    def _step(
        self,
        failed_step: PlanStep,
        *,
        suffix: str,
        title: str,
        action_type: str,
        parameters: dict[str, Any],
        success: str,
        target: ActionTarget | None = None,
    ) -> PlanStep:
        return PlanStep(
            step_id=f"{failed_step.step_id}_{suffix}",
            title=title,
            action_type=action_type,
            target=target,
            intent=ActionIntent(verb=action_type, description=title),
            parameters=parameters,
            confidence=0.8,
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            success_criteria=success,
            fallback_hint=failed_step.fallback_hint,
            control_modes=list(failed_step.control_modes),
        )

    def _clone_step(
        self,
        failed_step: PlanStep,
        *,
        suffix: str,
        title: str,
        parameter_patch: dict[str, Any] | None = None,
    ) -> PlanStep:
        params = dict(failed_step.parameters)
        if parameter_patch:
            for key, value in parameter_patch.items():
                params[key] = value
        return PlanStep(
            step_id=f"{failed_step.step_id}_{suffix}",
            title=title,
            action_type=failed_step.action_type,
            target=failed_step.target,
            intent=failed_step.intent,
            parameters=params,
            confidence=max(0.75, failed_step.confidence),
            risk_level=failed_step.risk_level,
            requires_approval=failed_step.requires_approval,
            success_criteria=failed_step.success_criteria,
            fallback_hint=failed_step.fallback_hint,
            control_modes=list(failed_step.control_modes),
        )
