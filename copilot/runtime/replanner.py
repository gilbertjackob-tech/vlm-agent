from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from copilot.runtime.desktop_state import DesktopState
from copilot.runtime.task_state import TaskState
from copilot.schemas import ExecutionPlan, TaskSpec


@dataclass
class ReplanResult:
    reason: str
    old_plan_step_ids: list[str]
    new_plan_step_ids: list[str]
    pending_preserved: list[str]
    fragment: list[Any]
    full_restart_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "old_plan_step_ids": list(self.old_plan_step_ids),
            "new_plan_step_ids": list(self.new_plan_step_ids),
            "pending_preserved": list(self.pending_preserved),
            "full_restart_required": self.full_restart_required,
        }


class Replanner:
    def __init__(self, compiler) -> None:
        self.compiler = compiler

    def replan(
        self,
        *,
        user_goal: str,
        desktop_state: DesktopState,
        task_state: TaskState,
        failure_reason: str,
        original_plan: ExecutionPlan,
        observation=None,
        environment: dict[str, Any] | None = None,
    ) -> ReplanResult | None:
        if not desktop_state:
            return None

        replanned_task = TaskSpec(
            prompt=original_plan.task.prompt,
            goal=user_goal,
            constraints=list(task_state.active_constraints),
            trust_mode=original_plan.task.trust_mode,
        )
        new_plan = self.compiler.compile(
            task=replanned_task,
            observation=observation,
            environment=environment or {},
        )
        completed = set(task_state.completed_steps)
        filtered_steps = [step for step in new_plan.steps if step.step_id not in completed]
        if not filtered_steps:
            return None
        return ReplanResult(
            reason=failure_reason,
            old_plan_step_ids=[step.step_id for step in original_plan.steps],
            new_plan_step_ids=[step.step_id for step in filtered_steps],
            pending_preserved=list(task_state.pending_steps),
            fragment=filtered_steps,
            full_restart_required=False,
        )
