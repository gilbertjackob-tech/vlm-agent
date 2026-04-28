from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import time

from copilot.runtime.desktop_state import DesktopState
from copilot.schemas import ExecutionPlan


@dataclass
class TaskState:
    user_goal: str
    current_subgoal: str = ""
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    active_constraints: list[str] = field(default_factory=list)
    expected_next_state: str = ""
    timeline_entry_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskStateManager:
    def initialize(self, plan: ExecutionPlan) -> TaskState:
        return TaskState(
            user_goal=plan.task.goal,
            current_subgoal=plan.summary,
            pending_steps=[step.step_id for step in plan.steps],
            active_constraints=list(plan.task.constraints),
            expected_next_state=plan.steps[0].success_criteria if plan.steps else "",
        )

    def update_task_state_after_step(
        self,
        task_state: TaskState,
        step,
        *,
        ok: bool,
        notes: str = "",
    ) -> TaskState:
        current = TaskState(**task_state.to_dict())
        step_id = step.step_id
        current.timeline_entry_at = time.time()
        if step_id in current.pending_steps:
            current.pending_steps = [item for item in current.pending_steps if item != step_id]
        if ok:
            if step_id not in current.completed_steps:
                current.completed_steps.append(step_id)
        else:
            if step_id not in current.failed_steps:
                current.failed_steps.append(step_id)
        current.current_subgoal = notes or step.title
        current.expected_next_state = current.pending_steps[0] if current.pending_steps else ""
        return current

    def detect_plan_drift(
        self,
        task_state: TaskState,
        desktop_state: DesktopState | None,
    ) -> bool:
        if not desktop_state:
            return False
        if not task_state.failed_steps:
            return False
        expected = str(task_state.expected_next_state or "").strip().lower()
        if not expected:
            return False
        last_change = str(desktop_state.last_verified_change or "").strip().lower()
        focused = str(desktop_state.focused_element or "").strip().lower()
        active_title = str((desktop_state.active_window or {}).get("title", "")).strip().lower()
        blob = " ".join(part for part in [last_change, focused, active_title] if part)
        return bool(blob) and expected not in blob

    def prevent_repeating_completed_step(self, task_state: TaskState, step_id: str, recovery_required: bool = False) -> bool:
        if recovery_required:
            return False
        return step_id in task_state.completed_steps

    def replan_from_current_state(self, plan: ExecutionPlan, task_state: TaskState) -> dict[str, Any]:
        pending = [step for step in plan.steps if step.step_id not in set(task_state.completed_steps)]
        return {
            "needs_replan": bool(task_state.failed_steps),
            "pending_step_ids": [step.step_id for step in pending],
            "completed_step_ids": list(task_state.completed_steps),
            "failed_step_ids": list(task_state.failed_steps),
        }
