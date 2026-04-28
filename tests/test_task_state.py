from __future__ import annotations

import unittest

from copilot.runtime.desktop_state import DesktopState
from copilot.runtime.task_state import TaskStateManager
from copilot.schemas import ExecutionPlan, PlanStep, TaskSpec


def plan() -> ExecutionPlan:
    task = TaskSpec(prompt="open explorer", goal="open explorer")
    return ExecutionPlan(
        task=task,
        steps=[
            PlanStep(step_id="s1", title="Route", action_type="route_window", success_criteria="Explorer focused."),
            PlanStep(step_id="s2", title="Parse", action_type="parse_ui", success_criteria="Explorer parsed."),
            PlanStep(step_id="s3", title="Open Downloads", action_type="click_node", success_criteria="Downloads visible."),
        ],
        summary="Open explorer workflow",
    )


class TaskStateTests(unittest.TestCase):
    def test_initialize_builds_pending_steps(self) -> None:
        state = TaskStateManager().initialize(plan())
        self.assertEqual(state.pending_steps, ["s1", "s2", "s3"])
        self.assertEqual(state.expected_next_state, "Explorer focused.")

    def test_update_after_success_moves_step_to_completed(self) -> None:
        manager = TaskStateManager()
        state = manager.initialize(plan())
        updated = manager.update_task_state_after_step(state, plan().steps[0], ok=True, notes="Explorer focused.")
        self.assertEqual(updated.completed_steps, ["s1"])
        self.assertEqual(updated.pending_steps, ["s2", "s3"])

    def test_update_after_failure_moves_step_to_failed(self) -> None:
        manager = TaskStateManager()
        state = manager.initialize(plan())
        updated = manager.update_task_state_after_step(state, plan().steps[0], ok=False, notes="route failed")
        self.assertEqual(updated.failed_steps, ["s1"])
        self.assertEqual(updated.pending_steps, ["s2", "s3"])

    def test_prevent_repeating_completed_step_blocks_without_recovery(self) -> None:
        manager = TaskStateManager()
        state = manager.initialize(plan())
        state.completed_steps.append("s1")
        self.assertTrue(manager.prevent_repeating_completed_step(state, "s1"))
        self.assertFalse(manager.prevent_repeating_completed_step(state, "s1", recovery_required=True))

    def test_detect_plan_drift_uses_expected_state_against_desktop_state(self) -> None:
        manager = TaskStateManager()
        state = manager.initialize(plan())
        desktop = DesktopState(
            state_id="a",
            timestamp=0.0,
            active_window={"title": "Other"},
            focused_element="Search",
            screen_hash="1",
            ui_tree_hash="2",
            dom_hash="3",
            last_action="after:click_node",
            last_verified_change="Unrelated page",
            confidence=0.8,
        )
        state.expected_next_state = "Downloads visible."
        state.failed_steps.append("s2")
        self.assertTrue(manager.detect_plan_drift(state, desktop))

    def test_detect_plan_drift_ignores_successful_pending_future_state(self) -> None:
        manager = TaskStateManager()
        state = manager.initialize(plan())
        desktop = DesktopState(
            state_id="a",
            timestamp=0.0,
            active_window={"title": "File Explorer"},
            focused_element="",
            screen_hash="1",
            ui_tree_hash="2",
            dom_hash="3",
            last_action="after:route_window",
            last_verified_change="Window routed.",
            confidence=0.8,
        )
        state.expected_next_state = "Explorer parsed."
        self.assertFalse(manager.detect_plan_drift(state, desktop))

    def test_replan_from_current_state_reports_pending_and_failed(self) -> None:
        manager = TaskStateManager()
        state = manager.initialize(plan())
        state.completed_steps.append("s1")
        state.failed_steps.append("s2")
        replanned = manager.replan_from_current_state(plan(), state)
        self.assertTrue(replanned["needs_replan"])
        self.assertEqual(replanned["pending_step_ids"], ["s2", "s3"])

    def test_completed_step_is_removed_from_pending_only_once(self) -> None:
        manager = TaskStateManager()
        state = manager.initialize(plan())
        updated = manager.update_task_state_after_step(state, plan().steps[0], ok=True, notes="done")
        updated = manager.update_task_state_after_step(updated, plan().steps[0], ok=True, notes="done")
        self.assertEqual(updated.pending_steps, ["s2", "s3"])
        self.assertEqual(updated.completed_steps, ["s1"])


if __name__ == "__main__":
    unittest.main()
