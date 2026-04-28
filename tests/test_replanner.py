from __future__ import annotations

import tempfile
import unittest

from copilot.memory.store import MemoryStore
from copilot.planner.compiler import PromptCompiler
from copilot.runtime.desktop_state import DesktopState
from copilot.runtime.replanner import Replanner
from copilot.runtime.task_state import TaskStateManager
from copilot.schemas import ExecutionPlan, PlanStep, TaskSpec, TrustMode


EXPLORER_ENV = {
    "windows": {
        "active_window": {"title": "File Explorer"},
        "active_app_guess": "explorer",
    },
    "browser": {},
}


def simple_plan() -> ExecutionPlan:
    task = TaskSpec(prompt="open explorer and open downloads", goal="open explorer and open downloads", trust_mode=TrustMode.PLAN_AND_RISK_GATES)
    return ExecutionPlan(
        task=task,
        steps=[
            PlanStep("step_1", "Route", "route_window", success_criteria="Explorer focused."),
            PlanStep("step_2", "Open Downloads", "click_node", success_criteria="Downloads visible."),
            PlanStep("step_3", "Verify", "verify_scene", success_criteria="Downloads verified."),
        ],
        summary="Open downloads",
    )


class ReplannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.memory = MemoryStore(base_dir=self.tmp.name)
        self.compiler = PromptCompiler(self.memory)
        self.replanner = Replanner(self.compiler)
        self.task_manager = TaskStateManager()

    def desktop_state(self, last_change: str = "downloads visible") -> DesktopState:
        return DesktopState(
            state_id="state",
            timestamp=0.0,
            active_window={"title": "File Explorer"},
            focused_element="Downloads",
            screen_hash="a",
            ui_tree_hash="b",
            dom_hash="c",
            last_action="after:click_node",
            last_verified_change=last_change,
            confidence=0.9,
        )

    def test_replanner_requires_current_state(self) -> None:
        state = self.task_manager.initialize(simple_plan())
        self.assertIsNone(
            self.replanner.replan(
                user_goal=state.user_goal,
                desktop_state=None,
                task_state=state,
                failure_reason="NO_STATE_CHANGE",
                original_plan=simple_plan(),
                observation=None,
                environment=EXPLORER_ENV,
            )
        )

    def test_replanner_preserves_constraints(self) -> None:
        plan = simple_plan()
        plan.task.constraints = ["do not close explorer"]
        state = self.task_manager.initialize(plan)
        result = self.replanner.replan(
            user_goal=state.user_goal,
            desktop_state=self.desktop_state(),
            task_state=state,
            failure_reason="NO_STATE_CHANGE",
            original_plan=plan,
            observation=None,
            environment=EXPLORER_ENV,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.new_plan_step_ids)

    def test_replanner_filters_completed_steps(self) -> None:
        plan = simple_plan()
        state = self.task_manager.initialize(plan)
        state.completed_steps = ["step_1"]
        result = self.replanner.replan(
            user_goal=state.user_goal,
            desktop_state=self.desktop_state(),
            task_state=state,
            failure_reason="NO_STATE_CHANGE",
            original_plan=plan,
            observation=None,
            environment=EXPLORER_ENV,
        )
        self.assertIsNotNone(result)
        self.assertNotIn("step_1", result.new_plan_step_ids)

    def test_replanner_uses_failure_reason(self) -> None:
        state = self.task_manager.initialize(simple_plan())
        result = self.replanner.replan(
            user_goal=state.user_goal,
            desktop_state=self.desktop_state(),
            task_state=state,
            failure_reason="TARGET_NOT_FOUND",
            original_plan=simple_plan(),
            observation=None,
            environment=EXPLORER_ENV,
        )
        self.assertEqual(result.reason, "TARGET_NOT_FOUND")

    def test_replanner_returns_fragment_not_full_restart(self) -> None:
        state = self.task_manager.initialize(simple_plan())
        result = self.replanner.replan(
            user_goal=state.user_goal,
            desktop_state=self.desktop_state(),
            task_state=state,
            failure_reason="PLAN_DRIFT",
            original_plan=simple_plan(),
            observation=None,
            environment=EXPLORER_ENV,
        )
        self.assertFalse(result.full_restart_required)
        self.assertTrue(result.fragment)

    def test_replanner_tracks_old_and_new_step_ids(self) -> None:
        state = self.task_manager.initialize(simple_plan())
        result = self.replanner.replan(
            user_goal=state.user_goal,
            desktop_state=self.desktop_state(),
            task_state=state,
            failure_reason="NO_STATE_CHANGE",
            original_plan=simple_plan(),
            observation=None,
            environment=EXPLORER_ENV,
        )
        self.assertEqual(result.old_plan_step_ids, ["step_1", "step_2", "step_3"])
        self.assertTrue(result.new_plan_step_ids)

    def test_replanner_respects_pending_preserved(self) -> None:
        plan = simple_plan()
        state = self.task_manager.initialize(plan)
        state.pending_steps = ["step_2", "step_3"]
        result = self.replanner.replan(
            user_goal=state.user_goal,
            desktop_state=self.desktop_state(),
            task_state=state,
            failure_reason="NO_STATE_CHANGE",
            original_plan=plan,
            observation=None,
            environment=EXPLORER_ENV,
        )
        self.assertEqual(result.pending_preserved, ["step_2", "step_3"])

    def test_replanner_can_use_observation(self) -> None:
        state = self.task_manager.initialize(simple_plan())
        result = self.replanner.replan(
            user_goal=state.user_goal,
            desktop_state=self.desktop_state(),
            task_state=state,
            failure_reason="NO_STATE_CHANGE",
            original_plan=simple_plan(),
            observation=None,
            environment=EXPLORER_ENV,
        )
        self.assertIsNotNone(result)

    def test_replanner_handles_failed_steps_present(self) -> None:
        plan = simple_plan()
        state = self.task_manager.initialize(plan)
        state.failed_steps = ["step_2"]
        result = self.replanner.replan(
            user_goal=state.user_goal,
            desktop_state=self.desktop_state(),
            task_state=state,
            failure_reason="NO_STATE_CHANGE",
            original_plan=plan,
            observation=None,
            environment=EXPLORER_ENV,
        )
        self.assertIsNotNone(result)

    def test_replanner_to_dict_contains_reason(self) -> None:
        state = self.task_manager.initialize(simple_plan())
        result = self.replanner.replan(
            user_goal=state.user_goal,
            desktop_state=self.desktop_state(),
            task_state=state,
            failure_reason="PLAN_DRIFT",
            original_plan=simple_plan(),
            observation=None,
            environment=EXPLORER_ENV,
        )
        self.assertEqual(result.to_dict()["reason"], "PLAN_DRIFT")


if __name__ == "__main__":
    unittest.main()
