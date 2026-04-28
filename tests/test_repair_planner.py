from __future__ import annotations

import unittest

from copilot.runtime.desktop_state import DesktopState
from copilot.runtime.repair_planner import RepairPlanner
from copilot.runtime.task_state import TaskState
from copilot.schemas import ActionTarget, PlanStep


def desktop_state() -> DesktopState:
    return DesktopState(
        state_id="state",
        timestamp=0.0,
        active_window={"title": "File Explorer"},
        focused_element="Downloads",
        screen_hash="a",
        ui_tree_hash="b",
        dom_hash="c",
        last_action="after:click_node",
        last_verified_change="downloads visible",
        confidence=0.9,
    )


def task_state() -> TaskState:
    return TaskState(
        user_goal="open downloads",
        current_subgoal="open downloads",
        completed_steps=["step_1"],
        pending_steps=["step_2"],
        failed_steps=[],
        active_constraints=["do not close explorer"],
        expected_next_state="Downloads visible.",
    )


def click_step() -> PlanStep:
    return PlanStep(
        step_id="step_2",
        title="Open Downloads",
        action_type="click_node",
        target=ActionTarget(kind="ui_node", value="Downloads", filters={"label_contains": "Downloads", "region": "left_menu"}),
        parameters={"filters": {"label_contains": "Downloads", "region": "left_menu"}, "selector_candidates": []},
        success_criteria="Downloads visible.",
    )


class RepairPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = RepairPlanner()

    def test_target_not_found_repairs_only_target_resolution(self) -> None:
        repair = self.planner.repair(
            failed_step=click_step(),
            failure_reason="TARGET_NOT_FOUND",
            desktop_state=desktop_state(),
            task_state=task_state(),
        )
        self.assertEqual([step.action_type for step in repair.fragment], ["parse_ui", "click_node"])

    def test_target_ambiguous_repairs_only_disambiguation(self) -> None:
        repair = self.planner.repair(
            failed_step=click_step(),
            failure_reason="TARGET_AMBIGUOUS",
            desktop_state=desktop_state(),
            task_state=task_state(),
            available_targets=[{"name": "Downloads", "role": "menu_item", "region": "left_menu"}],
        )
        self.assertEqual(len(repair.fragment), 1)
        self.assertEqual(repair.fragment[0].action_type, "click_node")
        self.assertEqual(repair.fragment[0].parameters["filters"]["region"], "left_menu")

    def test_focus_not_confirmed_repairs_focus_only(self) -> None:
        step = click_step()
        step.parameters["app_id"] = "explorer"
        repair = self.planner.repair(
            failed_step=step,
            failure_reason="FOCUS_NOT_CONFIRMED",
            desktop_state=desktop_state(),
            task_state=task_state(),
        )
        self.assertEqual([step.action_type for step in repair.fragment], ["confirm_focus", "click_node"])

    def test_no_state_change_prefers_dom_retry_only(self) -> None:
        step = click_step()
        step.parameters["selector_candidates"] = ["#downloads"]
        repair = self.planner.repair(
            failed_step=step,
            failure_reason="NO_STATE_CHANGE",
            desktop_state=desktop_state(),
            task_state=task_state(),
        )
        self.assertEqual(len(repair.fragment), 1)
        self.assertEqual(repair.fragment[0].parameters["selector_candidates"], ["#downloads"])

    def test_no_state_change_for_route_retries_route_and_focus(self) -> None:
        step = PlanStep(
            step_id="step_route",
            title="Route to Chrome",
            action_type="route_window",
            target=ActionTarget(kind="application", value="chrome"),
            parameters={"app_id": "chrome", "window_title": "Google Chrome"},
            success_criteria="Chrome focused.",
        )
        repair = self.planner.repair(
            failed_step=step,
            failure_reason="NO_STATE_CHANGE",
            desktop_state=desktop_state(),
            task_state=task_state(),
        )

        self.assertEqual([step.action_type for step in repair.fragment], ["route_window", "confirm_focus"])
        self.assertEqual(repair.fragment[-1].parameters["expected"], "chrome")

    def test_route_repair_does_not_loop_forever(self) -> None:
        step = PlanStep(
            step_id="step_route_repair_route",
            title="Retry Route to Chrome",
            action_type="route_window",
            target=ActionTarget(kind="application", value="chrome"),
            parameters={"app_id": "chrome", "window_title": "Google Chrome"},
        )
        repair = self.planner.repair(
            failed_step=step,
            failure_reason="NO_STATE_CHANGE",
            desktop_state=desktop_state(),
            task_state=task_state(),
        )

        self.assertTrue(repair.stop_required)
        self.assertEqual(repair.fragment, [])

    def test_no_state_change_for_type_text_checks_field_first(self) -> None:
        step = PlanStep(
            step_id="step_type",
            title="Type Query",
            action_type="type_text",
            target=ActionTarget(kind="text", value="hello"),
            parameters={"text": "hello", "selector": "#omnibox"},
            success_criteria="Query should be present.",
        )
        repair = self.planner.repair(
            failed_step=step,
            failure_reason="NO_STATE_CHANGE",
            desktop_state=desktop_state(),
            task_state=task_state(),
        )
        self.assertEqual([step.action_type for step in repair.fragment], ["verify_scene", "type_text"])

    def test_timeout_repairs_with_bounded_wait_and_check(self) -> None:
        step = PlanStep(
            step_id="step_wait",
            title="Wait",
            action_type="wait_for",
            parameters={"seconds": 5.0, "expected_focus": "explorer", "timeout": 10.0},
            success_criteria="Explorer focused.",
        )
        repair = self.planner.repair(
            failed_step=step,
            failure_reason="TIMEOUT",
            desktop_state=desktop_state(),
            task_state=task_state(),
        )
        self.assertEqual([step.action_type for step in repair.fragment], ["wait_for", "verify_scene"])
        self.assertLessEqual(repair.fragment[0].parameters["timeout"], 1.5)

    def test_policy_blocked_requires_stop(self) -> None:
        repair = self.planner.repair(
            failed_step=click_step(),
            failure_reason="POLICY_BLOCKED",
            desktop_state=desktop_state(),
            task_state=task_state(),
        )
        self.assertTrue(repair.stop_required)
        self.assertEqual(repair.fragment, [])

    def test_no_repair_without_current_state(self) -> None:
        repair = self.planner.repair(
            failed_step=click_step(),
            failure_reason="TARGET_NOT_FOUND",
            desktop_state=None,
            task_state=task_state(),
        )
        self.assertIsNone(repair)

    def test_repair_clones_failed_step_without_resetting_target(self) -> None:
        repair = self.planner.repair(
            failed_step=click_step(),
            failure_reason="TARGET_NOT_FOUND",
            desktop_state=desktop_state(),
            task_state=task_state(),
        )
        self.assertEqual(repair.fragment[-1].target.value, "Downloads")

    def test_repair_preserves_constraints_via_task_state_presence(self) -> None:
        repair = self.planner.repair(
            failed_step=click_step(),
            failure_reason="TARGET_NOT_FOUND",
            desktop_state=desktop_state(),
            task_state=task_state(),
        )
        self.assertEqual(repair.reason, "TARGET_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
