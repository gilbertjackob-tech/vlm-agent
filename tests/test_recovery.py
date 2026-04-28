from __future__ import annotations

import unittest

from copilot.runtime.recovery import RecoveryPlanner
from copilot.schemas import ActionTarget, ObservationGraph, PlanStep


class RegionReasoner:
    def _score_node(self, node, filters):
        label = str(filters.get("label_contains", "")).lower()
        if label and label not in node.display_label().lower():
            return 0.0
        return 1.0

    def resolve_ambiguity(self, candidates, task_prompt, scene=None):
        prompt = task_prompt.lower()
        for node in candidates:
            if node.region and node.region.lower() in prompt:
                return node
        return candidates[0] if candidates else None


class RecoveryPlannerTests(unittest.TestCase):
    def test_target_not_found_uses_ordered_fallback_sources(self) -> None:
        plan = RecoveryPlanner().plan("TARGET_NOT_FOUND", PlanStep("s1", "Click missing", "click_node"))
        self.assertEqual(plan.strategy[:3], ["retry_with_uia", "retry_with_ocr", "retry_with_vision"])
        self.assertFalse(plan.stop_required)

    def test_unsafe_coordinate_stops_before_retrying_blindly(self) -> None:
        plan = RecoveryPlanner().plan("UNSAFE_COORDINATE", PlanStep("s1", "Click point", "click_point"))
        self.assertTrue(plan.stop_required)
        self.assertIn("retry_with_node_target", plan.strategy)

    def test_ambiguous_target_uses_region_context_recovery(self) -> None:
        graph = ObservationGraph.from_raw(
            [
                {"id": "top_settings", "label": "Settings", "type": "button", "semantic_role": "menu_item", "region": "top_menu", "center": {"x": 10, "y": 10}},
                {"id": "left_settings", "label": "Settings", "type": "button", "semantic_role": "menu_item", "region": "left_menu", "center": {"x": 10, "y": 80}},
            ],
            metadata={"app_id": "demo"},
        )
        step = PlanStep(
            "s1",
            "Open left_menu Settings",
            "click_node",
            target=ActionTarget(kind="ui_node", value="Settings", filters={"label_contains": "Settings"}),
        )
        plan = RecoveryPlanner().plan("TARGET_AMBIGUOUS", step)
        recovered = RecoveryPlanner().recover_target(
            strategy=plan.strategy,
            step=step,
            graph=graph,
            reasoner=RegionReasoner(),
            scene={"app_id": "demo"},
        )
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.node.node_id, "left_settings")
        self.assertIn(recovered.resolver_used, {"uia", "ocr", "vision"})


if __name__ == "__main__":
    unittest.main()
