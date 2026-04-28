from __future__ import annotations

import unittest

from copilot.ui.overlay import OverlayState, ghost_preview_from_plan, mind_line_from_event, overlay_state_from_event, priority_for_event


class OverlayStateTests(unittest.TestCase):
    def test_high_priority_approval_expands_and_highlights(self) -> None:
        state = overlay_state_from_event(
            OverlayState(),
            {
                "phase": "approval",
                "message": "Approve step",
                "metadata": {"approval_id": "approval_1", "step_id": "s1", "target": "Downloads"},
            },
        )

        self.assertEqual(state.priority, "HIGH")
        self.assertTrue(state.expanded)
        self.assertTrue(state.highlighted)
        self.assertEqual(state.approval_id, "approval_1")

    def test_low_priority_observe_does_not_replace_narration(self) -> None:
        previous = OverlayState(narration="I am clicking it.")
        state = overlay_state_from_event(previous, {"phase": "observe", "message": "Observed UI", "metadata": {}})

        self.assertEqual(state.priority, "LOW")
        self.assertEqual(state.narration, "I am clicking it.")

    def test_priority_mapping(self) -> None:
        self.assertEqual(priority_for_event({"phase": "recovery"}), "HIGH")
        self.assertEqual(priority_for_event({"phase": "step"}), "MEDIUM")
        self.assertEqual(priority_for_event({"phase": "parse"}), "LOW")

    def test_agent_bar_fields_are_derived_from_step_event(self) -> None:
        state = overlay_state_from_event(
            OverlayState(),
            {
                "phase": "step",
                "message": "Executing: Search",
                "metadata": {
                    "goal": "Searching order page",
                    "step_id": "step_3",
                    "step_index": 3,
                    "step_total": 7,
                    "target": "Search box",
                    "trust_mode": "plan_and_risk_gates",
                    "score_gap": 0.5,
                    "focus_confidence": 0.9,
                },
            },
        )

        self.assertEqual(state.goal, "Searching order page")
        self.assertEqual(state.step_index, 3)
        self.assertEqual(state.step_total, 7)
        self.assertEqual(state.mode, "Safe Mode")
        self.assertEqual(state.target_summary, "Search box")

    def test_live_mind_line_summarizes_dom_and_target_events(self) -> None:
        self.assertEqual(mind_line_from_event({"phase": "contract", "message": "DOM click contract verified.", "metadata": {}}), "DOM is connected.")
        self.assertEqual(
            mind_line_from_event({"phase": "target_found", "message": "", "metadata": {"target": "search box"}}),
            "I found search box.",
        )

    def test_ghost_preview_extracts_action_sequence(self) -> None:
        preview = ghost_preview_from_plan(
            {
                "steps": [
                    {"step_id": "s1", "title": "Click", "action_type": "click_node", "target": {"value": "Search"}, "risk_level": "low", "confidence": 0.91},
                    {"step_id": "s2", "title": "Type", "action_type": "type_text", "target": {"value": "query"}, "risk_level": "medium", "requires_approval": True},
                ]
            }
        )

        self.assertEqual([step["action_type"] for step in preview], ["click_node", "type_text"])
        self.assertTrue(preview[1]["requires_approval"])


if __name__ == "__main__":
    unittest.main()
