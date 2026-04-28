from __future__ import annotations

import unittest

from copilot.runtime.action_contract import build_click_contract
from copilot.runtime.recovery import RecoveryPlanner
from copilot.runtime.target_identity import (
    ambiguous_identity_matches,
    create_target_identity,
    detect_target_drift,
    match_target_identity,
    resolve_same_target_again,
)
from copilot.schemas import ActionTarget, ObservationGraph, PlanStep


class SimpleReasoner:
    def _score_node(self, node, filters):
        label = str(filters.get("label_contains", "")).lower()
        return 1.0 if not label or label in node.display_label().lower() else 0.0

    def resolve_ambiguity(self, candidates, task_prompt, scene=None):
        return candidates[0] if candidates else None


def graph(nodes, app_id: str = "demo") -> ObservationGraph:
    return ObservationGraph.from_raw(nodes, metadata={"app_id": app_id, "output_filename": "screen.png"})


def button(node_id: str, label: str, x: int = 10, y: int = 10, region: str = "main_page"):
    return {
        "id": node_id,
        "label": label,
        "type": "button",
        "semantic_role": "menu_item",
        "entity_type": "button",
        "region": region,
        "center": {"x": x, "y": y},
        "box": {"x": x, "y": y, "width": 80, "height": 30},
        "visual_id": f"visual_{node_id}",
        "affordances": ["click"],
    }


class TargetIdentityTests(unittest.TestCase):
    def test_create_target_identity_has_stable_signature(self) -> None:
        g = graph([button("open_1", "Open")])
        identity = create_target_identity(g.flatten()[0], g)
        self.assertTrue(identity.target_id.startswith("target_"))
        self.assertEqual(identity.name, "Open")
        self.assertTrue(identity.stable_signature)

    def test_same_target_matches_high_confidence(self) -> None:
        g = graph([button("open_1", "Open")])
        identity = create_target_identity(g.flatten()[0], g)
        self.assertGreaterEqual(match_target_identity(identity, g.flatten()[0], g), 0.9)

    def test_changed_name_detects_drift(self) -> None:
        g1 = graph([button("open_1", "Open")])
        g2 = graph([button("open_1", "Submit")])
        identity = create_target_identity(g1.flatten()[0], g1)
        self.assertTrue(detect_target_drift(identity, g2.flatten()[0], g2))

    def test_changed_window_detects_drift(self) -> None:
        g1 = graph([button("open_1", "Open")], app_id="app_a")
        g2 = graph([button("open_1", "Open")], app_id="app_b")
        identity = create_target_identity(g1.flatten()[0], g1)
        self.assertTrue(detect_target_drift(identity, g2.flatten()[0], g2))

    def test_resolve_same_target_again_returns_unique_match(self) -> None:
        g = graph([button("open_1", "Open"), button("save_1", "Save", x=120)])
        identity = create_target_identity(g.flatten()[0], g)
        self.assertEqual(resolve_same_target_again(identity, g).node_id, "open_1")

    def test_resolve_same_target_again_blocks_ambiguous_match(self) -> None:
        g = graph([button("open_1", "Open"), button("open_2", "Open", x=120)])
        identity = create_target_identity(g.flatten()[0], g)
        self.assertIsNone(resolve_same_target_again(identity, g))

    def test_ambiguous_identity_matches_returns_duplicates(self) -> None:
        g = graph([button("open_1", "Open"), button("open_2", "Open", x=120)])
        identity = create_target_identity(g.flatten()[0], g)
        self.assertEqual(len(ambiguous_identity_matches(identity, g)), 2)

    def test_click_contract_blocks_ambiguous_identity(self) -> None:
        g = graph([button("open_1", "Open"), button("open_2", "Open", x=120)])
        contract = build_click_contract("s1", "click open", g.flatten()[0], g, 1)
        self.assertEqual(contract.status, "blocked")
        self.assertTrue(contract.identity_ambiguous)
        self.assertEqual(contract.failure_reason, "TARGET_AMBIGUOUS")

    def test_click_contract_includes_target_identity(self) -> None:
        g = graph([button("open_1", "Open")])
        contract = build_click_contract("s1", "click open", g.flatten()[0], g, 1)
        self.assertEqual(contract.target_identity["name"], "Open")
        self.assertIn("stable_signature", contract.target_identity)

    def test_recovery_blocks_ambiguous_identity(self) -> None:
        g = graph([button("open_1", "Open"), button("open_2", "Open", x=120)])
        identity = create_target_identity(g.flatten()[0], g).to_dict()
        step = PlanStep("s1", "Click Open", "click_node", target=ActionTarget(kind="ui_node", value="Open", filters={"label_contains": "Open"}))
        recovered = RecoveryPlanner().recover_target(
            strategy=["retry_with_ocr"],
            step=step,
            graph=g,
            reasoner=SimpleReasoner(),
            failed_contract={"target_identity": identity},
        )
        self.assertIsNone(recovered)

    def test_recovery_blocks_drifted_identity(self) -> None:
        g = graph([button("open_1", "Open")])
        step = PlanStep("s1", "Click Open", "click_node", target=ActionTarget(kind="ui_node", value="Open", filters={"label_contains": "Open"}))
        recovered = RecoveryPlanner().recover_target(
            strategy=["retry_with_ocr"],
            step=step,
            graph=g,
            reasoner=SimpleReasoner(),
            failed_contract={"identity_drifted": True},
        )
        self.assertIsNone(recovered)


if __name__ == "__main__":
    unittest.main()
