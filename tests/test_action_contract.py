from __future__ import annotations

import unittest

from copilot.runtime.action_contract import (
    build_click_contract,
    build_click_point_contract,
    build_dom_click_contract,
    build_press_key_contract,
    build_type_text_contract,
    build_wait_contract,
    verify_dom_click_contract,
    verify_press_key_contract,
    verify_type_text_contract,
    verify_wait_contract,
)
from copilot.schemas import ObservationGraph


def graph(label: str, app_id: str = "explorer") -> ObservationGraph:
    return ObservationGraph.from_raw(
        [
            {
                "id": "node",
                "label": label,
                "type": "button",
                "semantic_role": "menu_item",
                "entity_type": "navigation_item",
                "app_id": app_id,
                "region": "left_menu",
                "visual_id": "visual_node",
                "center": {"x": 10, "y": 20},
                "box": {"x": 0, "y": 0, "width": 40, "height": 30},
                "affordances": ["click"],
            }
        ],
        metadata={"app_id": app_id, "scene_summary": label},
    )


class ActionContractTests(unittest.TestCase):
    def test_click_contract_accepts_evidenced_node(self) -> None:
        g = graph("Downloads")
        contract = build_click_contract("s1", "open downloads", g.flatten()[0], g, 1)
        self.assertEqual(contract.status, "planned")
        self.assertTrue(contract.before_checks["target_clickable"])
        self.assertTrue(any(item.startswith("label:") for item in contract.evidence))
        self.assertGreaterEqual(contract.evidence_score, 0.55)
        self.assertIn(contract.evidence_grade, {"medium", "strong"})

    def test_click_contract_blocks_unsafe_click_count(self) -> None:
        g = graph("Downloads")
        contract = build_click_contract("s1", "open downloads", g.flatten()[0], g, 5)
        self.assertEqual(contract.status, "blocked")
        self.assertEqual(contract.failure_reason, "TARGET_NOT_FOUND")
        self.assertTrue(contract.recovery_strategy)

    def test_click_point_requires_screenshot_region_and_reason(self) -> None:
        contract = build_click_point_contract("s1", "raw click", 10, 20, ["screenshot:a.png", "region:toolbar", "reason:test safe button"])
        self.assertEqual(contract.status, "planned")

    def test_click_point_blocks_blind_coordinates(self) -> None:
        contract = build_click_point_contract("s1", "raw click", 10, 20, [])
        self.assertEqual(contract.status, "blocked")
        self.assertFalse(contract.before_checks["screenshot_evidence"])
        self.assertEqual(contract.failure_reason, "UNSAFE_COORDINATE")

    def test_dom_click_requires_dom_availability(self) -> None:
        contract = build_dom_click_contract("s1", "click search", ["#q"], dom_available=True)
        self.assertEqual(contract.status, "planned")
        self.assertIn("dom:available", contract.evidence)

    def test_dom_click_blocks_without_selectors(self) -> None:
        contract = build_dom_click_contract("s1", "click search", [], dom_available=True)
        self.assertEqual(contract.status, "blocked")

    def test_dom_click_verifies_with_observed_after_graph(self) -> None:
        contract = build_dom_click_contract("s1", "click search", ["#q"], dom_available=True)
        verified = verify_dom_click_contract(contract, "#q", True, dom_before={}, dom_after={}, after=graph("Search"))
        self.assertTrue(verified.verified)

    def test_type_text_requires_text_and_target(self) -> None:
        contract = build_type_text_contract("s1", "type query", "hello", selector="#q", dom_available=True)
        self.assertEqual(contract.status, "planned")
        self.assertTrue(contract.before_checks["editable_target_exists"])

    def test_type_text_blocks_empty_text(self) -> None:
        contract = build_type_text_contract("s1", "type query", "", selector="#q", dom_available=True)
        self.assertEqual(contract.status, "blocked")

    def test_type_text_verifies_typed_value_in_dom(self) -> None:
        contract = build_type_text_contract("s1", "type query", "hello", selector="#q", dom_available=True)
        verified = verify_type_text_contract(
            contract,
            "hello",
            True,
            dom_before={"items": [{"text": ""}]},
            dom_after={"items": [{"text": "hello"}]},
        )
        self.assertTrue(verified.verified)
        self.assertEqual(verified.failure_reason, "")

    def test_press_key_blocks_destructive_ambiguous_hotkey(self) -> None:
        contract = build_press_key_contract("s1", "select all", ["ctrl", "a"], hotkey=True, active_window="Editor")
        self.assertEqual(contract.status, "blocked")

    def test_press_key_allows_expected_destructive_hotkey(self) -> None:
        contract = build_press_key_contract("s1", "replace field", ["ctrl", "a"], hotkey=True, active_window="Editor", expected_change="field will be selected")
        self.assertEqual(contract.status, "planned")

    def test_press_key_blocks_permanent_delete_without_expected_change(self) -> None:
        contract = build_press_key_contract("s1", "permanent delete", ["shift", "delete"], hotkey=True, active_window="Explorer")
        self.assertEqual(contract.status, "blocked")

    def test_press_key_requires_verified_state_delta(self) -> None:
        contract = build_press_key_contract("s1", "submit search", ["enter"], hotkey=False, active_window="Chrome")
        verified = verify_press_key_contract(contract, True)
        self.assertFalse(verified.verified)
        self.assertEqual(verified.failure_reason, "NO_STATE_CHANGE")

    def test_press_key_verifies_dom_delta(self) -> None:
        contract = build_press_key_contract("s1", "submit search", ["enter"], hotkey=False, active_window="Chrome")
        verified = verify_press_key_contract(
            contract,
            True,
            dom_before={"url": "https://example.test", "items": []},
            dom_after={"url": "https://example.test/search?q=hello", "items": []},
        )
        self.assertTrue(verified.verified)

    def test_wait_contract_is_traceable(self) -> None:
        contract = build_wait_contract("s1", "wait for focus", 0.1, expected_focus="chrome", timeout=1.0)
        verified = verify_wait_contract(contract, True)
        self.assertEqual(verified.status, "verified")
        self.assertTrue(verified.verification["traceable_wait"])

    def test_wait_timeout_gets_failure_reason_and_recovery(self) -> None:
        contract = build_wait_contract("s1", "wait for focus", 0.1, expected_focus="chrome", timeout=1.0)
        failed = verify_wait_contract(contract, False)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.failure_reason, "TIMEOUT")
        self.assertIn("wait_and_reparse", failed.recovery_strategy)


if __name__ == "__main__":
    unittest.main()
