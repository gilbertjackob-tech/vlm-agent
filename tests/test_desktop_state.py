from __future__ import annotations

import time
import tempfile
import unittest

from copilot.runtime.desktop_state import DesktopStateManager, DesktopStateStore, stable_hash
from copilot.schemas import ObservationGraph


class StateBridge:
    def __init__(self) -> None:
        self.title = "Editor"
        self.dom_text = "empty"
        self.uia_hash = "uia-1"

    def observe_environment(self):
        return {
            "windows": {
                "active_window": {"title": self.title},
                "active_app_guess": self.title.lower(),
                "uia_elements": [{"stable_hash": self.uia_hash}],
            }
        }

    def read_browser_dom(self):
        return {"items": [{"text": self.dom_text}]}

    def read_browser_state_hash(self):
        return stable_hash(self.read_browser_dom())

    def read_uia_state_hash(self):
        return self.uia_hash

    def read_focused_element(self):
        return "focused-demo"


def graph(label: str, output: str = "screen.png") -> ObservationGraph:
    return ObservationGraph.from_raw(
        [{"id": "n1", "label": label, "type": "button", "center": {"x": 10, "y": 10}}],
        metadata={"app_id": "demo", "output_filename": output},
    )


class DesktopStateTests(unittest.TestCase):
    def test_stable_hash_is_deterministic(self) -> None:
        self.assertEqual(stable_hash({"b": 2, "a": 1}), stable_hash({"a": 1, "b": 2}))

    def test_observe_builds_state_hashes(self) -> None:
        manager = DesktopStateManager(StateBridge())
        state = manager.observe(graph=graph("A"), last_action="before:click")
        self.assertEqual(state.active_window["title"], "Editor")
        self.assertEqual(state.active_app, "editor")
        self.assertTrue(state.screen_hash)
        self.assertTrue(state.ui_tree_hash)
        self.assertTrue(state.dom_hash)
        self.assertTrue(state.uia_hash)
        self.assertTrue(state.state_signature)
        self.assertGreater(state.confidence, 0)

    def test_probe_builds_lightweight_state_signature(self) -> None:
        manager = DesktopStateManager(StateBridge())
        state = manager.probe(last_action="parse_current_ui")
        self.assertEqual(state.active_app, "editor")
        self.assertEqual(state.uia_hash, "uia-1")
        self.assertEqual(state.focused_element, "focused-demo")
        self.assertTrue(state.state_signature)
        self.assertEqual(state.state_hash, state.state_signature)
        self.assertEqual(state.last_action, "parse_current_ui")

    def test_state_store_persists_latest_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/desktop_state.json"
            manager = DesktopStateManager(StateBridge(), state_path=path)
            state = manager.observe(graph=graph("A"), last_action="parse_current_ui")
            loaded = DesktopStateStore(path).load()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.state_hash, state.state_hash)
            self.assertEqual(loaded.active_app, "editor")
            self.assertEqual(loaded.last_action, "parse_current_ui")

    def test_state_diff_detects_dom_change(self) -> None:
        bridge = StateBridge()
        manager = DesktopStateManager(bridge)
        before = manager.observe(graph=graph("A"))
        bridge.dom_text = "changed"
        after = manager.observe(graph=graph("A"))
        diff = manager.state_diff(before, after)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["dom_changed"])

    def test_state_diff_detects_uia_change(self) -> None:
        bridge = StateBridge()
        manager = DesktopStateManager(bridge)
        before = manager.probe()
        bridge.uia_hash = "uia-2"
        after = manager.probe()
        diff = manager.state_diff(before, after)
        self.assertTrue(diff["uia_changed"])
        self.assertTrue(diff["state_signature_changed"])

    def test_state_diff_detects_ui_tree_change(self) -> None:
        manager = DesktopStateManager(StateBridge())
        before = manager.observe(graph=graph("A"))
        after = manager.observe(graph=graph("B"))
        diff = manager.state_diff(before, after)
        self.assertTrue(diff["ui_tree_changed"])

    def test_missing_state_is_stale(self) -> None:
        manager = DesktopStateManager(StateBridge())
        self.assertTrue(manager.is_state_stale(None))

    def test_old_state_is_stale(self) -> None:
        manager = DesktopStateManager(StateBridge())
        state = manager.observe(graph=graph("A"))
        state.timestamp = time.time() - 100
        self.assertTrue(manager.is_state_stale(state, max_age_seconds=1))

    def test_window_change_makes_state_stale(self) -> None:
        bridge = StateBridge()
        manager = DesktopStateManager(bridge)
        state = manager.observe(graph=graph("A"))
        bridge.title = "Other"
        self.assertTrue(manager.is_state_stale(state, max_age_seconds=100))

    def test_before_after_helpers_record_action_and_verified_change(self) -> None:
        manager = DesktopStateManager(StateBridge())
        before = manager.observe_before_action("click_node", graph=graph("A"))
        after = manager.observe_after_action("click_node", graph=graph("B"), verified_change="opened")
        self.assertEqual(before.last_action, "before:click_node")
        self.assertEqual(after.last_action, "after:click_node")
        self.assertEqual(after.last_verified_change, "opened")


if __name__ == "__main__":
    unittest.main()
