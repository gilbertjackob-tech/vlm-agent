from __future__ import annotations

import json
import os
import tempfile
import unittest

from copilot.memory.store import MemoryStore
from copilot.perception import bridge as bridge_module


class StubAgent:
    def __init__(self) -> None:
        self.managed_semantic_memory = False
        self.semantic_memory_path = ""
        self.semantic_memory = {}
        self.parse_calls = 0
        self.last_parse_health = {}

    def parse_interface(self, output_filename: str = "ui.png"):
        self.parse_calls += 1
        self.last_parse_health = {
            "parse_mode": "vision",
            "screen_hash": f"hash-{self.parse_calls}",
            "ocr_calls": 2,
            "ocr_cache_hits": 1,
            "ocr_timeouts": 0,
            "ocr_errors": 0,
            "ocr_elapsed_seconds": 0.25,
            "element_count": 1,
        }
        return [
            {
                "id": "root",
                "label": f"Parsed {output_filename}",
                "type": "container",
                "semantic_role": "main_page",
                "region": "main_page",
                "children": [],
                "box": {"x": 0, "y": 0, "width": 10, "height": 10},
                "center": {"x": 5, "y": 5},
            }
        ]


class StubWindowsAdapter:
    def __init__(self, payload):
        self.payload = payload

    def observe(self):
        return self.payload


class StubBrowserAdapter:
    def __init__(self, payload, snapshot):
        self.payload = payload
        self._snapshot = snapshot
        self.debug_url = "http://127.0.0.1:9222/json"

    def observe(self):
        return self.payload

    def snapshot_dom(self):
        return self._snapshot


class StubRouteWindowsAdapter:
    def route_to_application(self, app_id: str, window_title: str = "", launch_callback=None):
        return {"ok": True, "launched": False, "window": {"title": window_title or app_id}}

    def focus_window(self, *args, **kwargs):
        return True


class VisionRuntimeBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_factory = bridge_module.create_legacy_vlm_agent
        self.created_agents: list[StubAgent] = []

        def factory():
            agent = StubAgent()
            self.created_agents.append(agent)
            return agent

        bridge_module.create_legacy_vlm_agent = factory

    def tearDown(self) -> None:
        bridge_module.create_legacy_vlm_agent = self.original_factory

    def _bridge(self):
        memory = MemoryStore(base_dir="tests_tmp_bridge")
        return bridge_module.VisionRuntimeBridge(memory)

    def test_dom_evidence_skips_ocr_and_worker(self) -> None:
        bridge = self._bridge()
        bridge.windows_adapter = StubWindowsAdapter(
            {
                "active_window": {"title": "Google Chrome"},
                "active_app_guess": "chrome",
                "open_windows": [],
                "adapter_mode": "native_window_metadata",
            }
        )
        bridge.browser_adapter = StubBrowserAdapter(
            {"cdp_available": True, "available": True, "active_title": "Google"},
            {
                "title": "Google",
                "items": [
                    {
                        "tag": "textarea",
                        "text": "Search",
                        "selectors": ["#q"],
                        "selector": "#q",
                        "placeholder": "Search",
                        "box": {"x": 10, "y": 20, "width": 100, "height": 30},
                        "visible": True,
                        "enabled": True,
                        "stable_hash": "dom-search",
                    },
                    {
                        "tag": "button",
                        "text": "Google Search",
                        "selectors": ["button"],
                        "selector": "button",
                        "box": {"x": 120, "y": 20, "width": 90, "height": 30},
                        "visible": True,
                        "enabled": True,
                        "stable_hash": "dom-button",
                    },
                ],
            },
        )

        graph = bridge.parse_ui(output_filename="dom.png")

        self.assertEqual(graph.metadata["parse_health"]["parse_mode"], "browser_dom")
        self.assertTrue(graph.metadata["parse_health"]["ocr_skipped_by_dom"])
        self.assertFalse(graph.metadata["parse_health"]["worker_used"])
        node = graph.flatten()[1]
        self.assertEqual(node.source_frame_id, "#q")
        self.assertEqual(node.tag, "textarea")
        self.assertEqual(node.role, "textarea")
        self.assertEqual(node.text, "Search")
        self.assertEqual(node.placeholder, "Search")
        self.assertEqual(node.selector, "#q")
        self.assertEqual(node.rect, {"x": 10, "y": 20, "w": 100, "h": 30})
        self.assertTrue(node.visible)
        self.assertTrue(node.enabled)
        self.assertEqual(node.stable_hash, bridge._dom_stable_hash(tag="textarea", role="textarea", text="Search", selector="#q"))
        self.assertEqual(node.visual_id, node.stable_hash)
        self.assertEqual(node.box["width"], 100)
        self.assertTrue(bridge.dom_identity.verify(node.node_id, node.stable_hash, node.rect))
        self.assertEqual(len(self.created_agents), 0)

    def test_chrome_without_cdp_raises_instead_of_ocr_fallback(self) -> None:
        bridge = self._bridge()
        bridge.windows_adapter = StubWindowsAdapter(
            {
                "active_window": {"title": "Google Chrome"},
                "active_app_guess": "chrome",
                "open_windows": [],
                "adapter_mode": "native_window_metadata",
            }
        )
        bridge.browser_adapter = StubBrowserAdapter({"cdp_available": False, "available": False}, {})
        bridge._ensure_chrome_dom_available = lambda: None  # type: ignore[method-assign]

        with self.assertRaises(bridge_module.BrowserDOMUnavailable):
            bridge.parse_ui(output_filename="chrome_no_cdp.png")
        self.assertEqual(len(self.created_agents), 0)

    def test_uia_evidence_skips_ocr_and_worker(self) -> None:
        bridge = self._bridge()
        bridge.windows_adapter = StubWindowsAdapter(
            {
                "active_window": {"title": "File Explorer"},
                "active_app_guess": "explorer",
                "open_windows": [],
                "adapter_mode": "native_window_metadata",
                "uia_elements": [
                    {
                        "name": "Downloads",
                        "automation_id": "DownloadsItem",
                        "control_type": "button",
                        "rectangle": {"x": 5, "y": 8, "width": 80, "height": 24},
                        "clickable_point": {"x": 45, "y": 20},
                        "enabled": True,
                        "visible": True,
                        "parent_window": "File Explorer",
                        "stable_hash": "uia-downloads",
                    },
                    {
                        "name": "Search",
                        "automation_id": "SearchBox",
                        "control_type": "text_field",
                        "rectangle": {"x": 90, "y": 8, "width": 120, "height": 24},
                        "clickable_point": {"x": 150, "y": 20},
                        "enabled": True,
                        "visible": True,
                        "parent_window": "File Explorer",
                        "stable_hash": "uia-search",
                    },
                ],
            }
        )
        bridge.browser_adapter = StubBrowserAdapter({"cdp_available": False, "available": False}, {})

        graph = bridge.parse_ui(output_filename="uia.png")

        self.assertEqual(graph.metadata["parse_health"]["parse_mode"], "windows_uia")
        self.assertTrue(graph.metadata["parse_health"]["ocr_skipped_by_uia"])
        self.assertFalse(graph.metadata["parse_health"]["worker_used"])
        node = graph.flatten()[1]
        self.assertEqual(node.visual_id, "uia-downloads")
        self.assertEqual(node.source_frame_id, "File Explorer")
        self.assertEqual(node.center, {"x": 45, "y": 20})
        self.assertEqual(len(self.created_agents), 0)

    def test_screen_hash_cache_avoids_second_worker_parse(self) -> None:
        bridge = self._bridge()
        bridge.windows_adapter = StubWindowsAdapter(
            {
                "active_window": {"title": "File Explorer"},
                "active_app_guess": "explorer",
                "open_windows": [],
                "adapter_mode": "native_window_metadata",
            }
        )
        bridge.browser_adapter = StubBrowserAdapter({"cdp_available": False, "available": False}, {})
        bridge._current_screen_hash = lambda: "screen-1"  # type: ignore[method-assign]

        first = bridge.parse_ui(output_filename="first.png")
        second = bridge.parse_ui(output_filename="second.png")

        agent = self.created_agents[0]
        self.assertEqual(agent.parse_calls, 1)
        self.assertTrue(first.metadata["parse_health"]["worker_used"])
        self.assertEqual(second.metadata["parse_health"]["parse_mode"], "vision_cache")
        self.assertTrue(second.metadata["parse_health"]["cache_hit"])
        self.assertTrue(second.metadata["parse_health"]["screen_hash_cache_hit"])
        self.assertEqual(second.metadata["parse_health"]["ocr_calls"], 0)

    def test_resident_worker_reuses_agent_and_records_timings(self) -> None:
        bridge = self._bridge()
        bridge.windows_adapter = StubWindowsAdapter(
            {
                "active_window": {"title": "File Explorer"},
                "active_app_guess": "explorer",
                "open_windows": [],
                "adapter_mode": "native_window_metadata",
            }
        )
        bridge.browser_adapter = StubBrowserAdapter({"cdp_available": False, "available": False}, {})
        hashes = iter(["screen-a", "screen-b"])
        bridge._current_screen_hash = lambda: next(hashes)  # type: ignore[method-assign]

        first = bridge.parse_ui(output_filename="first.png")
        second = bridge.parse_ui(output_filename="second.png")

        self.assertEqual(len(self.created_agents), 1)
        self.assertEqual(self.created_agents[0].parse_calls, 2)
        for graph in (first, second):
            health = graph.metadata["parse_health"]
            self.assertTrue(health["worker_used"])
            self.assertIn("worker_queue_wait_seconds", health)
            self.assertIn("worker_exec_seconds", health)
            self.assertIn("worker_boot_seconds", health)
            self.assertIn("screen_hash_elapsed_seconds", health)

    def test_route_chrome_auto_launches_debug_profile_when_cdp_missing(self) -> None:
        bridge = self._bridge()
        bridge.windows_adapter = StubRouteWindowsAdapter()
        bridge.browser_adapter = StubBrowserAdapter({"cdp_available": False, "available": False}, {})
        original_popen = bridge_module.subprocess.Popen
        calls = []

        def fake_popen(*args, **kwargs):
            calls.append(args)
            raise OSError("chrome unavailable in test")

        bridge_module.subprocess.Popen = fake_popen
        try:
            result = bridge.route_window("chrome")
        finally:
            bridge_module.subprocess.Popen = original_popen

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(len(calls), 1)
        self.assertIn("--remote-debugging-port=9222", calls[0][0])

    def test_chrome_profile_hint_selects_closest_existing_profile(self) -> None:
        bridge = self._bridge()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_state = {
                "profile": {
                    "info_cache": {
                        "Default": {"name": "Personal"},
                        "Profile 3": {"name": "Client Work"},
                    }
                }
            }
            with open(os.path.join(tmpdir, "Local State"), "w", encoding="utf-8") as handle:
                json.dump(local_state, handle)

            self.assertEqual(bridge._choose_chrome_profile_dir("work", tmpdir), "Profile 3")
            self.assertEqual(bridge._choose_chrome_profile_dir("", tmpdir), "Default")


if __name__ == "__main__":
    unittest.main()
