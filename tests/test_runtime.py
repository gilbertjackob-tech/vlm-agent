from __future__ import annotations

import importlib
import tempfile
import types
import sys
import time
import unittest

from copilot.memory.store import MemoryStore
from copilot.planner.compiler import PromptCompiler
from copilot.runtime.policy import PolicyEngine
from copilot.runtime.voice_narrator import VoiceConfig, VoiceNarrator
from copilot.schemas import ActionTarget, ExecutionPlan, ObservationGraph, PlanStep, RunTrace, TaskSpec, TrustMode


EXPLORER_GRAPH = [
    {
        "id": "main_page",
        "label": "Main Page",
        "box": {"x": 0, "y": 0, "width": 1200, "height": 800},
        "type": "container",
        "center": {"x": 600, "y": 400},
        "semantic_role": "main_page",
        "region": "main_page",
        "children": [
            {
                "id": "row_awake",
                "label": "Row: awake.py",
                "box": {"x": 260, "y": 120, "width": 720, "height": 52},
                "type": "container",
                "center": {"x": 620, "y": 146},
                "semantic_role": "list_row",
                "entity_type": "python_file",
                "region": "main_page",
                "affordances": ["open"],
                "children": [
                    {
                        "id": "awake_label",
                        "label": "awake.py",
                        "box": {"x": 300, "y": 126, "width": 160, "height": 28},
                        "type": "button",
                        "center": {"x": 380, "y": 140},
                        "region": "main_page",
                    }
                ],
            },
            {
                "id": "row_video",
                "label": "Row: demo.mp4",
                "box": {"x": 260, "y": 200, "width": 720, "height": 52},
                "type": "container",
                "center": {"x": 620, "y": 226},
                "semantic_role": "list_row",
                "entity_type": "video",
                "region": "main_page",
                "affordances": ["open"],
                "children": [
                    {
                        "id": "video_label",
                        "label": "demo.mp4",
                        "box": {"x": 300, "y": 206, "width": 160, "height": 28},
                        "type": "button",
                        "center": {"x": 380, "y": 220},
                        "region": "main_page",
                    }
                ],
            },
        ],
    },
    {
        "id": "nav_desktop",
        "label": "Desktop",
        "box": {"x": 20, "y": 160, "width": 160, "height": 36},
        "type": "button",
        "center": {"x": 100, "y": 178},
        "semantic_role": "menu_item",
        "entity_type": "navigation_item",
        "region": "left_menu",
        "affordances": ["navigate"],
    },
    {
        "id": "nav_downloads",
        "label": "Downloads",
        "box": {"x": 20, "y": 220, "width": 160, "height": 36},
        "type": "button",
        "center": {"x": 100, "y": 238},
        "semantic_role": "menu_item",
        "entity_type": "navigation_item",
        "region": "left_menu",
        "affordances": ["navigate"],
    },
    {
        "id": "search_explorer",
        "label": "Search",
        "box": {"x": 960, "y": 60, "width": 180, "height": 30},
        "type": "text_field",
        "center": {"x": 1050, "y": 75},
        "semantic_role": "text_field",
        "entity_type": "search_field",
        "region": "top_menu",
        "affordances": ["focus"],
    },
]

CHROME_GRAPH = [
    {
        "id": "chrome_top",
        "label": "Chrome top bar",
        "box": {"x": 0, "y": 0, "width": 1280, "height": 120},
        "type": "container",
        "center": {"x": 640, "y": 60},
        "semantic_role": "top_menu",
        "region": "top_menu",
        "children": [
            {
                "id": "omnibox_node",
                "label": "Search or type web address",
                "box": {"x": 220, "y": 36, "width": 720, "height": 30},
                "type": "text_field",
                "center": {"x": 580, "y": 51},
                "semantic_role": "text_field",
                "entity_type": "omnibox",
                "region": "top_menu",
                "affordances": ["focus"],
            }
        ],
    },
    {
        "id": "chrome_main",
        "label": "Search Results",
        "box": {"x": 0, "y": 120, "width": 1280, "height": 720},
        "type": "container",
        "center": {"x": 640, "y": 460},
        "semantic_role": "main_page",
        "region": "main_page",
        "children": [
            {
                "id": "result_1",
                "label": "Lo-fi Coding Mix - YouTube",
                "box": {"x": 160, "y": 220, "width": 700, "height": 40},
                "type": "button",
                "center": {"x": 510, "y": 240},
                "semantic_role": "clickable_container",
                "region": "main_page",
                "affordances": ["open"],
            }
        ],
    },
]


class FakeBridge:
    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.current_app = ""
        self.chrome_available = True
        self.agent = types.SimpleNamespace(semantic_memory=memory_store.semantic_memory)
        self.parse_count = 0
        self.selector_clicks: list[str] = []
        self.clicked_nodes: list[dict[str, object]] = []
        self.modified_clicks: list[dict[str, object]] = []
        self.typed_events: list[dict[str, object]] = []
        self.pressed_events: list[dict[str, object]] = []
        self.hovered_nodes: list[str] = []
        self.chrome_query = ""
        self.chrome_stage = "start"
        self.explorer_location = "desktop"
        self.tooltip_label = ""

    def observe_environment(self):
        if self.current_app == "explorer":
            title = "File Explorer"
        elif self.current_app == "chrome":
            title = "Google Chrome"
        else:
            title = ""
        return {
            "windows": {
                "active_window": {"title": title},
                "active_app_guess": self.current_app,
                "open_windows": [],
                "adapter_mode": "native_window_metadata",
            },
            "browser": {
                "available": self.chrome_available,
                "cdp_available": self.chrome_available,
                "tab_count": 1 if self.chrome_available else 0,
                "tab_titles": [self.read_browser_dom().get("title", "")] if self.chrome_available else [],
                "adapter_mode": "browser_dom_metadata",
            },
        }

    def browser_dom_available(self):
        return self.chrome_available

    def read_browser_dom(self):
        if not self.chrome_available:
            return {}
        title = "Google"
        url = "https://www.google.com/"
        result_text = ""
        if self.chrome_stage == "typed" and self.chrome_query:
            title = f"{self.chrome_query} - Search"
        elif self.chrome_stage == "results" and self.chrome_query:
            title = f"{self.chrome_query} - Google Search"
            url = f"https://www.google.com/search?q={self.chrome_query.replace(' ', '+')}"
            result_text = f"{self.chrome_query} - YouTube"
        return {
            "title": title,
            "url": url,
            "items": [
                {
                    "tag": "textarea",
                    "id": "omnibox",
                    "role": "combobox",
                    "aria_label": "Search",
                    "placeholder": "Search Google or type a URL",
                    "text": self.chrome_query if self.chrome_stage in {"typed", "results"} else "",
                    "selectors": ["#omnibox", "textarea[name='q']", "textarea[aria-label='Search']"],
                },
                {
                    "tag": "button",
                    "aria_label": "Google Search",
                    "text": "Google Search",
                    "selectors": ["button[aria-label='Google Search']"],
                },
                {
                    "tag": "a",
                    "text": result_text,
                    "href": "https://www.youtube.com/watch?v=abc" if result_text else "",
                    "selectors": ["a[href='https://www.youtube.com/watch?v=abc']"],
                },
            ],
        }

    def read_browser_state_hash(self):
        dom = self.read_browser_dom()
        return f"{dom.get('title', '')}|{dom.get('url', '')}|{self.chrome_stage}|{self.chrome_query}"

    def read_uia_state_hash(self):
        return f"{self.current_app}|{self.explorer_location}|{self.tooltip_label}"

    def read_focused_element(self):
        if self.current_app == "chrome":
            return "#omnibox"
        return ""

    def parse_ui(self, output_filename="ui.png"):
        self.parse_count += 1
        raw = CHROME_GRAPH if self.current_app == "chrome" else EXPLORER_GRAPH
        if self.current_app == "explorer" and self.explorer_location == "downloads":
            raw = list(raw) + [
                {
                    "id": "downloads_contents",
                    "label": "Downloads folder contents",
                    "box": {"x": 260, "y": 280, "width": 720, "height": 52},
                    "type": "container",
                    "center": {"x": 620, "y": 306},
                    "semantic_role": "main_page",
                    "entity_type": "folder_view",
                    "region": "main_page",
                    "affordances": ["inspect"],
                }
            ]
        if self.current_app == "explorer" and self.tooltip_label:
            raw = list(raw) + [
                {
                    "id": "tooltip",
                    "label": self.tooltip_label,
                    "box": {"x": 220, "y": 210, "width": 260, "height": 42},
                    "type": "container",
                    "center": {"x": 350, "y": 231},
                    "semantic_role": "clickable_container",
                    "entity_type": "tooltip",
                    "region": "main_page",
                    "affordances": ["inspect"],
                }
            ]
        parse_mode = "browser_dom" if self.current_app == "chrome" else "windows_uia"
        return ObservationGraph.from_raw(
            raw,
            metadata={
                "output_filename": output_filename,
                "environment": self.observe_environment(),
                "parse_health": {
                    "parse_mode": parse_mode,
                    "cache_hit": False,
                    "worker_used": False,
                    "parse_elapsed_seconds": 0.01,
                },
            },
        )

    def execute_legacy_command(self, command_text, debug=False):
        return True

    def route_window(self, app_id: str, window_title: str = ""):
        self.current_app = app_id
        if app_id == "chrome":
            self.chrome_stage = "start"
            self.chrome_query = ""
        return {"ok": True, "launched": False, "window": {"title": window_title or app_id}}

    def confirm_focus(self, expected: str) -> bool:
        return expected.lower() in {"", self.current_app.lower(), "file explorer", "google chrome"}

    def wait_for(self, seconds: float = 0.0, expected_focus: str = "", timeout: float = 0.0):
        return expected_focus.lower() in {"", self.current_app.lower()}

    def click_node(self, node, clicks=1):
        self.clicked_nodes.append({"label": node.display_label(), "clicks": clicks})
        label = node.display_label().strip().lower()
        if self.current_app == "explorer" and label in {"downloads", "desktop"}:
            self.explorer_location = label
        return True

    def click_node_with_modifiers(self, node, modifiers, clicks=1):
        self.modified_clicks.append({"label": node.display_label(), "modifiers": list(modifiers), "clicks": clicks})
        return True

    def open_explorer_location(self, location: str):
        self.current_app = "explorer"
        self.explorer_location = str(location or "").strip().lower()
        return bool(self.explorer_location)

    def current_explorer_location(self):
        return self.explorer_location

    def click_point(self, x: int, y: int, clicks: int = 1):
        return True

    def hover_node(self, node):
        label = node.display_label()
        self.hovered_nodes.append(label)
        if label == "Downloads":
            self.tooltip_label = "Downloads Opens your downloads folder"
        elif label == "Desktop":
            self.tooltip_label = "Desktop Shows desktop files and folders"
        else:
            self.tooltip_label = ""
        return True

    def click_selector(self, selector: str):
        return bool(self.click_first_selector([selector]))

    def click_first_selector(self, selectors):
        available = {
            selector
            for item in self.read_browser_dom().get("items", [])
            if isinstance(item, dict)
            for selector in item.get("selectors", [])
            if isinstance(selector, str)
        }
        for selector in selectors:
            if selector in available:
                self.selector_clicks.append(selector)
                return selector
        return ""

    def type_text(self, text: str, selector: str = "", clear_first: bool = False):
        self.typed_events.append({"text": text, "selector": selector, "clear_first": clear_first})
        if self.current_app == "chrome":
            self.chrome_query = text
            self.chrome_stage = "typed"
        return True

    def press_key(self, keys, hotkey=False):
        normalized = [str(key).lower() for key in keys]
        self.pressed_events.append({"keys": normalized, "hotkey": hotkey})
        if self.current_app == "chrome" and "enter" in normalized:
            self.chrome_stage = "results"
        if self.current_app == "explorer" and hotkey and normalized == ["alt", "left"]:
            self.explorer_location = "desktop"
        return True

    def clear_focus(self):
        self.current_app = ""


class RuntimeExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_module = types.ModuleType("copilot.perception.bridge")
        fake_module.VisionRuntimeBridge = FakeBridge
        self.original_bridge_module = sys.modules.get("copilot.perception.bridge")
        sys.modules["copilot.perception.bridge"] = fake_module
        sys.modules.pop("copilot.runtime.engine", None)
        self.engine_module = importlib.import_module("copilot.runtime.engine")

    def tearDown(self) -> None:
        if self.original_bridge_module is not None:
            sys.modules["copilot.perception.bridge"] = self.original_bridge_module
        else:
            sys.modules.pop("copilot.perception.bridge", None)
        sys.modules.pop("copilot.runtime.engine", None)

    def _build_engine(self, tmpdir: str):
        engine = self.engine_module.CopilotEngine.__new__(self.engine_module.CopilotEngine)
        engine.memory = MemoryStore(base_dir=tmpdir)
        engine.bridge = FakeBridge(engine.memory)
        engine.planner = PromptCompiler(engine.memory)
        engine.planner.browser.snapshot_dom = lambda: engine.bridge.read_browser_dom()
        engine.policy = PolicyEngine(engine.memory)
        engine.voice_narrator = VoiceNarrator(
            VoiceConfig(mode="tts", cache_dir=tempfile.mkdtemp(dir=tmpdir)),
            http_post=lambda endpoint, payload, timeout: (b"RIFF", "audio/wav"),
        )
        engine.voice_narrator._write_cache = lambda path, audio, content_type: None  # type: ignore[method-assign]
        engine.voice_narrator._play_audio_file = lambda path: None  # type: ignore[method-assign]
        engine.stop_requested = False
        engine.last_observation = None
        engine.last_plan = None
        return engine

    def test_execute_prompt_records_outcomes_and_scene_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)

            trace = self.engine_module.CopilotEngine.execute_prompt(
                engine,
                "Open explorer and parse screen",
                trust_mode=TrustMode.PLAN_AND_RISK_GATES,
                approval_callback=lambda _prompt, _payload: True,
                dry_run=False,
            )

            self.assertEqual(trace.status.value, "success")
            self.assertEqual(trace.mission.status.value, "success")
            self.assertGreaterEqual(len(trace.action_outcomes), 3)
            self.assertGreaterEqual(len(trace.scene_deltas), 3)
            self.assertGreaterEqual(engine.bridge.parse_count, 1)
            voice_events = trace.outputs.get("voice_events", [])
            self.assertGreaterEqual(len(voice_events), 4)
            self.assertTrue(all(event["status"] in {"spoken", "throttled", "dropped"} for event in voice_events))
            self.assertFalse(any(event["status"] == "queued" for event in voice_events))
            self.assertTrue(any(event["event_type"] == "done" and event["status"] == "spoken" for event in voice_events))
            self.assertTrue(any(event["event_type"] in {"observe", "parse_ui", "checkpoint", "scene_diff"} for event in voice_events))

    def test_state_cache_reuses_unchanged_browser_dom_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "chrome"

            first = self.engine_module.CopilotEngine.parse_current_ui(engine, "first.png")
            second = self.engine_module.CopilotEngine.parse_current_ui(engine, "second.png")

            self.assertEqual(engine.bridge.parse_count, 1)
            self.assertFalse(first.metadata["parse_health"].get("state_cache_hit"))
            self.assertTrue(second.metadata["parse_health"]["state_cache_hit"])
            self.assertEqual(second.metadata["parse_health"]["parse_elapsed_seconds"], 0.0)

    def test_state_cache_invalidates_when_browser_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "chrome"

            self.engine_module.CopilotEngine.parse_current_ui(engine, "first.png")
            engine.bridge.chrome_query = "changed"
            engine.bridge.chrome_stage = "typed"
            second = self.engine_module.CopilotEngine.parse_current_ui(engine, "second.png")

            self.assertEqual(engine.bridge.parse_count, 2)
            self.assertFalse(second.metadata["parse_health"].get("state_cache_hit"))
            self.assertEqual(second.metadata["parse_health"].get("state_cache_reason"), "signature_changed")

    def test_state_cache_reuses_unchanged_uia_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"

            self.engine_module.CopilotEngine.parse_current_ui(engine, "first.png")
            second = self.engine_module.CopilotEngine.parse_current_ui(engine, "second.png")

            self.assertEqual(engine.bridge.parse_count, 1)
            self.assertTrue(second.metadata["parse_health"]["state_cache_hit"])

    def test_state_cache_invalidates_when_active_window_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"

            self.engine_module.CopilotEngine.parse_current_ui(engine, "first.png")
            engine.bridge.current_app = "chrome"
            second = self.engine_module.CopilotEngine.parse_current_ui(engine, "second.png")

            self.assertEqual(engine.bridge.parse_count, 2)
            self.assertFalse(second.metadata["parse_health"].get("state_cache_hit"))

    def test_failed_action_invalidates_state_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "chrome"
            self.engine_module.CopilotEngine.parse_current_ui(engine, "first.png")
            task = TaskSpec(prompt="click missing", goal="click missing")
            step = PlanStep("step_missing_dom", "Click missing", "click_node", parameters={"selector_candidates": ["#missing"]})
            trace = RunTrace(run_id="run_missing_dom", task=task, plan=ExecutionPlan(task=task, steps=[step]))

            ok, _mode, _target, _notes = self.engine_module.CopilotEngine._execute_step(engine, step, trace, None)
            self.engine_module.CopilotEngine.parse_current_ui(engine, "after_failed.png")

            self.assertFalse(ok)
            self.assertEqual(engine.bridge.parse_count, 2)

    def test_execute_chrome_search_uses_ctrl_l_and_verified_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)

            trace = self.engine_module.CopilotEngine.execute_prompt(
                engine,
                "Open chrome and search youtube for lo-fi coding mix",
                trust_mode=TrustMode.PLAN_AND_RISK_GATES,
                approval_callback=lambda _prompt, _payload: True,
                dry_run=False,
            )

            action_types = [step.action_type for step in trace.plan.steps]
            focus_index = action_types.index("press_key")
            type_index = action_types.index("type_text")
            submit_index = action_types.index("press_key", focus_index + 1)

            self.assertEqual(trace.status.value, "success")
            self.assertNotIn("click_node", action_types[:type_index])
            self.assertNotIn("click_point", action_types)
            self.assertEqual(engine.bridge.pressed_events[0]["keys"], ["ctrl", "l"])
            self.assertTrue(engine.bridge.pressed_events[0]["hotkey"])
            self.assertEqual(action_types[type_index + 1], "verify_scene")
            self.assertEqual(action_types[submit_index + 1 : submit_index + 4], ["wait_for", "verify_scene", "scene_diff"])
            self.assertEqual(engine.bridge.typed_events[0]["selector"], "")
            self.assertFalse(engine.bridge.typed_events[0]["clear_first"])
            self.assertIn("enter", engine.bridge.pressed_events[1]["keys"])
            self.assertFalse(engine.bridge.selector_clicks)
            self.assertTrue(any(outcome.action_type == "type_text" and outcome.control_mode == "native" for outcome in trace.action_outcomes))
            contracts = trace.outputs.get("action_contracts", [])
            contract_types = [contract["action_type"] for contract in contracts]
            self.assertIn("type_text", contract_types)
            self.assertGreaterEqual(contract_types.count("hotkey"), 1)
            self.assertIn("press_key", contract_types)
            self.assertIn("wait_for", contract_types)
            self.assertTrue(all(contract["status"] == "verified" for contract in contracts))
            self.assertTrue(trace.outputs.get("state_snapshots"))
            self.assertTrue(trace.outputs.get("state_diffs"))
            self.assertTrue(trace.outputs.get("task_state_timeline"))
            self.assertTrue(all("state_id" in item for item in trace.outputs["state_snapshots"]))
            voice_lines = [event["line"] for event in trace.outputs.get("voice_events", [])]
            self.assertIn("I found the input field. I am typing now.", voice_lines)

    def test_execute_explorer_modifier_click_holds_modifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)

            trace = self.engine_module.CopilotEngine.execute_prompt(
                engine,
                "Open explorer and ctrl click awake.py",
                trust_mode=TrustMode.PLAN_AND_RISK_GATES,
                approval_callback=lambda _prompt, _payload: True,
                dry_run=False,
            )

            self.assertEqual(trace.status.value, "success")
            self.assertIn("awake.py", engine.bridge.modified_clicks[0]["label"])
            self.assertEqual(engine.bridge.modified_clicks[0]["modifiers"], ["ctrl"])

    def test_execute_learning_session_records_tooltip_feedback_and_review_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)

            trace = self.engine_module.CopilotEngine.execute_prompt(
                engine,
                "Open explorer and learn the current UI by hovering",
                trust_mode=TrustMode.PLAN_AND_RISK_GATES,
                approval_callback=lambda _prompt, _payload: True,
                dry_run=False,
            )

            action_types = [step.action_type for step in trace.plan.steps]
            self.assertEqual(trace.status.value, "success")
            self.assertIn("learning_session", action_types)
            self.assertNotIn("hover_probe", action_types)
            self.assertNotIn("explore_safe", action_types)
            self.assertTrue(engine.bridge.hovered_nodes)
            self.assertTrue(engine.memory.semantic_memory.get("hover_feedback"))
            self.assertTrue(engine.memory.semantic_memory.get("learning_sessions"))
            self.assertTrue(engine.memory.semantic_memory.get("review_queue"))
            self.assertIn("learning_sessions", trace.outputs)
            feedback_blob = str(engine.memory.semantic_memory["hover_feedback"]).lower()
            self.assertIn("downloads", feedback_blob)

    def test_execute_interaction_learning_records_rewarded_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)

            trace = self.engine_module.CopilotEngine.execute_prompt(
                engine,
                "Open explorer and learn what clicking safe folders opens",
                trust_mode=TrustMode.MOSTLY_AUTONOMOUS,
                approval_callback=lambda _prompt, _payload: True,
                dry_run=False,
            )

            action_types = [step.action_type for step in trace.plan.steps]
            self.assertEqual(trace.status.value, "success")
            self.assertIn("interaction_learning", action_types)
            self.assertTrue(engine.bridge.clicked_nodes)
            graph = engine.memory.semantic_memory["interaction_graph"]
            self.assertTrue(graph["edges"])
            self.assertTrue(graph["action_values"])
            self.assertTrue(trace.outputs.get("interaction_learning"))
            rewards = [edge["reward_avg"] for edge in graph["edges"].values()]
            self.assertTrue(any(reward > 0 for reward in rewards))

    def test_execute_prompt_replays_rewarded_interaction_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)

            self.engine_module.CopilotEngine.execute_prompt(
                engine,
                "Open explorer and learn what clicking safe folders opens",
                trust_mode=TrustMode.MOSTLY_AUTONOMOUS,
                approval_callback=lambda _prompt, _payload: True,
                dry_run=False,
            )
            engine.bridge.explorer_location = "desktop"
            engine.last_observation = None

            trace = self.engine_module.CopilotEngine.execute_prompt(
                engine,
                "open downloads",
                trust_mode=TrustMode.PLAN_AND_RISK_GATES,
                approval_callback=lambda _prompt, _payload: True,
                dry_run=False,
            )

            action_types = [step.action_type for step in trace.plan.steps]
            self.assertEqual(trace.plan.source, "interaction_graph_replay")
            self.assertIn("replay_interaction", action_types)
            self.assertEqual(trace.status.value, "success")
            self.assertTrue(any(outcome.action_type == "replay_interaction" and outcome.ok for outcome in trace.action_outcomes))

    def test_click_node_requires_action_contract_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            step = PlanStep(
                step_id="step_contract_click",
                title="Open Downloads",
                action_type="click_node",
                target=ActionTarget(kind="ui_node", value="Downloads", filters={"label_contains": "Downloads"}),
                success_criteria="Downloads folder contents should be visible.",
            )
            trace = RunTrace(
                run_id="run_contract",
                task=task,
                plan=ExecutionPlan(task=task, steps=[step]),
            )

            ok, mode, target, notes = self.engine_module.CopilotEngine._execute_step(engine, step, trace, None)

            self.assertTrue(ok)
            self.assertEqual(mode, "hybrid")
            self.assertIsNotNone(target)
            self.assertIn("Verified click", notes)
            contracts = trace.outputs.get("action_contracts", [])
            self.assertEqual(len(contracts), 1)
            self.assertEqual(contracts[0]["status"], "verified")
            self.assertTrue(contracts[0]["before_checks"]["target_exists"])
            self.assertTrue(contracts[0]["verification"]["verified"])
            self.assertGreaterEqual(contracts[0]["evidence_score"], 0.55)
            self.assertEqual(contracts[0]["failure_reason"], "")

    def test_click_node_blocks_low_gap_duplicate_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            step = PlanStep(
                step_id="step_ambiguous_downloads",
                title="Open Downloads",
                action_type="click_node",
                target=ActionTarget(kind="ui_node", value="Downloads", filters={"label_contains": "Downloads", "min_score": 0.4}),
            )
            trace = RunTrace(run_id="run_ambiguous", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            engine.last_observation = ObservationGraph.from_raw(
                [
                    {
                        "id": "nav_downloads",
                        "label": "Downloads",
                        "type": "button",
                        "semantic_role": "menu_item",
                        "entity_type": "navigation_item",
                        "app_id": "explorer",
                        "region": "left_menu",
                        "affordances": ["navigate", "click"],
                        "center": {"x": 90, "y": 220},
                        "box": {"x": 20, "y": 200, "width": 140, "height": 32},
                    },
                    {
                        "id": "main_downloads",
                        "label": "Downloads",
                        "type": "button",
                        "semantic_role": "list_row",
                        "entity_type": "folder",
                        "app_id": "explorer",
                        "region": "main_page",
                        "affordances": ["open", "click"],
                        "center": {"x": 380, "y": 260},
                        "box": {"x": 260, "y": 238, "width": 240, "height": 44},
                    },
                ],
                metadata={"app_id": "explorer", "output_filename": "ambiguous.png"},
            )

            ok, _mode, target, notes = self.engine_module.CopilotEngine._execute_step(engine, step, trace, None)

            self.assertFalse(ok)
            self.assertIsNone(target)
            self.assertIn("ambiguous", notes.lower())
            self.assertTrue(trace.outputs["target_rankings"][0]["ambiguous"])
            self.assertEqual(engine.bridge.clicked_nodes, [])

    def test_failed_click_point_records_classified_recovery_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            task = TaskSpec(prompt="click raw point", goal="click raw point")
            step = PlanStep(
                step_id="step_raw_point",
                title="Click raw point",
                action_type="click_point",
                parameters={"x": 10, "y": 20},
            )
            trace = RunTrace(
                run_id="run_recovery",
                task=task,
                plan=ExecutionPlan(task=task, steps=[step]),
            )

            ok, _mode, _target, notes = self.engine_module.CopilotEngine._execute_step(engine, step, trace, None)
            failed_contract = trace.outputs["action_contracts"][0]
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(
                engine,
                trace,
                step,
                self.engine_module.CopilotEngine._classify_failure_reason(engine, notes, failed_contract),
                failed_contract,
            )

            self.assertFalse(ok)
            self.assertEqual(failed_contract["failure_reason"], "UNSAFE_COORDINATE")
            self.assertEqual(recovery["failure_reason"], "UNSAFE_COORDINATE")
            self.assertTrue(recovery["stop_required"])

    def test_target_not_found_uses_ocr_recovery_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            step = PlanStep(
                step_id="step_missing_downloads",
                title="Open Downloads",
                action_type="click_node",
                target=ActionTarget(kind="ui_node", value="Downloads", filters={"label_contains": "Downloads"}),
            )
            trace = RunTrace(run_id="run_missing", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            engine.last_observation = ObservationGraph.from_raw(
                [{"id": "only_desktop", "label": "Desktop", "type": "button", "semantic_role": "menu_item", "entity_type": "navigation_item", "region": "left_menu", "center": {"x": 100, "y": 178}}],
                metadata={"app_id": "explorer", "output_filename": "missing.png"},
            )

            ok, _mode, _target, notes = self.engine_module.CopilotEngine._execute_step(engine, step, trace, None)
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(
                engine,
                trace,
                step,
                self.engine_module.CopilotEngine._classify_failure_reason(engine, notes, None),
                None,
            )
            retry_ok, retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, None, None)

            self.assertFalse(ok)
            self.assertTrue(retry_ok)
            self.assertIn("Recovered via", retry_note)
            attempt = trace.outputs["recovery_attempts"][0]
            self.assertEqual(attempt["failure_reason"], "TARGET_NOT_FOUND")
            self.assertIn(attempt["resolver_used"], {"ocr", "vision", "uia"})
            self.assertTrue(attempt["retry_success"])

    def test_no_state_change_retries_with_alternate_resolver_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            step = PlanStep(
                step_id="step_no_change",
                title="Open Downloads",
                action_type="click_node",
                target=ActionTarget(kind="ui_node", value="Downloads", filters={"label_contains": "Downloads"}),
            )
            trace = RunTrace(run_id="run_no_change", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            failed_contract = {
                "step_id": step.step_id,
                "target": "Dead Downloads",
                "failure_reason": "NO_STATE_CHANGE",
                "evidence_score": 0.4,
                "evidence_grade": "weak",
            }
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "NO_STATE_CHANGE", failed_contract)
            retry_ok, retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, failed_contract, None)

            self.assertTrue(retry_ok)
            self.assertIn("Recovered via", retry_note)
            self.assertTrue(trace.outputs["recovery_attempts"][0]["retry_success"])

    def test_unsafe_coordinate_recovery_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            task = TaskSpec(prompt="click raw point", goal="click raw point")
            step = PlanStep("step_point", "Click point", "click_point", parameters={"x": 10, "y": 20})
            trace = RunTrace(run_id="run_point_no_retry", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            failed_contract = {"failure_reason": "UNSAFE_COORDINATE", "target": "10,20"}
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "UNSAFE_COORDINATE", failed_contract)
            retry_ok, retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, failed_contract, None)

            self.assertFalse(retry_ok)
            self.assertIn("operator", retry_note.lower())
            self.assertNotIn("recovery_attempts", trace.outputs)

    def test_type_text_recovery_checks_field_before_retyping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "chrome"
            engine.bridge.chrome_query = "hello"
            engine.bridge.chrome_stage = "typed"
            task = TaskSpec(prompt="type hello", goal="type hello")
            step = PlanStep("step_type", "Type hello", "type_text", target=ActionTarget(kind="text", value="hello"), parameters={"selector": "#omnibox", "text": "hello"})
            trace = RunTrace(run_id="run_type_check", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "FOCUS_NOT_CONFIRMED", {"failure_reason": "FOCUS_NOT_CONFIRMED"})

            retry_ok, retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, None, None)

            self.assertTrue(retry_ok)
            self.assertIn("already present", retry_note)
            self.assertEqual(len(engine.bridge.typed_events), 0)
            self.assertEqual(trace.outputs["recovery_attempts"][0]["resolver_used"], "field_check")

    def test_type_text_recovery_retypes_after_content_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "chrome"
            task = TaskSpec(prompt="type hello", goal="type hello")
            step = PlanStep("step_type_retry", "Type hello", "type_text", target=ActionTarget(kind="text", value="hello"), parameters={"selector": "#omnibox", "text": "hello", "clear_first": True})
            trace = RunTrace(run_id="run_type_retry", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "FOCUS_NOT_CONFIRMED", {"failure_reason": "FOCUS_NOT_CONFIRMED"})

            retry_ok, _retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, None, None)

            self.assertTrue(retry_ok)
            self.assertEqual(engine.bridge.typed_events[-1]["text"], "hello")
            self.assertTrue(trace.outputs["recovery_attempts"][0]["retry_success"])

    def test_press_key_recovery_retries_safe_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "chrome"
            engine.bridge.chrome_query = "hello"
            engine.bridge.chrome_stage = "typed"
            task = TaskSpec(prompt="press enter", goal="press enter")
            step = PlanStep("step_enter", "Press Enter", "press_key", parameters={"keys": ["enter"]}, success_criteria="Search results should load.")
            trace = RunTrace(run_id="run_enter_retry", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "NO_STATE_CHANGE", {"failure_reason": "NO_STATE_CHANGE"})

            retry_ok, _retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, None, None)

            self.assertTrue(retry_ok)
            self.assertIn("enter", engine.bridge.pressed_events[-1]["keys"])
            self.assertEqual(trace.outputs["recovery_attempts"][0]["resolver_used"], "focus_key_retry")

    def test_destructive_hotkey_recovery_is_blocked_without_expected_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "chrome"
            task = TaskSpec(prompt="close tab", goal="close tab")
            step = PlanStep("step_hotkey", "Close tab", "press_key", parameters={"keys": ["ctrl", "w"], "hotkey": True})
            trace = RunTrace(run_id="run_hotkey_block", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "NO_STATE_CHANGE", {"failure_reason": "NO_STATE_CHANGE"})

            retry_ok, retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, None, None)

            self.assertFalse(retry_ok)
            self.assertIn("destructive", retry_note.lower())
            self.assertEqual(trace.outputs["recovery_attempts"][0]["resolver_used"], "safety_gate")

    def test_wait_for_recovery_retries_bounded_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "chrome"
            task = TaskSpec(prompt="wait for chrome", goal="wait for chrome")
            step = PlanStep("step_wait", "Wait for Chrome", "wait_for", parameters={"seconds": 0.1, "expected_focus": "chrome", "timeout": 0.2})
            trace = RunTrace(run_id="run_wait_retry", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "TIMEOUT", {"failure_reason": "TIMEOUT"})

            retry_ok, _retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, None, None)

            self.assertTrue(retry_ok)
            self.assertEqual(trace.outputs["recovery_attempts"][0]["resolver_used"], "bounded_wait_retry")

    def test_dom_click_recovery_uses_selector_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "chrome"
            task = TaskSpec(prompt="click search", goal="click search")
            step = PlanStep("step_dom_retry", "Click search", "click_node", parameters={"selector_candidates": ["#omnibox"]})
            trace = RunTrace(run_id="run_dom_retry", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "TARGET_NOT_FOUND", {"failure_reason": "TARGET_NOT_FOUND"})

            retry_ok, _retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, None, None)

            self.assertTrue(retry_ok)
            self.assertEqual(trace.outputs["recovery_attempts"][0]["resolver_used"], "dom")
            self.assertIn("#omnibox", engine.bridge.selector_clicks)

    def test_click_point_recovery_upgrades_to_resolved_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            step = PlanStep("step_point_upgrade", "Click point near Downloads", "click_point", target=ActionTarget(kind="ui_node", value="Downloads", filters={"label_contains": "Downloads"}), parameters={"x": 100, "y": 238})
            trace = RunTrace(run_id="run_point_upgrade", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "NO_STATE_CHANGE", {"failure_reason": "NO_STATE_CHANGE", "target": "100,238"})

            retry_ok, _retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, None, None)

            self.assertTrue(retry_ok)
            self.assertNotEqual(trace.outputs["recovery_attempts"][0]["resolver_used"], "")
            self.assertEqual(engine.bridge.clicked_nodes[-1]["label"], "Downloads")

    def test_retry_limit_is_enforced_per_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            task = TaskSpec(prompt="open missing", goal="open missing")
            step = PlanStep("step_limit", "Open missing", "click_node", target=ActionTarget(kind="ui_node", value="Missing", filters={"label_contains": "Missing"}))
            trace = RunTrace(run_id="run_limit", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            trace.outputs["recovery_attempts"] = [{"step_id": "step_limit"}, {"step_id": "step_limit"}]
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "TARGET_NOT_FOUND", {"failure_reason": "TARGET_NOT_FOUND"})

            retry_ok, retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, None, None)

            self.assertFalse(retry_ok)
            self.assertIn("limit", retry_note.lower())

    def test_wait_for_failure_classifies_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            task = TaskSpec(prompt="wait", goal="wait")
            step = PlanStep("step_wait_fail", "Wait for missing app", "wait_for", parameters={"seconds": 0.1, "expected_focus": "missing", "timeout": 0.1})
            trace = RunTrace(run_id="run_wait_fail", task=task, plan=ExecutionPlan(task=task, steps=[step]))

            ok, _mode, _target, notes = self.engine_module.CopilotEngine._execute_step(engine, step, trace, None)

            self.assertFalse(ok)
            contract = trace.outputs["action_contracts"][0]
            self.assertEqual(contract["failure_reason"], "TIMEOUT")
            self.assertIn("timed out", notes.lower())

    def test_route_failure_classifies_focus_not_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)

            reason = self.engine_module.CopilotEngine._classify_failure_reason(
                engine,
                "Could not route the requested window.",
                None,
            )

            self.assertEqual(reason, "FOCUS_NOT_CONFIRMED")

    def test_dom_click_failed_selector_records_target_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "chrome"
            task = TaskSpec(prompt="click missing", goal="click missing")
            step = PlanStep("step_missing_dom", "Click missing", "click_node", parameters={"selector_candidates": ["#missing"]})
            trace = RunTrace(run_id="run_missing_dom", task=task, plan=ExecutionPlan(task=task, steps=[step]))

            ok, _mode, _target, _notes = self.engine_module.CopilotEngine._execute_step(engine, step, trace, None)

            self.assertFalse(ok)
            self.assertEqual(trace.outputs["action_contracts"][0]["failure_reason"], "TARGET_NOT_FOUND")

    def test_recovery_retry_records_state_snapshots_and_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            step = PlanStep(
                "step_state_retry",
                "Open Downloads",
                "click_node",
                target=ActionTarget(kind="ui_node", value="Downloads", filters={"label_contains": "Downloads"}),
            )
            trace = RunTrace(run_id="run_state_retry", task=task, plan=ExecutionPlan(task=task, steps=[step]))
            recovery = self.engine_module.CopilotEngine._record_failure_recovery(engine, trace, step, "TARGET_NOT_FOUND", {"failure_reason": "TARGET_NOT_FOUND"})

            retry_ok, _retry_note = self.engine_module.CopilotEngine._retry_contract(engine, trace, step, recovery, None, None)

            self.assertTrue(retry_ok)
            phases = [item["phase"] for item in trace.outputs["state_snapshots"]]
            self.assertIn("retry_1_before", phases)
            self.assertIn("retry_1_after", phases)
            self.assertTrue(trace.outputs["state_diffs"])

    def test_plan_drops_stale_observation_before_compiling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            engine.last_observation = ObservationGraph.from_raw(
                [{"id": "old", "label": "Old", "type": "button"}],
                metadata={"app_id": "explorer", "output_filename": "old.png"},
            )
            engine.last_state = self.engine_module.CopilotEngine._state_manager(engine).observe(graph=engine.last_observation)
            engine.last_state.timestamp = time.time() - 100

            plan = self.engine_module.CopilotEngine.plan_prompt(engine, "Open explorer and parse screen")

            self.assertIsNotNone(plan)
            self.assertIsNone(engine.last_observation)

    def test_runtime_records_task_state_timeline_and_replan_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            task = TaskSpec(prompt="click raw point", goal="click raw point")
            step = PlanStep("step_task_fail", "Click point", "click_point", parameters={"x": 10, "y": 20})
            plan = ExecutionPlan(task=task, steps=[step])
            trace = RunTrace(run_id="run_task_fail", task=task, plan=plan)
            engine.current_task_state = self.engine_module.CopilotEngine._task_state_manager(engine).initialize(plan)
            self.engine_module.CopilotEngine._record_task_state(engine, trace, "initialized", engine.current_task_state)

            ok, _mode, _target, notes = self.engine_module.CopilotEngine._execute_step(engine, step, trace, None)
            engine.current_task_state = self.engine_module.CopilotEngine._task_state_manager(engine).update_task_state_after_step(
                engine.current_task_state, step, ok=ok, notes=notes
            )
            self.engine_module.CopilotEngine._record_task_state(engine, trace, "after_step_task_fail", engine.current_task_state)
            trace.outputs.setdefault("task_replans", []).append(
                {
                    "step_id": step.step_id,
                    "reason": self.engine_module.CopilotEngine._classify_failure_reason(engine, notes, trace.outputs["action_contracts"][0]),
                    **self.engine_module.CopilotEngine._task_state_manager(engine).replan_from_current_state(plan, engine.current_task_state),
                }
            )

            self.assertFalse(ok)
            self.assertEqual(trace.outputs["task_state_timeline"][0]["phase"], "initialized")
            self.assertEqual(trace.outputs["task_state_timeline"][-1]["failed_steps"], ["step_task_fail"])
            self.assertEqual(trace.outputs["task_replans"][0]["reason"], "UNSAFE_COORDINATE")

    def test_completed_step_is_skipped_without_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            task = TaskSpec(prompt="open explorer", goal="open explorer")
            step = PlanStep("step_repeat", "Parse UI", "parse_ui")
            plan = ExecutionPlan(task=task, steps=[step])
            trace = RunTrace(run_id="run_repeat", task=task, plan=plan)
            engine.current_task_state = self.engine_module.CopilotEngine._task_state_manager(engine).initialize(plan)
            engine.current_task_state.completed_steps.append("step_repeat")

            should_skip = self.engine_module.CopilotEngine._task_state_manager(engine).prevent_repeating_completed_step(engine.current_task_state, "step_repeat")

            self.assertTrue(should_skip)

    def test_attempt_replan_records_old_and_new_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            plan = self.engine_module.CopilotEngine.plan_prompt(engine, "open explorer and open downloads")
            trace = RunTrace(run_id="run_replan", task=plan.task, plan=plan)
            engine.current_task_state = self.engine_module.CopilotEngine._task_state_manager(engine).initialize(plan)
            engine.last_observation = ObservationGraph.from_raw(
                [{"id": "downloads", "label": "Downloads", "type": "button", "semantic_role": "menu_item", "entity_type": "navigation_item", "region": "left_menu"}],
                metadata={"app_id": "explorer", "output_filename": "replan.png"},
            )
            engine.last_state = self.engine_module.CopilotEngine._state_manager(engine).observe(graph=engine.last_observation, last_action="after:click_node")
            engine.current_task_state.completed_steps.append("step_1")

            fragment = self.engine_module.CopilotEngine._attempt_replan(engine, trace, plan, "PLAN_DRIFT", "step_1")

            self.assertTrue(fragment)
            replacement = trace.outputs["plan_replacements"][0]
            self.assertEqual(replacement["reason"], "PLAN_DRIFT")
            self.assertEqual(replacement["old_plan_step_ids"][0], "step_1")
            self.assertTrue(replacement["new_plan_step_ids"])

    def test_attempt_replan_returns_none_without_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            plan = ExecutionPlan(task=TaskSpec(prompt="open", goal="open"), steps=[])
            trace = RunTrace(run_id="run_no_state_replan", task=plan.task, plan=plan)

            fragment = self.engine_module.CopilotEngine._attempt_replan(engine, trace, plan, "PLAN_DRIFT", "step_x")

            self.assertIsNone(fragment)

    def test_plan_replacement_preserves_completed_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            plan = self.engine_module.CopilotEngine.plan_prompt(engine, "open explorer and open downloads")
            trace = RunTrace(run_id="run_preserve", task=plan.task, plan=plan)
            engine.current_task_state = self.engine_module.CopilotEngine._task_state_manager(engine).initialize(plan)
            engine.current_task_state.completed_steps = ["step_1", "step_2"]
            engine.last_observation = ObservationGraph.from_raw(
                [{"id": "downloads", "label": "Downloads", "type": "button", "semantic_role": "menu_item", "entity_type": "navigation_item", "region": "left_menu"}],
                metadata={"app_id": "explorer", "output_filename": "preserve.png"},
            )
            engine.last_state = self.engine_module.CopilotEngine._state_manager(engine).observe(graph=engine.last_observation)

            fragment = self.engine_module.CopilotEngine._attempt_replan(engine, trace, plan, "NO_STATE_CHANGE", "step_2")

            self.assertTrue(fragment)
            self.assertNotIn("step_1", [step.step_id for step in fragment])
            self.assertNotIn("step_2", [step.step_id for step in fragment])

    def test_failed_replan_reason_is_stored_in_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            plan = self.engine_module.CopilotEngine.plan_prompt(engine, "open explorer and open downloads")
            trace = RunTrace(run_id="run_failed_replan", task=plan.task, plan=plan)
            engine.current_task_state = self.engine_module.CopilotEngine._task_state_manager(engine).initialize(plan)
            engine.current_task_state.failed_steps = ["step_2"]
            engine.last_observation = ObservationGraph.from_raw(
                [{"id": "downloads", "label": "Downloads", "type": "button", "semantic_role": "menu_item", "entity_type": "navigation_item", "region": "left_menu"}],
                metadata={"app_id": "explorer", "output_filename": "failed_replan.png"},
            )
            engine.last_state = self.engine_module.CopilotEngine._state_manager(engine).observe(graph=engine.last_observation)

            fragment = self.engine_module.CopilotEngine._attempt_replan(engine, trace, plan, "UNSAFE_COORDINATE", "step_2")

            self.assertTrue(fragment)
            self.assertEqual(trace.outputs["plan_replacements"][0]["reason"], "UNSAFE_COORDINATE")

    def test_execute_prompt_records_plan_replacement_on_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            trace = self.engine_module.CopilotEngine.execute_prompt(
                engine,
                "open explorer and open downloads",
                trust_mode=TrustMode.PLAN_AND_RISK_GATES,
                approval_callback=lambda _prompt, _payload: True,
                dry_run=False,
            )

            self.assertTrue("plan_replacements" in trace.outputs or trace.status.value == "success")

    def test_attempt_replan_prefers_repair_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            plan = self.engine_module.CopilotEngine.plan_prompt(engine, "open explorer and open downloads")
            trace = RunTrace(run_id="run_repair_first", task=plan.task, plan=plan)
            engine.current_task_state = self.engine_module.CopilotEngine._task_state_manager(engine).initialize(plan)
            engine.last_observation = ObservationGraph.from_raw(
                [{"id": "downloads", "label": "Downloads", "type": "button", "semantic_role": "menu_item", "entity_type": "navigation_item", "region": "left_menu"}],
                metadata={"app_id": "explorer", "output_filename": "repair_first.png"},
            )
            engine.last_state = self.engine_module.CopilotEngine._state_manager(engine).observe(graph=engine.last_observation)

            fragment = self.engine_module.CopilotEngine._attempt_replan(engine, trace, plan, "TARGET_NOT_FOUND", "step_2")

            self.assertTrue(fragment)
            self.assertEqual(trace.outputs["plan_replacements"][0]["planner_type"], "repair")

    def test_attempt_replan_falls_back_to_broad_replan_when_no_repair_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._build_engine(tmpdir)
            engine.bridge.current_app = "explorer"
            plan = self.engine_module.CopilotEngine.plan_prompt(engine, "open explorer and open downloads")
            trace = RunTrace(run_id="run_replan_fallback", task=plan.task, plan=plan)
            engine.current_task_state = self.engine_module.CopilotEngine._task_state_manager(engine).initialize(plan)
            engine.last_observation = ObservationGraph.from_raw(
                [{"id": "downloads", "label": "Downloads", "type": "button", "semantic_role": "menu_item", "entity_type": "navigation_item", "region": "left_menu"}],
                metadata={"app_id": "explorer", "output_filename": "replan_fallback.png"},
            )
            engine.last_state = self.engine_module.CopilotEngine._state_manager(engine).observe(graph=engine.last_observation)

            fragment = self.engine_module.CopilotEngine._attempt_replan(engine, trace, plan, "PLAN_DRIFT", "step_2")

            self.assertTrue(fragment)
            self.assertEqual(trace.outputs["plan_replacements"][0]["planner_type"], "replan")


if __name__ == "__main__":
    unittest.main()
