from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from copilot.adapters.browser import BrowserAdapter
from copilot.memory.store import MemoryStore
from copilot.planner.compiler import PromptCompiler
from copilot.schemas import ObservationGraph, ObservationNode, TaskSpec, TrustMode


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

EXPLORER_ENV = {
    "windows": {
        "active_window": {"title": "File Explorer"},
        "active_app_guess": "explorer",
    },
    "browser": {},
}

CHROME_ENV = {
    "windows": {
        "active_window": {"title": "Google Chrome"},
        "active_app_guess": "chrome",
    },
    "browser": {
        "available": True,
        "cdp_available": True,
    },
}

UNKNOWN_ENV = {
    "windows": {
        "active_window": {"title": "Some App"},
        "active_app_guess": "",
    },
    "browser": {},
}


def load_browser_fixture(name: str = "browser_dom_chrome.json") -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class PromptCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.memory = MemoryStore(base_dir=self.tmp.name)
        self.compiler = PromptCompiler(self.memory)

    def _compile(self, prompt: str, environment: dict | None = None):
        return self.compiler.compile(
            task=TaskSpec(prompt=prompt, goal=prompt, trust_mode=TrustMode.PLAN_AND_RISK_GATES),
            observation=None,
            environment=environment or EXPLORER_ENV,
        )

    def test_open_explorer_and_open_downloads_emits_verified_navigation_chain(self) -> None:
        plan = self._compile("open explorer and open downloads", environment=EXPLORER_ENV)
        action_types = [step.action_type for step in plan.steps]

        self.assertEqual(plan.source, "reasoner_compiler_v2")
        self.assertEqual(plan.required_apps, ["explorer"])
        self.assertEqual(action_types, ["route_window", "confirm_focus", "press_key", "open_explorer_location", "wait_for", "verify_scene", "scene_diff"])
        self.assertNotIn("click_point", action_types)

        stable_view_step = plan.steps[2]
        self.assertEqual(stable_view_step.parameters["shortcut_id"], "explorer_stable_view")
        self.assertEqual(stable_view_step.parameters["keys"], ["ctrl", "shift", "6"])

        location_step = plan.steps[3]
        self.assertEqual(location_step.parameters["location"], "downloads")
        self.assertEqual(location_step.parameters["ranking"]["top_candidate_score"], 1.0)
        self.assertEqual(location_step.parameters["ranking"]["runner_up_score"], 0.0)
        self.assertEqual(location_step.parameters["ranking"]["score_gap"], 1.0)
        self.assertTrue(location_step.parameters["ranking"]["duplicate_disambiguation_used"])

        self.assertEqual(plan.steps[5].parameters["expected_labels"], ["downloads"])
        self.assertEqual(plan.steps[6].parameters["expected_scene"], "Explorer should show the downloads location.")

    def test_identify_videos_on_desktop_adds_classification_checkpoint(self) -> None:
        plan = self._compile("identify which files are videos on desktop", environment=EXPLORER_ENV)
        action_types = [step.action_type for step in plan.steps]

        self.assertEqual(
            action_types,
            ["route_window", "confirm_focus", "press_key", "open_explorer_location", "wait_for", "verify_scene", "scene_diff", "parse_ui", "checkpoint"],
        )
        self.assertNotIn("click_point", action_types)

        checkpoint = plan.steps[-1]
        self.assertEqual(checkpoint.parameters["filters"]["app_id"], "explorer")
        self.assertEqual(checkpoint.parameters["filters"]["region"], "main_page")
        self.assertEqual(checkpoint.parameters["filters"]["entity_type"], "video")
        self.assertEqual(checkpoint.parameters["expected_labels"], ["video"])
        self.assertEqual(checkpoint.parameters["recovery_hint"], "Reparse Explorer and teach ambiguous rows if needed.")

    def test_learning_prompt_uses_learning_session_not_random_click_exploration(self) -> None:
        plan = self._compile("open explorer and learn the current UI by hovering", environment=EXPLORER_ENV)
        action_types = [step.action_type for step in plan.steps]

        self.assertIn("learning_session", action_types)
        self.assertNotIn("hover_probe", action_types)
        self.assertNotIn("explore_safe", action_types)
        self.assertNotIn("click_point", action_types)
        session_step = plan.steps[action_types.index("learning_session")]
        self.assertEqual(session_step.parameters["app_id"], "explorer")
        self.assertEqual(session_step.parameters["max_nodes"], 8)
        self.assertTrue(session_step.parameters["filters"]["exclude_destructive"])

    def test_interaction_learning_prompt_uses_rewarded_click_learning(self) -> None:
        plan = self._compile("open explorer and learn what clicking safe folders opens", environment=EXPLORER_ENV)
        action_types = [step.action_type for step in plan.steps]

        self.assertIn("interaction_learning", action_types)
        self.assertNotIn("explore_safe", action_types)
        self.assertNotIn("click_point", action_types)
        step = plan.steps[action_types.index("interaction_learning")]
        self.assertEqual(step.parameters["app_id"], "explorer")
        self.assertEqual(step.parameters["max_actions"], 5)
        self.assertTrue(step.parameters["recover_after_each"])
        self.assertTrue(step.parameters["filters"]["exclude_destructive"])

    def test_current_ui_learning_uses_active_app_without_opening_explorer(self) -> None:
        plan = self._compile("learn the current UI by hovering", environment=CHROME_ENV)
        action_types = [step.action_type for step in plan.steps]

        self.assertEqual(action_types, ["learning_session"])
        self.assertEqual(plan.steps[0].parameters["app_id"], "chrome")
        self.assertEqual(plan.required_apps, [])

    def test_current_ui_hover_learning_can_map_unknown_current_window(self) -> None:
        plan = self._compile("learn the current UI by hovering", environment=UNKNOWN_ENV)
        action_types = [step.action_type for step in plan.steps]

        self.assertEqual(action_types, ["learning_session"])
        self.assertEqual(plan.steps[0].parameters["app_id"], "current_window")

    def test_open_file_target_ignores_application_route_words(self) -> None:
        plan = self._compile("open explorer and open awake.py", environment=EXPLORER_ENV)
        click_steps = [step for step in plan.steps if step.action_type == "click_node"]

        self.assertTrue(click_steps)
        self.assertEqual(click_steps[-1].parameters["filters"]["label_contains"], "awake.py")
        self.assertEqual(click_steps[-1].parameters["click_count"], 2)

    def test_chrome_search_uses_ctrl_l_instead_of_page_click(self) -> None:
        snapshot = load_browser_fixture()
        self.compiler.browser.snapshot_dom = lambda: snapshot

        plan = self._compile("open chrome and search youtube for lo-fi coding mix", environment=CHROME_ENV)
        action_types = [step.action_type for step in plan.steps]

        self.assertEqual(plan.required_apps, ["chrome"])
        self.assertEqual(
            action_types,
            [
                "route_window",
                "confirm_focus",
                "press_key",
                "type_text",
                "verify_scene",
                "press_key",
                "wait_for",
                "verify_scene",
                "scene_diff",
            ],
        )
        self.assertNotIn("click_point", action_types)
        self.assertNotIn("click_node", action_types[:4])

        focus_step = plan.steps[2]
        self.assertEqual(focus_step.parameters["keys"], ["ctrl", "l"])
        self.assertTrue(focus_step.parameters["hotkey"])
        self.assertEqual(focus_step.parameters["expected_app"], "chrome")

        type_step = plan.steps[3]
        self.assertEqual(type_step.parameters["text"], "youtube lo-fi coding mix")
        self.assertEqual(type_step.parameters["selector"], "")
        self.assertFalse(type_step.parameters["clear_first"])
        self.assertEqual(type_step.parameters["focused_target"], "Chrome address bar")
        self.assertEqual(type_step.parameters["deterministic_focus"], "ctrl+l")

    def test_chrome_results_plan_verifies_transition_after_key_submission(self) -> None:
        snapshot = load_browser_fixture()
        self.compiler.browser.snapshot_dom = lambda: snapshot

        plan = self._compile("search chrome for release notes and verify results page", environment=CHROME_ENV)
        action_types = [step.action_type for step in plan.steps]
        press_index = action_types.index("press_key", action_types.index("type_text") + 1)

        self.assertEqual(action_types[press_index + 1 : press_index + 4], ["wait_for", "verify_scene", "scene_diff"])
        self.assertEqual(plan.steps[press_index + 2].parameters["expected_labels"], ["release notes", "release", "notes", "search"])
        self.assertEqual(plan.steps[-1].parameters["expected_scene"], "Chrome should run the requested search and show a results page.")

    def test_chrome_new_tab_intent_uses_shortcut(self) -> None:
        plan = self._compile("open chrome and open a new tab", environment=CHROME_ENV)
        shortcut_steps = [step for step in plan.steps if step.action_type == "press_key" and step.parameters.get("shortcut_id") == "new_tab"]

        self.assertEqual(plan.required_apps, ["chrome"])
        self.assertEqual(len(shortcut_steps), 1)
        self.assertEqual(shortcut_steps[0].parameters["keys"], ["ctrl", "t"])
        self.assertTrue(shortcut_steps[0].parameters["hotkey"])
        self.assertNotIn("click_node", [step.action_type for step in plan.steps])

    def test_chrome_downloads_intent_uses_shortcut_without_routing_explorer(self) -> None:
        plan = self._compile("open chrome downloads page", environment=CHROME_ENV)
        shortcut_steps = [step for step in plan.steps if step.action_type == "press_key" and step.parameters.get("shortcut_id") == "downloads"]

        self.assertEqual(plan.required_apps, ["chrome"])
        self.assertEqual(len(shortcut_steps), 1)
        self.assertEqual(shortcut_steps[0].parameters["keys"], ["ctrl", "j"])
        self.assertFalse(any(step.parameters.get("app_id") == "explorer" for step in plan.steps))

    def test_explorer_new_folder_intent_uses_shortcut(self) -> None:
        plan = self._compile("open explorer and create a new folder", environment=EXPLORER_ENV)
        shortcut_steps = [step for step in plan.steps if step.action_type == "press_key" and step.parameters.get("shortcut_id") == "new_folder"]

        self.assertEqual(plan.required_apps, ["explorer"])
        self.assertEqual(len(shortcut_steps), 1)
        self.assertEqual(shortcut_steps[0].parameters["keys"], ["ctrl", "shift", "n"])
        self.assertTrue(shortcut_steps[0].requires_approval)

    def test_explorer_route_always_sets_stable_details_view(self) -> None:
        plan = self._compile("open explorer and parse screen", environment=EXPLORER_ENV)
        stable_view_steps = [step for step in plan.steps if step.action_type == "press_key" and step.parameters.get("shortcut_id") == "explorer_stable_view"]

        self.assertEqual(len(stable_view_steps), 1)
        self.assertEqual(stable_view_steps[0].parameters["keys"], ["ctrl", "shift", "6"])
        self.assertLess([step.action_type for step in plan.steps].index("confirm_focus"), plan.steps.index(stable_view_steps[0]))

    def test_explorer_preview_and_details_pane_intents_use_shortcuts(self) -> None:
        preview_plan = self._compile("open explorer and toggle preview pane", environment=EXPLORER_ENV)
        details_plan = self._compile("open explorer and toggle details pane", environment=EXPLORER_ENV)
        preview_step = next(step for step in preview_plan.steps if step.parameters.get("shortcut_id") == "preview_pane")
        details_step = next(step for step in details_plan.steps if step.parameters.get("shortcut_id") == "details_pane")

        self.assertEqual(preview_step.parameters["keys"], ["alt", "p"])
        self.assertEqual(details_step.parameters["keys"], ["alt", "shift", "p"])

    def test_explorer_view_intent_uses_requested_view_shortcut(self) -> None:
        plan = self._compile("open explorer and use details view", environment=EXPLORER_ENV)
        explicit_view_steps = [step for step in plan.steps if step.parameters.get("shortcut_id") == "view_stable_details"]

        self.assertEqual(len(explicit_view_steps), 1)
        self.assertEqual(explicit_view_steps[0].parameters["keys"], ["ctrl", "shift", "6"])

    def test_explorer_search_query_uses_f3_shortcut_to_focus_search(self) -> None:
        plan = self._compile("open explorer and search files for python", environment=EXPLORER_ENV)
        action_types = [step.action_type for step in plan.steps]
        search_step = next(step for step in plan.steps if step.action_type == "press_key" and step.parameters.get("shortcut_id") == "search")

        self.assertEqual(search_step.parameters["keys"], ["f3"])
        self.assertNotIn("click_node", action_types[: action_types.index("type_text")])

    def test_explorer_modifier_click_intent_uses_modified_click_node(self) -> None:
        plan = self._compile("open explorer and ctrl click awake.py", environment=EXPLORER_ENV)
        step = next(step for step in plan.steps if step.action_type == "modified_click_node")

        self.assertEqual(step.parameters["modifiers"], ["ctrl"])
        self.assertEqual(step.parameters["filters"]["label_contains"], "awake.py")

    def test_memory_bias_enriches_explorer_filters(self) -> None:
        self.memory.teach_node(
            ObservationNode(
                node_id="downloads_nav",
                label="Downloads",
                node_type="button",
                semantic_role="menu_item",
                entity_type="navigation_item",
                app_id="explorer",
                region="left_menu",
                affordances=["navigate"],
                visual_id="downloads_visual",
            ),
            label="Downloads",
            concepts=["navigation", "sidebar"],
            app_identity="explorer",
            entity_type="navigation_item",
            affordances=["navigate"],
        )
        self.memory.remember_negative_example(
            ObservationNode(
                node_id="delete_nav",
                label="Delete",
                node_type="button",
                semantic_role="menu_item",
                entity_type="navigation_item",
                app_id="explorer",
                region="left_menu",
                affordances=["navigate"],
                visual_id="danger_visual",
            ),
            note="Unsafe destructive target",
        )

        plan = self._compile("open explorer and open downloads", environment=EXPLORER_ENV)
        location_step = next(step for step in plan.steps if step.action_type == "open_explorer_location")
        ranking = location_step.parameters["ranking"]

        self.assertEqual(ranking["selected_label"], "Explorer downloads")
        self.assertGreater(ranking["top_candidate_score"], ranking["runner_up_score"])
        self.assertEqual(ranking["score_gap"], 1.0)

    def test_rewarded_interaction_graph_replays_matching_prompt(self) -> None:
        before = ObservationGraph.from_raw(
            [
                {
                    "id": "downloads",
                    "label": "Downloads",
                    "type": "button",
                    "semantic_role": "menu_item",
                    "entity_type": "navigation_item",
                    "app_id": "explorer",
                    "region": "left_menu",
                    "visual_id": "visual_downloads",
                }
            ],
            metadata={"app_id": "explorer"},
        )
        after = ObservationGraph.from_raw(
            [{"id": "contents", "label": "Downloads folder contents", "type": "container"}],
            metadata={"app_id": "explorer"},
        )
        self.memory.record_interaction_outcome(
            before=before,
            after=after,
            node=before.flatten()[0],
            action_type="click_1",
            reward=1.0,
            outcome="opened_or_navigated",
            app_id="explorer",
            recovery="back",
        )

        plan = self._compile("open downloads", environment=EXPLORER_ENV)
        action_types = [step.action_type for step in plan.steps]

        self.assertEqual(plan.source, "interaction_graph_replay")
        self.assertIn("replay_interaction", action_types)
        replay_step = plan.steps[action_types.index("replay_interaction")]
        self.assertEqual(replay_step.parameters["filters"]["label_contains"], "Downloads")
        self.assertEqual(replay_step.parameters["filters"]["app_id"], "explorer")

    def test_rewarded_interaction_graph_compiles_multistep_path(self) -> None:
        home = ObservationGraph.from_raw(
            [{"id": "projects", "label": "Projects", "type": "button", "semantic_role": "menu_item", "entity_type": "folder", "app_id": "explorer", "region": "main_page"}],
            metadata={"app_id": "explorer"},
        )
        projects = ObservationGraph.from_raw(
            [{"id": "reports", "label": "Reports", "type": "button", "semantic_role": "list_row", "entity_type": "folder", "app_id": "explorer", "region": "main_page"}],
            metadata={"app_id": "explorer"},
        )
        reports = ObservationGraph.from_raw(
            [{"id": "final", "label": "Final Report", "type": "container", "semantic_role": "main_page", "app_id": "explorer", "region": "main_page"}],
            metadata={"app_id": "explorer"},
        )
        self.memory.record_interaction_outcome(home, projects, home.flatten()[0], "click_1", 1.0, "opened_projects", app_id="explorer")
        self.memory.record_interaction_outcome(projects, reports, projects.flatten()[0], "click_2", 1.0, "opened_reports", app_id="explorer")

        plan = self.compiler.compile(
            task=TaskSpec(prompt="open final report", goal="open final report", trust_mode=TrustMode.PLAN_AND_RISK_GATES),
            observation=home,
            environment=EXPLORER_ENV,
        )
        action_types = [step.action_type for step in plan.steps]

        self.assertEqual(plan.source, "interaction_graph_path")
        self.assertEqual(action_types.count("replay_interaction"), 2)
        replay_labels = [step.target.value for step in plan.steps if step.action_type == "replay_interaction"]
        self.assertEqual(replay_labels, ["Projects", "Reports"])

    def test_browser_adapter_ranks_omnibox_and_link_selectors(self) -> None:
        adapter = BrowserAdapter()
        snapshot = load_browser_fixture()

        omnibox_ranked = adapter.rank_selector_candidates(snapshot, purpose="omnibox", query="lo-fi")
        link_ranked = adapter.rank_selector_candidates(snapshot, purpose="link", query="lo-fi")
        modal_ranked = adapter.rank_selector_candidates(snapshot, purpose="modal_dismiss", blocked_terms=["delete", "payment", "account"])

        self.assertEqual(omnibox_ranked[0], "#omnibox")
        self.assertEqual(link_ranked[0], "a[href='https://www.youtube.com/watch?v=abc']")
        self.assertEqual(modal_ranked[0], "button[aria-label='Close']")


if __name__ == "__main__":
    unittest.main()
