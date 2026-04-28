from __future__ import annotations

import tempfile
import unittest

from copilot.memory.store import MemoryStore
from copilot.schemas import ActionTarget, ExecutionPlan, ObservationGraph, PlanStep, RunStatus, RunTrace, TaskSpec


class MemoryLearningTests(unittest.TestCase):
    def test_observation_graph_populates_label_visual_concept_and_control_memory(self) -> None:
        raw_graph = [
            {
                "id": "row_demo",
                "label": "Row: demo.mp4",
                "type": "container",
                "semantic_role": "list_row",
                "entity_type": "video",
                "app_id": "explorer",
                "region": "main_page",
                "visual_id": "visual_demo",
                "affordances": ["open", "inspect"],
                "box": {"x": 10, "y": 20, "width": 300, "height": 40},
                "center": {"x": 160, "y": 40},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            graph = ObservationGraph.from_raw(raw_graph, metadata={"app_id": "explorer"})

            memory.remember_observation_graph(graph)

            self.assertIn("row: demo.mp4", memory.semantic_memory["labels"])
            self.assertIn("visual_demo", memory.semantic_memory["visuals"])
            self.assertIn("video", memory.semantic_memory["concepts"])
            self.assertTrue(memory.semantic_memory["controls"])
            self.assertIn("explorer", memory.semantic_memory["entities"])

    def test_hover_feedback_records_tooltip_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            graph = ObservationGraph.from_raw(
                [
                    {
                        "id": "nav_downloads",
                        "label": "[ICON]",
                        "type": "icon",
                        "semantic_role": "menu_item",
                        "entity_type": "navigation_item",
                        "app_id": "explorer",
                        "region": "left_menu",
                        "visual_id": "visual_downloads",
                        "affordances": ["click", "navigate"],
                        "box": {"x": 10, "y": 20, "width": 40, "height": 40},
                        "center": {"x": 30, "y": 40},
                    }
                ],
                metadata={"app_id": "explorer"},
            )
            node = graph.flatten()[0]

            memory.remember_hover_feedback(node, ["Downloads", "Opens your downloads folder"], app_id="explorer")

            self.assertEqual(memory.semantic_memory["hover_feedback"][0]["feedback_labels"][0], "Downloads")
            self.assertIn("downloads", memory.semantic_memory["labels"])
            self.assertIn("hover_feedback", memory.semantic_memory["concepts"])
            self.assertIn("visual_downloads", memory.semantic_memory["visuals"])

    def test_review_queue_accept_correct_and_unsafe_update_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            graph = ObservationGraph.from_raw(
                [
                    {
                        "id": "ambiguous_icon",
                        "label": "[ICON]",
                        "type": "icon",
                        "semantic_role": "menu_item",
                        "entity_type": "",
                        "app_id": "explorer",
                        "region": "left_menu",
                        "visual_id": "visual_ambiguous",
                        "visual_ids": ["visual_ambiguous", "visual_child"],
                        "box": {"x": 10, "y": 20, "width": 40, "height": 40},
                        "center": {"x": 30, "y": 40},
                    }
                ],
                metadata={"app_id": "explorer"},
            )
            node = graph.flatten()[0]

            accepted = memory.enqueue_review_item(node, "low_confidence", ["Downloads"], confidence=0.4, app_id="explorer")
            corrected = memory.enqueue_review_item(node, "wrong_label", [], confidence=0.3, app_id="explorer")
            unsafe = memory.enqueue_review_item(node, "risky", [], confidence=0.2, app_id="explorer")

            self.assertTrue(memory.resolve_review_item(accepted["review_id"], "accepted", label="Downloads", concepts=["reviewed"], entity_type="navigation_item"))
            self.assertTrue(memory.resolve_review_item(corrected["review_id"], "corrected", label="Desktop", concepts=["reviewed"], entity_type="navigation_item"))
            self.assertTrue(memory.resolve_review_item(unsafe["review_id"], "unsafe", note="unsafe control"))

            self.assertIn("downloads", memory.semantic_memory["labels"])
            self.assertIn("desktop", memory.semantic_memory["labels"])
            self.assertIn("visual_child", memory.semantic_memory["visuals"])
            self.assertTrue(memory.semantic_memory["negative_examples"])
            self.assertEqual(len(memory.list_review_items()), 0)

    def test_interaction_outcome_builds_rewarded_action_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
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
                [
                    {
                        "id": "folder_view",
                        "label": "Downloads folder contents",
                        "type": "container",
                        "semantic_role": "main_page",
                        "entity_type": "folder_view",
                        "app_id": "explorer",
                        "region": "main_page",
                    }
                ],
                metadata={"app_id": "explorer"},
            )
            node = before.flatten()[0]

            edge = memory.record_interaction_outcome(
                before=before,
                after=after,
                node=node,
                action_type="click_1",
                reward=1.0,
                outcome="opened_or_navigated",
                app_id="explorer",
                recovery="back",
            )

            graph = memory.semantic_memory["interaction_graph"]
            self.assertEqual(edge["successes"], 1)
            self.assertTrue(graph["scene_nodes"])
            self.assertTrue(graph["control_nodes"])
            self.assertTrue(graph["edges"])
            self.assertEqual(memory.preferred_interaction_labels("explorer"), ["Downloads"])

    def test_interaction_graph_finds_multistep_path_and_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
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

            memory.record_interaction_outcome(home, projects, home.flatten()[0], "click_1", 1.0, "opened_projects", app_id="explorer")
            memory.record_interaction_outcome(projects, reports, projects.flatten()[0], "click_2", 1.0, "opened_reports", app_id="explorer")

            path = memory.find_interaction_path("open final report", app_id="explorer", start_scene=memory._scene_signature(home))
            dashboard = memory.interaction_dashboard()

            self.assertEqual([edge["target_label"] for edge in path], ["Projects", "Reports"])
            self.assertTrue(dashboard["ready_for_multistep"])
            self.assertEqual(dashboard["positive_edges"], 2)
            self.assertEqual(dashboard["success_rate"], 1.0)

    def test_operator_status_exposes_readiness_and_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)

            cold = memory.operator_status()
            self.assertEqual(cold["level"], "cold-start")
            self.assertFalse(cold["safe_to_replay"])
            self.assertIn("readiness_score", cold)

            before = ObservationGraph.from_raw(
                [{"id": "downloads", "label": "Downloads", "type": "button", "semantic_role": "menu_item", "entity_type": "navigation_item", "app_id": "explorer", "region": "left_menu", "visual_id": "visual_downloads"}],
                metadata={"app_id": "explorer"},
            )
            after = ObservationGraph.from_raw(
                [{"id": "folder_view", "label": "Downloads folder contents", "type": "container", "semantic_role": "main_page", "entity_type": "folder_view", "app_id": "explorer", "region": "main_page"}],
                metadata={"app_id": "explorer"},
            )
            memory.remember_observation_graph(before)
            memory.record_interaction_outcome(before, after, before.flatten()[0], "click_1", 1.0, "opened_or_navigated", app_id="explorer")

            learned = memory.operator_status()
            self.assertGreater(learned["readiness_score"], cold["readiness_score"])
            self.assertIn(learned["level"], {"learning", "supervised"})
            self.assertTrue(learned["learning"]["ready_for_replay"])

    def test_workflow_promotion_requires_approval_five_successes_and_zero_wrong_clicks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            plan = ExecutionPlan(task=task, steps=[PlanStep("s1", "Route", "route_window")])

            workflow = None
            for _ in range(5):
                workflow = memory.record_workflow_run(task, plan, success=True)

            self.assertEqual(workflow["promotion_state"], "verified")
            self.assertFalse(memory.promote_workflow(workflow["workflow_id"], "trusted"))
            self.assertIn("user_approval_required", workflow["promotion_blockers"])

            self.assertTrue(memory.approve_workflow(workflow["workflow_id"]))
            self.assertTrue(memory.promote_workflow(workflow["workflow_id"], "trusted"))
            self.assertEqual(workflow["promotion_state"], "trusted")

    def test_workflow_with_wrong_click_cannot_be_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            plan = ExecutionPlan(task=task, steps=[PlanStep("s1", "Route", "route_window")])
            workflow = None
            for _ in range(5):
                workflow = memory.record_workflow_run(task, plan, success=True)
            workflow["wrong_click_count"] = 1
            memory.approve_workflow(workflow["workflow_id"])

            self.assertFalse(memory.promote_workflow(workflow["workflow_id"], "trusted"))
            self.assertIn("wrong_click_count_must_be_zero", workflow["promotion_blockers"])

    def test_find_workflow_prefers_lower_latency_for_same_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            task = TaskSpec(prompt="open chrome downloads", goal="open chrome downloads")
            fast_plan = ExecutionPlan(task=task, steps=[PlanStep("s1", "Open downloads", "press_key", parameters={"shortcut_id": "downloads"})])
            slow_plan = ExecutionPlan(
                task=task,
                steps=[
                    PlanStep("s1", "Route", "route_window"),
                    PlanStep("s2", "Confirm", "confirm_focus"),
                    PlanStep("s3", "Open downloads", "press_key", parameters={"shortcut_id": "downloads"}),
                ],
            )

            memory.record_workflow_run(task, slow_plan, True, latency_seconds=3.5)
            memory.record_workflow_run(task, fast_plan, True, latency_seconds=1.2)

            selected = memory.find_workflow("open chrome downloads")
            self.assertEqual(selected["plan_signature"], memory._plan_signature(fast_plan))
            self.assertEqual(selected["avg_latency_seconds"], 1.2)

    def test_successful_trace_extracts_shortcut_and_fragment_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            task = TaskSpec(prompt="open chrome downloads", goal="open chrome downloads")
            plan = ExecutionPlan(
                task=task,
                steps=[
                    PlanStep("s1", "Route to chrome", "route_window", parameters={"app_id": "chrome"}),
                    PlanStep("s2", "Confirm chrome focus", "confirm_focus", parameters={"expected": "chrome"}),
                    PlanStep("s3", "Open Chrome downloads", "press_key", parameters={"keys": ["ctrl", "j"], "shortcut_id": "downloads"}),
                ],
            )
            trace = RunTrace(run_id="run_shortcut", task=task, plan=plan, status=RunStatus.SUCCESS)
            trace.finished_at = trace.started_at + 1.5

            memory.record_workflow_trace(task, plan, trace)
            workflows = memory.list_workflows()
            shortcut_workflows = [workflow for workflow in workflows if workflow.get("workflow_type") == "shortcut"]
            fragment_workflows = [workflow for workflow in workflows if workflow.get("workflow_type") == "fragment"]

            self.assertTrue(shortcut_workflows)
            self.assertEqual(shortcut_workflows[0]["objective_key"], "downloads")
            self.assertEqual(shortcut_workflows[0]["avg_latency_seconds"], 1.5)
            self.assertTrue(fragment_workflows)

    def test_skill_capsule_exposes_trigger_app_targets_and_approval_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            task = TaskSpec(prompt="search invoices", goal="search invoices")
            plan = ExecutionPlan(
                task=task,
                steps=[
                    PlanStep(
                        "s1",
                        "Click search",
                        "click_node",
                        target=ActionTarget(kind="label", value="Search", filters={"automation_id": "SearchBox"}),
                        parameters={"selector_candidates": ["#search", "input[aria-label='Search']"]},
                    )
                ],
                required_apps=["chrome"],
            )

            workflow = memory.record_workflow_run(task, plan, success=True)
            capsule = memory.build_skill_capsule(workflow)

            self.assertEqual(capsule["capsule_type"], "skill_capsule")
            self.assertEqual(capsule["skill_name"], "search invoices")
            self.assertEqual(capsule["trigger_phrase"], "search invoices")
            self.assertEqual(capsule["app"], "chrome")
            self.assertEqual(capsule["approval_status"], "pending")
            self.assertEqual(capsule["success_rate"], 1.0)
            self.assertEqual(capsule["selectors_uia_targets"][0]["selectors"][0], "#search")
            self.assertEqual(capsule["selectors_uia_targets"][0]["uia"]["automation_id"], "SearchBox")

    def test_skill_replay_scores_variants_and_updates_last_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            task = TaskSpec(prompt="open reports", goal="open reports")
            plan = ExecutionPlan(task=task, steps=[PlanStep("s1", "Open", "click_node")])
            workflow = memory.save_plan_as_workflow(task, plan, name="Open Reports")

            scored = memory.record_skill_replay(workflow["workflow_id"], success=True, variant_count=3, latency_seconds=0.8)

            self.assertIsNotNone(scored)
            assert scored is not None
            self.assertEqual(scored["replay_count"], 1)
            self.assertEqual(scored["variant_count"], 3)
            self.assertEqual(scored["success_rate"], 1.0)
            self.assertTrue(scored["last_verified"])

    def test_unapproved_skill_remains_verified_but_not_trusted_for_auto_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            plan = ExecutionPlan(task=task, steps=[PlanStep("s1", "Route", "route_window")])
            for _ in range(5):
                memory.record_workflow_run(task, plan, success=True)

            selected = memory.find_workflow("open downloads")

            self.assertEqual(selected["promotion_state"], "verified")
            self.assertFalse(selected["user_approved"])


if __name__ == "__main__":
    unittest.main()
