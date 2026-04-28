from __future__ import annotations

import tempfile
import unittest

from copilot.memory.store import MemoryStore
from copilot.schemas import ExecutionPlan, PlanStep, RunStatus, RunTrace, TaskSpec


class SkillManifestTests(unittest.TestCase):
    def test_export_import_skill_manifest_round_trips_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source = MemoryStore(base_dir=source_dir)
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            plan = ExecutionPlan(task=task, steps=[PlanStep("s1", "Route", "route_window")])
            workflow = source.save_plan_as_workflow(task, plan, name="Open Downloads")

            manifest = source.build_skill_manifest([workflow["workflow_id"]])
            target = MemoryStore(base_dir=target_dir)
            result = target.import_skill_manifest(manifest)

            self.assertEqual(result["imported"], 1)
            self.assertEqual(target.list_workflows()[0]["workflow_id"], workflow["workflow_id"])

    def test_workflow_trace_wrong_click_blocks_trusted_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(base_dir=tmpdir)
            task = TaskSpec(prompt="open downloads", goal="open downloads")
            plan = ExecutionPlan(task=task, steps=[PlanStep("s1", "Route", "route_window")])
            trace = RunTrace(run_id="run_1", task=task, plan=plan, status=RunStatus.SUCCESS)
            trace.outputs["wrong_click_count"] = 1

            workflow = memory.record_workflow_trace(task, plan, trace)
            workflow["success_count"] = 5
            memory.approve_workflow(workflow["workflow_id"])

            self.assertFalse(memory.promote_workflow(workflow["workflow_id"], "trusted"))
            self.assertIn("wrong_click_count_must_be_zero", workflow["promotion_blockers"])


if __name__ == "__main__":
    unittest.main()
