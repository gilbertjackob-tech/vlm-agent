from __future__ import annotations

import time
import unittest

from copilot.runtime.run_control import CancelLevel, RunRegistry
from copilot.schemas import ExecutionPlan, RunStatus, RunTrace, TaskSpec


class FakeEngine:
    def __init__(self) -> None:
        self.cancel_level = ""

    def request_cancel(self, level: str = "soft") -> None:
        self.cancel_level = level

    def execute_prompt(self, prompt, trust_mode=None, approval_callback=None, trace_callback=None, dry_run=False, voice_mode=None):
        task = TaskSpec(prompt=prompt, goal=prompt)
        trace = RunTrace(run_id="fake_trace", task=task, plan=ExecutionPlan(task=task, steps=[]), status=RunStatus.SUCCESS)
        if trace_callback:
            trace_callback({"phase": "test", "message": "started", "metadata": {}})
        trace.finished_at = time.time()
        return trace


class ApprovalEngine(FakeEngine):
    def execute_prompt(self, prompt, trust_mode=None, approval_callback=None, trace_callback=None, dry_run=False, voice_mode=None):
        approved = bool(approval_callback and approval_callback("Approve step", {"risk_level": "high"}))
        task = TaskSpec(prompt=prompt, goal=prompt)
        status = RunStatus.SUCCESS if approved else RunStatus.CANCELLED
        trace = RunTrace(run_id="approval_trace", task=task, plan=ExecutionPlan(task=task, steps=[]), status=status)
        trace.finished_at = time.time()
        return trace


class SlowEngine(FakeEngine):
    def execute_prompt(self, prompt, trust_mode=None, approval_callback=None, trace_callback=None, dry_run=False, voice_mode=None):
        time.sleep(0.2)
        return super().execute_prompt(prompt, trust_mode, approval_callback, trace_callback, dry_run, voice_mode)


class RunControlTests(unittest.TestCase):
    def test_registry_starts_and_records_run(self) -> None:
        engine = FakeEngine()
        registry = RunRegistry(lambda: engine)

        run_id = registry.start_task("observe", dry_run=True)
        deadline = time.time() + 2.0
        record = registry.get(run_id)
        while record and record["status"] in {"queued", "running", "waiting_approval"} and time.time() < deadline:
            time.sleep(0.01)
            record = registry.get(run_id)

        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["lifecycle"], "success")
        self.assertEqual(record["snapshot"]["status"], "success")
        self.assertEqual(record["trace"]["task"]["prompt"], "observe")
        self.assertTrue(record["events"])

    def test_registry_cancels_engine_with_level(self) -> None:
        engine = SlowEngine()
        registry = RunRegistry(lambda: engine)
        run_id = registry.start_task("observe", dry_run=True)

        ok = registry.cancel(run_id, CancelLevel.HARD)

        self.assertTrue(ok)
        self.assertEqual(engine.cancel_level, "hard")
        record = registry.get(run_id)
        self.assertEqual(record["lifecycle"], "cancelling")
        self.assertEqual(record["cancel"]["level"], "hard")

    def test_registry_blocks_run_until_approval_decision(self) -> None:
        engine = ApprovalEngine()
        registry = RunRegistry(lambda: engine)

        run_id = registry.start_task("approve me", dry_run=True, approval_timeout=2.0)
        deadline = time.time() + 2.0
        pending = []
        while time.time() < deadline:
            pending = registry.list_approvals(run_id, pending_only=True)
            if pending:
                break
            time.sleep(0.01)

        self.assertEqual(len(pending), 1)
        self.assertTrue(registry.decide_approval(pending[0]["approval_id"], True))

        record = registry.get(run_id)
        while record and record["status"] in {"queued", "running", "waiting_approval"} and time.time() < deadline:
            time.sleep(0.01)
            record = registry.get(run_id)

        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["lifecycle"], "success")
        self.assertEqual(record["approvals"][0]["status"], "approved")


if __name__ == "__main__":
    unittest.main()
