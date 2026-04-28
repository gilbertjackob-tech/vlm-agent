from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from copilot.runtime.daemon import CopilotDaemonHandler
from copilot.runtime.run_control import RunRegistry
from copilot.schemas import ExecutionPlan, ObservationGraph, PlanStep, RunStatus, RunTrace, TaskSpec


class FakeEngine:
    def __init__(self) -> None:
        self.cancelled = ""
        self.approved = ""
        self.promoted = {}
        self.replayed = {}
        self.high_risk_allowed = []

    def request_cancel(self, level: str = "soft") -> None:
        self.cancelled = level

    def get_workflows(self):
        return [{"workflow_id": "wf_1", "promotion_state": "verified"}]

    def get_skill_capsules(self):
        return [{"workflow_id": "wf_1", "capsule_type": "skill_capsule", "approval_status": "pending"}]

    def approve_workflow(self, workflow_id: str) -> bool:
        self.approved = workflow_id
        return workflow_id == "wf_1"

    def promote_workflow(self, workflow_id: str, promotion_state: str = "trusted") -> bool:
        self.promoted[workflow_id] = promotion_state
        return workflow_id == "wf_1" and promotion_state == "trusted"

    def record_skill_replay(self, workflow_id: str, **kwargs):
        self.replayed[workflow_id] = kwargs
        if workflow_id != "wf_1":
            return None
        return {"workflow_id": workflow_id, "capsule_type": "skill_capsule", "replay_count": 1, "success_rate": 1.0}

    def allow_high_risk_for_app(self, app_id: str) -> bool:
        self.high_risk_allowed.append(app_id)
        return True

    def plan_prompt(self, prompt, trust_mode=None):
        task = TaskSpec(prompt=prompt, goal=prompt)
        return ExecutionPlan(task=task, steps=[PlanStep("s1", "Click Search", "click_node")], summary="Preview")

    def execute_prompt(self, prompt, **kwargs):
        trace_callback = kwargs.get("trace_callback")
        if trace_callback:
            trace_callback({"phase": "step", "message": "Executing fake step.", "metadata": {"step_id": "fake_step"}})
        approval_callback = kwargs.get("approval_callback")
        if prompt == "needs approval" and approval_callback:
            approved = approval_callback("Approve risky step", {"risk_level": "high", "app_id": "chrome", "choices": ["allow_once", "always_allow_app", "cancel"]})
            status = RunStatus.SUCCESS if approved else RunStatus.CANCELLED
        else:
            status = RunStatus.SUCCESS
        task = TaskSpec(prompt=prompt, goal=prompt)
        trace = RunTrace(run_id="daemon_trace", task=task, plan=ExecutionPlan(task=task, steps=[]), status=status)
        graph = ObservationGraph.from_raw(
            [
                {
                    "id": "search",
                    "label": "Search",
                    "type": "text_field",
                    "semantic_role": "text_field",
                    "box": {"x": 10, "y": 20, "width": 100, "height": 30},
                    "affordances": ["focus"],
                }
            ]
        )
        trace.outputs["last_observation"] = graph.to_dict()
        trace.finished_at = time.time()
        return trace


class DaemonTests(unittest.TestCase):
    def _server(self):
        engine = FakeEngine()
        factory = lambda: engine

        class Handler(CopilotDaemonHandler):
            registry = RunRegistry(factory)
            engine_factory = staticmethod(factory)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, engine

    def test_post_task_and_get_run(self) -> None:
        server, _engine = self._server()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            request = urllib.request.Request(
                f"{base}/tasks",
                data=json.dumps({"prompt": "observe", "dry_run": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            created = json.loads(urllib.request.urlopen(request, timeout=2).read().decode("utf-8"))
            run_id = created["run_id"]
            deadline = time.time() + 2.0
            record = {}
            while time.time() < deadline:
                record = json.loads(urllib.request.urlopen(f"{base}/runs/{run_id}", timeout=2).read().decode("utf-8"))
                if record["status"] not in {"queued", "running", "waiting_approval", "cancelling"}:
                    break
                time.sleep(0.01)

            self.assertEqual(record["status"], "success")
            self.assertEqual(record["lifecycle"], "success")
            self.assertEqual(record["snapshot"]["status"], "success")
            self.assertEqual(record["trace"]["task"]["prompt"], "observe")
        finally:
            server.shutdown()
            server.server_close()

    def test_run_stream_replays_events_and_terminal_done(self) -> None:
        server, _engine = self._server()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            request = urllib.request.Request(
                f"{base}/tasks",
                data=json.dumps({"prompt": "observe", "dry_run": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            run_id = json.loads(urllib.request.urlopen(request, timeout=2).read().decode("utf-8"))["run_id"]
            deadline = time.time() + 2.0
            while time.time() < deadline:
                record = json.loads(urllib.request.urlopen(f"{base}/runs/{run_id}", timeout=2).read().decode("utf-8"))
                if record["lifecycle"] == "success":
                    break
                time.sleep(0.01)

            stream = urllib.request.urlopen(f"{base}/runs/{run_id}/stream", timeout=2).read().decode("utf-8")
            self.assertIn('"phase": "step"', stream)
            self.assertIn('"phase": "done"', stream)
        finally:
            server.shutdown()
            server.server_close()

    def test_cancel_endpoint_passes_level(self) -> None:
        server, engine = self._server()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            request = urllib.request.Request(
                f"{base}/tasks",
                data=json.dumps({"prompt": "observe", "dry_run": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            run_id = json.loads(urllib.request.urlopen(request, timeout=2).read().decode("utf-8"))["run_id"]
            cancel = urllib.request.Request(
                f"{base}/runs/{run_id}/cancel",
                data=json.dumps({"level": "hard"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            result = json.loads(urllib.request.urlopen(cancel, timeout=2).read().decode("utf-8"))

            self.assertTrue(result["ok"])
            self.assertEqual(engine.cancelled, "hard")
        finally:
            server.shutdown()
            server.server_close()

    def test_plan_preview_and_xray_endpoints(self) -> None:
        server, _engine = self._server()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            preview_request = urllib.request.Request(
                f"{base}/plans",
                data=json.dumps({"prompt": "observe"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            preview = json.loads(urllib.request.urlopen(preview_request, timeout=2).read().decode("utf-8"))
            self.assertEqual(preview["plan"]["steps"][0]["step_id"], "s1")

            run_request = urllib.request.Request(
                f"{base}/tasks",
                data=json.dumps({"prompt": "observe", "dry_run": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            run_id = json.loads(urllib.request.urlopen(run_request, timeout=2).read().decode("utf-8"))["run_id"]
            deadline = time.time() + 2.0
            while time.time() < deadline:
                record = json.loads(urllib.request.urlopen(f"{base}/runs/{run_id}", timeout=2).read().decode("utf-8"))
                if record["status"] == "success":
                    break
                time.sleep(0.01)
            xray = json.loads(urllib.request.urlopen(f"{base}/runs/{run_id}/xray", timeout=2).read().decode("utf-8"))
            self.assertEqual(xray["nodes"][0]["label"], "Search")
        finally:
            server.shutdown()
            server.server_close()

    def test_skills_endpoint_approves_and_promotes_workflow(self) -> None:
        server, engine = self._server()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            approve = urllib.request.Request(
                f"{base}/skills",
                data=json.dumps({"workflow_id": "wf_1", "action": "approve"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            approve_result = json.loads(urllib.request.urlopen(approve, timeout=2).read().decode("utf-8"))
            promote = urllib.request.Request(
                f"{base}/skills",
                data=json.dumps({"workflow_id": "wf_1", "action": "promote", "promotion_state": "trusted"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            promote_result = json.loads(urllib.request.urlopen(promote, timeout=2).read().decode("utf-8"))

            self.assertTrue(approve_result["ok"])
            self.assertTrue(promote_result["ok"])
            self.assertEqual(engine.approved, "wf_1")
            self.assertEqual(engine.promoted["wf_1"], "trusted")
        finally:
            server.shutdown()
            server.server_close()

    def test_skills_endpoint_lists_capsules_and_records_replay_score(self) -> None:
        server, engine = self._server()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            listed = json.loads(urllib.request.urlopen(f"{base}/skills", timeout=2).read().decode("utf-8"))
            self.assertEqual(listed["skills"][0]["capsule_type"], "skill_capsule")

            replay = urllib.request.Request(
                f"{base}/skills",
                data=json.dumps({"workflow_id": "wf_1", "action": "record_replay", "success": True, "variant_count": 2}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            replay_result = json.loads(urllib.request.urlopen(replay, timeout=2).read().decode("utf-8"))

            self.assertTrue(replay_result["ok"])
            self.assertTrue(engine.replayed["wf_1"]["success"])
            self.assertEqual(engine.replayed["wf_1"]["variant_count"], 2)
        finally:
            server.shutdown()
            server.server_close()

    def test_approval_endpoint_unblocks_waiting_run(self) -> None:
        server, _engine = self._server()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            request = urllib.request.Request(
                f"{base}/tasks",
                data=json.dumps({"prompt": "needs approval", "dry_run": True, "approval_timeout": 2.0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            run_id = json.loads(urllib.request.urlopen(request, timeout=2).read().decode("utf-8"))["run_id"]
            deadline = time.time() + 2.0
            pending = []
            while time.time() < deadline:
                pending = json.loads(urllib.request.urlopen(f"{base}/approvals?pending=true", timeout=2).read().decode("utf-8"))["approvals"]
                if pending:
                    break
                time.sleep(0.01)

            self.assertEqual(pending[0]["run_id"], run_id)
            approval = urllib.request.Request(
                f"{base}/approvals/{pending[0]['approval_id']}",
                data=json.dumps({"approved": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            result = json.loads(urllib.request.urlopen(approval, timeout=2).read().decode("utf-8"))
            self.assertTrue(result["ok"])

            record = {}
            while time.time() < deadline:
                record = json.loads(urllib.request.urlopen(f"{base}/runs/{run_id}", timeout=2).read().decode("utf-8"))
                if record["status"] not in {"queued", "running", "waiting_approval", "cancelling"}:
                    break
                time.sleep(0.01)
            self.assertEqual(record["status"], "success")
        finally:
            server.shutdown()
            server.server_close()

    def test_approval_endpoint_can_always_allow_app(self) -> None:
        server, engine = self._server()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            request = urllib.request.Request(
                f"{base}/tasks",
                data=json.dumps({"prompt": "needs approval", "dry_run": True, "approval_timeout": 2.0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            run_id = json.loads(urllib.request.urlopen(request, timeout=2).read().decode("utf-8"))["run_id"]
            deadline = time.time() + 2.0
            pending = []
            while time.time() < deadline:
                pending = json.loads(urllib.request.urlopen(f"{base}/approvals?pending=true", timeout=2).read().decode("utf-8"))["approvals"]
                if pending:
                    break
                time.sleep(0.01)

            approval = urllib.request.Request(
                f"{base}/approvals/{pending[0]['approval_id']}",
                data=json.dumps({"approved": True, "decision": "always_allow_app"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            result = json.loads(urllib.request.urlopen(approval, timeout=2).read().decode("utf-8"))

            self.assertTrue(result["ok"])
            self.assertEqual(engine.high_risk_allowed, ["chrome"])
            self.assertEqual(run_id[:7], "daemon_")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
