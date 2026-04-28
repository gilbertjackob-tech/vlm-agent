from __future__ import annotations

import json
import unittest

from copilot.ui.overlay import DaemonOverlayClient


class FakeResponse:
    def __init__(self, payload=None, lines=None) -> None:
        self.payload = payload or {}
        self.lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __iter__(self):
        return iter([line.encode("utf-8") for line in self.lines])


class FakeOpener:
    def __init__(self) -> None:
        self.requests = []

    def __call__(self, request, timeout=0):
        self.requests.append((request, timeout))
        if isinstance(request, str):
            return FakeResponse(lines=['data: {"phase":"step","message":"Running","metadata":{"step_id":"s1"}}\n', "\n", 'data: {"phase":"done","message":"Run success.","metadata":{}}\n'])
        path = request.full_url
        if path.endswith("/tasks"):
            return FakeResponse({"run_id": "run_1"})
        if path.endswith("/plans"):
            return FakeResponse({"plan": {"steps": [{"step_id": "s1", "action_type": "click_node"}]}})
        if path.endswith("/xray"):
            return FakeResponse({"nodes": [{"label": "Search", "box": {"x": 1, "y": 2, "width": 3, "height": 4}}]})
        if path.endswith("/cancel"):
            return FakeResponse({"ok": True})
        if "/approvals/" in path:
            return FakeResponse({"ok": True})
        return FakeResponse({"run_id": "run_1", "status": "running"})


class OverlayDaemonClientTests(unittest.TestCase):
    def test_client_posts_task_cancel_and_approval(self) -> None:
        opener = FakeOpener()
        client = DaemonOverlayClient("http://daemon", opener=opener)

        self.assertEqual(client.start_task("observe", dry_run=True), "run_1")
        self.assertEqual(client.preview_plan("observe")["plan"]["steps"][0]["step_id"], "s1")
        self.assertEqual(client.get_xray("run_1")[0]["label"], "Search")
        self.assertTrue(client.cancel_run("run_1", "hard"))
        self.assertTrue(client.decide_approval("approval_1", False))

        methods = [request.get_method() for request, _timeout in opener.requests if not isinstance(request, str)]
        self.assertEqual(methods, ["POST", "POST", "GET", "POST", "POST"])

    def test_client_stream_parses_sse_data_lines(self) -> None:
        opener = FakeOpener()
        client = DaemonOverlayClient("http://daemon", opener=opener)
        events = []

        client.stream_run("run_1", events.append)

        self.assertEqual([event["phase"] for event in events], ["step", "done"])


if __name__ == "__main__":
    unittest.main()
