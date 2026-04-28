from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from copilot.ui.shell_state import OperatorShellState, discover_benchmark_reports


class ShellStateTests(unittest.TestCase):
    def test_developer_mode_controls_visible_tabs(self) -> None:
        state = OperatorShellState(developer_mode=False)
        self.assertIn("Prompt", state.visible_tabs)
        self.assertNotIn("Trace", state.visible_tabs)

        state.developer_mode = True
        self.assertIn("Trace", state.visible_tabs)

    def test_daemon_run_rows_use_lifecycle_snapshot_and_confidence(self) -> None:
        state = OperatorShellState()
        state.apply_daemon_payload(
            runs=[
                {
                    "run_id": "run_1",
                    "lifecycle": "waiting_approval",
                    "prompt": "observe",
                    "snapshot": {"message": "Approve", "confidence": {"level": "LOW"}},
                }
            ],
            approvals=[{"approval_id": "a1", "status": "pending"}, {"approval_id": "a2", "status": "approved"}],
        )

        self.assertEqual(state.run_rows()[0]["status"], "waiting_approval")
        self.assertEqual(state.run_rows()[0]["confidence"], "LOW")
        self.assertEqual(len(state.approval_rows()), 1)

    def test_discovers_benchmark_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run" / "benchmark_report.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"generated_at": 1, "live_actions": False, "metrics": {"stable_success_rate": 1.0}}), encoding="utf-8")

            reports = discover_benchmark_reports(tmpdir)

            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0]["metrics"]["stable_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
