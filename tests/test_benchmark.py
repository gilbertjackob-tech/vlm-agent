from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from copilot.benchmark import BenchmarkRunner, DEFAULT_MISSIONS, compute_report, extract_trace_metrics
from copilot.benchmark.missions import BenchmarkMission
from copilot.schemas import ActionOutcome, ExecutionPlan, PlanStep, RunStatus, RunTrace, TaskSpec


def make_trace(status: RunStatus = RunStatus.SUCCESS, outputs: dict | None = None, outcome_count: int = 3) -> RunTrace:
    task = TaskSpec(prompt="benchmark prompt", goal="benchmark prompt")
    steps = [PlanStep(f"step_{idx}", f"Step {idx}", "parse_ui") for idx in range(1, outcome_count + 1)]
    trace = RunTrace(
        run_id="run_test",
        task=task,
        plan=ExecutionPlan(task=task, steps=steps),
        status=status,
    )
    trace.started_at = 100.0
    trace.finished_at = 112.0
    trace.outputs = outputs or {}
    trace.action_outcomes = [
        ActionOutcome(step_id=f"step_{idx}", action_type="parse_ui", ok=status == RunStatus.SUCCESS)
        for idx in range(1, outcome_count + 1)
    ]
    return trace


class FakeEngine:
    def __init__(self, traces: list[RunTrace]) -> None:
        self.traces = list(traces)
        self.prompts: list[str] = []
        self.call_kwargs: list[dict] = []

    def execute_prompt(self, prompt, trust_mode=None, approval_callback=None, trace_callback=None, dry_run=True, **kwargs):
        self.prompts.append(prompt)
        self.call_kwargs.append({"trust_mode": trust_mode, "dry_run": dry_run, **kwargs})
        if trace_callback:
            trace_callback({"phase": "test", "message": "fake event", "metadata": {}})
        return self.traces.pop(0)


class InterruptingEngine(FakeEngine):
    def execute_prompt(self, prompt, trust_mode=None, approval_callback=None, trace_callback=None, dry_run=True, **kwargs):
        self.prompts.append(prompt)
        self.call_kwargs.append({"trust_mode": trust_mode, "dry_run": dry_run, **kwargs})
        if trace_callback:
            trace_callback({"phase": "test", "message": "before interrupt", "metadata": {}})
        if self.traces:
            return self.traces.pop(0)
        raise KeyboardInterrupt()


class BenchmarkHarnessTests(unittest.TestCase):
    def test_default_catalog_covers_required_v099_mission_classes(self) -> None:
        categories = {mission.category for mission in DEFAULT_MISSIONS}

        self.assertGreaterEqual(len(DEFAULT_MISSIONS), 20)
        self.assertTrue(
            {
                "file_explorer",
                "notepad",
                "browser_search",
                "form_filling",
                "copy_paste",
                "window_switching",
                "dialog_handling",
                "wrong_target_recovery",
                "timeout_recovery",
                "duplicate_label_disambiguation",
            }.issubset(categories)
        )

    def test_trace_metrics_count_contract_failures_recovery_and_replanning(self) -> None:
        trace = make_trace(
            RunStatus.FAILED,
            outputs={
                "action_contracts": [
                    {
                        "action_type": "click_node",
                        "failure_reason": "NO_STATE_CHANGE",
                        "during_checks": {"executor_accepted": True},
                        "verification": {"verified": False},
                    },
                    {
                        "action_type": "wait_for",
                        "failure_reason": "TIMEOUT",
                        "during_checks": {"executor_accepted": False},
                        "verification": {"verified": False},
                    },
                ],
                "recovery_attempts": [
                    {"step_id": "step_1", "retry_success": True, "target_switch_after_recovery": True},
                    {"step_id": "step_1", "retry_success": False},
                ],
                "target_rankings": [
                    {"step_id": "step_1", "candidate_count": 2, "score_gap": 0.03, "ambiguous": True, "duplicate_disambiguation_used": False},
                    {"step_id": "step_2", "candidate_count": 2, "score_gap": 0.4, "ambiguous": False, "duplicate_disambiguation_used": True},
                ],
                "perception_quality": [
                    {"step_id": "step_3", "action_type": "type_text", "focus_confirmed": False, "focus_confidence": 0.2},
                ],
                "plan_replacements": [
                    {"planner_type": "repair"},
                    {"planner_type": "replan"},
                    {"planner_type": "repair"},
                ],
                "parse_health": [
                    {
                        "parse_mode": "vision",
                        "worker_used": True,
                        "worker_exec_seconds": 1.2,
                        "worker_queue_wait_seconds": 0.1,
                        "ocr_calls": 3,
                        "ocr_cache_hits": 2,
                        "ocr_timeouts": 1,
                        "ocr_elapsed_seconds": 2.5,
                        "parse_elapsed_seconds": 4.0,
                        "screen_hash_elapsed_seconds": 0.05,
                    },
                    {
                        "parse_mode": "vision_cache",
                        "cache_hit": True,
                        "ocr_calls": 0,
                        "ocr_elapsed_seconds": 0.0,
                        "parse_elapsed_seconds": 0.1,
                        "screen_hash_cache_hit": True,
                        "screen_hash_elapsed_seconds": 0.02,
                        "state_cache_hit": True,
                        "state_probe_elapsed_seconds": 0.03,
                    },
                ],
            },
            outcome_count=4,
        )

        metrics = extract_trace_metrics(trace)

        self.assertFalse(metrics["success"])
        self.assertEqual(metrics["wrong_click_count"], 1)
        self.assertEqual(metrics["recovery_attempt_count"], 2)
        self.assertEqual(metrics["recovery_success_count"], 1)
        self.assertEqual(metrics["recovery_depth_per_step"], 0.5)
        self.assertEqual(metrics["focus_failure_count"], 1)
        self.assertEqual(metrics["target_ambiguity_count"], 1)
        self.assertEqual(metrics["target_switch_after_recovery_count"], 1)
        self.assertEqual(metrics["duplicate_disambiguation_count"], 1)
        self.assertEqual(metrics["avg_score_gap"], 0.215)
        self.assertEqual(metrics["repair_vs_replan_count"], {"repair": 2, "replan": 1})
        self.assertEqual(metrics["failure_reason_distribution"]["NO_STATE_CHANGE"], 1)
        self.assertEqual(metrics["failure_reason_distribution"]["TIMEOUT"], 1)
        self.assertEqual(metrics["step_count"], 4)
        self.assertEqual(metrics["latency_per_step"], 3.0)
        self.assertEqual(metrics["parse_count"], 2)
        self.assertEqual(metrics["ocr_call_count"], 3)
        self.assertEqual(metrics["ocr_cache_hit_count"], 2)
        self.assertEqual(metrics["ocr_timeout_count"], 1)
        self.assertEqual(metrics["parse_cache_hit_count"], 1)
        self.assertEqual(metrics["state_cache_hit_count"], 1)
        self.assertEqual(metrics["state_cache_miss_count"], 0)
        self.assertEqual(metrics["state_probe_elapsed_seconds"], 0.03)
        self.assertEqual(metrics["parse_count_avoided_by_state_cache"], 1)
        self.assertEqual(metrics["ocr_elapsed_seconds"], 2.5)
        self.assertEqual(metrics["parse_elapsed_seconds"], 4.1)
        self.assertEqual(metrics["parse_worker_count"], 1)
        self.assertEqual(metrics["parse_worker_elapsed_seconds"], 1.2)
        self.assertEqual(metrics["parse_worker_queue_wait_seconds"], 0.1)
        self.assertEqual(metrics["screen_hash_elapsed_seconds"], 0.07)

    def test_stable_success_is_not_invalidated_by_nonfatal_ocr_timeout(self) -> None:
        trace = make_trace(
            RunStatus.SUCCESS,
            outputs={
                "parse_health": [
                    {
                        "parse_mode": "vision_worker",
                        "ocr_timeouts": 1,
                        "parse_elapsed_seconds": 1.0,
                    }
                ]
            },
            outcome_count=2,
        )

        metrics = extract_trace_metrics(trace)

        self.assertTrue(metrics["success"])
        self.assertTrue(metrics["stable_success"])
        self.assertEqual(metrics["ocr_timeout_count"], 1)

    def test_report_aggregates_minimum_required_metrics(self) -> None:
        missions = [
            BenchmarkMission("m1", "file_explorer", "Open explorer and parse screen"),
            BenchmarkMission("m2", "timeout_recovery", "Wait for missing benchmark window"),
        ]
        results = [
            {
                "mission_id": "m1",
                "category": "file_explorer",
                "iteration": 1,
                "success": True,
                "stable_success": True,
                "wrong_click_count": 0,
                "recovery_attempt_count": 0,
                "recovery_success_count": 0,
                "focus_failure_count": 0,
                "target_ambiguity_count": 0,
                "target_switch_after_recovery_count": 0,
                "duplicate_disambiguation_count": 1,
                "avg_score_gap": 0.8,
                "repair_vs_replan_count": {},
                "failure_reason_distribution": {},
                "step_count": 4,
                "latency_per_step": 0.5,
                "ocr_elapsed_seconds": 0.5,
                "parse_elapsed_seconds": 1.0,
                "parse_count": 2,
                "parse_worker_count": 1,
                "parse_worker_elapsed_seconds": 0.7,
                "parse_worker_queue_wait_seconds": 0.05,
                "structured_skip_count": 1,
                "screen_hash_elapsed_seconds": 0.04,
            },
            {
                "mission_id": "m2",
                "category": "timeout_recovery",
                "iteration": 1,
                "success": False,
                "stable_success": False,
                "wrong_click_count": 1,
                "recovery_attempt_count": 2,
                "recovery_success_count": 1,
                "focus_failure_count": 1,
                "target_ambiguity_count": 1,
                "target_switch_after_recovery_count": 1,
                "duplicate_disambiguation_count": 0,
                "avg_score_gap": 0.2,
                "repair_vs_replan_count": {"repair": 1},
                "failure_reason_distribution": {"TIMEOUT": 1},
                "step_count": 2,
                "latency_per_step": 1.5,
                "ocr_elapsed_seconds": 2.5,
                "parse_elapsed_seconds": 3.0,
                "parse_count": 2,
                "ocr_timeout_count": 1,
                "ocr_call_count": 3,
                "ocr_cache_hit_count": 1,
                "parse_cache_hit_count": 1,
                "state_cache_hit_count": 1,
                "state_cache_miss_count": 2,
                "state_probe_elapsed_seconds": 0.04,
                "parse_count_avoided_by_state_cache": 1,
                "dom_parse_count": 0,
                "parse_worker_count": 1,
                "parse_worker_elapsed_seconds": 1.1,
                "parse_worker_queue_wait_seconds": 0.15,
                "structured_skip_count": 0,
                "screen_hash_elapsed_seconds": 0.06,
                "artifact_dir": "failed",
            },
        ]

        report = compute_report(
            results=results,
            missions=missions,
            repeat_count=1,
            live_actions=False,
            output_dir="bench",
        )

        metrics = report["metrics"]
        self.assertEqual(metrics["success_rate"], 0.5)
        self.assertEqual(metrics["stable_success_rate"], 0.5)
        self.assertEqual(metrics["unstable_success_count"], 0)
        self.assertEqual(metrics["wrong_click_count"], 1)
        self.assertEqual(metrics["recovery_count"], 2)
        self.assertEqual(metrics["recovery_rate_per_successful_task"], 2.0)
        self.assertEqual(metrics["recovery_depth_per_step"], 0.333333)
        self.assertEqual(metrics["focus_failure_count"], 1)
        self.assertEqual(metrics["target_ambiguity_count"], 1)
        self.assertEqual(metrics["target_switch_after_recovery_count"], 1)
        self.assertEqual(metrics["duplicate_disambiguation_count"], 1)
        self.assertEqual(metrics["avg_score_gap"], 0.5)
        self.assertEqual(metrics["recovery_success_rate"], 0.5)
        self.assertEqual(metrics["repair_vs_replan_count"], {"repair": 1})
        self.assertEqual(metrics["avg_steps_per_task"], 3.0)
        self.assertEqual(metrics["avg_latency_per_step"], 0.833333)
        self.assertEqual(metrics["avg_ocr_time_per_task"], 1.5)
        self.assertEqual(metrics["avg_parse_time_per_task"], 2.0)
        self.assertEqual(metrics["avg_parse_time_per_parse"], 1.0)
        self.assertEqual(metrics["ocr_timeout_count"], 1)
        self.assertEqual(metrics["ocr_call_count"], 3)
        self.assertEqual(metrics["ocr_cache_hit_count"], 1)
        self.assertEqual(metrics["parse_cache_hit_count"], 1)
        self.assertEqual(metrics["state_cache_hit_count"], 1)
        self.assertEqual(metrics["state_cache_miss_count"], 2)
        self.assertEqual(metrics["state_probe_elapsed_seconds"], 0.04)
        self.assertEqual(metrics["parse_count_avoided_by_state_cache"], 1)
        self.assertEqual(metrics["parse_worker_count"], 2)
        self.assertEqual(metrics["parse_worker_elapsed_seconds"], 1.8)
        self.assertEqual(metrics["parse_worker_queue_wait_seconds"], 0.2)
        self.assertEqual(metrics["structured_skip_count"], 1)
        self.assertEqual(metrics["screen_hash_elapsed_seconds"], 0.1)
        self.assertEqual(metrics["failure_reason_distribution"], {"TIMEOUT": 1})
        self.assertEqual(len(report["failed_artifacts"]), 1)

    def test_runner_repeats_missions_and_writes_benchmark_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            traces = [make_trace(RunStatus.SUCCESS), make_trace(RunStatus.SUCCESS)]
            engine = FakeEngine(traces)
            mission = BenchmarkMission("m1", "file_explorer", "Open explorer and parse screen")
            runner = BenchmarkRunner(
                engine_factory=lambda: engine,
                missions=[mission],
                output_dir=tmpdir,
                repeat_count=2,
                live_actions=False,
            )

            report = runner.run()

            report_path = Path(tmpdir) / "benchmark_report.json"
            self.assertTrue(report_path.exists())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["attempt_count"], 2)
            self.assertEqual(saved["metrics"]["success_rate"], 1.0)
            self.assertEqual(engine.prompts, [mission.prompt, mission.prompt])

    def test_runner_passes_voice_mode_to_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FakeEngine([make_trace(RunStatus.SUCCESS)])
            mission = BenchmarkMission("m1", "file_explorer", "Open explorer and parse screen")
            runner = BenchmarkRunner(
                engine_factory=lambda: engine,
                missions=[mission],
                output_dir=tmpdir,
                repeat_count=1,
                live_actions=True,
                voice_mode="console",
            )

            runner.run()

            self.assertEqual(engine.call_kwargs[0]["voice_mode"], "console")
            self.assertEqual(engine.call_kwargs[0]["narration_context"], "benchmark")

    def test_runner_flushes_partial_report_when_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = InterruptingEngine([make_trace(RunStatus.SUCCESS)])
            mission = BenchmarkMission("m1", "file_explorer", "Open explorer and parse screen")
            runner = BenchmarkRunner(
                engine_factory=lambda: engine,
                missions=[mission],
                output_dir=tmpdir,
                repeat_count=3,
                live_actions=False,
            )

            report = runner.run()

            report_path = Path(tmpdir) / "benchmark_report.json"
            self.assertTrue(report_path.exists())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(saved["complete"])
            self.assertTrue(saved["interrupted"])
            self.assertEqual(saved["attempt_count"], 2)
            self.assertEqual(saved["expected_attempt_count"], 3)
            self.assertEqual(saved["results"][-1]["status"], "interrupted")
            self.assertEqual(saved["results"][-1]["failure_reason_distribution"], {"BENCHMARK_INTERRUPTED": 1})
            self.assertEqual(report["attempt_count"], saved["attempt_count"])

    def test_failed_attempt_exports_trace_and_screenshot_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "bench"
            source_dir.mkdir()
            trace_source = source_dir / "trace.json"
            screenshot = source_dir / "after.png"
            raw_screenshot = source_dir / "after_raw.png"
            trace_source.write_text("{}", encoding="utf-8")
            screenshot.write_bytes(b"png")
            raw_screenshot.write_bytes(b"raw")

            failed = make_trace(
                RunStatus.FAILED,
                outputs={
                    "trace_path": str(trace_source),
                    "action_contracts": [
                        {
                            "action_type": "click_node",
                            "failure_reason": "NO_STATE_CHANGE",
                            "during_checks": {"executor_accepted": True},
                            "verification": {
                                "verified": False,
                                "screenshot_after": str(screenshot),
                            },
                        }
                    ],
                },
            )
            engine = FakeEngine([failed])
            mission = BenchmarkMission("m_failed", "wrong_target_recovery", "Recover wrong target")
            runner = BenchmarkRunner(
                engine_factory=lambda: engine,
                missions=[mission],
                output_dir=output_dir,
                repeat_count=1,
                live_actions=False,
            )

            report = runner.run()
            result = report["results"][0]

            self.assertFalse(result["success"])
            self.assertTrue(Path(result["trace_copy"]).exists())
            self.assertEqual(len(result["screenshot_copies"]), 2)
            self.assertTrue(all(Path(path).exists() for path in result["screenshot_copies"]))
            self.assertTrue((Path(result["artifact_dir"]) / "failure_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
