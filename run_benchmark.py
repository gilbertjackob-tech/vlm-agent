from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force stable Paddle engine on Windows.
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from copilot.benchmark import BenchmarkRunner, DEFAULT_MISSIONS, validate_live_design
from copilot.schemas import TrustMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v0.9.9 real-world desktop benchmark harness.")
    parser.add_argument(
        "--output-dir",
        default="benchmark_runs",
        help="Directory where benchmark_report.json and failed artifacts are written.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of times to repeat each selected mission.",
    )
    parser.add_argument(
        "--mission",
        action="append",
        default=[],
        help="Mission id to run. Can be passed more than once.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Mission category to run. Can be passed more than once.",
    )
    parser.add_argument(
        "--max-missions",
        type=int,
        default=0,
        help="Limit selected missions for smoke runs.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute real desktop actions. Without this flag the harness runs dry plans only.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve policy-gated live benchmark steps.",
    )
    parser.add_argument(
        "--new-engine-per-attempt",
        action="store_true",
        help="Create a fresh CopilotEngine for every mission attempt.",
    )
    parser.add_argument(
        "--per-mission-timeout",
        type=float,
        default=180.0,
        help="Maximum runtime budget per mission attempt in seconds.",
    )
    parser.add_argument(
        "--trust-mode",
        choices=[mode.value for mode in TrustMode],
        default=TrustMode.PLAN_AND_RISK_GATES.value,
        help="Runtime trust mode used for mission attempts.",
    )
    parser.add_argument(
        "--voice",
        choices=["off", "console", "tts"],
        default=None,
        help="Narrate benchmark progress. 'tts' uses LocalAI NeuTTS Air; 'console' prints voice lines.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List benchmark missions and exit.",
    )
    parser.add_argument(
        "--design-only",
        action="store_true",
        help="Validate live mission design coverage without running plans or desktop actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        for mission in DEFAULT_MISSIONS:
            print(f"{mission.mission_id:34} {mission.category:30} {mission.prompt}")
        return

    output_dir = Path(args.output_dir)
    if output_dir.name == "benchmark_runs":
        output_dir = output_dir / time.strftime("%Y%m%d_%H%M%S")

    if args.design_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = validate_live_design(DEFAULT_MISSIONS)
        report_path = output_dir / "live_design_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("=" * 60)
        print("LIVE MISSION DESIGN CHECK")
        print("=" * 60)
        print(f"Report: {report_path}")
        print(f"Passed: {report['passed']}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    runner = BenchmarkRunner(
        output_dir=output_dir,
        repeat_count=max(1, args.repeat),
        live_actions=bool(args.live),
        trust_mode=TrustMode(args.trust_mode),
        auto_approve=bool(args.auto_approve),
        new_engine_per_attempt=bool(args.new_engine_per_attempt),
        per_mission_timeout_seconds=max(1.0, args.per_mission_timeout),
        voice_mode=args.voice,
    )

    report = runner.run(
        mission_ids=set(args.mission) if args.mission else None,
        categories=set(args.category) if args.category else None,
        max_missions=args.max_missions or None,
    )
    report_path = output_dir / "benchmark_report.json"
    metrics = report["metrics"]

    print("=" * 60)
    print("VLM AGENT BENCHMARK")
    print("=" * 60)
    print(f"Report: {report_path}")
    print(f"Live actions: {report['live_actions']}")
    print(f"Missions: {report['mission_count']} | Attempts: {report['attempt_count']} | Repeat: {report['repeat_count']}")
    print(f"Complete: {report.get('complete', True)} | Interrupted: {report.get('interrupted', False)}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if not args.live:
        print()
        print("Dry-run report only. Use --live to execute real desktop actions.")


if __name__ == "__main__":
    main()
