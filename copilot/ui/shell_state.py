from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


PRIMARY_TABS = ["Prompt", "Runs", "Approvals", "Settings", "Benchmarks", "Skills"]
DEVELOPER_TABS = ["Plan", "Trace", "Nodes", "Memory", "Review"]


@dataclass
class OperatorShellState:
    developer_mode: bool = False
    selected_run_id: str = ""
    runs: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    benchmark_reports: list[dict[str, Any]] = field(default_factory=list)

    @property
    def visible_tabs(self) -> list[str]:
        return PRIMARY_TABS + (DEVELOPER_TABS if self.developer_mode else [])

    def apply_daemon_payload(self, *, runs=None, approvals=None, skills=None) -> None:
        if runs is not None:
            self.runs = list(runs)
        if approvals is not None:
            self.approvals = list(approvals)
        if skills is not None:
            self.skills = list(skills)

    def run_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "run_id": run.get("run_id", ""),
                "status": run.get("lifecycle", run.get("status", "")),
                "prompt": run.get("prompt", ""),
                "message": (run.get("snapshot") or {}).get("message", ""),
                "confidence": ((run.get("snapshot") or {}).get("confidence") or {}).get("level", ""),
            }
            for run in self.runs
        ]

    def approval_rows(self) -> list[dict[str, Any]]:
        return [
            approval
            for approval in self.approvals
            if approval.get("status") == "pending"
        ]


def discover_benchmark_reports(root: str | Path = "benchmark_runs") -> list[dict[str, Any]]:
    base = Path(root)
    if not base.exists():
        return []
    reports = []
    for path in sorted(base.rglob("benchmark_report.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        reports.append(
            {
                "path": str(path),
                "generated_at": payload.get("generated_at", 0.0),
                "live_actions": bool(payload.get("live_actions", False)),
                "metrics": payload.get("metrics", {}),
            }
        )
    return reports
