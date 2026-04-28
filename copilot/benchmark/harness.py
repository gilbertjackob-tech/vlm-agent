from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import shutil
import time

from copilot.benchmark.missions import BenchmarkMission, DEFAULT_MISSIONS
from copilot.schemas import RunTrace, TrustMode


EngineFactory = Callable[[], Any]
ApprovalCallback = Callable[[str, dict[str, Any]], bool]

BENCHMARK_VERSION = "v0.9.9"
HARD_RULE = 'No claim of "general PC assistant" without benchmark evidence.'


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _trace_outputs(trace: RunTrace | dict[str, Any] | None) -> dict[str, Any]:
    if trace is None:
        return {}
    if isinstance(trace, dict):
        outputs = trace.get("outputs", {})
    else:
        outputs = getattr(trace, "outputs", {})
    return outputs if isinstance(outputs, dict) else {}


def _trace_status(trace: RunTrace | dict[str, Any] | None) -> str:
    if trace is None:
        return "exception"
    if isinstance(trace, dict):
        return _status_value(trace.get("status", ""))
    return _status_value(getattr(trace, "status", ""))


def _trace_started_at(trace: RunTrace | dict[str, Any] | None) -> float:
    if trace is None:
        return time.time()
    if isinstance(trace, dict):
        return float(trace.get("started_at", 0.0) or 0.0)
    return float(getattr(trace, "started_at", 0.0) or 0.0)


def _trace_finished_at(trace: RunTrace | dict[str, Any] | None) -> float:
    if trace is None:
        return time.time()
    if isinstance(trace, dict):
        return float(trace.get("finished_at", 0.0) or 0.0)
    return float(getattr(trace, "finished_at", 0.0) or 0.0)


def _trace_plan_steps(trace: RunTrace | dict[str, Any] | None) -> int:
    if trace is None:
        return 0
    if isinstance(trace, dict):
        plan = trace.get("plan", {}) if isinstance(trace.get("plan", {}), dict) else {}
        return len(plan.get("steps", []) or [])
    plan = getattr(trace, "plan", None)
    return len(getattr(plan, "steps", []) or [])


def _trace_action_outcomes(trace: RunTrace | dict[str, Any] | None) -> int:
    if trace is None:
        return 0
    if isinstance(trace, dict):
        return len(trace.get("action_outcomes", []) or [])
    return len(getattr(trace, "action_outcomes", []) or [])


def extract_trace_metrics(trace: RunTrace | dict[str, Any] | None) -> dict[str, Any]:
    outputs = _trace_outputs(trace)
    contracts = [item for item in outputs.get("action_contracts", []) if isinstance(item, dict)]
    recovery_attempts = [item for item in outputs.get("recovery_attempts", []) if isinstance(item, dict)]
    plan_replacements = [item for item in outputs.get("plan_replacements", []) if isinstance(item, dict)]
    failure_recovery = [item for item in outputs.get("failure_recovery", []) if isinstance(item, dict)]
    parse_health = [item for item in outputs.get("parse_health", []) if isinstance(item, dict)]
    target_rankings = [item for item in outputs.get("target_rankings", []) if isinstance(item, dict)]
    perception_quality = [item for item in outputs.get("perception_quality", []) if isinstance(item, dict)]

    failure_reasons: Counter[str] = Counter()
    for contract in contracts:
        reason = str(contract.get("failure_reason", "") or "")
        if reason:
            failure_reasons[reason] += 1
    for record in failure_recovery:
        reason = str(record.get("failure_reason", "") or "")
        if reason and reason not in failure_reasons:
            failure_reasons[reason] += 1

    wrong_click_count = 0
    for contract in contracts:
        action_type = str(contract.get("action_type", ""))
        if action_type not in {"click_node", "click_point"}:
            continue
        verification = contract.get("verification", {}) if isinstance(contract.get("verification", {}), dict) else {}
        during = contract.get("during_checks", {}) if isinstance(contract.get("during_checks", {}), dict) else {}
        executor_accepted = bool(during.get("executor_accepted"))
        identity_problem = bool(contract.get("identity_drifted") or contract.get("identity_ambiguous"))
        unverified_accepted_click = executor_accepted and not bool(verification.get("verified"))
        if identity_problem or unverified_accepted_click:
            wrong_click_count += 1

    repair_vs_replan = Counter()
    for replacement in plan_replacements:
        planner_type = str(replacement.get("planner_type", "") or "unknown")
        repair_vs_replan[planner_type] += 1

    step_count = _trace_action_outcomes(trace) or _trace_plan_steps(trace)
    started_at = _trace_started_at(trace)
    finished_at = _trace_finished_at(trace) or time.time()
    latency_seconds = max(0.0, finished_at - started_at) if started_at else 0.0
    recovery_success_count = sum(1 for item in recovery_attempts if bool(item.get("retry_success")))
    focus_failure_step_ids = {
        str(contract.get("step_id", ""))
        for contract in contracts
        if str(contract.get("failure_reason", "")) == "FOCUS_NOT_CONFIRMED"
    }
    focus_failure_step_ids.update(
        str(item.get("step_id", ""))
        for item in perception_quality
        if item.get("focus_confirmed") is False and str(item.get("action_type", "")) in {"route_window", "confirm_focus", "type_text"}
    )
    target_ambiguity_step_ids = {
        str(contract.get("step_id", ""))
        for contract in contracts
        if str(contract.get("failure_reason", "")) == "TARGET_AMBIGUOUS" or bool(contract.get("identity_ambiguous"))
    }
    target_ambiguity_step_ids.update(str(item.get("step_id", "")) for item in target_rankings if bool(item.get("ambiguous")))
    target_switch_after_recovery_count = sum(1 for item in recovery_attempts if bool(item.get("target_switch_after_recovery")))
    score_gaps = [float(item.get("score_gap", 0.0) or 0.0) for item in target_rankings if item.get("score_gap") is not None]
    duplicate_disambiguation_count = sum(1 for item in target_rankings if bool(item.get("duplicate_disambiguation_used")))
    ocr_elapsed_seconds = sum(float(item.get("ocr_elapsed_seconds", 0.0) or 0.0) for item in parse_health)
    parse_elapsed_seconds = sum(float(item.get("parse_elapsed_seconds", 0.0) or 0.0) for item in parse_health)
    ocr_timeout_count = sum(int(item.get("ocr_timeouts", 0) or 0) for item in parse_health)
    ocr_call_count = sum(int(item.get("ocr_calls", 0) or 0) for item in parse_health)
    ocr_cache_hit_count = sum(int(item.get("ocr_cache_hits", 0) or 0) for item in parse_health)
    parse_cache_hit_count = sum(1 for item in parse_health if bool(item.get("cache_hit")))
    state_cache_hit_count = sum(1 for item in parse_health if bool(item.get("state_cache_hit")))
    state_cache_miss_count = sum(1 for item in parse_health if bool(item.get("state_cache_miss")))
    state_probe_elapsed_seconds = sum(float(item.get("state_probe_elapsed_seconds", 0.0) or 0.0) for item in parse_health)
    dom_parse_count = sum(1 for item in parse_health if item.get("parse_mode") == "browser_dom")
    uia_parse_count = sum(1 for item in parse_health if item.get("parse_mode") == "windows_uia")
    ocr_fallback_count = sum(1 for item in parse_health if not bool(item.get("ocr_skipped_by_dom")) and not bool(item.get("ocr_skipped_by_uia")) and int(item.get("ocr_calls", 0) or 0) > 0)
    verification_failure_count = sum(1 for contract in contracts if not bool((contract.get("verification", {}) if isinstance(contract.get("verification", {}), dict) else {}).get("verified", True)))
    parse_worker_count = sum(1 for item in parse_health if bool(item.get("worker_used")))
    parse_worker_elapsed_seconds = sum(float(item.get("worker_exec_seconds", 0.0) or 0.0) for item in parse_health)
    parse_worker_queue_wait_seconds = sum(float(item.get("worker_queue_wait_seconds", 0.0) or 0.0) for item in parse_health)
    structured_skip_count = sum(
        1
        for item in parse_health
        if bool(item.get("ocr_skipped_by_dom")) or bool(item.get("ocr_skipped_by_uia"))
    )
    screen_hash_elapsed_seconds = sum(float(item.get("screen_hash_elapsed_seconds", 0.0) or 0.0) for item in parse_health)
    stable_success = (
        _trace_status(trace) == "success"
        and len(recovery_attempts) == 0
        and wrong_click_count == 0
        and not failure_reasons
    )

    return {
        "status": _trace_status(trace),
        "success": _trace_status(trace) == "success",
        "stable_success": stable_success,
        "step_count": step_count,
        "latency_seconds": round(latency_seconds, 6),
        "latency_per_step": round(latency_seconds / step_count, 6) if step_count else 0.0,
        "wrong_click_count": wrong_click_count,
        "wrong_action_count": wrong_click_count,
        "verification_failure_count": verification_failure_count,
        "verification_failure_rate": round(verification_failure_count / len(contracts), 6) if contracts else 0.0,
        "recovery_attempt_count": len(recovery_attempts),
        "recovery_success_count": recovery_success_count,
        "recovery_depth_per_step": round(len(recovery_attempts) / step_count, 6) if step_count else 0.0,
        "focus_failure_count": len([item for item in focus_failure_step_ids if item]),
        "target_ambiguity_count": len([item for item in target_ambiguity_step_ids if item]),
        "target_switch_after_recovery_count": target_switch_after_recovery_count,
        "duplicate_disambiguation_count": duplicate_disambiguation_count,
        "avg_score_gap": round(sum(score_gaps) / len(score_gaps), 6) if score_gaps else 0.0,
        "repair_vs_replan_count": dict(repair_vs_replan),
        "failure_reason_distribution": dict(failure_reasons),
        "trace_path": str(outputs.get("trace_path", "") or ""),
        "contract_count": len(contracts),
        "state_snapshot_count": len(outputs.get("state_snapshots", []) or []),
        "state_diff_count": len(outputs.get("state_diffs", []) or []),
        "parse_count": len(parse_health),
        "ocr_elapsed_seconds": round(ocr_elapsed_seconds, 6),
        "parse_elapsed_seconds": round(parse_elapsed_seconds, 6),
        "ocr_timeout_count": ocr_timeout_count,
        "ocr_call_count": ocr_call_count,
        "ocr_cache_hit_count": ocr_cache_hit_count,
        "parse_cache_hit_count": parse_cache_hit_count,
        "state_cache_hit_count": state_cache_hit_count,
        "state_cache_miss_count": state_cache_miss_count,
        "state_probe_elapsed_seconds": round(state_probe_elapsed_seconds, 6),
        "parse_count_avoided_by_state_cache": state_cache_hit_count,
        "dom_parse_count": dom_parse_count,
        "uia_parse_count": uia_parse_count,
        "ocr_fallback_count": ocr_fallback_count,
        "parse_worker_count": parse_worker_count,
        "parse_worker_elapsed_seconds": round(parse_worker_elapsed_seconds, 6),
        "parse_worker_queue_wait_seconds": round(parse_worker_queue_wait_seconds, 6),
        "structured_skip_count": structured_skip_count,
        "screen_hash_elapsed_seconds": round(screen_hash_elapsed_seconds, 6),
    }


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = len(results)
    successes = sum(1 for item in results if item.get("success"))
    recovery_attempts = sum(int(item.get("recovery_attempt_count", 0) or 0) for item in results)
    recovery_successes = sum(int(item.get("recovery_success_count", 0) or 0) for item in results)
    total_steps = sum(float(item.get("step_count", 0) or 0) for item in results)
    stable_successes = sum(1 for item in results if item.get("stable_success"))
    total_score_gap_count = sum(1 for item in results if float(item.get("avg_score_gap", 0.0) or 0.0) > 0.0)
    total_ocr_elapsed = sum(float(item.get("ocr_elapsed_seconds", 0.0) or 0.0) for item in results)
    total_parse_elapsed = sum(float(item.get("parse_elapsed_seconds", 0.0) or 0.0) for item in results)
    total_parse_count = sum(int(item.get("parse_count", 0) or 0) for item in results)
    total_latency = 0.0
    for item in results:
        if item.get("latency_seconds") is not None:
            total_latency += float(item.get("latency_seconds", 0.0) or 0.0)
        else:
            total_latency += float(item.get("latency_per_step", 0.0) or 0.0) * float(item.get("step_count", 0) or 0)
    repair_vs_replan: Counter[str] = Counter()
    failures: Counter[str] = Counter()

    for item in results:
        repair_vs_replan.update(item.get("repair_vs_replan_count", {}) or {})
        failures.update(item.get("failure_reason_distribution", {}) or {})

    return {
        "success_rate": round(successes / attempts, 4) if attempts else 0.0,
        "stable_success_rate": round(stable_successes / attempts, 4) if attempts else 0.0,
        "unstable_success_count": successes - stable_successes,
        "wrong_click_count": sum(int(item.get("wrong_click_count", 0) or 0) for item in results),
        "wrong_action_count": sum(int(item.get("wrong_action_count", item.get("wrong_click_count", 0)) or 0) for item in results),
        "verification_failure_count": sum(int(item.get("verification_failure_count", 0) or 0) for item in results),
        "verification_failure_rate": round(
            sum(int(item.get("verification_failure_count", 0) or 0) for item in results)
            / sum(int(item.get("contract_count", 0) or 0) for item in results),
            6,
        ) if sum(int(item.get("contract_count", 0) or 0) for item in results) else 0.0,
        "recovery_count": recovery_attempts,
        "recovery_rate_per_successful_task": round(recovery_attempts / successes, 4) if successes else 0.0,
        "recovery_depth_per_step": round(recovery_attempts / total_steps, 6) if total_steps else 0.0,
        "focus_failure_count": sum(int(item.get("focus_failure_count", 0) or 0) for item in results),
        "target_ambiguity_count": sum(int(item.get("target_ambiguity_count", 0) or 0) for item in results),
        "target_switch_after_recovery_count": sum(int(item.get("target_switch_after_recovery_count", 0) or 0) for item in results),
        "duplicate_disambiguation_count": sum(int(item.get("duplicate_disambiguation_count", 0) or 0) for item in results),
        "avg_score_gap": round(
            sum(float(item.get("avg_score_gap", 0.0) or 0.0) for item in results) / total_score_gap_count,
            6,
        ) if total_score_gap_count else 0.0,
        "recovery_success_rate": round(recovery_successes / recovery_attempts, 4) if recovery_attempts else 0.0,
        "repair_vs_replan_count": dict(repair_vs_replan),
        "avg_steps_per_task": round(total_steps / attempts, 4) if attempts else 0.0,
        "avg_latency_per_step": round(total_latency / total_steps, 6) if total_steps else 0.0,
        "avg_ocr_time_per_task": round(total_ocr_elapsed / attempts, 6) if attempts else 0.0,
        "avg_parse_time_per_task": round(total_parse_elapsed / attempts, 6) if attempts else 0.0,
        "avg_parse_time_per_parse": round(total_parse_elapsed / total_parse_count, 6) if total_parse_count else 0.0,
        "ocr_timeout_count": sum(int(item.get("ocr_timeout_count", 0) or 0) for item in results),
        "ocr_call_count": sum(int(item.get("ocr_call_count", 0) or 0) for item in results),
        "ocr_cache_hit_count": sum(int(item.get("ocr_cache_hit_count", 0) or 0) for item in results),
        "parse_cache_hit_count": sum(int(item.get("parse_cache_hit_count", 0) or 0) for item in results),
        "state_cache_hit_count": sum(int(item.get("state_cache_hit_count", 0) or 0) for item in results),
        "state_cache_miss_count": sum(int(item.get("state_cache_miss_count", 0) or 0) for item in results),
        "state_probe_elapsed_seconds": round(sum(float(item.get("state_probe_elapsed_seconds", 0.0) or 0.0) for item in results), 6),
        "parse_count_avoided_by_state_cache": sum(int(item.get("parse_count_avoided_by_state_cache", 0) or 0) for item in results),
        "dom_parse_count": sum(int(item.get("dom_parse_count", 0) or 0) for item in results),
        "uia_parse_count": sum(int(item.get("uia_parse_count", 0) or 0) for item in results),
        "ocr_fallback_count": sum(int(item.get("ocr_fallback_count", 0) or 0) for item in results),
        "parse_worker_count": sum(int(item.get("parse_worker_count", 0) or 0) for item in results),
        "parse_worker_elapsed_seconds": round(sum(float(item.get("parse_worker_elapsed_seconds", 0.0) or 0.0) for item in results), 6),
        "parse_worker_queue_wait_seconds": round(sum(float(item.get("parse_worker_queue_wait_seconds", 0.0) or 0.0) for item in results), 6),
        "structured_skip_count": sum(int(item.get("structured_skip_count", 0) or 0) for item in results),
        "screen_hash_elapsed_seconds": round(sum(float(item.get("screen_hash_elapsed_seconds", 0.0) or 0.0) for item in results), 6),
        "failure_reason_distribution": dict(failures),
    }


def compute_report(
    *,
    results: list[dict[str, Any]],
    missions: list[BenchmarkMission],
    repeat_count: int,
    live_actions: bool,
    output_dir: str,
    complete: bool = True,
    interrupted: bool = False,
) -> dict[str, Any]:
    category_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        category_results[str(item.get("category", "unknown"))].append(item)

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": time.time(),
        "hard_rule": HARD_RULE,
        "live_actions": bool(live_actions),
        "repeat_count": int(repeat_count),
        "mission_count": len(missions),
        "attempt_count": len(results),
        "expected_attempt_count": len(missions) * int(repeat_count),
        "complete": bool(complete),
        "interrupted": bool(interrupted),
        "output_dir": output_dir,
        "metrics": _aggregate_results(results),
        "category_metrics": {
            category: _aggregate_results(items)
            for category, items in sorted(category_results.items())
        },
        "missions": [mission.to_dict() for mission in missions],
        "results": results,
        "failed_artifacts": [
            {
                "mission_id": item.get("mission_id"),
                "iteration": item.get("iteration"),
                "artifact_dir": item.get("artifact_dir", ""),
                "trace_copy": item.get("trace_copy", ""),
                "screenshots": item.get("screenshot_copies", []),
            }
            for item in results
            if not item.get("success")
        ],
    }


def _default_engine_factory() -> Any:
    from copilot.runtime.engine import CopilotEngine

    return CopilotEngine()


class BenchmarkRunner:
    def __init__(
        self,
        *,
        engine_factory: EngineFactory | None = None,
        missions: Iterable[BenchmarkMission] | None = None,
        output_dir: str | Path = "benchmark_runs",
        repeat_count: int = 5,
        live_actions: bool = False,
        trust_mode: TrustMode = TrustMode.PLAN_AND_RISK_GATES,
        auto_approve: bool = False,
        new_engine_per_attempt: bool = False,
        per_mission_timeout_seconds: float = 180.0,
        voice_mode: str | None = None,
    ) -> None:
        self.engine_factory = engine_factory or _default_engine_factory
        self.missions = list(missions or DEFAULT_MISSIONS)
        self.output_dir = Path(output_dir)
        self.repeat_count = repeat_count
        self.live_actions = live_actions
        self.trust_mode = trust_mode
        self.auto_approve = auto_approve
        self.new_engine_per_attempt = new_engine_per_attempt
        self.per_mission_timeout_seconds = float(per_mission_timeout_seconds)
        self.voice_mode = voice_mode
        self._shared_engine = None

    def _approval_callback(self, prompt: str, payload: dict[str, Any]) -> bool:
        return bool(self.auto_approve)

    def _engine(self) -> Any:
        if self.new_engine_per_attempt:
            return self.engine_factory()
        if self._shared_engine is None:
            self._shared_engine = self.engine_factory()
        return self._shared_engine

    def select_missions(
        self,
        *,
        mission_ids: set[str] | None = None,
        categories: set[str] | None = None,
        max_missions: int | None = None,
    ) -> list[BenchmarkMission]:
        selected = []
        for mission in self.missions:
            if mission_ids and mission.mission_id not in mission_ids:
                continue
            if categories and mission.category not in categories:
                continue
            selected.append(mission)
            if max_missions is not None and len(selected) >= max_missions:
                break
        return selected

    def run(
        self,
        *,
        mission_ids: set[str] | None = None,
        categories: set[str] | None = None,
        max_missions: int | None = None,
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        selected = self.select_missions(mission_ids=mission_ids, categories=categories, max_missions=max_missions)
        results: list[dict[str, Any]] = []

        for mission in selected:
            for iteration in range(1, self.repeat_count + 1):
                result = self._run_one(mission, iteration)
                results.append(result)
                if result.get("status") == "interrupted":
                    return self._write_report(selected, results, complete=False, interrupted=True)
                self._write_report(selected, results, complete=False)

        return self._write_report(selected, results, complete=True)

    def _write_report(
        self,
        selected: list[BenchmarkMission],
        results: list[dict[str, Any]],
        *,
        complete: bool,
        interrupted: bool = False,
    ) -> dict[str, Any]:
        report = compute_report(
            results=results,
            missions=selected,
            repeat_count=self.repeat_count,
            live_actions=self.live_actions,
            output_dir=str(self.output_dir),
            complete=complete,
            interrupted=interrupted,
        )
        report_path = self.output_dir / "benchmark_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def _run_one(self, mission: BenchmarkMission, iteration: int) -> dict[str, Any]:
        trace = None
        error = ""
        started_at = time.time()
        events: list[dict[str, Any]] = []

        try:
            engine = self._engine()
            trace = engine.execute_prompt(
                mission.prompt,
                trust_mode=self.trust_mode,
                approval_callback=self._approval_callback,
                trace_callback=lambda event: events.append(event),
                dry_run=not self.live_actions,
                max_runtime_seconds=self.per_mission_timeout_seconds,
                voice_mode=self.voice_mode,
                narration_context="benchmark",
            )
            metrics = extract_trace_metrics(trace)
        except KeyboardInterrupt:
            error = "KeyboardInterrupt"
            metrics = self._failure_metrics(
                status="interrupted",
                started_at=started_at,
                failure_reason="BENCHMARK_INTERRUPTED",
            )
        except Exception as exc:
            error = str(exc)
            metrics = self._failure_metrics(
                status="exception",
                started_at=started_at,
                failure_reason="HARNESS_EXCEPTION",
            )

        result = {
            "mission_id": mission.mission_id,
            "category": mission.category,
            "iteration": iteration,
            "prompt": mission.prompt,
            "expected_apps": list(mission.expected_apps),
            "setup": mission.setup,
            "success_signal": mission.success_signal,
            "tags": list(mission.tags),
            "events": len(events),
            "error": error,
            **metrics,
        }
        if not result["success"]:
            result.update(self._export_failed_artifacts(mission, iteration, trace, events, error))
        return result

    def _failure_metrics(self, *, status: str, started_at: float, failure_reason: str) -> dict[str, Any]:
        return {
            "status": status,
            "success": False,
            "stable_success": False,
            "step_count": 0,
            "latency_seconds": round(time.time() - started_at, 6),
            "latency_per_step": 0.0,
            "wrong_click_count": 0,
            "recovery_attempt_count": 0,
            "recovery_success_count": 0,
            "recovery_depth_per_step": 0.0,
            "focus_failure_count": 0,
            "target_ambiguity_count": 0,
            "target_switch_after_recovery_count": 0,
            "duplicate_disambiguation_count": 0,
            "avg_score_gap": 0.0,
            "repair_vs_replan_count": {},
            "failure_reason_distribution": {failure_reason: 1},
            "trace_path": "",
            "contract_count": 0,
            "state_snapshot_count": 0,
            "state_diff_count": 0,
            "parse_count": 0,
            "ocr_elapsed_seconds": 0.0,
            "parse_elapsed_seconds": 0.0,
            "ocr_timeout_count": 0,
            "ocr_call_count": 0,
            "ocr_cache_hit_count": 0,
            "parse_cache_hit_count": 0,
            "state_cache_hit_count": 0,
            "state_cache_miss_count": 0,
            "state_probe_elapsed_seconds": 0.0,
            "parse_count_avoided_by_state_cache": 0,
            "dom_parse_count": 0,
            "parse_worker_count": 0,
            "parse_worker_elapsed_seconds": 0.0,
            "parse_worker_queue_wait_seconds": 0.0,
            "structured_skip_count": 0,
            "screen_hash_elapsed_seconds": 0.0,
        }

    def _export_failed_artifacts(
        self,
        mission: BenchmarkMission,
        iteration: int,
        trace: RunTrace | dict[str, Any] | None,
        events: list[dict[str, Any]],
        error: str,
    ) -> dict[str, Any]:
        artifact_dir = self.output_dir / "failed_artifacts" / f"{mission.mission_id}_run_{iteration:02d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "artifact_dir": str(artifact_dir),
            "trace_copy": "",
            "screenshot_copies": [],
        }

        outputs = _trace_outputs(trace)
        trace_path = str(outputs.get("trace_path", "") or "")
        copied_trace = self._copy_optional_file(trace_path, artifact_dir)
        if copied_trace:
            payload["trace_copy"] = copied_trace

        if trace is not None and not copied_trace:
            trace_dump = artifact_dir / "trace_inline.json"
            trace_payload = trace if isinstance(trace, dict) else trace.to_dict()
            trace_dump.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            payload["trace_copy"] = str(trace_dump)

        manifest = {
            "mission": mission.to_dict(),
            "iteration": iteration,
            "error": error,
            "events": events,
        }
        (artifact_dir / "failure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        screenshots = []
        for artifact_name in sorted(self._collect_screenshot_names(outputs)):
            copied = self._copy_optional_file(artifact_name, artifact_dir)
            if copied:
                screenshots.append(copied)
            raw_name = self._raw_screenshot_name(artifact_name)
            if raw_name:
                copied_raw = self._copy_optional_file(raw_name, artifact_dir)
                if copied_raw:
                    screenshots.append(copied_raw)
        payload["screenshot_copies"] = screenshots
        return payload

    def _copy_optional_file(self, source: str, target_dir: Path) -> str:
        if not source:
            return ""
        source_path = Path(source)
        candidates = []
        if source_path.is_absolute():
            candidates.append(source_path)
        else:
            candidates.extend(
                [
                    Path.cwd() / source_path,
                    Path.cwd() / "debug_steps" / source_path.name,
                ]
            )
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                target = target_dir / candidate.name
                shutil.copy2(candidate, target)
                return str(target)
        return ""

    def _collect_screenshot_names(self, payload: Any) -> set[str]:
        names: set[str] = set()
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in {"screenshot_before", "screenshot_after", "output_filename"} and isinstance(value, str):
                    suffix = Path(value).suffix.lower()
                    if suffix in {".png", ".jpg", ".jpeg"}:
                        names.add(value)
                names.update(self._collect_screenshot_names(value))
        elif isinstance(payload, list):
            for item in payload:
                names.update(self._collect_screenshot_names(item))
        return names

    def _raw_screenshot_name(self, artifact_name: str) -> str:
        path = Path(artifact_name)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            return ""
        if path.stem.endswith("_raw"):
            return ""
        return str(path.with_name(f"{path.stem}_raw{path.suffix}"))
