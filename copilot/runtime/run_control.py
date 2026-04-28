from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from threading import Event, Lock, Thread
from typing import Any, Callable
import time
import uuid

from copilot.schemas import RunStatus, RunTrace, TrustMode
from copilot.runtime.confidence import confidence_from_trace_event


class CancelLevel(str, Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


class RunLifecycle(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    CANCELLING = "cancelling"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class CancelState:
    level: CancelLevel = CancelLevel.NONE
    requested_at: float = 0.0
    effective_at: float = 0.0
    cancelled_step_id: str = ""
    forced_cancel: bool = False

    def requested(self) -> bool:
        return self.level != CancelLevel.NONE

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["level"] = self.level.value
        return payload


@dataclass
class ApprovalRequest:
    approval_id: str
    run_id: str
    prompt: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    decided_at: float = 0.0
    approved: bool | None = None
    status: str = "pending"
    _event: Event = field(default_factory=Event, repr=False)

    def decide(self, approved: bool) -> None:
        self.approved = bool(approved)
        self.status = "approved" if approved else "rejected"
        self.decided_at = time.time()
        self._event.set()

    def wait(self, timeout: float) -> bool:
        if not self._event.wait(timeout=max(0.1, timeout)):
            self.status = "timeout"
            self.decided_at = time.time()
            return False
        return bool(self.approved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "prompt": self.prompt,
            "payload": dict(self.payload),
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "approved": self.approved,
            "status": self.status,
        }


class RunRegistry:
    def __init__(self, engine_factory: Callable[[], Any]) -> None:
        self.engine_factory = engine_factory
        self._lock = Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._approvals: dict[str, ApprovalRequest] = {}

    def start_task(
        self,
        prompt: str,
        *,
        trust_mode: TrustMode = TrustMode.PLAN_AND_RISK_GATES,
        dry_run: bool = False,
        voice_mode: str | None = None,
        auto_approve: bool = False,
        approval_timeout: float = 600.0,
    ) -> str:
        run_id = f"daemon_{uuid.uuid4().hex[:10]}"
        engine = self.engine_factory()
        now = time.time()
        record = {
            "run_id": run_id,
            "prompt": prompt,
            "status": RunLifecycle.QUEUED.value,
            "lifecycle": RunLifecycle.QUEUED.value,
            "created_at": now,
            "updated_at": now,
            "engine": engine,
            "trace": None,
            "events": [],
            "approvals": [],
            "cancel": CancelState().to_dict(),
            "snapshot": self._initial_snapshot(run_id, prompt),
            "error": "",
        }
        with self._lock:
            self._runs[run_id] = record

        def approval_callback(prompt: str, payload: dict[str, Any]) -> bool:
            if auto_approve:
                return True
            return self.request_approval(run_id, prompt, payload, timeout=approval_timeout)

        def worker() -> None:
            try:
                self._set_lifecycle(run_id, RunLifecycle.RUNNING)
                trace = engine.execute_prompt(
                    prompt,
                    trust_mode=trust_mode,
                    approval_callback=approval_callback,
                    trace_callback=lambda event: self.add_event(run_id, event),
                    dry_run=dry_run,
                    voice_mode=voice_mode,
                )
                with self._lock:
                    record["trace"] = trace
                    record["status"] = trace.status.value
                    record["lifecycle"] = self._lifecycle_from_trace_status(trace.status.value).value
                    self._sync_snapshot_from_trace(record, trace)
                    record["updated_at"] = time.time()
            except Exception as exc:
                with self._lock:
                    record["status"] = RunStatus.FAILED.value
                    record["lifecycle"] = RunLifecycle.FAILED.value
                    record["error"] = str(exc)
                    record["snapshot"]["status"] = RunLifecycle.FAILED.value
                    record["snapshot"]["message"] = str(exc)
                    record["updated_at"] = time.time()

        Thread(target=worker, name=f"copilot-run-{run_id}", daemon=True).start()
        return run_id

    def add_event(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if not record:
                return
            normalized = self._normalize_event(run_id, event)
            record.setdefault("events", []).append(normalized)
            record["snapshot"] = self._snapshot_from_event(record.get("snapshot", {}), normalized)
            record["updated_at"] = time.time()

    def request_approval(self, run_id: str, prompt: str, payload: dict[str, Any], *, timeout: float = 600.0) -> bool:
        approval = ApprovalRequest(
            approval_id=f"approval_{uuid.uuid4().hex[:10]}",
            run_id=run_id,
            prompt=prompt,
            payload=dict(payload or {}),
        )
        with self._lock:
            record = self._runs.get(run_id)
            if not record:
                return False
            self._approvals[approval.approval_id] = approval
            record["status"] = RunLifecycle.WAITING_APPROVAL.value
            record["lifecycle"] = RunLifecycle.WAITING_APPROVAL.value
            record.setdefault("approvals", []).append(approval.approval_id)
            record.setdefault("events", []).append(
                {
                    "phase": "approval",
                    "message": prompt,
                    "metadata": {"approval_id": approval.approval_id, **dict(payload or {})},
                    "timestamp": approval.created_at,
                }
            )
            record["snapshot"] = self._snapshot_from_event(record.get("snapshot", {}), record["events"][-1])
            record["updated_at"] = time.time()
        approved = approval.wait(timeout)
        with self._lock:
            record = self._runs.get(run_id)
            if record:
                if record.get("lifecycle") == RunLifecycle.WAITING_APPROVAL.value:
                    record["status"] = RunLifecycle.RUNNING.value
                    record["lifecycle"] = RunLifecycle.RUNNING.value
                record.setdefault("events", []).append(
                    {
                        "phase": "approval",
                        "message": approval.status,
                        "metadata": {"approval_id": approval.approval_id, "approved": approved},
                        "timestamp": approval.decided_at,
                    }
                )
                record["snapshot"] = self._snapshot_from_event(record.get("snapshot", {}), record["events"][-1])
                record["updated_at"] = time.time()
        return approved

    def decide_approval(self, approval_id: str, approved: bool, decision: str = "allow_once") -> bool:
        with self._lock:
            approval = self._approvals.get(approval_id)
            run_record = self._runs.get(approval.run_id) if approval else None
        if not approval or approval.status != "pending":
            return False
        decision = str(decision or "allow_once").strip().lower()
        if approved and decision == "always_allow_app":
            app_id = str(approval.payload.get("app_id") or "").strip()
            engine = run_record.get("engine") if run_record else None
            if app_id and engine and hasattr(engine, "allow_high_risk_for_app"):
                engine.allow_high_risk_for_app(app_id)
                approval.payload["decision"] = decision
                approval.payload["policy_update"] = f"high_risk_allowed_app:{app_id}"
        approval.decide(approved)
        return True

    def list_approvals(self, run_id: str = "", *, pending_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            approvals = list(self._approvals.values())
        if run_id:
            approvals = [approval for approval in approvals if approval.run_id == run_id]
        if pending_only:
            approvals = [approval for approval in approvals if approval.status == "pending"]
        return [approval.to_dict() for approval in approvals]

    def cancel(self, run_id: str, level: CancelLevel = CancelLevel.SOFT) -> bool:
        with self._lock:
            record = self._runs.get(run_id)
            if not record:
                return False
            engine = record.get("engine")
            record["status"] = RunLifecycle.CANCELLING.value
            record["lifecycle"] = RunLifecycle.CANCELLING.value
            record["cancel"] = {
                "level": level.value,
                "requested_at": time.time(),
                "effective_at": 0.0,
                "cancelled_step_id": "",
                "forced_cancel": False,
            }
            record.setdefault("events", []).append(
                {
                    "phase": "cancel",
                    "message": f"{level.value} cancel requested",
                    "metadata": {"run_id": run_id, "cancel_level": level.value},
                    "timestamp": record["cancel"]["requested_at"],
                }
            )
            record["snapshot"] = self._snapshot_from_event(record.get("snapshot", {}), record["events"][-1])
            record["updated_at"] = time.time()
        if hasattr(engine, "request_cancel"):
            engine.request_cancel(level.value)
            return True
        if hasattr(engine, "request_stop"):
            engine.request_stop()
            return True
        return False

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._runs.get(run_id)
            if not record:
                return None
            return self._public_record(record)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_record(record) for record in self._runs.values()]

    def _public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        trace = record.get("trace")
        return {
            "run_id": record.get("run_id", ""),
            "prompt": record.get("prompt", ""),
            "status": record.get("status", ""),
            "lifecycle": record.get("lifecycle", record.get("status", "")),
            "created_at": record.get("created_at", 0.0),
            "updated_at": record.get("updated_at", 0.0),
            "error": record.get("error", ""),
            "snapshot": dict(record.get("snapshot", {})),
            "cancel": dict(record.get("cancel", {})),
            "events": list(record.get("events", []))[-200:],
            "approvals": [
                self._approvals[approval_id].to_dict()
                for approval_id in record.get("approvals", [])
                if approval_id in self._approvals
            ],
            "trace": trace.to_dict() if hasattr(trace, "to_dict") else trace,
        }

    def _set_lifecycle(self, run_id: str, lifecycle: RunLifecycle) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if not record:
                return
            record["status"] = lifecycle.value
            record["lifecycle"] = lifecycle.value
            record["snapshot"]["status"] = lifecycle.value
            record["updated_at"] = time.time()

    def _initial_snapshot(self, run_id: str, prompt: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "status": RunLifecycle.QUEUED.value,
            "current_step": "queued",
            "target_summary": "",
            "message": prompt,
            "confidence": {"level": "MEDIUM", "score": 0.5, "reasons": ["queued"]},
            "pending_approval_id": "",
            "last_event": None,
        }

    def _normalize_event(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(event or {})
        metadata = normalized.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("run_id", run_id)
        normalized["metadata"] = metadata
        normalized.setdefault("phase", "runtime")
        normalized.setdefault("message", "")
        normalized.setdefault("timestamp", time.time())
        return normalized

    def _snapshot_from_event(self, previous: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        snapshot = dict(previous or {})
        metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
        signal = confidence_from_trace_event(event).to_dict()
        snapshot.update(
            {
                "run_id": metadata.get("run_id", snapshot.get("run_id", "")),
                "current_step": metadata.get("step_id", snapshot.get("current_step", "")),
                "target_summary": metadata.get("target", metadata.get("app_id", snapshot.get("target_summary", ""))),
                "message": event.get("message", snapshot.get("message", "")),
                "confidence": signal,
                "last_event": event,
            }
        )
        if event.get("phase") == "approval" and metadata.get("approval_id") and event.get("message") not in {"approved", "rejected", "timeout"}:
            snapshot["pending_approval_id"] = metadata.get("approval_id", "")
        elif event.get("phase") == "approval" and event.get("message") in {"approved", "rejected", "timeout"}:
            snapshot["pending_approval_id"] = ""
        if event.get("phase") == "cancel":
            snapshot["status"] = RunLifecycle.CANCELLING.value
        return snapshot

    def _sync_snapshot_from_trace(self, record: dict[str, Any], trace: RunTrace) -> None:
        snapshot = dict(record.get("snapshot", {}))
        snapshot["status"] = self._lifecycle_from_trace_status(trace.status.value).value
        snapshot["pending_approval_id"] = ""
        outputs = getattr(trace, "outputs", {}) if trace else {}
        if isinstance(outputs, dict):
            cancel_level = outputs.get("cancel_level")
            if cancel_level:
                record["cancel"] = {
                    "level": cancel_level,
                    "requested_at": outputs.get("cancel_requested_at", 0.0),
                    "effective_at": outputs.get("cancel_effective_at", 0.0),
                    "cancelled_step_id": outputs.get("cancelled_step_id", ""),
                    "forced_cancel": bool(outputs.get("forced_cancel", False)),
                }
        snapshot["message"] = f"Run {trace.status.value}."
        record["snapshot"] = snapshot

    def _lifecycle_from_trace_status(self, status: str) -> RunLifecycle:
        normalized = str(status or "").lower()
        if normalized == RunStatus.SUCCESS.value:
            return RunLifecycle.SUCCESS
        if normalized == RunStatus.BLOCKED.value:
            return RunLifecycle.BLOCKED
        if normalized == RunStatus.CANCELLED.value:
            return RunLifecycle.CANCELLED
        if normalized == RunStatus.FAILED.value:
            return RunLifecycle.FAILED
        return RunLifecycle.RUNNING
