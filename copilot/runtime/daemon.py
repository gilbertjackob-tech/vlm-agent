from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import json
import time
import urllib.parse

from copilot.benchmark import BenchmarkRunner
from copilot.runtime.engine import CopilotEngine
from copilot.runtime.run_control import CancelLevel, RunRegistry
from copilot.schemas import TrustMode


class LocalDaemon:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, engine_factory=None) -> None:
        self.host = host
        self.port = int(port)
        self.registry = RunRegistry(engine_factory or CopilotEngine)

    def serve_forever(self) -> None:
        daemon = self
        factory = self.registry.engine_factory

        class Handler(CopilotDaemonHandler):
            registry = daemon.registry
            engine_factory = staticmethod(factory)

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        server.serve_forever()


class CopilotDaemonHandler(BaseHTTPRequestHandler):
    registry: RunRegistry
    engine_factory = CopilotEngine
    server_version = "CopilotLocalDaemon/0.1"

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path.strip("/")
        payload = self._read_json()
        if path == "tasks":
            prompt = str(payload.get("prompt") or payload.get("task") or "").strip()
            if not prompt:
                return self._send_json({"error": "prompt_required"}, status=400)
            try:
                trust_mode = TrustMode(str(payload.get("trust_mode") or TrustMode.PLAN_AND_RISK_GATES.value))
            except ValueError:
                return self._send_json({"error": "invalid_trust_mode"}, status=400)
            run_id = self.registry.start_task(
                prompt,
                trust_mode=trust_mode,
                dry_run=bool(payload.get("dry_run", False)),
                voice_mode=payload.get("voice_mode"),
                auto_approve=bool(payload.get("auto_approve", False)),
                approval_timeout=float(payload.get("approval_timeout", 600.0) or 600.0),
            )
            return self._send_json({"run_id": run_id, "status": "running"})

        if path == "plans":
            prompt = str(payload.get("prompt") or payload.get("task") or "").strip()
            if not prompt:
                return self._send_json({"error": "prompt_required"}, status=400)
            try:
                trust_mode = TrustMode(str(payload.get("trust_mode") or TrustMode.PLAN_AND_RISK_GATES.value))
            except ValueError:
                return self._send_json({"error": "invalid_trust_mode"}, status=400)
            engine = self.engine_factory()
            plan = engine.plan_prompt(prompt, trust_mode=trust_mode)
            return self._send_json({"plan": plan.to_dict()})

        parts = path.split("/")
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "cancel":
            try:
                level = CancelLevel(str(payload.get("level") or CancelLevel.SOFT.value))
            except ValueError:
                return self._send_json({"error": "invalid_cancel_level"}, status=400)
            ok = self.registry.cancel(parts[1], level)
            return self._send_json({"ok": ok, "run_id": parts[1], "cancel_level": level.value}, status=200 if ok else 404)

        if len(parts) == 2 and parts[0] == "approvals":
            if "approved" not in payload:
                return self._send_json({"error": "approved_required"}, status=400)
            decision = str(payload.get("decision") or ("allow_once" if payload.get("approved") else "cancel")).strip().lower()
            if decision not in {"allow_once", "always_allow_app", "cancel"}:
                return self._send_json({"error": "invalid_approval_decision"}, status=400)
            ok = self.registry.decide_approval(parts[1], bool(payload.get("approved")), decision=decision)
            return self._send_json({"ok": ok, "approval_id": parts[1]}, status=200 if ok else 404)

        if path == "skills":
            workflow_id = str(payload.get("workflow_id") or "").strip()
            action = str(payload.get("action") or "").strip().lower()
            if action == "export":
                engine = self.engine_factory()
                workflow_ids = [str(item) for item in payload.get("workflow_ids", [])] if isinstance(payload.get("workflow_ids", []), list) else []
                return self._send_json(engine.build_skill_manifest(workflow_ids or None))
            if action == "import":
                manifest = payload.get("manifest", {})
                if not isinstance(manifest, dict):
                    return self._send_json({"error": "manifest_required"}, status=400)
                engine = self.engine_factory()
                return self._send_json(engine.import_skill_manifest(manifest))
            if action in {"record_replay", "score_replay"}:
                if not workflow_id:
                    return self._send_json({"error": "workflow_id_required"}, status=400)
                engine = self.engine_factory()
                replay = engine.record_skill_replay(
                    workflow_id,
                    success=bool(payload.get("success", False)),
                    variant_count=max(1, int(payload.get("variant_count", 1) or 1)),
                    latency_seconds=(float(payload["latency_seconds"]) if payload.get("latency_seconds") is not None else None),
                    trace=payload.get("trace") if isinstance(payload.get("trace"), dict) else None,
                )
                if not replay:
                    return self._send_json({"ok": False, "workflow_id": workflow_id, "action": action}, status=404)
                return self._send_json({"ok": True, "workflow_id": workflow_id, "action": action, "skill": replay})
            if not workflow_id or action not in {"approve", "promote", "demote"}:
                return self._send_json({"error": "workflow_id_and_valid_action_required"}, status=400)
            engine = self.engine_factory()
            if action == "approve":
                ok = engine.approve_workflow(workflow_id)
            else:
                state = "draft" if action == "demote" else str(payload.get("promotion_state") or "trusted").strip().lower()
                if state not in {"draft", "verified", "trusted"}:
                    return self._send_json({"error": "invalid_promotion_state"}, status=400)
                ok = engine.promote_workflow(workflow_id, state)
            return self._send_json({"ok": ok, "workflow_id": workflow_id, "action": action}, status=200 if ok else 409)

        if path == "benchmarks/run":
            output_dir = str(payload.get("output_dir") or "benchmark_runs/daemon_dry")
            runner = BenchmarkRunner(
                output_dir=output_dir,
                repeat_count=max(1, int(payload.get("repeat", 1) or 1)),
                live_actions=False,
                auto_approve=bool(payload.get("auto_approve", False)),
                voice_mode=payload.get("voice_mode"),
            )
            report = runner.run(
                mission_ids=set(payload.get("missions", [])) if payload.get("missions") else None,
                max_missions=int(payload.get("max_missions", 0) or 0) or None,
            )
            return self._send_json(report)

        return self._send_json({"error": "not_found"}, status=404)

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path.strip("/")
        if path == "runs":
            return self._send_json({"runs": self.registry.list_runs()})
        if path == "approvals":
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            return self._send_json(
                {
                    "approvals": self.registry.list_approvals(
                        run_id=(query.get("run_id", [""])[0] or ""),
                        pending_only=(query.get("pending", ["false"])[0].lower() in {"1", "true", "yes"}),
                    )
                }
            )
        if path == "skills":
            engine = self.engine_factory()
            if hasattr(engine, "get_skill_capsules"):
                return self._send_json({"skills": engine.get_skill_capsules()})
            return self._send_json({"skills": engine.get_workflows()})

        parts = path.split("/")
        if len(parts) >= 2 and parts[0] == "runs":
            record = self.registry.get(parts[1])
            if not record:
                return self._send_json({"error": "run_not_found"}, status=404)
            if len(parts) == 3 and parts[2] == "xray":
                return self._send_json({"nodes": self._xray_nodes(record)})
            if len(parts) == 3 and parts[2] == "stream":
                return self._send_stream(parts[1])
            return self._send_json(record)

        return self._send_json({"status": "ok"})

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, run_id: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        sent = 0
        terminal = {"success", "failed", "blocked", "cancelled"}
        while True:
            record = self.registry.get(run_id)
            if not record:
                return
            events = record.get("events", [])
            for event in events[sent:]:
                line = f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            sent = len(events)
            if record.get("lifecycle") in terminal:
                snapshot = {
                    "phase": "done",
                    "message": f"Run {record.get('lifecycle')}.",
                    "metadata": {"run_id": run_id, "status": record.get("lifecycle"), "snapshot": record.get("snapshot", {})},
                }
                self.wfile.write(f"data: {json.dumps(snapshot, ensure_ascii=False, default=str)}\n\n".encode("utf-8"))
                self.wfile.flush()
                return
            time.sleep(0.1)

    def _xray_nodes(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        trace = record.get("trace")
        payload = trace.to_dict() if hasattr(trace, "to_dict") else trace
        if not isinstance(payload, dict):
            payload = {}
        outputs = payload.get("outputs", {}) if isinstance(payload.get("outputs", {}), dict) else {}
        observation = outputs.get("last_observation", {}) if isinstance(outputs.get("last_observation", {}), dict) else {}
        nodes = []
        for node in self._flatten_observation_nodes(observation.get("nodes", [])):
            box = node.get("box", {}) if isinstance(node.get("box", {}), dict) else {}
            if not box:
                continue
            node_type = str(node.get("type", ""))
            role = str(node.get("semantic_role", ""))
            affordances = [str(item).lower() for item in node.get("affordances", []) if str(item)]
            clickable = bool({"click", "open", "focus", "navigate"} & set(affordances))
            if not clickable and node_type not in {"button", "text_field", "input", "textarea", "a"} and role not in {"button", "link", "text_field", "menu_item", "clickable_container"}:
                continue
            nodes.append(
                {
                    "id": str(node.get("node_id") or node.get("id", "")),
                    "label": str(node.get("label", "")),
                    "type": node_type,
                    "semantic_role": role,
                    "box": box,
                    "confidence": float(node.get("stability", 0.8) or 0.8),
                    "risk_level": "low",
                }
            )
        if nodes:
            return nodes[:200]

        contracts = outputs.get("action_contracts", []) if isinstance(outputs.get("action_contracts", []), list) else []
        for contract in contracts:
            if not isinstance(contract, dict):
                continue
            identity = contract.get("target_identity", {}) if isinstance(contract.get("target_identity", {}), dict) else {}
            bounds = identity.get("bounds", {}) if isinstance(identity.get("bounds", {}), dict) else {}
            if not bounds:
                continue
            nodes.append(
                {
                    "id": str(identity.get("target_id", "")),
                    "label": str(identity.get("name", contract.get("target", ""))),
                    "type": str(identity.get("role", contract.get("action_type", ""))),
                    "semantic_role": str(identity.get("role", "")),
                    "box": bounds,
                    "confidence": float(identity.get("confidence", contract.get("evidence_score", 0.0)) or 0.0),
                    "risk_level": "low" if not contract.get("failure_reason") else "medium",
                }
            )
        return nodes[:200]

    def _flatten_observation_nodes(self, raw_nodes: Any) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        if not isinstance(raw_nodes, list):
            return flattened
        for node in raw_nodes:
            if not isinstance(node, dict):
                continue
            flattened.append(node)
            flattened.extend(self._flatten_observation_nodes(node.get("children", [])))
        return flattened
