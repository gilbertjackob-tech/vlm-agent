from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import json
import tkinter as tk
import threading
import urllib.request
from tkinter import messagebox

from copilot.runtime.confidence import ConfidenceSignal, confidence_from_trace_event, derive_confidence


Priority = str


@dataclass
class OverlayState:
    run_id: str = ""
    goal: str = "Waiting for a task"
    current_step: str = "Idle"
    step_index: int = 0
    step_total: int = 0
    target_summary: str = ""
    status: str = "observing"
    mode: str = "Safe Mode"
    message: str = "Waiting for a task."
    narration: str = ""
    confidence: ConfidenceSignal = field(default_factory=lambda: derive_confidence(score_gap=1.0, focus_confidence=1.0))
    priority: Priority = "LOW"
    expanded: bool = False
    highlighted: bool = False
    approval_id: str = ""
    run_status: str = ""
    thought_lines: list[str] = field(default_factory=list)
    mind_lines: list[str] = field(default_factory=list)
    xray_nodes: list[dict[str, Any]] = field(default_factory=list)
    ghost_steps: list[dict[str, Any]] = field(default_factory=list)
    ghost_pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "current_step": self.current_step,
            "step_index": self.step_index,
            "step_total": self.step_total,
            "target_summary": self.target_summary,
            "status": self.status,
            "mode": self.mode,
            "message": self.message,
            "narration": self.narration,
            "confidence": self.confidence.to_dict(),
            "priority": self.priority,
            "expanded": self.expanded,
            "highlighted": self.highlighted,
            "approval_id": self.approval_id,
            "run_status": self.run_status,
            "thought_lines": list(self.thought_lines),
            "mind_lines": list(self.mind_lines),
            "xray_nodes": list(self.xray_nodes),
            "ghost_steps": list(self.ghost_steps),
            "ghost_pending": self.ghost_pending,
        }


def priority_for_event(event: dict[str, Any]) -> Priority:
    phase = str(event.get("phase") or "").lower()
    if phase in {"failure", "recovery", "replan", "approval", "cancel", "done", "blocked"}:
        return "HIGH"
    if phase in {"target", "action", "step", "policy", "route", "focus"}:
        return "MEDIUM"
    return "LOW"


def _step_index_from_id(step_id: str) -> int:
    digits = "".join(ch for ch in str(step_id or "") if ch.isdigit())
    return int(digits) if digits else 0


def mind_line_from_event(event: dict[str, Any]) -> str:
    phase = str(event.get("phase") or "").lower()
    metadata = event.get("metadata", {}) if isinstance(event.get("metadata", {}), dict) else {}
    message = str(event.get("message") or "")
    app_id = str(metadata.get("app_id") or metadata.get("target") or "").strip()
    action_type = str(metadata.get("action_type") or "").strip()
    if phase in {"focus", "route", "route_window"} and app_id:
        return f"I found {app_id.title()}."
    if phase in {"observation", "observe", "parse_ui"}:
        return "I mapped the visible controls."
    if phase == "target_found":
        target = str(metadata.get("target") or "").strip()
        return f"I found {target}." if target else "I found the target."
    if action_type == "type_text" or phase == "type_text":
        return "Typing query now."
    if phase in {"verify_scene", "checkpoint", "scene_diff"}:
        return "Verifying the result loaded."
    if phase == "contract" and "dom" in message.lower():
        return "DOM is connected."
    if phase == "recovery":
        return "Verification failed; trying recovery."
    if phase == "done":
        return message or "Run finished."
    return message


def overlay_state_from_event(previous: OverlayState, event: dict[str, Any]) -> OverlayState:
    metadata = event.get("metadata", {}) if isinstance(event.get("metadata", {}), dict) else {}
    priority = priority_for_event(event)
    phase = str(event.get("phase") or previous.status or "observing")
    message = str(event.get("message") or previous.message)
    step_id = str(metadata.get("step_id") or previous.current_step)
    step_index = int(metadata.get("step_index") or _step_index_from_id(step_id) or previous.step_index or 0)
    step_total = int(metadata.get("step_total") or metadata.get("total_steps") or previous.step_total or 0)
    action_type = str(metadata.get("action_type") or phase)
    target = str(metadata.get("target") or metadata.get("app_id") or previous.target_summary)
    approval_id = str(metadata.get("approval_id") or previous.approval_id)
    snapshot = metadata.get("snapshot", {}) if isinstance(metadata.get("snapshot", {}), dict) else {}
    goal = str(metadata.get("goal") or metadata.get("prompt") or snapshot.get("message") or previous.goal)
    mode = str(metadata.get("mode") or metadata.get("trust_mode") or previous.mode or "Safe Mode")
    if mode == "plan_and_risk_gates":
        mode = "Safe Mode"

    thought_lines = list(previous.thought_lines)
    if message:
        thought_lines.append(message)
        thought_lines = thought_lines[-8:]
    mind_lines = list(previous.mind_lines)
    mind_line = mind_line_from_event(event)
    if mind_line and (not mind_lines or mind_lines[-1] != mind_line):
        mind_lines.append(mind_line)
        mind_lines = mind_lines[-8:]

    xray_nodes = previous.xray_nodes
    if isinstance(metadata.get("xray_nodes"), list):
        xray_nodes = [item for item in metadata.get("xray_nodes", []) if isinstance(item, dict)]

    return OverlayState(
        run_id=str(metadata.get("run_id") or previous.run_id),
        goal=goal,
        current_step=step_id if step_id else previous.current_step,
        step_index=step_index,
        step_total=step_total,
        target_summary=target,
        status=action_type,
        mode=mode,
        message=message,
        narration=message if priority in {"HIGH", "MEDIUM"} else previous.narration,
        confidence=confidence_from_trace_event(event),
        priority=priority,
        expanded=priority == "HIGH" or previous.expanded,
        highlighted=priority == "HIGH",
        approval_id=approval_id,
        run_status=str(metadata.get("status") or previous.run_status),
        thought_lines=thought_lines,
        mind_lines=mind_lines,
        xray_nodes=xray_nodes,
        ghost_steps=previous.ghost_steps,
        ghost_pending=previous.ghost_pending,
    )


def ghost_preview_from_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    steps = []
    for idx, step in enumerate(plan.get("steps", []) if isinstance(plan, dict) else [], start=1):
        if not isinstance(step, dict):
            continue
        target = step.get("target", {}) if isinstance(step.get("target", {}), dict) else {}
        steps.append(
            {
                "index": idx,
                "step_id": str(step.get("step_id", "")),
                "title": str(step.get("title", "")),
                "action_type": str(step.get("action_type", "")),
                "target": str(target.get("value", "")),
                "risk_level": str(step.get("risk_level", "low")),
                "requires_approval": bool(step.get("requires_approval", False)),
                "confidence": float(step.get("confidence", 0.0) or 0.0),
            }
        )
    return steps


class OverlayBar:
    def __init__(
        self,
        *,
        on_submit: Callable[[str], None] | None = None,
        on_pause: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_hard_stop: Callable[[], None] | None = None,
        on_approve: Callable[[str, bool], None] | None = None,
        on_xray: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.state = OverlayState()
        self.on_submit = on_submit or (lambda _prompt: None)
        self.on_pause = on_pause or (lambda: None)
        self.on_stop = on_stop or (lambda: None)
        self.on_hard_stop = on_hard_stop or (lambda: None)
        self.on_approve = on_approve or (lambda _approval_id, _approved: None)
        self.on_xray = on_xray or (lambda: [])
        self.xray_window: tk.Toplevel | None = None
        self.root = tk.Toplevel()
        self.root.title("Operator")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.94)
        try:
            self.root.attributes("-toolwindow", True)
        except tk.TclError:
            pass
        self.root.configure(bg="#111827")
        self.root.bind("<Enter>", lambda _event: self.expand())
        self.root.bind("<Leave>", lambda _event: self.collapse_when_idle())
        self.root.bind("<F8>", lambda _event: self.toggle_xray())
        self._drag_start: tuple[int, int] | None = None
        self.root.bind("<ButtonPress-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._drag)
        self._build()
        self._dock_to_taskbar_band(expanded=False)
        self.render()

    def _build(self) -> None:
        self.status_var = tk.StringVar()
        self.goal_var = tk.StringVar()
        self.step_var = tk.StringVar()
        self.target_var = tk.StringVar()
        self.confidence_var = tk.StringVar()
        self.mode_var = tk.StringVar()
        self.message_var = tk.StringVar()
        self.prompt_var = tk.StringVar()

        self.frame = tk.Frame(self.root, bg="#111827", padx=10, pady=7)
        self.frame.pack(fill="both", expand=True)
        self.frame.columnconfigure(1, weight=1)
        self.frame.columnconfigure(3, weight=0)

        self.status_label = tk.Label(self.frame, textvariable=self.status_var, bg="#111827", fg="#93c5fd", font=("Segoe UI Semibold", 9))
        self.status_label.grid(row=0, column=0, sticky="w", padx=(0, 12))
        tk.Label(self.frame, textvariable=self.goal_var, bg="#111827", fg="#f9fafb", font=("Segoe UI Semibold", 10)).grid(row=0, column=1, sticky="w")
        tk.Label(self.frame, textvariable=self.step_var, bg="#111827", fg="#d1d5db", font=("Segoe UI", 9)).grid(row=0, column=2, sticky="e", padx=(12, 0))
        tk.Label(self.frame, textvariable=self.confidence_var, bg="#111827", fg="#fbbf24", font=("Segoe UI Semibold", 9)).grid(row=0, column=3, sticky="e", padx=(12, 0))
        tk.Label(self.frame, textvariable=self.mode_var, bg="#111827", fg="#3ff59b", font=("Segoe UI Semibold", 9)).grid(row=0, column=4, sticky="e", padx=(12, 0))
        tk.Label(self.frame, textvariable=self.message_var, bg="#111827", fg="#d1d5db", font=("Segoe UI", 9)).grid(row=1, column=0, columnspan=3, sticky="ew")

        self.controls = tk.Frame(self.frame, bg="#111827")
        self.controls.grid(row=0, column=5, rowspan=2, sticky="e", padx=(12, 0))
        tk.Button(self.controls, text="Pause", command=self.on_pause, width=7, relief="flat").grid(row=0, column=0, padx=2)
        tk.Button(self.controls, text="Stop", command=self.on_stop, width=6, relief="flat", bg="#7f1d1d", fg="#ffffff").grid(row=0, column=1, padx=2)
        tk.Button(self.controls, text="X-Ray", command=self.toggle_xray, width=6, relief="flat", bg="#164e63", fg="#ffffff").grid(row=0, column=2, padx=2)
        tk.Button(self.controls, text="Hard", command=self.on_hard_stop, width=6, relief="flat", bg="#5b1220", fg="#ffffff").grid(row=0, column=3, padx=2)
        self.approve_button = tk.Button(self.controls, text="Approve", command=lambda: self.on_approve(self.state.approval_id, True), width=8, relief="flat")
        self.reject_button = tk.Button(self.controls, text="Reject", command=lambda: self.on_approve(self.state.approval_id, False), width=7, relief="flat")

        self.prompt_row = tk.Frame(self.frame, bg="#111827")
        self.prompt_row.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(8, 0))
        self.prompt_row.columnconfigure(0, weight=1)
        self.prompt_entry = tk.Entry(
            self.prompt_row,
            textvariable=self.prompt_var,
            relief="flat",
            bg="#0b1623",
            fg="#f9fafb",
            insertbackground="#f9fafb",
            highlightthickness=1,
            highlightbackground="#223243",
            font=("Segoe UI", 10),
        )
        self.prompt_entry.grid(row=0, column=0, sticky="ew")
        self.prompt_entry.bind("<Return>", lambda _event: self.submit_prompt())
        self.voice_button = tk.Button(self.prompt_row, text="Mic", state="disabled", width=6, relief="flat")
        self.voice_button.grid(row=0, column=1, padx=(6, 0))
        self.submit_button = tk.Button(self.prompt_row, text="Send", command=self.submit_prompt, width=7, relief="flat", bg="#0f766e", fg="#f9fafb")
        self.submit_button.grid(row=0, column=2, padx=(6, 0))

        self.thought_view = tk.Text(
            self.frame,
            height=4,
            wrap="word",
            relief="flat",
            bg="#0b1623",
            fg="#d1d5db",
            insertbackground="#f9fafb",
            state="disabled",
            font=("Segoe UI", 9),
        )
        self.thought_view.grid(row=3, column=0, columnspan=6, sticky="nsew", pady=(8, 0))
        self.mind_view = tk.Text(
            self.frame,
            height=5,
            wrap="word",
            relief="flat",
            bg="#07121f",
            fg="#a7f3d0",
            insertbackground="#f9fafb",
            state="disabled",
            font=("Segoe UI", 9),
        )
        self.mind_view.grid(row=4, column=0, columnspan=6, sticky="nsew", pady=(8, 0))
        self.ghost_view = tk.Text(
            self.frame,
            height=5,
            wrap="word",
            relief="flat",
            bg="#160f24",
            fg="#e9d5ff",
            insertbackground="#f9fafb",
            state="disabled",
            font=("Segoe UI", 9),
        )
        self.ghost_view.grid(row=5, column=0, columnspan=6, sticky="nsew", pady=(8, 0))
        self.frame.rowconfigure(3, weight=1)

    def apply_event(self, event: dict[str, Any]) -> None:
        self.state = overlay_state_from_event(self.state, event)
        self.render()

    def render(self) -> None:
        self.status_var.set(self.state.status.upper())
        self.goal_var.set(self.state.goal[:72])
        step_text = f"Step {self.state.step_index}/{self.state.step_total}" if self.state.step_index and self.state.step_total else self.state.current_step
        self.step_var.set(step_text)
        target = f" | {self.state.target_summary}" if self.state.target_summary else ""
        self.message_var.set(f"{self.state.message}{target}")
        self.confidence_var.set(f"Confidence {int(round(self.state.confidence.score * 100))}%")
        self.mode_var.set(self.state.mode)
        bg = "#3f1d1d" if self.state.highlighted else "#111827"
        self.root.configure(bg=bg)
        self.frame.configure(bg=bg)
        for widget in self.frame.winfo_children():
            try:
                widget.configure(bg=bg)
            except tk.TclError:
                pass
        self.prompt_row.configure(bg=bg)
        self.controls.configure(bg=bg)
        self.thought_view.configure(state="normal")
        self.thought_view.delete("1.0", "end")
        if self.state.thought_lines:
            self.thought_view.insert("1.0", "\n".join(self.state.thought_lines[-6:]))
        self.thought_view.configure(state="disabled")
        self.mind_view.configure(state="normal")
        self.mind_view.delete("1.0", "end")
        if self.state.mind_lines:
            self.mind_view.insert("1.0", "\n".join(self.state.mind_lines[-6:]))
        self.mind_view.configure(state="disabled")
        self.ghost_view.configure(state="normal")
        self.ghost_view.delete("1.0", "end")
        if self.state.ghost_steps:
            lines = ["Ghost Replay"]
            for item in self.state.ghost_steps[:8]:
                risk = str(item.get("risk_level", "low")).upper()
                target = f" -> {item.get('target')}" if item.get("target") else ""
                lines.append(f"{item.get('index')}. {item.get('action_type')}{target} [{risk}]")
            self.ghost_view.insert("1.0", "\n".join(lines))
        self.ghost_view.configure(state="disabled")
        if self.state.approval_id and self.state.priority == "HIGH":
            self.approve_button.grid(row=1, column=0, padx=2, pady=(4, 0))
            self.reject_button.grid(row=1, column=1, padx=2, pady=(4, 0))
        else:
            self.approve_button.grid_remove()
            self.reject_button.grid_remove()
        self.thought_view.grid() if self.state.expanded else self.thought_view.grid_remove()
        self.mind_view.grid() if self.state.expanded else self.mind_view.grid_remove()
        self.ghost_view.grid() if self.state.expanded and self.state.ghost_steps else self.ghost_view.grid_remove()
        self.prompt_row.grid() if self.state.expanded else self.prompt_row.grid_remove()

    def expand(self) -> None:
        self.state.expanded = True
        self._dock_to_taskbar_band(expanded=True)
        self.render()

    def collapse_when_idle(self) -> None:
        if self.state.priority != "HIGH":
            self.state.expanded = False
            self._dock_to_taskbar_band(expanded=False)
            self.render()

    def submit_prompt(self) -> None:
        prompt = self.prompt_var.get().strip()
        if not prompt:
            return
        self.prompt_var.set("")
        self.on_submit(prompt)

    def _dock_to_taskbar_band(self, *, expanded: bool) -> None:
        width = min(max(int(self.root.winfo_screenwidth() * 0.62), 840), self.root.winfo_screenwidth() - 80)
        height = 300 if expanded else 58
        x = max(20, int((self.root.winfo_screenwidth() - width) / 2))
        y = max(20, self.root.winfo_screenheight() - height - 68)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _start_drag(self, event) -> None:
        self._drag_start = (event.x, event.y)

    def _drag(self, event) -> None:
        if not self._drag_start:
            return
        x = self.root.winfo_x() + event.x - self._drag_start[0]
        y = self.root.winfo_y() + event.y - self._drag_start[1]
        self.root.geometry(f"+{x}+{y}")

    def set_ghost_preview(self, steps: list[dict[str, Any]]) -> None:
        self.state.ghost_steps = list(steps)
        self.state.ghost_pending = bool(steps)
        if steps:
            self.state.expanded = True
            self._dock_to_taskbar_band(expanded=True)
        self.render()

    def toggle_xray(self) -> None:
        if self.xray_window and self.xray_window.winfo_exists():
            self.xray_window.destroy()
            self.xray_window = None
            return
        nodes = self.on_xray() or self.state.xray_nodes
        self.state.xray_nodes = nodes
        self.xray_window = tk.Toplevel(self.root)
        self.xray_window.title("X-Ray Overlay")
        self.xray_window.attributes("-topmost", True)
        self.xray_window.attributes("-alpha", 0.82)
        self.xray_window.configure(bg="#020617")
        width = min(960, self.root.winfo_screenwidth() - 80)
        height = min(640, self.root.winfo_screenheight() - 140)
        self.xray_window.geometry(f"{width}x{height}+40+40")
        canvas = tk.Canvas(self.xray_window, bg="#020617", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.update_idletasks()
        self._draw_xray(canvas, nodes, width, height)

    def _draw_xray(self, canvas: tk.Canvas, nodes: list[dict[str, Any]], width: int, height: int) -> None:
        if not nodes:
            canvas.create_text(24, 24, anchor="nw", fill="#e5e7eb", font=("Segoe UI", 12), text="No X-Ray nodes available yet.")
            return
        boxes = [dict(item.get("box", {}) or {}) for item in nodes if isinstance(item.get("box", {}), dict)]
        max_x = max([int(box.get("x", 0) or 0) + int(box.get("width", 1) or 1) for box in boxes] + [width])
        max_y = max([int(box.get("y", 0) or 0) + int(box.get("height", 1) or 1) for box in boxes] + [height])
        scale = min(width / max(1, max_x), height / max(1, max_y))
        palette = {"text_field": "#22d3ee", "button": "#3ff59b", "link": "#fbbf24", "clickable": "#a78bfa"}
        for item in nodes[:160]:
            box = dict(item.get("box", {}) or {})
            x = int(box.get("x", 0) or 0) * scale
            y = int(box.get("y", 0) or 0) * scale
            w = max(8, int(box.get("width", 1) or 1) * scale)
            h = max(8, int(box.get("height", 1) or 1) * scale)
            role = str(item.get("semantic_role") or item.get("type") or "clickable")
            color = palette.get(role, palette.get(str(item.get("type", "")), "#38bdf8"))
            canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=2)
            label = str(item.get("label") or item.get("target") or role)[:32]
            risk = str(item.get("risk_level", "low")).upper()
            conf = item.get("confidence", "")
            suffix = f" {int(float(conf) * 100)}%" if isinstance(conf, (float, int)) else ""
            canvas.create_text(x + 4, y + 4, anchor="nw", fill=color, font=("Segoe UI", 8), text=f"{label} [{risk}]{suffix}")


class DaemonOverlayClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", opener: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._opener = opener or urllib.request.urlopen
        self._stop_stream = threading.Event()

    def start_task(
        self,
        prompt: str,
        *,
        dry_run: bool = False,
        voice_mode: str | None = None,
        auto_approve: bool = False,
    ) -> str:
        payload: dict[str, Any] = {"prompt": prompt, "dry_run": dry_run, "auto_approve": auto_approve}
        if voice_mode:
            payload["voice_mode"] = voice_mode
        response = self._json_request("POST", "/tasks", payload)
        return str(response.get("run_id", ""))

    def preview_plan(self, prompt: str) -> dict[str, Any]:
        if not prompt:
            return {}
        return self._json_request("POST", "/plans", {"prompt": prompt})

    def get_xray(self, run_id: str) -> list[dict[str, Any]]:
        if not run_id:
            return []
        return list(self._json_request("GET", f"/runs/{run_id}/xray", None).get("nodes", []))

    def cancel_run(self, run_id: str, level: str = "soft") -> bool:
        response = self._json_request("POST", f"/runs/{run_id}/cancel", {"level": level})
        return bool(response.get("ok", False))

    def decide_approval(self, approval_id: str, approved: bool, decision: str = "allow_once") -> bool:
        if not approval_id:
            return False
        response = self._json_request("POST", f"/approvals/{approval_id}", {"approved": bool(approved), "decision": decision})
        return bool(response.get("ok", False))

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._json_request("GET", f"/runs/{run_id}", None)

    def list_runs(self) -> list[dict[str, Any]]:
        return list(self._json_request("GET", "/runs", None).get("runs", []))

    def list_approvals(self, *, pending_only: bool = False) -> list[dict[str, Any]]:
        suffix = "?pending=true" if pending_only else ""
        return list(self._json_request("GET", f"/approvals{suffix}", None).get("approvals", []))

    def list_skills(self) -> list[dict[str, Any]]:
        return list(self._json_request("GET", "/skills", None).get("skills", []))

    def ping(self) -> bool:
        try:
            payload = self._json_request("GET", "/", None)
        except Exception:
            return False
        return payload.get("status") == "ok"

    def stream_run(self, run_id: str, on_event: Callable[[dict[str, Any]], None]) -> None:
        self._stop_stream.clear()
        with self._opener(f"{self.base_url}/runs/{run_id}/stream", timeout=30) as response:
            for raw_line in response:
                if self._stop_stream.is_set():
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                on_event(event)

    def stream_run_background(self, run_id: str, on_event: Callable[[dict[str, Any]], None]) -> threading.Thread:
        thread = threading.Thread(target=lambda: self.stream_run(run_id, on_event), name=f"overlay-stream-{run_id}", daemon=True)
        thread.start()
        return thread

    def stop_stream(self) -> None:
        self._stop_stream.set()

    def _json_request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with self._opener(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))


class DaemonOverlayApp:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", run_id: str = "") -> None:
        self.client = DaemonOverlayClient(base_url)
        self.run_id = run_id
        self.overlay = OverlayBar(
            on_submit=self.submit_prompt,
            on_pause=lambda: self.client.cancel_run(self.run_id, "soft") if self.run_id else None,
            on_stop=lambda: self.client.cancel_run(self.run_id, "hard") if self.run_id else None,
            on_hard_stop=lambda: self.client.cancel_run(self.run_id, "hard") if self.run_id else None,
            on_approve=lambda approval_id, approved: self.client.decide_approval(approval_id, approved),
            on_xray=lambda: self.client.get_xray(self.run_id) if self.run_id else [],
        )
        if run_id:
            self.follow(run_id)

    def follow(self, run_id: str) -> None:
        self.run_id = run_id
        run = self.client.get_run(run_id)
        snapshot = run.get("snapshot", {})
        self.overlay.state.run_id = run_id
        self.overlay.state.goal = str(run.get("prompt", "") or self.overlay.state.goal)
        self.overlay.state.current_step = str(snapshot.get("current_step", self.overlay.state.current_step))
        self.overlay.state.target_summary = str(snapshot.get("target_summary", self.overlay.state.target_summary))
        self.overlay.state.message = str(snapshot.get("message", self.overlay.state.message))
        self.overlay.state.run_status = str(snapshot.get("status", self.overlay.state.run_status))
        self.overlay.render()

        def apply(event: dict[str, Any]) -> None:
            self.overlay.root.after(0, lambda: self.overlay.apply_event(event))

        self.client.stream_run_background(run_id, apply)

    def submit_prompt(self, prompt: str) -> None:
        try:
            preview = self.client.preview_plan(prompt)
        except Exception:
            preview = {}
        ghost_steps = ghost_preview_from_plan(preview.get("plan", preview))
        risky = any(step.get("requires_approval") or str(step.get("risk_level", "low")).lower() in {"high", "critical"} for step in ghost_steps)
        multi_step = len(ghost_steps) >= 3
        if ghost_steps and (risky or multi_step):
            self.overlay.set_ghost_preview(ghost_steps)
            summary = "\n".join(
                f"{step.get('index')}. {step.get('action_type')} {step.get('target')}".strip()
                for step in ghost_steps[:8]
            )
            approved = messagebox.askyesno("Ghost Replay", f"I will run this sequence:\n\n{summary}\n\nApprove?")
            if not approved:
                self.overlay.apply_event({"phase": "approval", "message": "Ghost replay rejected.", "metadata": {"run_id": self.run_id}})
                return
        run_id = self.client.start_task(prompt, dry_run=False)
        self.overlay.state.goal = prompt
        self.overlay.state.step_total = len(ghost_steps)
        self.overlay.state.ghost_pending = False
        self.follow(run_id)


def launch_overlay(base_url: str = "http://127.0.0.1:8765", run_id: str = "") -> None:
    root = tk.Tk()
    root.withdraw()
    DaemonOverlayApp(base_url=base_url, run_id=run_id)
    root.mainloop()
