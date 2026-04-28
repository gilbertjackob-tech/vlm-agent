from __future__ import annotations

import json
import threading
import traceback
import tkinter as tk
import time
from tkinter import filedialog, messagebox, ttk

from copilot.runtime.daemon import LocalDaemon
from copilot.runtime.engine import CopilotEngine
from copilot.schemas import ObservationNode, TrustMode
from copilot.ui.overlay import DaemonOverlayApp, DaemonOverlayClient
from copilot.ui.shell_state import OperatorShellState, discover_benchmark_reports


class Tooltip:
    def __init__(self, widget, text: str, delay_ms: int = 500) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._window = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self._window or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self._window = tk.Toplevel(self.widget)
        self._window.wm_overrideredirect(True)
        self._window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self._window,
            text=self.text,
            justify="left",
            bg="#061116",
            fg="#e8fbff",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            wraplength=320,
            font=("Segoe UI", 9),
        )
        label.pack()

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._window:
            self._window.destroy()
            self._window = None


class CopilotPanel:
    def __init__(self, engine: CopilotEngine, base_url: str = "http://127.0.0.1:8765") -> None:
        self.engine = engine
        self.base_url = base_url.rstrip("/")
        self.root = tk.Tk()
        self.root.title("VLM Autopilot Command Center")
        self.root.geometry("1440x900")
        self.root.minsize(920, 620)
        self.client = DaemonOverlayClient(self.base_url)
        self.shell_state = OperatorShellState()
        self.overlay_app: DaemonOverlayApp | None = None
        self._embedded_daemon_thread: threading.Thread | None = None
        self.colors = {
            "bg": "#071014",
            "panel": "#0d1b22",
            "panel_2": "#102833",
            "panel_3": "#132f3b",
            "text": "#e8fbff",
            "muted": "#7fa7b4",
            "cyan": "#22d3ee",
            "blue": "#38bdf8",
            "green": "#3ff59b",
            "amber": "#ffc857",
            "red": "#ff4d6d",
            "border": "#1e4f5e",
        }

        self.current_nodes: list[ObservationNode] = []
        self.current_review_items: list[dict] = []
        self.current_plan = None
        self.worker_active = False
        self.worker_started_at = 0.0
        self.restore_after_worker = False
        self.action_buttons: list[ttk.Button] = []
        self.stop_button: ttk.Button | None = None
        self.activity_steps: list[str] = []
        self.developer_mode = tk.BooleanVar(value=False)

        self._ensure_daemon()
        self._build_layout()
        self.overlay_app = DaemonOverlayApp(base_url=self.base_url)
        self._apply_developer_mode()
        self.refresh_memory_summary()
        self.refresh_learning_dashboard()
        self.refresh_review_queue()
        self.refresh_workflows()
        self.refresh_recent_runs()
        self.refresh_approvals()
        self.refresh_benchmarks()
        self.root.after(1500, self.refresh_remote_state)

    def _build_layout(self) -> None:
        self._configure_style()
        self.root.configure(bg=self.colors["bg"])
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=0)

        hero = ttk.Frame(self.root, padding=(16, 14), style="Hero.TFrame")
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 6))
        hero.columnconfigure(1, weight=1)
        ttk.Label(hero, text="VLM AUTOPILOT", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(hero, text="hybrid vision + deterministic control + reward graph", style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Checkbutton(hero, text="Developer Mode", variable=self.developer_mode, command=self._apply_developer_mode).grid(row=0, column=1, sticky="e", padx=(0, 14))
        self.hero_status = ttk.Label(hero, text="ONLINE", style="Online.TLabel")
        self.hero_status.grid(row=0, column=2, rowspan=2, sticky="e")

        top = ttk.Frame(self.root, padding=(12, 10), style="Card.TFrame")
        top.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 8))
        top.columnconfigure(0, weight=1)

        ttk.Label(top, text="Mission Prompt", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.prompt_text = tk.Text(
            top,
            height=2,
            wrap="word",
            bg=self.colors["panel_2"],
            fg=self.colors["text"],
            insertbackground=self.colors["cyan"],
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            font=("Cascadia Code", 10),
            selectbackground=self.colors["cyan"],
            selectforeground="#001116",
        )
        self.prompt_text.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        self.prompt_text.insert("1.0", "Open explorer and parse screen")

        button_bar = ttk.Frame(top, style="Card.TFrame")
        button_bar.grid(row=1, column=1, sticky="ns")
        button_bar.columnconfigure((0, 1), weight=1)

        self._add_action_button(button_bar, "Plan Task", self.on_plan, 0, 0, tooltip="Build the execution plan only. No desktop actions are taken.")
        self._add_action_button(button_bar, "Simulate", lambda: self.on_run(dry_run=True), 0, 1, tooltip="Run planning and policy checks without clicking or typing.")
        self._add_action_button(button_bar, "Run Task", lambda: self.on_run(dry_run=False), 1, 0, tooltip="Execute the prompt against the desktop using policy and approval gates.")
        self._add_action_button(button_bar, "Scan App", self.on_parse_ui, 1, 1, tooltip="Observe the current desktop/app and load detected UI nodes.")
        self._add_action_button(button_bar, "Learn Hover", self.on_hover_learn, 2, 0, tooltip="Minimize the panel and learn labels/tooltips by hovering safe controls.")
        self._add_action_button(button_bar, "Learn Clicks", self.on_interaction_learn, 2, 1, tooltip="Autonomously test safe clicks and record rewards. Use only in a safe app/window.")
        self._add_action_button(button_bar, "Explore App", self.on_explore, 3, 0, tooltip="Run a scripted exploratory pass. Best for developer testing, not production tasks.")
        self.stop_button = ttk.Button(button_bar, text="Stop", command=self.on_stop, style="Danger.TButton")
        self.stop_button.grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        Tooltip(self.stop_button, "Request stop at the next safe checkpoint.")
        self.stop_button.state(["disabled"])
        self._add_action_button(button_bar, "Save Workflow", self.on_save_skill, 4, 0, tooltip="Save the current plan as a reusable workflow.")
        self._add_action_button(button_bar, "Export Memory", self.on_export_memory, 4, 1, tooltip="Export learned memory to a portable JSON pack.")
        self._add_action_button(button_bar, "Import Memory", self.on_import_memory, 5, 0, columnspan=2, tooltip="Import a previously exported memory pack.")

        activity = ttk.Frame(top, style="Activity.TFrame", padding=(10, 7))
        activity.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        activity.columnconfigure(1, weight=1)
        ttk.Label(activity, text="Thinking", style="ActivityBadge.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.activity_message = tk.StringVar(value="Idle. Waiting for a mission.")
        ttk.Label(activity, textvariable=self.activity_message, style="Activity.TLabel").grid(row=0, column=1, sticky="ew")
        self.activity_count = tk.StringVar(value="0 steps")
        ttk.Label(activity, textvariable=self.activity_count, style="ActivityMuted.TLabel").grid(row=0, column=2, sticky="e", padx=(10, 0))

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))

        left_shell = ttk.Frame(body, style="Card.TFrame")
        right_shell = ttk.Frame(body, style="Card.TFrame")
        left_shell.rowconfigure(0, weight=1)
        left_shell.columnconfigure(0, weight=1)
        right_shell.rowconfigure(0, weight=1)
        right_shell.columnconfigure(0, weight=1)
        body.add(left_shell, weight=3)
        body.add(right_shell, weight=2)

        left = ttk.Notebook(left_shell)
        left.grid(row=0, column=0, sticky="nsew")
        self.left_notebook = left

        plan_frame = ttk.Frame(left, padding=10, style="Card.TFrame")
        activity_frame = ttk.Frame(left, padding=10, style="Card.TFrame")
        trace_frame = ttk.Frame(left, padding=10, style="Card.TFrame")
        left.add(plan_frame, text="Plan")
        left.add(activity_frame, text="Thinking")
        left.add(trace_frame, text="Trace")
        self.left_tabs = {
            "Plan": plan_frame,
            "Thinking": activity_frame,
            "Trace": trace_frame,
        }

        plan_frame.rowconfigure(0, weight=1)
        plan_frame.columnconfigure(0, weight=1)
        trace_frame.rowconfigure(0, weight=1)
        trace_frame.columnconfigure(0, weight=1)
        activity_frame.rowconfigure(0, weight=1)
        activity_frame.columnconfigure(0, weight=1)

        self.plan_view = self._make_console_text(plan_frame)
        self.plan_view.grid(row=0, column=0, sticky="nsew")

        self.activity_view = self._make_console_text(activity_frame)
        self.activity_view.grid(row=0, column=0, sticky="nsew")

        self.trace_view = self._make_console_text(trace_frame)
        self.trace_view.grid(row=0, column=0, sticky="nsew")
        self.trace_view.tag_configure("error", foreground="#b00020")
        self.trace_view.tag_configure("warn", foreground=self.colors["amber"])
        self.trace_view.tag_configure("success", foreground=self.colors["green"])
        self.trace_view.tag_configure("normal", foreground=self.colors["text"])

        right = ttk.Notebook(right_shell)
        right.grid(row=0, column=0, sticky="nsew")
        self.right_notebook = right

        memory_frame = ttk.Frame(right, padding=10, style="Card.TFrame")
        nodes_frame = ttk.Frame(right, padding=10, style="Card.TFrame")
        review_frame = ttk.Frame(right, padding=10, style="Card.TFrame")
        learning_frame = ttk.Frame(right, padding=10, style="Card.TFrame")
        workflows_frame = ttk.Frame(right, padding=10, style="Card.TFrame")
        recent_frame = ttk.Frame(right, padding=10, style="Card.TFrame")
        approvals_frame = ttk.Frame(right, padding=10, style="Card.TFrame")
        benchmark_frame = ttk.Frame(right, padding=10, style="Card.TFrame")
        right.add(memory_frame, text="Memory")
        right.add(nodes_frame, text="Detected Nodes")
        right.add(review_frame, text="Review Queue")
        right.add(learning_frame, text="Learning Graph")
        right.add(workflows_frame, text="Workflows")
        right.add(recent_frame, text="Recent Runs")
        right.add(approvals_frame, text="Approvals")
        right.add(benchmark_frame, text="Benchmarks")
        self.right_tabs = {
            "Memory": memory_frame,
            "Detected Nodes": nodes_frame,
            "Review Queue": review_frame,
            "Learning Graph": learning_frame,
            "Workflows": workflows_frame,
            "Recent Runs": recent_frame,
            "Approvals": approvals_frame,
            "Benchmarks": benchmark_frame,
        }

        memory_frame.rowconfigure(3, weight=1)
        memory_frame.columnconfigure(0, weight=1)
        ttk.Label(memory_frame, text="Operator Readiness", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.operator_status_view = self._make_console_text(memory_frame, height=8)
        self.operator_status_view.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        ttk.Label(memory_frame, text="Memory Summary", style="Section.TLabel").grid(row=2, column=0, sticky="w")
        self.memory_view = self._make_console_text(memory_frame, height=16)
        self.memory_view.grid(row=3, column=0, sticky="nsew")

        nodes_frame.rowconfigure(1, weight=1)
        nodes_frame.columnconfigure(0, weight=1)
        ttk.Label(nodes_frame, text="Observation Nodes", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.node_list = self._make_listbox(nodes_frame)
        self.node_list.grid(row=1, column=0, sticky="nsew")
        self.node_list.bind("<<ListboxSelect>>", lambda _event: self.on_node_selected())

        teach_box = ttk.LabelFrame(nodes_frame, text="Teach Selected Node", padding=10, style="Panel.TLabelframe")
        teach_box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        teach_box.columnconfigure(1, weight=1)

        ttk.Label(teach_box, text="Label").grid(row=0, column=0, sticky="w")
        self.label_entry = ttk.Entry(teach_box)
        self.label_entry.grid(row=0, column=1, sticky="ew")

        ttk.Label(teach_box, text="Concepts").grid(row=1, column=0, sticky="w")
        self.concepts_entry = ttk.Entry(teach_box)
        self.concepts_entry.grid(row=1, column=1, sticky="ew")

        ttk.Label(teach_box, text="Entity Type").grid(row=2, column=0, sticky="w")
        self.entity_type_entry = ttk.Entry(teach_box)
        self.entity_type_entry.grid(row=2, column=1, sticky="ew")

        ttk.Label(teach_box, text="Affordances").grid(row=3, column=0, sticky="w")
        self.affordances_entry = ttk.Entry(teach_box)
        self.affordances_entry.grid(row=3, column=1, sticky="ew")

        ttk.Label(teach_box, text="App").grid(row=4, column=0, sticky="w")
        self.app_entry = ttk.Entry(teach_box)
        self.app_entry.grid(row=4, column=1, sticky="ew")

        ttk.Label(teach_box, text="Risk").grid(row=5, column=0, sticky="w")
        self.risk_entry = ttk.Entry(teach_box)
        self.risk_entry.grid(row=5, column=1, sticky="ew")

        self.outcome_correct = tk.BooleanVar(value=True)
        ttk.Checkbutton(teach_box, text="Current outcome was correct", variable=self.outcome_correct).grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Button(teach_box, text="Teach", command=self.on_teach_selected).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        review_frame.rowconfigure(1, weight=1)
        review_frame.columnconfigure(0, weight=1)
        ttk.Label(review_frame, text="Pending Learning Reviews", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.review_list = self._make_listbox(review_frame)
        self.review_list.grid(row=1, column=0, sticky="nsew")
        self.review_list.bind("<<ListboxSelect>>", lambda _event: self.on_review_selected())

        review_box = ttk.LabelFrame(review_frame, text="Resolve Selected Review", padding=10, style="Panel.TLabelframe")
        review_box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        review_box.columnconfigure(1, weight=1)

        ttk.Label(review_box, text="Label").grid(row=0, column=0, sticky="w")
        self.review_label_entry = ttk.Entry(review_box)
        self.review_label_entry.grid(row=0, column=1, sticky="ew")

        ttk.Label(review_box, text="Concepts").grid(row=1, column=0, sticky="w")
        self.review_concepts_entry = ttk.Entry(review_box)
        self.review_concepts_entry.grid(row=1, column=1, sticky="ew")

        ttk.Label(review_box, text="Entity Type").grid(row=2, column=0, sticky="w")
        self.review_entity_entry = ttk.Entry(review_box)
        self.review_entity_entry.grid(row=2, column=1, sticky="ew")

        review_buttons = ttk.Frame(review_box)
        review_buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        review_buttons.columnconfigure((0, 1, 2, 3), weight=1)
        ttk.Button(review_buttons, text="Accept", command=lambda: self.on_resolve_review("accepted")).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(review_buttons, text="Correct", command=lambda: self.on_resolve_review("corrected")).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(review_buttons, text="Mark Unsafe", command=lambda: self.on_resolve_review("unsafe")).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(review_buttons, text="Skip", command=lambda: self.on_resolve_review("skipped")).grid(row=0, column=3, sticky="ew", padx=2)

        learning_frame.rowconfigure(1, weight=1)
        learning_frame.columnconfigure(0, weight=1)
        learning_top = ttk.Frame(learning_frame, style="Card.TFrame")
        learning_top.grid(row=0, column=0, sticky="ew")
        learning_top.columnconfigure(0, weight=1)
        ttk.Label(learning_top, text="Interaction Graph Dashboard", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(learning_top, text="Refresh", command=self.refresh_learning_dashboard).grid(row=0, column=1, sticky="e")
        self.learning_graph_view = self._make_console_text(learning_frame)
        self.learning_graph_view.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        workflows_frame.rowconfigure(0, weight=1)
        workflows_frame.columnconfigure(0, weight=1)
        self.workflow_view = self._make_console_text(workflows_frame)
        self.workflow_view.grid(row=0, column=0, sticky="nsew")

        recent_frame.rowconfigure(0, weight=1)
        recent_frame.columnconfigure(0, weight=1)
        self.recent_runs_view = self._make_console_text(recent_frame)
        self.recent_runs_view.grid(row=0, column=0, sticky="nsew")

        approvals_frame.rowconfigure(0, weight=1)
        approvals_frame.columnconfigure(0, weight=1)
        self.approvals_view = self._make_console_text(approvals_frame)
        self.approvals_view.grid(row=0, column=0, sticky="nsew")

        benchmark_frame.rowconfigure(0, weight=1)
        benchmark_frame.columnconfigure(0, weight=1)
        self.benchmark_view = self._make_console_text(benchmark_frame)
        self.benchmark_view.grid(row=0, column=0, sticky="nsew")

        status = ttk.Frame(self.root, padding=(12, 8), style="Status.TFrame")
        status.grid(row=3, column=0, columnspan=2, sticky="ew")
        status.columnconfigure(1, weight=1)
        self.status_badge = ttk.Label(status, text="IDLE", style="Idle.TLabel", width=12, anchor="center")
        self.status_badge.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.status_message = tk.StringVar(value="Ready. No task is running.")
        ttk.Label(status, textvariable=self.status_message).grid(row=0, column=1, sticky="ew")
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=2, sticky="e", padx=(8, 0))

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        c = self.colors
        style.configure(".", background=c["bg"], foreground=c["text"], font=("Segoe UI", 10))
        style.configure("Hero.TFrame", background=c["panel"])
        style.configure("Card.TFrame", background=c["panel"])
        style.configure("Activity.TFrame", background="#061d26")
        style.configure("Status.TFrame", background="#061116")
        style.configure("TNotebook", background=c["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=c["panel_2"], foreground=c["muted"], padding=(16, 8), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", c["panel_3"])], foreground=[("selected", c["cyan"])])
        style.configure("TButton", background=c["panel_3"], foreground=c["text"], padding=(12, 7), borderwidth=1, focusthickness=0)
        style.map("TButton", background=[("active", "#15546a"), ("disabled", "#17262d")], foreground=[("disabled", "#58717a")])
        style.configure("Danger.TButton", background="#4c1420", foreground="#ffdce4", padding=(12, 7))
        style.map("Danger.TButton", background=[("active", "#8a1f35"), ("disabled", "#2a1820")])
        style.configure("Title.TLabel", background=c["panel"], foreground=c["cyan"], font=("Segoe UI Semibold", 18))
        style.configure("Subtitle.TLabel", background=c["panel"], foreground=c["muted"], font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=c["panel"], foreground=c["green"], font=("Segoe UI Semibold", 11))
        style.configure("ActivityBadge.TLabel", background="#073642", foreground=c["cyan"], padding=(10, 4), font=("Segoe UI Semibold", 9))
        style.configure("Activity.TLabel", background="#061d26", foreground=c["text"], font=("Segoe UI", 10))
        style.configure("ActivityMuted.TLabel", background="#061d26", foreground=c["muted"], font=("Segoe UI", 9))
        style.configure("Online.TLabel", background="#052e2b", foreground=c["green"], padding=(14, 7), font=("Segoe UI Semibold", 10))
        style.configure("Idle.TLabel", background="#064e3b", foreground="white", padding=(10, 5), font=("Segoe UI Semibold", 9))
        style.configure("Busy.TLabel", background="#92400e", foreground="white", padding=(10, 5), font=("Segoe UI Semibold", 9))
        style.configure("Error.TLabel", background="#9b1c31", foreground="white", padding=(10, 5), font=("Segoe UI Semibold", 9))
        style.configure("Panel.TLabelframe", background=c["panel"], foreground=c["text"], bordercolor=c["border"])
        style.configure("Panel.TLabelframe.Label", background=c["panel"], foreground=c["cyan"], font=("Segoe UI Semibold", 10))
        style.configure("TEntry", fieldbackground=c["panel_2"], foreground=c["text"], insertcolor=c["cyan"])
        style.configure("TCheckbutton", background=c["panel"], foreground=c["text"])
        style.configure("Horizontal.TProgressbar", troughcolor=c["panel_2"], background=c["cyan"], bordercolor=c["border"], lightcolor=c["cyan"], darkcolor=c["blue"])

    def _make_console_text(self, parent, height: int | None = None) -> tk.Text:
        return tk.Text(
            parent,
            height=height or 12,
            wrap="word",
            bg="#061116",
            fg=self.colors["text"],
            insertbackground=self.colors["cyan"],
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            font=("Cascadia Code", 10),
            selectbackground=self.colors["cyan"],
            selectforeground="#001116",
        )

    def _make_listbox(self, parent) -> tk.Listbox:
        return tk.Listbox(
            parent,
            exportselection=False,
            bg="#061116",
            fg=self.colors["text"],
            selectbackground=self.colors["cyan"],
            selectforeground="#001116",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            font=("Cascadia Code", 9),
        )

    def _add_action_button(
        self,
        parent: ttk.Frame,
        text: str,
        command,
        row: int,
        column: int = 0,
        columnspan: int = 1,
        tooltip: str = "",
    ) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        button.grid(row=row, column=column, columnspan=columnspan, sticky="ew", padx=2, pady=2)
        if tooltip:
            Tooltip(button, tooltip)
        self.action_buttons.append(button)
        return button

    def _ensure_daemon(self) -> None:
        if self.client.ping():
            return

        def serve() -> None:
            LocalDaemon(host="127.0.0.1", port=8765).serve_forever()

        self._embedded_daemon_thread = threading.Thread(target=serve, name="copilot-ui-daemon", daemon=True)
        self._embedded_daemon_thread.start()
        deadline = time.time() + 4.0
        while time.time() < deadline:
            if self.client.ping():
                return
            time.sleep(0.1)

    def _apply_developer_mode(self) -> None:
        self.shell_state.developer_mode = bool(self.developer_mode.get())
        left_visible = {"Thinking"}
        if self.developer_mode.get():
            left_visible.update({"Plan", "Trace"})
        for label, frame in self.left_tabs.items():
            if label in left_visible:
                if str(frame) not in self.left_notebook.tabs():
                    self.left_notebook.add(frame, text=label)
            else:
                self.left_notebook.hide(frame)

        right_visible = {"Workflows", "Recent Runs", "Approvals", "Benchmarks"}
        if self.developer_mode.get():
            right_visible.update({"Memory", "Detected Nodes", "Review Queue", "Learning Graph"})
        for label, frame in self.right_tabs.items():
            if label in right_visible:
                if str(frame) not in self.right_notebook.tabs():
                    self.right_notebook.add(frame, text=label)
            else:
                self.right_notebook.hide(frame)

    def refresh_remote_state(self) -> None:
        try:
            runs = self.client.list_runs()
            approvals = self.client.list_approvals(pending_only=True)
            skills = self.client.list_skills()
            self.shell_state.apply_daemon_payload(runs=runs, approvals=approvals, skills=skills)
            self.refresh_recent_runs()
            self.refresh_approvals()
            self.refresh_workflows()
        except Exception as exc:
            self._set_status_message(f"Daemon sync issue: {exc}")
        finally:
            self.root.after(1500, self.refresh_remote_state)

    def _append_trace(self, phase: str, message: str, metadata: dict | None = None) -> None:
        metadata = metadata or {}
        line = f"[{phase}] {message}"
        if metadata:
            line += f" | {json.dumps(metadata, ensure_ascii=False)}"
        tag = "normal"
        if phase in {"error", "blocked", "failure"}:
            tag = "error"
        elif phase in {"busy", "approval", "policy"}:
            tag = "warn"
        elif phase in {"done", "status", "learning", "interaction_learning", "replay"}:
            tag = "success"
        self.trace_view.insert("end", line + "\n", tag)
        self.trace_view.see("end")
        self._append_activity(phase, message, metadata)
        if phase not in {"step_result", "policy"}:
            self._set_status_message(message)

    def _append_activity(self, phase: str, message: str, metadata: dict | None = None) -> None:
        if not hasattr(self, "activity_view"):
            return
        metadata = metadata or {}
        visible_phases = {
            "plan",
            "policy",
            "step",
            "observation",
            "learning",
            "interaction_learning",
            "replay",
            "recover",
            "step_result",
            "blocked",
            "failure",
            "done",
            "status",
            "error",
        }
        if phase not in visible_phases:
            return
        verb = "Thinking"
        if phase == "policy":
            verb = "Checking safety"
        elif phase == "step":
            verb = "Doing"
        elif phase == "observation":
            verb = "Seeing"
        elif phase in {"learning", "interaction_learning"}:
            verb = "Learning"
        elif phase == "replay":
            verb = "Replaying memory"
        elif phase == "recover":
            verb = "Recovering"
        elif phase in {"blocked", "failure", "error"}:
            verb = "Needs attention"
        elif phase in {"done", "status"}:
            verb = "Status"
        suffix = ""
        if metadata.get("step_id"):
            suffix = f" ({metadata['step_id']})"
        line = f"{time.strftime('%H:%M:%S')}  {verb}{suffix}: {message}"
        self.activity_steps.append(line)
        self.activity_steps = self.activity_steps[-300:]
        self.activity_view.insert("end", line + "\n")
        self.activity_view.see("end")
        self.activity_message.set(f"{verb}: {message[:180]}")
        self.activity_count.set(f"{len(self.activity_steps)} steps")

    def _approval_dialog(self, prompt: str, payload: dict) -> bool:
        details = "\n".join(f"{key}: {value}" for key, value in payload.items() if key != "choices")
        app_id = str(payload.get("app_id") or "").strip()
        if not app_id:
            return messagebox.askyesno("Approval Required", f"{prompt}\n\n{details}")

        dialog = tk.Toplevel(self.root)
        dialog.title("Approval Required")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        result = {"approved": False}

        ttk.Label(dialog, text=prompt, font=("Segoe UI", 10, "bold"), wraplength=420).grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(14, 6))
        ttk.Label(dialog, text=details, justify="left", wraplength=420).grid(row=1, column=0, columnspan=3, sticky="w", padx=14, pady=(0, 12))

        def finish(approved: bool, always_allow: bool = False) -> None:
            result["approved"] = approved
            if approved and always_allow and hasattr(self.engine, "allow_high_risk_for_app"):
                self.engine.allow_high_risk_for_app(app_id)
            dialog.destroy()

        ttk.Button(dialog, text="Allow once", command=lambda: finish(True, False)).grid(row=2, column=0, sticky="ew", padx=(14, 4), pady=(0, 14))
        ttk.Button(dialog, text="Always allow for this app", command=lambda: finish(True, True)).grid(row=2, column=1, sticky="ew", padx=4, pady=(0, 14))
        ttk.Button(dialog, text="Cancel", command=lambda: finish(False, False)).grid(row=2, column=2, sticky="ew", padx=(4, 14), pady=(0, 14))
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False, False))
        self.root.wait_window(dialog)
        return bool(result["approved"])

    def _approval_dialog_threadsafe(self, prompt: str, payload: dict) -> bool:
        result = {"approved": False}
        completed = threading.Event()

        def show_dialog() -> None:
            result["approved"] = self._approval_dialog(prompt, payload)
            completed.set()

        self.root.after(0, show_dialog)
        completed.wait()
        return result["approved"]

    def _set_worker_active(self, active: bool) -> None:
        self.worker_active = active
        for button in self.action_buttons:
            if active:
                button.state(["disabled"])
            else:
                button.state(["!disabled"])
        if self.stop_button:
            if active:
                self.stop_button.state(["!disabled"])
            else:
                self.stop_button.state(["disabled"])
        if active:
            self.worker_started_at = time.time()
            self.activity_steps = []
            if hasattr(self, "activity_view"):
                self.activity_view.delete("1.0", "end")
            if hasattr(self, "activity_count"):
                self.activity_count.set("0 steps")
            if hasattr(self, "activity_message"):
                self.activity_message.set("Thinking: starting mission.")
            self.status_badge.configure(text="RUNNING", style="Busy.TLabel")
            self.progress.start(12)
        else:
            elapsed = time.time() - self.worker_started_at if self.worker_started_at else 0.0
            self.status_badge.configure(text="IDLE", style="Idle.TLabel")
            self.progress.stop()
            if elapsed > 0:
                self._set_status_message(f"Ready. Last task duration: {elapsed:.1f}s.")
                if hasattr(self, "activity_message") and self.status_badge.cget("text") != "ERROR":
                    self.activity_message.set(f"Idle. Last task duration: {elapsed:.1f}s.")

    def _set_status_message(self, message: str) -> None:
        if hasattr(self, "status_message"):
            self.status_message.set(message[:220])

    def on_stop(self) -> None:
        active_run_id = self.shell_state.selected_run_id
        if active_run_id:
            self.client.cancel_run(active_run_id, "soft")
        else:
            self.engine.request_stop()
        self.status_badge.configure(text="STOPPING", style="Error.TLabel")
        self._set_status_message("Stop requested. Waiting for the current safe checkpoint.")
        self._append_trace("status", "Stop requested by operator.")

    def _start_worker(self, label: str, worker) -> None:
        if self.worker_active:
            self._append_trace("busy", f"{label} ignored because another task is still running.")
            return

        self.worker_active = True
        self._set_worker_active(True)
        self._set_status_message(f"{label} started.")
        self._append_trace("status", f"{label} started.")

        def runner() -> None:
            error_message = ""
            try:
                worker()
            except Exception as exc:
                error_message = f"{label} failed: {exc}"
                self.root.after(
                    0,
                    self._append_trace,
                    "error",
                    f"{label} failed.",
                    {
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    },
                )
            finally:
                def finish() -> None:
                    self._set_worker_active(False)
                    if self.restore_after_worker:
                        self.restore_after_worker = False
                        self.root.deiconify()
                        self.root.lift()
                    if error_message:
                        self.status_badge.configure(text="ERROR", style="Error.TLabel")
                        self._set_status_message(error_message)

                self.root.after(0, finish)

        threading.Thread(target=runner, daemon=True).start()

    def on_plan(self) -> None:
        prompt = self.prompt_text.get("1.0", "end").strip()
        plan = self.engine.plan_prompt(prompt, trust_mode=TrustMode.PLAN_AND_RISK_GATES)
        self.current_plan = plan

        self.plan_view.delete("1.0", "end")
        self.plan_view.insert("end", f"Summary: {plan.summary}\n")
        self.plan_view.insert("end", f"Source: {plan.source}\n")
        self.plan_view.insert("end", f"Required Apps: {', '.join(plan.required_apps) or 'None'}\n\n")

        for step in plan.steps:
            target = step.target.value if step.target else ""
            self.plan_view.insert(
                "end",
                f"- {step.step_id}: {step.title}\n"
                f"  action={step.action_type} target={target}\n"
                f"  risk={step.risk_level.value} approval={step.requires_approval} confidence={step.confidence:.2f}\n"
                f"  success={step.success_criteria or 'n/a'}\n\n",
            )

    def on_run(
        self,
        dry_run: bool,
        handoff_target: bool = False,
        trust_mode: TrustMode = TrustMode.PLAN_AND_RISK_GATES,
    ) -> None:
        if self.worker_active:
            self._append_trace("busy", "Run ignored because another task is still running.")
            return
        prompt = self.prompt_text.get("1.0", "end").strip()
        self.trace_view.delete("1.0", "end")
        if hasattr(self, "left_notebook"):
            try:
                self.left_notebook.select(1)
            except tk.TclError:
                pass

        def worker() -> None:
            if handoff_target:
                self.root.after(0, self._append_trace, "status", "Target handoff: runtime is shifting to the taskbar bar while the app face stays available.")
                self.root.after(0, self._set_status_message, "Runtime bar is taking over this task.")
            run_id = self.client.start_task(prompt, dry_run=dry_run)
            self.shell_state.selected_run_id = run_id
            self.root.after(0, self._append_trace, "status", "Daemon run started.", {"run_id": run_id, "dry_run": dry_run})
            if self.overlay_app is not None:
                self.root.after(0, lambda: self.overlay_app.follow(run_id))
            self.root.after(0, self.refresh_recent_runs)
            self.root.after(0, self.refresh_approvals)

        if not handoff_target:
            self.on_plan()
        else:
            self.plan_view.delete("1.0", "end")
            self.plan_view.insert("end", "Plan will be compiled after target handoff.\n")
        self._start_worker("Run", worker)

    def on_parse_ui(self) -> None:
        def worker() -> None:
            graph = self.engine.parse_current_ui("panel_parse.png")
            self.root.after(0, self.populate_nodes, graph.flatten())
            self.root.after(0, self.refresh_memory_summary)
            self.root.after(0, self.refresh_learning_dashboard)
            self.root.after(0, self._append_trace, "observation", "UI parsed and nodes loaded.", graph.metadata)

        self._start_worker("Parse UI", worker)

    def on_hover_learn(self) -> None:
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", "Learn the current UI by hovering")
        self.on_run(dry_run=False, handoff_target=True)

    def on_interaction_learn(self) -> None:
        approved = messagebox.askyesno(
            "Confirm Autonomous Click Learning",
            "Learn Clicks will minimize this panel, click controls judged safe, and record what changes.\n\n"
            "Use it only inside a disposable or trusted app state. Continue?",
        )
        if not approved:
            self._append_trace("status", "Learn Clicks cancelled by operator.")
            return
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", "Learn what clicking safe controls opens in the current app")
        self.on_run(dry_run=False, handoff_target=True, trust_mode=TrustMode.MOSTLY_AUTONOMOUS)

    def on_explore(self) -> None:
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", "Open explorer and randomly explore the current UI for 4 rounds, then parse")
        self.on_run(dry_run=False)

    def populate_nodes(self, nodes: list[ObservationNode]) -> None:
        self.current_nodes = nodes
        self.node_list.delete(0, "end")
        for idx, node in enumerate(nodes):
            label = node.display_label() or node.label or node.node_id
            concepts = ",".join(node.learned_concepts[:3])
            entity_type = node.entity_type or "-"
            self.node_list.insert("end", f"{idx:03d} | {node.semantic_role or node.node_type} | {entity_type} | {label} | {concepts}")

    def on_node_selected(self) -> None:
        selection = self.node_list.curselection()
        if not selection:
            return
        node = self.current_nodes[selection[0]]
        self.label_entry.delete(0, "end")
        self.label_entry.insert(0, node.display_label() or node.label)
        self.concepts_entry.delete(0, "end")
        self.concepts_entry.insert(0, ", ".join(node.learned_concepts))
        self.entity_type_entry.delete(0, "end")
        self.entity_type_entry.insert(0, node.entity_type or "")
        self.affordances_entry.delete(0, "end")
        self.affordances_entry.insert(0, ", ".join(node.affordances))
        self.app_entry.delete(0, "end")
        self.app_entry.insert(0, node.app_id or "")
        if not node.app_id and node.region == "top_menu":
            self.app_entry.insert(0, node.display_label() or node.label)
        self.outcome_correct.set(True)

    def on_teach_selected(self) -> None:
        selection = self.node_list.curselection()
        if not selection:
            messagebox.showinfo("Teach Node", "Select a node first.")
            return

        node = self.current_nodes[selection[0]]
        label = self.label_entry.get().strip() or node.display_label() or node.label
        concepts = [item.strip() for item in self.concepts_entry.get().split(",") if item.strip()]
        entity_type = self.entity_type_entry.get().strip()
        affordances = [item.strip() for item in self.affordances_entry.get().split(",") if item.strip()]
        app_identity = self.app_entry.get().strip()
        risk_level = self.risk_entry.get().strip()
        outcome_correct = bool(self.outcome_correct.get())

        if not label:
            messagebox.showerror("Teach Node", "A label is required.")
            return

        self.engine.teach_node(
            node=node,
            label=label,
            concepts=concepts,
            app_identity=app_identity,
            risk_level=risk_level,
            entity_type=entity_type,
            affordances=affordances,
            outcome_correct=outcome_correct,
        )
        self.refresh_memory_summary()
        self.refresh_learning_dashboard()
        self.refresh_review_queue()
        self._append_trace(
            "teach",
            f"Taught node '{label}'",
            {
                "concepts": concepts,
                "entity_type": entity_type,
                "affordances": affordances,
                "app": app_identity,
                "risk": risk_level,
                "outcome_correct": outcome_correct,
            },
        )

    def refresh_review_queue(self) -> None:
        if not hasattr(self, "review_list"):
            return
        self.current_review_items = self.engine.get_review_items(status="pending", limit=80)
        self.review_list.delete(0, "end")
        for item in self.current_review_items:
            label = item.get("label", "")
            app_id = item.get("app_id", "-")
            entity_type = item.get("entity_type", "-")
            confidence = float(item.get("confidence", 0.0) or 0.0)
            feedback = ", ".join(item.get("feedback_labels", [])[:2])
            self.review_list.insert("end", f"{confidence:.2f} | {app_id} | {entity_type} | {label} | {feedback}")

    def on_review_selected(self) -> None:
        selection = self.review_list.curselection()
        if not selection:
            return
        item = self.current_review_items[selection[0]]
        self.review_label_entry.delete(0, "end")
        feedback = item.get("feedback_labels", [])
        self.review_label_entry.insert(0, feedback[0] if feedback else item.get("label", ""))
        self.review_concepts_entry.delete(0, "end")
        concepts = ["reviewed"]
        if item.get("entity_type"):
            concepts.append(str(item.get("entity_type")))
        self.review_concepts_entry.insert(0, ", ".join(concepts))
        self.review_entity_entry.delete(0, "end")
        self.review_entity_entry.insert(0, item.get("entity_type", ""))

    def on_resolve_review(self, status: str) -> None:
        selection = self.review_list.curselection()
        if not selection:
            messagebox.showinfo("Review Queue", "Select a review item first.")
            return
        item = self.current_review_items[selection[0]]
        label = self.review_label_entry.get().strip() or item.get("label", "")
        concepts = [part.strip() for part in self.review_concepts_entry.get().split(",") if part.strip()]
        entity_type = self.review_entity_entry.get().strip() or item.get("entity_type", "")
        ok = self.engine.resolve_review_item(
            review_id=item.get("review_id", ""),
            status=status,
            label=label,
            concepts=concepts,
            entity_type=entity_type,
            affordances=list(item.get("affordances", [])),
            app_identity=item.get("app_id", ""),
            note=f"Resolved from panel as {status}",
        )
        if not ok:
            messagebox.showerror("Review Queue", "Could not resolve the selected review item.")
            return
        self.refresh_review_queue()
        self.refresh_memory_summary()
        self.refresh_learning_dashboard()
        self._append_trace("review", f"Review item {status}: {label}", {"review_id": item.get("review_id", "")})

    def refresh_learning_dashboard(self) -> None:
        if not hasattr(self, "learning_graph_view"):
            return
        dashboard = self.engine.get_learning_dashboard()
        self.learning_graph_view.delete("1.0", "end")
        readiness = "READY" if dashboard.get("ready_for_replay") else "NOT READY"
        multistep = "READY" if dashboard.get("ready_for_multistep") else "NEEDS MORE EDGES"
        lines = [
            f"Replay readiness: {readiness}",
            f"Multi-step readiness: {multistep}",
            "",
            f"Scenes: {dashboard.get('scene_nodes', 0)}",
            f"Controls: {dashboard.get('control_nodes', 0)}",
            f"Edges: {dashboard.get('edges', 0)}  positive={dashboard.get('positive_edges', 0)}  negative={dashboard.get('negative_edges', 0)}",
            f"Attempts: {dashboard.get('attempts', 0)}  success={dashboard.get('successes', 0)}  fail={dashboard.get('failures', 0)}  rate={dashboard.get('success_rate', 0.0)}",
            "",
            "Top Learned Actions",
        ]
        top_actions = dashboard.get("top_actions", [])
        if top_actions:
            for item in top_actions:
                lines.append(
                    f"- {item.get('label', '')} | app={item.get('app_id', '')} | reward={item.get('reward_avg', 0)} | "
                    f"ok={item.get('successes', 0)} fail={item.get('failures', 0)} | {item.get('last_outcome', '')}"
                )
        else:
            lines.append("- none yet")

        lines.extend(["", "Weak / Punished Actions"])
        weak_actions = dashboard.get("weak_actions", [])
        if weak_actions:
            for item in weak_actions:
                lines.append(
                    f"- {item.get('label', '')} | app={item.get('app_id', '')} | reward={item.get('reward_avg', 0)} | "
                    f"ok={item.get('successes', 0)} fail={item.get('failures', 0)} | {item.get('last_outcome', '')}"
                )
        else:
            lines.append("- none yet")

        self.learning_graph_view.insert("end", "\n".join(lines))

    def on_save_skill(self) -> None:
        if not self.current_plan:
            self.on_plan()
        if not self.current_plan:
            messagebox.showinfo("Save Skill", "Create a plan first.")
            return

        skill_name = self.prompt_text.get("1.0", "end").strip() or self.current_plan.summary
        workflow = self.engine.save_current_plan_as_skill(skill_name)
        if not workflow:
            messagebox.showerror("Save Skill", "No plan is available to save.")
            return
        self.refresh_workflows()
        self.refresh_memory_summary()
        self._append_trace("skill", f"Saved workflow '{workflow.get('name', '')}'", {"workflow_id": workflow.get("workflow_id", "")})
        messagebox.showinfo("Save Skill", f"Saved reusable workflow:\n{workflow.get('name', '')}")

    def refresh_memory_summary(self) -> None:
        self.refresh_operator_status()
        summary = self.engine.get_memory_summary()
        self.memory_view.delete("1.0", "end")
        self.memory_view.insert("end", json.dumps(summary, indent=2, ensure_ascii=False))

    def refresh_operator_status(self) -> None:
        if not hasattr(self, "operator_status_view"):
            return
        status = self.engine.get_operator_status()
        learning = status.get("learning", {})
        memory = status.get("memory", {})
        blockers = status.get("blockers", [])
        lines = [
            f"Readiness: {status.get('level', 'unknown')} ({status.get('readiness_score', 0)}/100)",
            f"Next step: {status.get('next_step', '')}",
            f"Replay: {'allowed' if status.get('safe_to_replay') else 'not ready'} | Multi-step: {'allowed' if status.get('safe_for_multistep') else 'not ready'}",
            f"Known controls: {memory.get('known_controls', 0)} | visuals: {memory.get('known_visuals', 0)} | pending reviews: {memory.get('review_queue', 0)}",
            f"Click learning: edges={learning.get('positive_edges', 0)} positive/{learning.get('negative_edges', 0)} weak | success rate={learning.get('success_rate', 0.0)}",
            f"Blockers: {', '.join(blockers) if blockers else 'none'}",
        ]
        self.operator_status_view.delete("1.0", "end")
        self.operator_status_view.insert("end", "\n".join(lines))

    def refresh_workflows(self) -> None:
        self.workflow_view.delete("1.0", "end")
        workflows = self.shell_state.skills or self.engine.get_workflows()
        for workflow in workflows:
            skill_name = workflow.get("skill_name") or workflow.get("name", "")
            trigger = workflow.get("trigger_phrase") or workflow.get("prompt_pattern", "")
            success_rate = float(workflow.get("success_rate", 0.0) or 0.0) * 100.0
            app = workflow.get("app") or ",".join(workflow.get("required_apps", []))
            self.workflow_view.insert(
                "end",
                f"{workflow.get('promotion_state', 'candidate'):>10} | "
                f"{workflow.get('approval_status', 'pending')} | "
                f"success={workflow.get('success_count', 0)} fail={workflow.get('failure_count', 0)} rate={success_rate:.0f}% | "
                f"{skill_name}\n"
                f"id={workflow.get('workflow_id', '')}\n"
                f"trigger={trigger}\n"
                f"app={app} targets={len(workflow.get('selectors_uia_targets', workflow.get('targets', [])))}\n\n",
            )

    def refresh_recent_runs(self) -> None:
        self.recent_runs_view.delete("1.0", "end")
        rows = self.shell_state.run_rows()
        if not rows:
            for run in self.engine.get_recent_runs():
                self.recent_runs_view.insert(
                    "end",
                    f"{run['status']:>9} | {run['prompt']}\nTrace: {run['trace_path']}\n\n",
                )
            return
        for run in rows:
            self.recent_runs_view.insert(
                "end",
                f"{run['status']:>18} | {run['prompt']}\n"
                f"run_id={run['run_id']}\n"
                f"message={run['message']}\n"
                f"confidence={run['confidence']}\n\n",
            )

    def refresh_approvals(self) -> None:
        if not hasattr(self, "approvals_view"):
            return
        self.approvals_view.delete("1.0", "end")
        for approval in self.shell_state.approval_rows():
            self.approvals_view.insert(
                "end",
                f"{approval.get('status', ''):>10} | {approval.get('prompt', '')}\n"
                f"approval_id={approval.get('approval_id', '')}\n"
                f"run_id={approval.get('run_id', '')}\n\n",
            )

    def refresh_benchmarks(self) -> None:
        if not hasattr(self, "benchmark_view"):
            return
        self.shell_state.benchmark_reports = discover_benchmark_reports()
        self.benchmark_view.delete("1.0", "end")
        for report in self.shell_state.benchmark_reports[:12]:
            metrics = report.get("metrics", {})
            self.benchmark_view.insert(
                "end",
                f"{'LIVE' if report.get('live_actions') else 'DRY ':>4} | {report.get('path', '')}\n"
                f"stable_success_rate={metrics.get('stable_success_rate', 0)} wrong_click_count={metrics.get('wrong_click_count', 0)}\n\n",
            )

    def on_export_memory(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        saved_path = self.engine.export_memory(path)
        messagebox.showinfo("Export Memory", f"Memory exported to:\n{saved_path}")

    def on_import_memory(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        self.engine.import_memory(path)
        self.refresh_memory_summary()
        self.refresh_learning_dashboard()
        self.refresh_review_queue()
        self.refresh_workflows()
        messagebox.showinfo("Import Memory", "Memory pack imported successfully.")

    def run(self) -> None:
        self.root.mainloop()


def launch_panel() -> None:
    panel = CopilotPanel(CopilotEngine())
    panel.run()
