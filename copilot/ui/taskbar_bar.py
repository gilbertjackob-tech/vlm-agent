from __future__ import annotations

from typing import Any

try:
    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
    from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget
except ImportError as exc:  # pragma: no cover - exercised only when GUI deps are absent
    raise ImportError("TaskbarBar requires PySide6. Install it with: pip install PySide6") from exc

from copilot.core import event_bus


TYPE_LABELS = {
    "thinking": "Thinking",
    "seeing": "Seeing",
    "action": "Action",
    "verify": "Verify",
    "error": "Error",
    "status": "Status",
}

TYPE_COLORS = {
    "thinking": "#facc15",
    "seeing": "#a78bfa",
    "action": "#22d3ee",
    "verify": "#45d483",
    "error": "#ff6b6b",
    "status": "#d7dee8",
}


class _FeedBridge(QObject):
    event_received = Signal(dict)


class TaskbarBar(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setFixedHeight(96)
        self.setGeometry(0, 0, self._screen_width(), 96)
        self.setObjectName("TaskbarBar")
        self._pending_events: list[dict[str, Any]] = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(150)
        self._flush_timer.timeout.connect(self._flush_events)

        title = QLabel("Agent Live Feed")
        title.setObjectName("Title")
        self.state_header = QLabel("Idle | Step -/- | Confidence -- | SAFE")
        self.state_header.setObjectName("StateHeader")
        self.status = QLabel("RUN")
        self.status.setObjectName("RunStatus")
        stop_button = QPushButton("STOP")
        stop_button.setObjectName("StopButton")
        stop_button.clicked.connect(event_bus.request_stop)

        header = QHBoxLayout()
        header.setContentsMargins(12, 5, 12, 0)
        header.addWidget(title)
        header.addSpacing(16)
        header.addWidget(self.state_header, stretch=1)
        header.addWidget(self.status)
        header.addWidget(stop_button)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFrameShape(QTextEdit.NoFrame)
        self.text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self.text)
        self.setLayout(layout)

        self.setStyleSheet(
            """
            #TaskbarBar {
                background: #101418;
                border-bottom: 1px solid #2b3440;
            }
            #Title {
                color: #f8fafc;
                font: 600 13px "Segoe UI";
            }
            #StateHeader {
                color: #aab6c5;
                font: 12px "Segoe UI";
            }
            #RunStatus {
                color: #45d483;
                font: 700 12px "Segoe UI";
                padding: 0 8px;
            }
            #StopButton {
                background: #7f1d1d;
                color: #fee2e2;
                border: 1px solid #ef4444;
                border-radius: 4px;
                padding: 2px 10px;
                font: 700 11px "Segoe UI";
            }
            QTextEdit {
                background: #101418;
                color: #d7dee8;
                font: 12px "Cascadia Mono", "Consolas", monospace;
                padding: 1px 12px 6px 12px;
            }
            """
        )

    def push(self, msg: str, event_type: str = "status") -> None:
        clean = str(msg or "").strip()
        if not clean:
            return
        label = TYPE_LABELS.get(event_type, TYPE_LABELS["status"])
        color = QColor(TYPE_COLORS.get(event_type, TYPE_COLORS["status"]))
        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        cursor.insertText(f"{label}: {clean}\n", fmt)
        self.text.setTextCursor(cursor)
        self.text.ensureCursorVisible()

    def queue_event(self, event: dict[str, Any]) -> None:
        self._pending_events.append(dict(event or {}))
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def set_mode(self, mode: str) -> None:
        normalized = str(mode or "run").upper()
        color = QColor("#45d483")
        if normalized in {"ERROR", "FAILED", "BLOCKED", "STOPPING"}:
            color = QColor("#ff6b6b")
        elif normalized in {"DONE", "SUCCESS"}:
            color = QColor("#7dd3fc")
        elif normalized in {"IDLE", "WAITING"}:
            color = QColor("#facc15")
        self.status.setText(normalized)
        self.status.setStyleSheet(f"color: {color.name()};")

    def update_header(self, event: dict[str, Any]) -> None:
        metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
        app = str(metadata.get("app_id") or metadata.get("active_app") or metadata.get("target") or "Desktop")
        if app.lower() in {"chrome", "google chrome"}:
            app = "Chrome"
        step = event.get("step") or metadata.get("step_index") or "-"
        total = event.get("step_total") or metadata.get("step_total") or metadata.get("step_count") or "-"
        confidence = _confidence_percent(event.get("confidence", metadata.get("confidence", 0.0)))
        risk = str(metadata.get("risk_level") or metadata.get("status") or "safe").upper()
        if risk == "LOW":
            risk = "SAFE"
        self.state_header.setText(f"{app} | Step {step}/{total} | Confidence {confidence} | {risk}")

    def _flush_events(self) -> None:
        if not self._pending_events:
            self._flush_timer.stop()
            return
        events = self._pending_events[:]
        self._pending_events.clear()
        for event in events[-8:]:
            self.update_header(event)
            event_type = str(event.get("type") or "status")
            message = event_message(event)
            if message:
                self.push(message, event_type=event_type)
            status = _status_from_event(event)
            if status:
                self.set_mode(status)

    def _screen_width(self) -> int:
        screen = QApplication.primaryScreen()
        if screen is None:
            return 1920
        return int(screen.availableGeometry().width())


def _confidence_percent(value: Any) -> str:
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    if score <= 0:
        return "--"
    if score <= 1:
        score *= 100
    return f"{round(score):.0f}%"


def _status_from_event(event: dict[str, Any]) -> str:
    metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
    status = str(metadata.get("status") or event.get("phase") or event.get("type") or "").lower()
    if event.get("type") == "error":
        return "error"
    if status in {"done", "success", "failed", "blocked", "error", "waiting", "idle", "stopping"}:
        return status
    return ""


def event_message(event: dict[str, Any]) -> str:
    return str(event.get("msg") or event.get("message") or "").strip()


def attach_event_bus(bar: TaskbarBar) -> _FeedBridge:
    bridge = _FeedBridge()

    def handle_event(event: dict[str, Any]) -> None:
        bridge.event_received.emit(dict(event or {}))

    bridge.event_received.connect(bar.queue_event)
    event_bus.subscribe(handle_event)
    return bridge


def launch_taskbar() -> None:
    app = QApplication.instance() or QApplication([])
    bar = TaskbarBar()
    bar.push("Agent live feed ready.")
    attach_event_bus(bar)
    bar.show()
    app.exec()


if __name__ == "__main__":
    launch_taskbar()
