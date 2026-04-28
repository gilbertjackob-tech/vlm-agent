from __future__ import annotations

from collections import deque
import json
import os
import time
from threading import Event, RLock
from typing import Any, Callable


EventListener = Callable[[dict[str, Any]], None]

_listeners: list[EventListener] = []
_recent_events: deque[dict[str, Any]] = deque(maxlen=200)
_stop_event = Event()
_lock = RLock()
_log_path = os.path.join("logs", "live_feed.log")


def subscribe(fn: EventListener) -> EventListener:
    """Register a process-local event listener."""
    with _lock:
        if fn not in _listeners:
            _listeners.append(fn)
    return fn


def unsubscribe(fn: EventListener) -> None:
    with _lock:
        if fn in _listeners:
            _listeners.remove(fn)


def emit(event: dict[str, Any]) -> None:
    normalized = normalize_event(event)
    with _lock:
        listeners = list(_listeners)
        _recent_events.append(normalized)
    _persist_recent_events()
    for listener in listeners:
        try:
            listener(dict(normalized))
        except Exception:
            continue


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = dict(event or {})
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    payload["metadata"] = metadata
    payload.setdefault("type", "status")
    payload.setdefault("msg", payload.get("message", ""))
    payload.setdefault("message", payload.get("msg", ""))
    payload.setdefault("timestamp", time.time())
    if "step" not in payload and metadata.get("step_index") is not None:
        payload["step"] = metadata.get("step_index")
    if "step_total" not in payload and metadata.get("step_total") is not None:
        payload["step_total"] = metadata.get("step_total")
    if "confidence" not in payload:
        payload["confidence"] = metadata.get("confidence", metadata.get("evidence_score", 0.0))
    return payload


def recent_events() -> list[dict[str, Any]]:
    with _lock:
        return list(_recent_events)


def request_stop() -> None:
    _stop_event.set()
    emit({"type": "error", "msg": "Stop requested by operator.", "phase": "stop", "metadata": {"status": "stopping"}})


def clear_stop() -> None:
    _stop_event.clear()


def stop_requested() -> bool:
    return _stop_event.is_set()


def _persist_recent_events() -> None:
    with _lock:
        events = list(_recent_events)
    try:
        os.makedirs(os.path.dirname(_log_path), exist_ok=True)
        with open(_log_path, "w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except OSError:
        return
