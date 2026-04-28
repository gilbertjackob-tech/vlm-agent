from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable
from hashlib import sha256
import json
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request


HttpPost = Callable[[str, dict[str, Any], float], tuple[bytes, str]]


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class VoiceConfig:
    mode: str = "off"
    provider: str = "localai"
    model: str = "voice-en-us-ryan-high"
    endpoint: str = "http://localhost:8080/v1/audio/speech"
    voice: str = "voice-en-us-ryan-high"
    timeout: float = 15.0
    speak_sensitive: bool = False
    throttle_seconds: float = 2.0
    cache_dir: str = os.path.join("debug_steps", "voice_cache")
    collapse_queue: bool = True

    @classmethod
    def from_env(cls, mode_override: str | None = None) -> "VoiceConfig":
        mode = str(mode_override or os.environ.get("COPILOT_VOICE_MODE", "off")).strip().lower()
        if mode in {"", "none", "disabled", "false", "0"}:
            mode = "off"
        return cls(
            mode=mode,
            provider=str(os.environ.get("COPILOT_TTS_PROVIDER", "localai")).strip().lower() or "localai",
            model=str(os.environ.get("COPILOT_TTS_MODEL", "voice-en-us-ryan-high")).strip() or "voice-en-us-ryan-high",
            endpoint=str(os.environ.get("COPILOT_TTS_ENDPOINT", "http://localhost:8080/v1/audio/speech")).strip(),
            voice=str(os.environ.get("COPILOT_TTS_VOICE", "voice-en-us-ryan-high")).strip() or "voice-en-us-ryan-high",
            timeout=max(0.1, _env_float("COPILOT_TTS_TIMEOUT", 15.0)),
            speak_sensitive=_env_bool("COPILOT_TTS_SPEAK_SENSITIVE", False),
            cache_dir=str(os.environ.get("COPILOT_TTS_CACHE_DIR", os.path.join("debug_steps", "voice_cache"))).strip() or os.path.join("debug_steps", "voice_cache"),
            collapse_queue=_env_bool("COPILOT_TTS_COLLAPSE_QUEUE", True),
        )


class VoiceNarrator:
    def __init__(self, config: VoiceConfig | None = None, http_post: HttpPost | None = None) -> None:
        self.config = config or VoiceConfig.from_env()
        self._http_post = http_post or self._default_http_post
        self._queue: Queue[dict[str, Any]] = Queue()
        self._worker_started = False
        self._worker_lock = threading.Lock()
        self._last_event_by_key: dict[str, float] = {}
        self._pending_events: list[dict[str, Any]] = []
        self._pending_lock = threading.Lock()
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, mode_override: str | None = None) -> "VoiceNarrator":
        return cls(VoiceConfig.from_env(mode_override=mode_override))

    @property
    def enabled(self) -> bool:
        return self.config.mode in {"console", "tts"}

    def speak(
        self,
        line: str,
        *,
        trace: Any | None = None,
        event_type: str = "runtime",
        throttle_key: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_line, redacted = self._sanitize(line)
        event = {
            "timestamp": time.time(),
            "mode": self.config.mode,
            "provider": self.config.provider,
            "model": self.config.model,
            "voice": self.config.voice,
            "event_type": event_type,
            "line": clean_line,
            "status": "disabled",
            "sensitive_redacted": redacted,
            "metadata": dict(metadata or {}),
        }
        self._record_trace_event(trace, event)

        if not self.enabled or not clean_line:
            return event
        if throttle_key and self._is_throttled(throttle_key):
            event["status"] = "throttled"
            return event
        if self.config.mode == "console" or self.config.provider != "localai":
            print(f"[voice] {clean_line}")
            event["status"] = "console"
            return event

        if self.config.collapse_queue:
            self._collapse_pending(event_type=event_type)
        event["status"] = "queued"
        with self._pending_lock:
            self._pending_events.append(event)
        self._start_worker()
        self._queue.put({"line": clean_line, "event": event})
        return event

    def speak_phase(
        self,
        phase: str,
        *,
        trace: Any | None = None,
        step: Any | None = None,
        metadata: dict[str, Any] | None = None,
        throttle_key: str = "",
    ) -> dict[str, Any]:
        event_metadata = dict(metadata or {})
        line = self._phase_line(phase, step=step, metadata=event_metadata)
        intent_key = throttle_key or self._phase_intent_key(phase, step=step, metadata=event_metadata)
        return self.speak(line, trace=trace, event_type=phase, throttle_key=intent_key, metadata=event_metadata)

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        if self.config.mode != "tts":
            return True
        deadline = time.time() + max(0.1, timeout_seconds)
        while time.time() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.05)
        with self._pending_lock:
            for event in self._pending_events:
                if event.get("status") == "queued":
                    event["status"] = "dropped"
        return False

    def cancel(self) -> None:
        with self._pending_lock:
            for event in self._pending_events:
                if event.get("status") == "queued":
                    event["status"] = "dropped"
            self._pending_events = []
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                break

    def _record_trace_event(self, trace: Any | None, event: dict[str, Any]) -> None:
        if trace is None:
            return
        outputs = getattr(trace, "outputs", None)
        if isinstance(outputs, dict):
            outputs.setdefault("voice_events", []).append(event)

    def _is_throttled(self, key: str) -> bool:
        now = time.time()
        previous = self._last_event_by_key.get(key, 0.0)
        if now - previous < self.config.throttle_seconds:
            return True
        self._last_event_by_key[key] = now
        return False

    def _start_worker(self) -> None:
        with self._worker_lock:
            if self._worker_started:
                return
            thread = threading.Thread(target=self._worker_loop, name="copilot-voice-narrator", daemon=True)
            thread.start()
            self._worker_started = True

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=60.0)
            except Empty:
                continue
            event = item["event"]
            try:
                if event.get("status") == "dropped":
                    continue
                cache_path = self._cache_path(item["line"])
                if cache_path.exists():
                    event["cache_hit"] = True
                    self._play_audio_file(cache_path)
                else:
                    audio, content_type = self._request_speech(item["line"])
                    self._write_cache(cache_path, audio, content_type)
                    event["cache_hit"] = False
                    self._play_audio_file(cache_path)
                event["status"] = "spoken"
            except Exception as exc:  # pragma: no cover - fallback path depends on LocalAI/audio device
                event["status"] = "fallback_console"
                event["error"] = str(exc)
                print(f"[voice fallback] {item['line']}")
            finally:
                with self._pending_lock:
                    self._pending_events = [candidate for candidate in self._pending_events if candidate is not event]
                self._queue.task_done()

    def _request_speech(self, line: str) -> tuple[bytes, str]:
        payload = {
            "model": self.config.model,
            "input": line,
            "voice": self.config.voice,
        }
        return self._http_post(self.config.endpoint, payload, self.config.timeout)

    def _default_http_post(self, endpoint: str, payload: dict[str, Any], timeout: float) -> tuple[bytes, str]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), str(response.headers.get("Content-Type", "audio/mpeg"))
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"LocalAI TTS request failed: {exc}") from exc

    def _play_audio(self, audio: bytes, content_type: str) -> None:
        if not audio:
            raise RuntimeError("LocalAI TTS returned empty audio.")
        suffix = ".wav" if "wav" in content_type.lower() else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(audio)
            path = handle.name
        try:
            self._play_audio_file(Path(path))
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    def _play_audio_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".wav":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return
        os.startfile(str(path))  # type: ignore[attr-defined]

    def _collapse_pending(self, event_type: str) -> None:
        if event_type in {"failure", "recovery", "done"}:
            return
        with self._pending_lock:
            for event in self._pending_events:
                if event.get("status") == "queued" and event.get("event_type") not in {"failure", "recovery", "done"}:
                    event["status"] = "dropped"

    def _cache_path(self, line: str) -> Path:
        key = f"{self.config.model}|{self.config.voice}|{line}"
        digest = sha256(key.encode("utf-8")).hexdigest()[:24]
        return Path(self.config.cache_dir) / f"{digest}.wav"

    def _write_cache(self, path: Path, audio: bytes, content_type: str) -> None:
        if not audio:
            raise RuntimeError("LocalAI TTS returned empty audio.")
        if "wav" not in content_type.lower():
            raise RuntimeError(f"Unsupported audio format for cache: {content_type}")
        path.write_bytes(audio)

    def _phase_line(self, phase: str, *, step: Any | None = None, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        app_name = self._safe_app_name(metadata.get("app_id") or metadata.get("expected_app") or metadata.get("app_name") or metadata.get("target"))
        target_name = self._safe_target_name(metadata.get("target_name") or metadata.get("target_label") or metadata.get("target"))
        if phase == "benchmark":
            return "I am starting the benchmark."
        if phase == "runtime":
            return "I am starting the task."
        if phase == "planning":
            return "I am planning the next steps."
        if phase in {"route_window", "focus", "confirm_focus"}:
            return f"I am switching to {app_name}." if app_name else "I am switching to the target application."
        if phase in {"observation", "parse_ui", "observe"}:
            return "I am reading the current screen."
        if phase == "target_found":
            return f"I found {target_name}." if target_name else "I found a stable target."
        if phase == "click":
            return "I am clicking it."
        if phase == "type_text":
            return "I found the input field. I am typing now."
        if phase == "press_key":
            return "I am sending the keyboard action now."
        if phase == "wait_for":
            return "I am waiting for the page to load."
        if phase in {"verify", "verify_scene", "checkpoint", "scene_diff"}:
            return "I am checking whether the screen changed."
        if phase == "step_result":
            return "That step completed successfully."
        if phase == "recovery":
            return "I did not get a stable result, so I am trying recovery."
        if phase == "replan":
            return "The current plan no longer matches the screen. I am updating the next steps."
        if phase == "repair":
            return "I found a local problem in the current step. I am repairing it."
        if phase == "failure":
            return "This step failed safely."
        if phase == "done":
            return "Mission completed successfully."
        return ""

    def _phase_intent_key(self, phase: str, *, step: Any | None = None, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        action_type = str(metadata.get("action_type") or getattr(step, "action_type", "") or "").strip().lower()
        target_name = self._safe_target_name(metadata.get("target_name") or metadata.get("target_label") or metadata.get("target"))
        return "|".join(part for part in [phase.strip().lower(), action_type, target_name.lower()] if part)

    def _safe_target_name(self, value: Any) -> str:
        text = " ".join(str(value or "").strip().split())
        if not text:
            return ""
        lowered = text.lower()
        if any(token in lowered for token in {"password", "token", "secret", "api key"}):
            return ""
        aliases = {
            "downloads": "the Downloads button",
            "desktop": "the Desktop button",
            "search": "the search field",
            "search or type web address": "the browser address bar",
            "chrome omnibox": "the browser address bar",
            "omnibox": "the browser address bar",
        }
        return aliases.get(lowered, f"the {text}")

    def _safe_app_name(self, value: Any) -> str:
        text = " ".join(str(value or "").strip().split())
        lowered = text.lower()
        aliases = {
            "explorer": "Explorer",
            "file explorer": "Explorer",
            "chrome": "Chrome",
            "google chrome": "Chrome",
            "notepad": "Notepad",
        }
        return aliases.get(lowered, text)

    def _sanitize(self, line: str) -> tuple[str, bool]:
        text = " ".join(str(line or "").strip().split())
        if self.config.speak_sensitive:
            return text, False
        original = text
        text = re.sub(r"(?i)\b(password|token|api[_ -]?key|secret)\b\s*[:=]?\s*\S+", r"\1 redacted", text)
        text = re.sub(r"[A-Za-z]:\\[^\s]+(?:\\[^\s]+)*", "[file path redacted]", text)
        text = re.sub(r"(?:/[^/\s]+){2,}", "[file path redacted]", text)
        text = re.sub(r"\b[A-Za-z0-9_\-]{24,}\b", "[sensitive value redacted]", text)
        return text, text != original
