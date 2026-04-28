from __future__ import annotations

import time
import tempfile
import unittest

from copilot.runtime.voice_narrator import VoiceConfig, VoiceNarrator
from copilot.schemas import ExecutionPlan, RunTrace, TaskSpec


def make_trace() -> RunTrace:
    task = TaskSpec(prompt="say progress", goal="say progress")
    return RunTrace(run_id="run_voice", task=task, plan=ExecutionPlan(task=task, steps=[]))


class VoiceNarratorTests(unittest.TestCase):
    def test_console_mode_records_voice_event_and_redacts_sensitive_text(self) -> None:
        trace = make_trace()
        narrator = VoiceNarrator(VoiceConfig(mode="console"))

        event = narrator.speak(
            r"Opening C:\Users\DELL\secret.txt with token abcdefghijklmnopqrstuvwxyz",
            trace=trace,
            event_type="runtime",
        )

        self.assertEqual(event["status"], "console")
        self.assertTrue(event["sensitive_redacted"])
        self.assertIn("[file path redacted]", event["line"])
        self.assertIn("token redacted", event["line"])
        self.assertEqual(trace.outputs["voice_events"][0], event)

    def test_tts_mode_queues_without_blocking_runtime(self) -> None:
        calls = []

        def slow_post(endpoint, payload, timeout):
            calls.append((endpoint, payload, timeout))
            time.sleep(0.2)
            raise RuntimeError("LocalAI unavailable")

        trace = make_trace()
        narrator = VoiceNarrator(VoiceConfig(mode="tts", timeout=0.1), http_post=slow_post)

        started = time.time()
        event = narrator.speak("I am starting the benchmark.", trace=trace, event_type="benchmark")
        elapsed = time.time() - started

        self.assertLess(elapsed, 0.1)
        self.assertEqual(event["status"], "queued")
        self.assertEqual(trace.outputs["voice_events"][0]["line"], "I am starting the benchmark.")

    def test_tts_worker_posts_localai_payload_and_updates_event(self) -> None:
        def post(endpoint, payload, timeout):
            self.assertEqual(endpoint, "http://localhost:8080/v1/audio/speech")
            self.assertEqual(payload["model"], "voice-en-us-ryan-high")
            self.assertEqual(payload["voice"], "voice-en-us-ryan-high")
            self.assertEqual(payload["input"], "Mission completed successfully.")
            return b"RIFF", "audio/wav"

        trace = make_trace()
        narrator = VoiceNarrator(VoiceConfig(mode="tts"), http_post=post)
        narrator._play_audio_file = lambda path: None  # type: ignore[method-assign]

        event = narrator.speak("Mission completed successfully.", trace=trace, event_type="done")
        narrator._queue.join()

        self.assertEqual(event["status"], "spoken")
        self.assertEqual(trace.outputs["voice_events"][0]["status"], "spoken")

    def test_recovery_messages_are_throttled(self) -> None:
        trace = make_trace()
        config = VoiceConfig(mode="console", throttle_seconds=60.0)
        narrator = VoiceNarrator(config)

        first = narrator.speak("The click did not change the screen. I am trying recovery.", trace=trace, event_type="recovery", throttle_key="click")
        second = narrator.speak("The click did not change the screen. I am trying recovery.", trace=trace, event_type="recovery", throttle_key="click")

        self.assertEqual(first["status"], "console")
        self.assertEqual(second["status"], "throttled")
        self.assertEqual(len(trace.outputs["voice_events"]), 2)

    def test_speak_phase_maps_safe_runtime_summary(self) -> None:
        trace = make_trace()
        narrator = VoiceNarrator(VoiceConfig(mode="console"))

        event = narrator.speak_phase("focus", trace=trace, metadata={"app_id": "explorer"})

        self.assertEqual(event["line"], "I am switching to Explorer.")

    def test_flush_marks_queued_events_as_dropped_when_timeout_expires(self) -> None:
        trace = make_trace()

        def blocked_post(endpoint, payload, timeout):
            time.sleep(0.3)
            return b"RIFF", "audio/wav"

        narrator = VoiceNarrator(VoiceConfig(mode="tts", timeout=0.1), http_post=blocked_post)
        narrator._play_audio_file = lambda path: time.sleep(0.3)  # type: ignore[method-assign]

        event = narrator.speak("Mission completed successfully.", trace=trace, event_type="done")
        flushed = narrator.flush(timeout_seconds=0.05)

        self.assertFalse(flushed)
        self.assertEqual(event["status"], "dropped")

    def test_cached_phrase_skips_http_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = make_trace()
            calls = []

            def post(endpoint, payload, timeout):
                calls.append(payload)
                return b"RIFF", "audio/wav"

            narrator = VoiceNarrator(VoiceConfig(mode="tts", cache_dir=tmpdir), http_post=post)
            narrator._play_audio_file = lambda path: None  # type: ignore[method-assign]

            narrator.speak("I am reading the current screen.", trace=trace, event_type="observe")
            narrator._queue.join()
            narrator.speak("I am reading the current screen.", trace=trace, event_type="observe")
            narrator._queue.join()

            self.assertEqual(len(calls), 1)

    def test_collapse_queue_drops_stale_noncritical_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = make_trace()

            def slow_post(endpoint, payload, timeout):
                time.sleep(0.2)
                return b"RIFF", "audio/wav"

            narrator = VoiceNarrator(VoiceConfig(mode="tts", timeout=0.1, cache_dir=tmpdir), http_post=slow_post)
            narrator._play_audio_file = lambda path: time.sleep(0.2)  # type: ignore[method-assign]

            first = narrator.speak("I am reading the current screen.", trace=trace, event_type="observe")
            second = narrator.speak("I am checking whether the screen changed.", trace=trace, event_type="verify_scene")
            third = narrator.speak("I found a stable target.", trace=trace, event_type="target_found")
            narrator._queue.join()

            self.assertIn(first["status"], {"spoken", "dropped"})
            self.assertIn(second["status"], {"spoken", "dropped"})
            self.assertIn(third["status"], {"spoken", "dropped"})
            self.assertEqual(second["status"], "dropped")
            self.assertFalse(any(event["status"] == "queued" for event in trace.outputs["voice_events"]))


if __name__ == "__main__":
    unittest.main()
