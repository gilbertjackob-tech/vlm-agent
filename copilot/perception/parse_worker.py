from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable
import threading
import time


ParseFn = Callable[[str], list[dict[str, Any]]]
HealthFn = Callable[[], dict[str, Any]]
WarmupFn = Callable[[], Any]


@dataclass
class ParseWorkerResult:
    raw_graph: list[dict[str, Any]]
    parse_health: dict[str, Any]
    worker_metrics: dict[str, Any]


class ResidentParseWorker:
    def __init__(self, parse_fn: ParseFn, health_fn: HealthFn, warmup_fn: WarmupFn | None = None) -> None:
        self._parse_fn = parse_fn
        self._health_fn = health_fn
        self._warmup_fn = warmup_fn
        self._queue: Queue[dict[str, Any]] = Queue()
        self._worker_started = False
        self._worker_lock = threading.Lock()
        self._booted = False
        self._boot_time_seconds = 0.0

    def parse(self, output_filename: str) -> ParseWorkerResult:
        self._start_worker()
        submitted_at = time.time()
        response_queue: Queue[tuple[str, Any]] = Queue(maxsize=1)
        self._queue.put(
            {
                "output_filename": output_filename,
                "submitted_at": submitted_at,
                "response_queue": response_queue,
            }
        )
        status, payload = response_queue.get()
        if status == "error":
            raise RuntimeError(str(payload))
        return payload

    def _start_worker(self) -> None:
        with self._worker_lock:
            if self._worker_started:
                return
            thread = threading.Thread(target=self._worker_loop, name="copilot-parse-worker", daemon=True)
            thread.start()
            self._worker_started = True

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            response_queue: Queue[tuple[str, Any]] = item["response_queue"]
            started_at = time.time()
            queue_wait_seconds = max(0.0, started_at - float(item.get("submitted_at", started_at)))
            try:
                if not self._booted:
                    boot_started = time.time()
                    if self._warmup_fn is not None:
                        self._warmup_fn()
                    self._boot_time_seconds = max(0.0, time.time() - boot_started)
                    self._booted = True
                raw_graph = self._parse_fn(str(item.get("output_filename", "ui_parsed_map.png")))
                exec_seconds = max(0.0, time.time() - started_at)
                worker_metrics = {
                    "worker_used": True,
                    "worker_queue_wait_seconds": round(queue_wait_seconds, 6),
                    "worker_exec_seconds": round(exec_seconds, 6),
                    "worker_boot_seconds": round(self._boot_time_seconds, 6),
                }
                response_queue.put(
                    (
                        "ok",
                        ParseWorkerResult(
                            raw_graph=raw_graph,
                            parse_health=dict(self._health_fn() or {}),
                            worker_metrics=worker_metrics,
                        ),
                    )
                )
            except Exception as exc:
                response_queue.put(("error", str(exc)))
            finally:
                self._queue.task_done()
