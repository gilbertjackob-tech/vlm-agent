from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from copilot.memory.store import MemoryStore
from copilot.profiles import AppProfileRegistry, ChromeProfile, ExplorerProfile
from copilot.schemas import ObservationGraph


def _load_graph(path: Path) -> ObservationGraph | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    raw_graph = payload.get("ui_graph")
    if not isinstance(raw_graph, list):
        return None

    return ObservationGraph.from_raw(
        raw_graph,
        metadata={
            "artifact": path.name,
            "timestamp": payload.get("timestamp", ""),
            "element_count": payload.get("element_count", 0),
        },
    )


def _environment_for_graph(graph: ObservationGraph) -> dict[str, Any]:
    labels = " ".join(node.display_label().lower() for node in graph.flatten() if node.display_label())
    if "youtube" in labels or "google chrome" in labels or "chrome" in labels:
        app_id = "chrome"
        title = "Google Chrome"
    else:
        app_id = "explorer"
        title = "File Explorer"

    return {
        "windows": {
            "active_window": {"title": title},
            "active_app_guess": app_id,
        },
        "browser": {
            "available": app_id == "chrome",
            "cdp_available": False,
        },
    }


def bootstrap_memory(debug_dir: Path, memory_dir: Path) -> dict[str, Any]:
    memory = MemoryStore(base_dir=str(memory_dir))
    registry = AppProfileRegistry([ExplorerProfile(), ChromeProfile()])

    imported = 0
    skipped = 0
    app_counts: dict[str, int] = {}
    entity_counts: dict[str, int] = {}
    label_count_before = len(memory.semantic_memory.get("controls", {}))

    for path in sorted(debug_dir.glob("*_parse.json")):
        graph = _load_graph(path)
        if graph is None:
            skipped += 1
            continue

        environment = _environment_for_graph(graph)
        graph, profile = registry.annotate(
            graph,
            environment,
            app_id=environment["windows"]["active_app_guess"],
        )
        graph.metadata["source"] = "bootstrap_debug_steps"
        graph.metadata["app_id"] = profile.app_id if profile else environment["windows"]["active_app_guess"]
        memory.remember_observation_graph(graph)

        imported += 1
        app_id = graph.metadata.get("app_id", "")
        app_counts[app_id] = app_counts.get(app_id, 0) + 1
        for node in graph.flatten():
            if node.entity_type:
                entity_counts[node.entity_type] = entity_counts.get(node.entity_type, 0) + 1

    memory.save_all()
    summary = memory.summary()
    return {
        "debug_dir": str(debug_dir),
        "memory_dir": str(memory_dir),
        "imported_artifacts": imported,
        "skipped_artifacts": skipped,
        "apps": app_counts,
        "entity_types": dict(sorted(entity_counts.items())),
        "controls_before": label_count_before,
        "controls_after": summary.get("known_controls", 0),
        "memory_summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed copilot semantic memory from saved debug parse artifacts.")
    parser.add_argument("--debug-dir", default="debug_steps", help="Directory containing *_parse.json artifacts.")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory to update.")
    parser.add_argument("--report", default="", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = bootstrap_memory(Path(args.debug_dir), Path(args.memory_dir))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
