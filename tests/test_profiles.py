from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest

from copilot.profiles import AppProfileRegistry, ChromeProfile, ExplorerProfile
from copilot.schemas import ObservationGraph


ROOT = Path(__file__).resolve().parents[1]


def load_graph(path: Path) -> ObservationGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ObservationGraph.from_raw(payload["ui_graph"], metadata={"artifact": path.name})


class ProfileClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = AppProfileRegistry([ExplorerProfile(), ChromeProfile()])

    def test_explorer_artifact_classifies_rows(self) -> None:
        graph = load_graph(ROOT / "debug_steps" / "random_round_0_parse.json")
        graph, profile = self.registry.annotate(
            graph,
            {"windows": {"active_window": {"title": "File Explorer"}, "active_app_guess": "explorer"}},
            app_id="explorer",
        )
        self.assertIsNotNone(profile)
        self.assertEqual(profile.app_id, "explorer")

        labels_to_type = {
            node.display_label(): node.entity_type
            for node in graph.flatten()
            if node.semantic_role == "list_row"
        }
        self.assertEqual(labels_to_type.get("Row: excell"), "folder")
        self.assertEqual(labels_to_type.get("Row: 6figure.zip"), "archive")
        self.assertEqual(labels_to_type.get("Row: awake.py"), "python_file")

    def test_desktop_artifact_preserves_clickable_structure(self) -> None:
        graph = load_graph(ROOT / "debug_steps" / "random_round_6_parse.json")
        roles = Counter(node.semantic_role for node in graph.flatten() if node.semantic_role)
        labels = {node.display_label() for node in graph.flatten()}
        self.assertGreaterEqual(roles["clickable_container"], 3)
        self.assertIn("Zoom", labels)
        self.assertIn("Desktop", labels)


if __name__ == "__main__":
    unittest.main()
