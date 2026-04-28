from __future__ import annotations

import unittest

from copilot.perception.target_ranking import rank_action_targets
from copilot.schemas import ObservationGraph


def duplicate_downloads_graph() -> ObservationGraph:
    return ObservationGraph.from_raw(
        [
            {
                "id": "nav_downloads",
                "label": "Downloads",
                "type": "button",
                "semantic_role": "menu_item",
                "entity_type": "navigation_item",
                "app_id": "explorer",
                "region": "left_menu",
                "affordances": ["navigate", "click"],
                "center": {"x": 90, "y": 220},
                "box": {"x": 20, "y": 200, "width": 140, "height": 32},
            },
            {
                "id": "main_downloads",
                "label": "Downloads",
                "type": "button",
                "semantic_role": "list_row",
                "entity_type": "folder",
                "app_id": "explorer",
                "region": "main_page",
                "affordances": ["open", "click"],
                "center": {"x": 380, "y": 260},
                "box": {"x": 260, "y": 238, "width": 240, "height": 44},
            },
        ],
        metadata={"app_id": "explorer"},
    )


class TargetRankingTests(unittest.TestCase):
    def test_duplicate_label_is_disambiguated_by_region_role_and_entity(self) -> None:
        result = rank_action_targets(
            {
                "label_contains": "Downloads",
                "region": "left_menu",
                "semantic_role": "menu_item",
                "entity_type": "navigation_item",
                "affordance": "navigate",
                "min_score": 0.4,
            },
            duplicate_downloads_graph(),
        )

        self.assertEqual(result.selected_node.node_id, "nav_downloads")
        self.assertTrue(result.duplicate_disambiguation_used)
        self.assertFalse(result.ambiguous)
        self.assertGreater(result.score_gap, 0)

    def test_duplicate_label_without_context_is_ambiguous(self) -> None:
        result = rank_action_targets(
            {
                "label_contains": "Downloads",
                "min_score": 0.4,
            },
            duplicate_downloads_graph(),
        )

        self.assertEqual(result.candidate_count, 2)
        self.assertTrue(result.ambiguous)
        self.assertEqual(result.ambiguity_reason, "small_score_gap")


if __name__ == "__main__":
    unittest.main()
