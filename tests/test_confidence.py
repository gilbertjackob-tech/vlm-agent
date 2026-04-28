from __future__ import annotations

import unittest

from copilot.runtime.confidence import confidence_from_trace_event, derive_confidence


class ConfidenceTests(unittest.TestCase):
    def test_high_confidence_requires_clean_gap_focus_and_no_recovery(self) -> None:
        signal = derive_confidence(score_gap=0.45, focus_confidence=0.9, recovery_count=0)
        self.assertEqual(signal.level, "HIGH")
        self.assertIn("stable_target_and_focus", signal.reasons)

    def test_recovery_or_moderate_gap_lowers_to_medium(self) -> None:
        signal = derive_confidence(score_gap=0.15, focus_confidence=0.85, recovery_count=1)
        self.assertEqual(signal.level, "MEDIUM")
        self.assertIn("one_recovery", signal.reasons)

    def test_ambiguous_low_focus_target_is_low(self) -> None:
        signal = derive_confidence(score_gap=0.03, focus_confidence=0.2, recovery_count=2, ambiguous=True)
        self.assertEqual(signal.level, "LOW")
        self.assertIn("target_ambiguous", signal.reasons)

    def test_event_metadata_derives_confidence(self) -> None:
        signal = confidence_from_trace_event(
            {
                "metadata": {
                    "focus_confidence": 0.8,
                    "target_ranking": {"score_gap": 0.04, "ambiguous": True},
                }
            }
        )
        self.assertEqual(signal.level, "LOW")


if __name__ == "__main__":
    unittest.main()
