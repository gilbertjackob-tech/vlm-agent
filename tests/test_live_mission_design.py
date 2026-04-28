from __future__ import annotations

import unittest

from copilot.benchmark.live_design import LIVE_DESIGN_REQUIREMENTS, validate_live_design
from copilot.benchmark.missions import DEFAULT_MISSIONS


class LiveMissionDesignTests(unittest.TestCase):
    def test_default_missions_cover_required_live_design_cases(self) -> None:
        report = validate_live_design(DEFAULT_MISSIONS)

        self.assertTrue(report["passed"])
        self.assertEqual(report["requirement_count"], len(LIVE_DESIGN_REQUIREMENTS))
        self.assertTrue(all(item["missions"] for item in report["results"]))


if __name__ == "__main__":
    unittest.main()
