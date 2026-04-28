from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from copilot.benchmark.missions import BenchmarkMission, DEFAULT_MISSIONS


@dataclass(frozen=True)
class LiveDesignRequirement:
    requirement_id: str
    description: str
    required_tags: set[str] = field(default_factory=set)
    required_categories: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_tags"] = sorted(self.required_tags)
        payload["required_categories"] = sorted(self.required_categories)
        return payload


LIVE_DESIGN_REQUIREMENTS: list[LiveDesignRequirement] = [
    LiveDesignRequirement("wrong_click_prevention", "Wrong clicks are prevented or counted.", {"wrong_target"}),
    LiveDesignRequirement("duplicate_disambiguation", "Duplicate labels require region/context disambiguation.", {"duplicate_label", "region_disambiguation"}),
    LiveDesignRequirement("focus_before_typing", "Typing missions verify focus before text entry.", {"focus", "typing", "verification"}),
    LiveDesignRequirement("recovery_depth_tracking", "Recovery missions exercise bounded repair/replan depth.", {"recovery"}),
    LiveDesignRequirement("soft_hard_cancel", "Cancel behavior is represented by a bounded timeout/control mission.", {"timeout", "negative_control"}),
    LiveDesignRequirement("dialog_approval_safety", "Dialogs and approvals are represented by safe dialog missions.", {"dialog", "safety"}),
]


def validate_live_design(missions: list[BenchmarkMission] | None = None) -> dict[str, Any]:
    missions = list(missions or DEFAULT_MISSIONS)
    results = []
    for requirement in LIVE_DESIGN_REQUIREMENTS:
        matching = [
            mission
            for mission in missions
            if requirement.required_tags.issubset(set(mission.tags))
            and (not requirement.required_categories or mission.category in requirement.required_categories)
        ]
        results.append(
            {
                "requirement": requirement.to_dict(),
                "passed": bool(matching),
                "missions": [mission.mission_id for mission in matching],
            }
        )
    return {
        "design_only": True,
        "mission_count": len(missions),
        "requirement_count": len(results),
        "passed": all(item["passed"] for item in results),
        "results": results,
    }
