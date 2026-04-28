from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfidenceSignal:
    level: str
    score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "score": round(self.score, 3), "reasons": list(self.reasons)}


def derive_confidence(
    *,
    score_gap: float = 0.0,
    focus_confidence: float = 0.0,
    recovery_count: int = 0,
    ambiguous: bool = False,
    focus_confirmed: bool = True,
) -> ConfidenceSignal:
    reasons: list[str] = []
    score = 1.0

    gap = max(0.0, min(float(score_gap or 0.0), 1.0))
    focus = max(0.0, min(float(focus_confidence or 0.0), 1.0))
    recoveries = max(0, int(recovery_count or 0))

    if gap < 0.08:
        score -= 0.35
        reasons.append("small_target_score_gap")
    elif gap < 0.2:
        score -= 0.18
        reasons.append("moderate_target_score_gap")

    if focus < 0.45 or not focus_confirmed:
        score -= 0.35
        reasons.append("low_focus_confidence")
    elif focus < 0.75:
        score -= 0.15
        reasons.append("moderate_focus_confidence")

    if recoveries >= 2:
        score -= 0.35
        reasons.append("multiple_recoveries")
    elif recoveries == 1:
        score -= 0.16
        reasons.append("one_recovery")

    if ambiguous:
        score -= 0.25
        reasons.append("target_ambiguous")

    score = max(0.0, min(1.0, score))
    if score >= 0.78 and not reasons:
        level = "HIGH"
    elif score >= 0.48:
        level = "MEDIUM"
    else:
        level = "LOW"
    if not reasons:
        reasons.append("stable_target_and_focus")
    return ConfidenceSignal(level=level, score=score, reasons=reasons)


def confidence_from_trace_event(event: dict[str, Any]) -> ConfidenceSignal:
    metadata = event.get("metadata", {}) if isinstance(event, dict) else {}
    ranking = metadata.get("target_ranking", {}) if isinstance(metadata, dict) else {}
    return derive_confidence(
        score_gap=float(ranking.get("score_gap", metadata.get("score_gap", 1.0)) or 0.0),
        focus_confidence=float(metadata.get("focus_confidence", 1.0) or 0.0),
        recovery_count=int(metadata.get("recovery_count", 0) or 0),
        ambiguous=bool(ranking.get("ambiguous", metadata.get("ambiguous", False))),
        focus_confirmed=bool(metadata.get("focus_confirmed", True)),
    )
