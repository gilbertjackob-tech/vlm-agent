from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from copilot.schemas import ObservationGraph, ObservationNode


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


@dataclass
class TargetCandidate:
    node: ObservationNode
    score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node.node_id,
            "label": self.node.display_label(),
            "semantic_role": self.node.semantic_role,
            "entity_type": self.node.entity_type,
            "region": self.node.region,
            "app_id": self.node.app_id,
            "score": round(float(self.score), 4),
            "reasons": list(self.reasons),
            "center": dict(self.node.center or {}),
        }


@dataclass
class TargetRankResult:
    selected_node: ObservationNode | None
    candidates: list[TargetCandidate]
    top_candidate_score: float = 0.0
    runner_up_score: float = 0.0
    score_gap: float = 0.0
    duplicate_disambiguation_used: bool = False
    ambiguous: bool = False
    ambiguity_reason: str = ""

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_node_id": self.selected_node.node_id if self.selected_node else "",
            "selected_label": self.selected_node.display_label() if self.selected_node else "",
            "candidate_count": self.candidate_count,
            "top_candidate_score": round(float(self.top_candidate_score), 4),
            "runner_up_score": round(float(self.runner_up_score), 4),
            "score_gap": round(float(self.score_gap), 4),
            "duplicate_disambiguation_used": bool(self.duplicate_disambiguation_used),
            "ambiguous": bool(self.ambiguous),
            "ambiguity_reason": self.ambiguity_reason,
            "candidates": [candidate.to_dict() for candidate in self.candidates[:8]],
        }


def _score_node(node: ObservationNode, filters: dict[str, Any]) -> tuple[float, list[str]]:
    if filters.get("exclude_destructive") and "destructive" in node.state_tags:
        return -1.0, ["excluded:destructive"]

    score = 0.0
    max_score = 0.0
    reasons: list[str] = []
    label = normalize_text(node.display_label())

    def add(weight: float, matched: bool, reason: str) -> None:
        nonlocal score, max_score
        max_score += weight
        if matched:
            score += weight
            reasons.append(reason)

    if filters.get("label_contains"):
        expected = normalize_text(str(filters["label_contains"]))
        add(2.0, bool(expected and expected in label), "label")

    if filters.get("entity_type"):
        add(1.5, node.entity_type == filters["entity_type"], "entity_type")

    if filters.get("semantic_role"):
        add(1.0, node.semantic_role == filters["semantic_role"], "semantic_role")

    if filters.get("region"):
        add(1.0, node.region == filters["region"], "region")

    if filters.get("app_id"):
        add(0.75, node.app_id == filters["app_id"], "app_id")

    if filters.get("affordance"):
        add(1.0, filters["affordance"] in node.affordances, "affordance")

    if filters.get("state_tag"):
        add(0.75, filters["state_tag"] in node.state_tags, "state_tag")

    if filters.get("concept"):
        add(0.75, filters["concept"] in node.learned_concepts, "concept")

    preferred_labels = [normalize_text(str(item)) for item in filters.get("preferred_labels", []) if item]
    if preferred_labels:
        add(1.0, any(pref in label or label in pref for pref in preferred_labels), "preferred_label")

    avoid_labels = [normalize_text(str(item)) for item in filters.get("avoid_labels", []) if item]
    if avoid_labels:
        max_score += 1.25
        if any(avoid in label or label in avoid for avoid in avoid_labels):
            score -= 1.25
            reasons.append("avoid_label")

    avoid_visual_ids = {str(item) for item in filters.get("avoid_visual_ids", []) if item}
    if avoid_visual_ids:
        max_score += 1.0
        if node.visual_id and node.visual_id in avoid_visual_ids:
            score -= 1.0
            reasons.append("avoid_visual_id")

    if filters.get("y_max") is not None:
        add(0.5, int(node.center.get("y", 99999) or 99999) <= int(filters["y_max"]), "y_max")

    if filters.get("x_max") is not None:
        add(0.5, int(node.center.get("x", 99999) or 99999) <= int(filters["x_max"]), "x_max")

    region_priors = filters.get("region_priors", {})
    if isinstance(region_priors, dict) and region_priors:
        max_score += max(abs(float(value or 0.0)) for value in region_priors.values())
        prior = float(region_priors.get(node.region, 0.0) or 0.0)
        if prior:
            score += prior
            reasons.append("region_prior" if prior > 0 else "region_penalty")

    entity_priors = filters.get("entity_priors", {})
    if isinstance(entity_priors, dict) and entity_priors:
        max_score += max(abs(float(value or 0.0)) for value in entity_priors.values())
        prior = float(entity_priors.get(node.entity_type, 0.0) or 0.0)
        if prior:
            score += prior
            reasons.append("entity_prior" if prior > 0 else "entity_penalty")

    role_priors = filters.get("semantic_role_priors", {})
    if isinstance(role_priors, dict) and role_priors:
        max_score += max(abs(float(value or 0.0)) for value in role_priors.values())
        prior = float(role_priors.get(node.semantic_role, 0.0) or 0.0)
        if prior:
            score += prior
            reasons.append("role_prior" if prior > 0 else "role_penalty")

    if max_score <= 0:
        return 0.0, reasons
    ranked = score / max_score
    ranked += float(node.stability or 0.0) * 0.1
    return ranked, reasons


def rank_action_targets(
    filters: dict[str, Any],
    observation: ObservationGraph | None,
    *,
    ambiguity_gap: float = 0.08,
) -> TargetRankResult:
    if not observation:
        return TargetRankResult(selected_node=None, candidates=[], ambiguity_reason="missing_observation")

    min_score = float(filters.get("min_score", 0.55))
    candidates: list[TargetCandidate] = []
    scored_nodes: list[TargetCandidate] = []
    for node in observation.flatten():
        score, reasons = _score_node(node, filters)
        if score > 0:
            scored_nodes.append(TargetCandidate(node=node, score=score, reasons=reasons))
        if score >= min_score:
            candidates.append(TargetCandidate(node=node, score=score, reasons=reasons))

    if not candidates:
        return TargetRankResult(selected_node=None, candidates=[], ambiguity_reason="no_candidates")

    candidates.sort(key=lambda item: item.score, reverse=True)
    top_score = float(candidates[0].score)
    runner_up_score = float(candidates[1].score) if len(candidates) > 1 else 0.0
    score_gap = round(top_score - runner_up_score, 4) if len(candidates) > 1 else round(top_score, 4)

    duplicate_disambiguation_used = _duplicate_disambiguation_used(candidates, filters, duplicate_pool=scored_nodes)
    ambiguous = bool(len(candidates) > 1 and score_gap <= ambiguity_gap and not duplicate_disambiguation_used)
    ambiguity_reason = "small_score_gap" if ambiguous else ""

    return TargetRankResult(
        selected_node=candidates[0].node,
        candidates=candidates,
        top_candidate_score=top_score,
        runner_up_score=runner_up_score,
        score_gap=score_gap,
        duplicate_disambiguation_used=duplicate_disambiguation_used,
        ambiguous=ambiguous,
        ambiguity_reason=ambiguity_reason,
    )


def _duplicate_disambiguation_used(
    candidates: list[TargetCandidate],
    filters: dict[str, Any],
    *,
    duplicate_pool: list[TargetCandidate] | None = None,
) -> bool:
    if len(candidates) < 2:
        duplicate_pool = duplicate_pool or candidates
        if len(duplicate_pool) < 2:
            return False
    selected = candidates[0]
    selected_label = normalize_text(selected.node.display_label())
    duplicate_pool = duplicate_pool or candidates
    duplicate_labels = [
        candidate
        for candidate in duplicate_pool
        if normalize_text(candidate.node.display_label()) == selected_label
    ]
    if len(duplicate_labels) < 2:
        return False

    deterministic_reasons = {
        "region",
        "region_prior",
        "entity_type",
        "entity_prior",
        "semantic_role",
        "role_prior",
        "app_id",
        "affordance",
        "preferred_label",
        "x_max",
        "y_max",
    }
    requested = {key for key, value in filters.items() if value not in ("", None, [], {})}
    reason_set = set(selected.reasons)
    return bool((requested & deterministic_reasons) and (reason_set & deterministic_reasons))


def result_to_contract_metrics(result: TargetRankResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    return {
        key: value
        for key, value in result.to_dict().items()
        if key != "candidates"
    }
