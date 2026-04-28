from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from copilot.schemas import ObservationGraph, ObservationNode
from copilot.runtime.target_identity import ambiguous_identity_matches, detect_target_drift

from copilot.runtime.action_contract import (
    FAILURE_FOCUS_NOT_CONFIRMED,
    FAILURE_NO_STATE_CHANGE,
    FAILURE_POLICY_BLOCKED,
    FAILURE_TARGET_AMBIGUOUS,
    FAILURE_TARGET_NOT_FOUND,
    FAILURE_TIMEOUT,
    FAILURE_UNSAFE_COORDINATE,
    recovery_strategy_for,
)


@dataclass
class RecoveryPlan:
    failure_reason: str
    strategy: list[str]
    stop_required: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_reason": self.failure_reason,
            "strategy": list(self.strategy),
            "stop_required": self.stop_required,
            "note": self.note,
        }


@dataclass
class RecoveredTarget:
    resolver_used: str
    node: ObservationNode | None = None
    selectors: list[str] | None = None
    evidence: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolver_used": self.resolver_used,
            "target_label": self.node.display_label() if self.node else "",
            "target_node_id": self.node.node_id if self.node else "",
            "selectors": list(self.selectors or []),
            "evidence": list(self.evidence or []),
        }


class RecoveryPlanner:
    max_retries_per_action = 2
    max_coordinate_fallbacks = 1

    def plan(self, failure_reason: str, step, contract: dict[str, Any] | None = None) -> RecoveryPlan:
        reason = failure_reason or FAILURE_NO_STATE_CHANGE
        strategy = recovery_strategy_for(reason)
        stop_required = reason in {FAILURE_POLICY_BLOCKED, FAILURE_UNSAFE_COORDINATE}
        if reason == FAILURE_TARGET_NOT_FOUND:
            note = "Target was not found. Reparse and retry with stronger metadata sources."
        elif reason == FAILURE_TARGET_AMBIGUOUS:
            note = "Target was ambiguous. Resolve by role, text, region, proximity, and task context."
        elif reason == FAILURE_FOCUS_NOT_CONFIRMED:
            note = "Focus was not confirmed. Refocus the expected app before retry."
        elif reason == FAILURE_TIMEOUT:
            note = "Wait timed out. Re-observe before retrying."
        elif reason == FAILURE_UNSAFE_COORDINATE:
            note = "Coordinate action lacked safety evidence. Stop for operator review or retry with a resolved node target."
        else:
            note = "No verified state change. Reparse before selecting a fallback action."
        return RecoveryPlan(reason, strategy, stop_required=stop_required, note=note)

    def recover_target(
        self,
        *,
        strategy: list[str],
        step,
        graph: ObservationGraph | None,
        reasoner,
        scene: dict | None = None,
        failed_contract: dict[str, Any] | None = None,
    ) -> RecoveredTarget | None:
        selectors = [str(item) for item in step.parameters.get("selector_candidates", []) if str(item).strip()]
        if selectors and "retry_with_uia" in strategy:
            return RecoveredTarget("dom", selectors=selectors, evidence=[f"dom_selector:{selector}" for selector in selectors])

        if not graph:
            return None
        failed_identity = (failed_contract or {}).get("target_identity") or {}
        if (failed_contract or {}).get("identity_drifted"):
            return None
        if failed_identity and len(ambiguous_identity_matches(failed_identity, graph)) > 1:
            return None

        filters = dict(step.target.filters if step.target else {})
        filters.update(step.parameters.get("filters", {}))
        if not filters and step.target and step.target.value:
            filters["label_contains"] = step.target.value

        failed_node_id = str((failed_contract or {}).get("target_node_id", ""))
        failed_target = str((failed_contract or {}).get("target", "")).strip().lower()

        def candidate_without_failed(source: str) -> ObservationNode | None:
            source_filters = dict(filters)
            if source == "ocr":
                source_filters.setdefault("min_score", 0.45)
            elif source == "vision":
                source_filters.setdefault("min_score", 0.35)
            else:
                source_filters.setdefault("min_score", 0.55)
            candidates = []
            for node in graph.flatten():
                if failed_node_id and node.node_id == failed_node_id:
                    continue
                if failed_target and (node.display_label() or node.label or "").strip().lower() == failed_target:
                    continue
                score = reasoner._score_node(node, source_filters) if hasattr(reasoner, "_score_node") else 0.0
                if score >= float(source_filters.get("min_score", 0.55)):
                    candidates.append(node)
            if not candidates:
                return None
            chosen = reasoner.resolve_ambiguity(candidates, getattr(step, "title", ""), scene or {})
            if chosen and failed_identity and detect_target_drift(failed_identity, chosen, graph):
                return None
            return chosen

        source_order = [
            ("retry_with_uia", "uia", ["uia:accessibility_tree"]),
            ("retry_with_ocr", "ocr", ["ocr:screen_text"]),
            ("retry_with_vision", "vision", ["visual:vision_box"]),
        ]
        for strategy_name, resolver_name, evidence in source_order:
            if strategy_name not in strategy:
                continue
            node = candidate_without_failed(resolver_name)
            if node:
                return RecoveredTarget(resolver_name, node=node, evidence=evidence)
        return None
