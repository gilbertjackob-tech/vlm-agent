from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any
import json

from copilot.schemas import ObservationGraph, ObservationNode


def _hash(parts: dict[str, Any]) -> str:
    return sha256(json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()[:16]


@dataclass
class TargetIdentity:
    target_id: str
    role: str
    name: str
    bounds: dict[str, int]
    parent_window: str
    source: str
    stable_signature: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_from_node(node: ObservationNode, graph: ObservationGraph | None = None) -> str:
    if graph and graph.metadata.get("dom_selector"):
        return "DOM"
    if node.visual_id:
        return "Vision"
    if node.label:
        return "OCR"
    return "UIA"


def create_target_identity(
    node: ObservationNode,
    graph: ObservationGraph | None = None,
    source: str = "",
) -> TargetIdentity:
    name = node.display_label() or node.label or node.node_id
    parent_window = ""
    if graph:
        parent_window = str(graph.metadata.get("app_id") or graph.metadata.get("scene", {}).get("app_id", ""))
    bounds = {key: int(value or 0) for key, value in dict(node.box or {}).items() if key in {"x", "y", "width", "height"}}
    if not bounds and node.center:
        bounds = {"x": int(node.center.get("x", 0) or 0), "y": int(node.center.get("y", 0) or 0), "width": 0, "height": 0}
    signature_payload = {
        "role": node.semantic_role or node.node_type,
        "name": name.strip().lower(),
        "parent_window": parent_window.strip().lower(),
        "region": node.region,
        "entity_type": node.entity_type,
    }
    stable_signature = _hash(signature_payload)
    confidence = 0.0
    if name:
        confidence += 0.3
    if node.semantic_role or node.node_type:
        confidence += 0.2
    if parent_window:
        confidence += 0.2
    if bounds:
        confidence += 0.15
    if node.visual_id or getattr(node, "visual_ids", []):
        confidence += 0.15
    return TargetIdentity(
        target_id=f"target_{stable_signature}",
        role=node.semantic_role or node.node_type or "",
        name=name,
        bounds=bounds,
        parent_window=parent_window,
        source=source or _source_from_node(node, graph),
        stable_signature=stable_signature,
        confidence=round(min(1.0, confidence), 3),
    )


def match_target_identity(identity: TargetIdentity | dict[str, Any], node: ObservationNode, graph: ObservationGraph | None = None) -> float:
    if isinstance(identity, dict):
        identity = TargetIdentity(**identity)
    candidate = create_target_identity(node, graph, source=identity.source)
    score = 0.0
    if candidate.stable_signature == identity.stable_signature:
        score += 0.55
    if candidate.name.strip().lower() == identity.name.strip().lower():
        score += 0.2
    if candidate.role == identity.role:
        score += 0.1
    if candidate.parent_window == identity.parent_window:
        score += 0.1
    if _bounds_close(candidate.bounds, identity.bounds):
        score += 0.05
    return round(min(1.0, score), 3)


def detect_target_drift(
    identity: TargetIdentity | dict[str, Any],
    node: ObservationNode,
    graph: ObservationGraph | None = None,
    threshold: float = 0.7,
) -> bool:
    return match_target_identity(identity, node, graph) < threshold


def resolve_same_target_again(
    identity: TargetIdentity | dict[str, Any],
    graph: ObservationGraph,
    threshold: float = 0.7,
) -> ObservationNode | None:
    matches = []
    for node in graph.flatten():
        score = match_target_identity(identity, node, graph)
        if score >= threshold:
            matches.append((score, node))
    if len(matches) != 1:
        return None
    return matches[0][1]


def ambiguous_identity_matches(
    identity: TargetIdentity | dict[str, Any],
    graph: ObservationGraph,
    threshold: float = 0.7,
) -> list[ObservationNode]:
    return [node for node in graph.flatten() if match_target_identity(identity, node, graph) >= threshold]


def _bounds_close(a: dict[str, int], b: dict[str, int]) -> bool:
    if not a or not b:
        return False
    return abs(int(a.get("x", 0)) - int(b.get("x", 0))) <= 12 and abs(int(a.get("y", 0)) - int(b.get("y", 0))) <= 12
