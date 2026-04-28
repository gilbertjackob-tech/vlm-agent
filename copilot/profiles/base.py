from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from copilot.schemas import ObservationGraph, ObservationNode


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


@dataclass
class AppProfile:
    app_id: str
    display_name: str
    window_keywords: list[str] = field(default_factory=list)
    trusted_for_exploration: bool = False
    risky_terms: set[str] = field(default_factory=set)

    def match_score(self, environment: dict, observation: ObservationGraph | None) -> float:
        active_title = normalize_text(environment.get("windows", {}).get("active_window", {}).get("title", ""))
        score = 0.0
        for keyword in self.window_keywords:
            if normalize_text(keyword) and normalize_text(keyword) in active_title:
                score += 2.0

        if observation:
            labels = " ".join(normalize_text(node.display_label()) for node in observation.flatten())
            for keyword in self.window_keywords:
                if normalize_text(keyword) and normalize_text(keyword) in labels:
                    score += 0.5

        return score

    def annotate(self, graph: ObservationGraph, environment: dict) -> ObservationGraph:
        for node in graph.flatten():
            node.app_id = self.app_id
            entity_type = self.classify_node(node)
            if entity_type and not node.entity_type:
                node.entity_type = entity_type
            affordances = self.affordances_for(node)
            if affordances:
                node.affordances = sorted(set(node.affordances + affordances))
            state_tags = self.state_tags_for(node)
            if state_tags:
                node.state_tags = sorted(set(node.state_tags + state_tags))
            node.stability = max(node.stability, self.estimate_stability(node))

        graph.metadata["app_id"] = self.app_id
        graph.metadata["app_name"] = self.display_name
        return graph

    def classify_node(self, node: ObservationNode) -> str:
        return node.entity_type

    def affordances_for(self, node: ObservationNode) -> list[str]:
        affordances: list[str] = []
        if node.semantic_role in {"menu_item", "clickable_container", "list_row"}:
            affordances.extend(["click", "select"])
        if node.node_type == "text_field":
            affordances.extend(["focus", "type"])
        return affordances

    def state_tags_for(self, node: ObservationNode) -> list[str]:
        tags: list[str] = []
        if node.region:
            tags.append(node.region)
        if node.semantic_role:
            tags.append(node.semantic_role)
        return tags

    def estimate_stability(self, node: ObservationNode) -> float:
        score = 0.15
        if node.visual_id:
            score += 0.20
        if node.display_label() and not node.display_label().startswith("["):
            score += 0.25
        if node.semantic_role:
            score += 0.20
        if node.entity_type:
            score += 0.20
        return min(1.0, score)

    def is_safe_target(self, node: ObservationNode) -> bool:
        label = normalize_text(node.display_label())
        if any(term in label for term in self.risky_terms):
            return False
        if node.semantic_role not in {"menu_item", "clickable_container", "list_row"}:
            return False
        if "destructive" in node.state_tags:
            return False
        return "click" in node.affordances or "select" in node.affordances

    def describe_scene(self, graph: ObservationGraph) -> str:
        labels = [node.display_label() for node in graph.flatten() if node.display_label() and not node.display_label().startswith("[")]
        preview = ", ".join(labels[:6]) if labels else "no readable controls"
        return f"{self.display_name} scene with {len(graph.flatten())} nodes; highlights: {preview}"

    def labels_for(self, node: ObservationNode) -> list[str]:
        labels: list[str] = []
        for current in node.flatten():
            label = normalize_text(current.display_label())
            if label:
                labels.append(label)
        return labels

    def label_blob(self, node: ObservationNode) -> str:
        return " ".join(self.labels_for(node))

    def safe_nodes(self, graph: ObservationGraph) -> list[ObservationNode]:
        safe = [node for node in graph.flatten() if self.is_safe_target(node)]
        safe.sort(key=lambda item: (-item.stability, item.region, item.display_label()))
        return safe


class AppProfileRegistry:
    def __init__(self, profiles: Iterable[AppProfile] | None = None) -> None:
        self._profiles = {profile.app_id: profile for profile in (profiles or [])}

    def register(self, profile: AppProfile) -> None:
        self._profiles[profile.app_id] = profile

    def get(self, app_id: str) -> AppProfile | None:
        return self._profiles.get(app_id)

    def all(self) -> list[AppProfile]:
        return list(self._profiles.values())

    def detect(self, environment: dict, observation: ObservationGraph | None = None) -> AppProfile | None:
        scored = [(profile.match_score(environment, observation), profile) for profile in self._profiles.values()]
        scored = [item for item in scored if item[0] > 0]
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def annotate(self, graph: ObservationGraph, environment: dict, app_id: str = "") -> tuple[ObservationGraph, AppProfile | None]:
        profile = self.get(app_id) if app_id else self.detect(environment, graph)
        if profile:
            return profile.annotate(graph, environment), profile
        return graph, None
