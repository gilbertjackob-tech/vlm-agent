from __future__ import annotations

from copilot.profiles.base import AppProfile, normalize_text
from copilot.schemas import ObservationNode


class ChromeProfile(AppProfile):
    def __init__(self) -> None:
        super().__init__(
            app_id="chrome",
            display_name="Google Chrome",
            window_keywords=[
                "chrome",
                "google chrome",
                "microsoft edge",
                "edge",
                "mozilla firefox",
                "firefox",
                "brave",
                "opera",
                "youtube",
                "google search",
            ],
            trusted_for_exploration=False,
            risky_terms={"close", "delete", "remove", "payment", "checkout", "purchase", "account", "sign out", "submit"},
        )

    def classify_node(self, node: ObservationNode) -> str:
        label = normalize_text(node.display_label())
        text = f"{label} {self.label_blob(node)}"

        if node.region == "top_menu" and ("search" in text or "address" in text or "http" in text):
            return "omnibox"
        if node.region == "top_menu" and node.semantic_role == "menu_item":
            if "tab" in text or node.node_type == "container":
                return "tab"
        if node.node_type == "text_field" and ("search" in text or "youtube" in text):
            return "search_field"
        if "dialog" in text or "modal" in text:
            return "modal_dialog"
        if node.region == "main_page" and node.semantic_role in {"clickable_container", "list_row"}:
            return "page_content"
        return node.entity_type

    def affordances_for(self, node: ObservationNode) -> list[str]:
        affordances = super().affordances_for(node)
        if node.entity_type == "omnibox":
            affordances.extend(["focus", "type", "navigate"])
        if node.entity_type == "tab":
            affordances.extend(["activate", "select"])
        if node.entity_type == "search_field":
            affordances.extend(["focus", "type", "search"])
        if node.entity_type == "page_content":
            affordances.extend(["open", "inspect"])
        return sorted(set(affordances))

    def state_tags_for(self, node: ObservationNode) -> list[str]:
        tags = super().state_tags_for(node)
        label = normalize_text(node.display_label())
        if node.entity_type == "modal_dialog":
            tags.append("high_attention")
        if any(term in label for term in self.risky_terms):
            tags.append("destructive")
        return sorted(set(tags))

    def is_safe_target(self, node: ObservationNode) -> bool:
        if not super().is_safe_target(node):
            return False
        if node.entity_type in {"omnibox", "search_field", "tab"}:
            return True
        if node.entity_type == "page_content":
            return "destructive" not in node.state_tags
        return False
