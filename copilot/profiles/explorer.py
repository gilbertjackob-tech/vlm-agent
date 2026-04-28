from __future__ import annotations

import os

from copilot.profiles.base import AppProfile, normalize_text
from copilot.schemas import ObservationNode


class ExplorerProfile(AppProfile):
    def __init__(self) -> None:
        super().__init__(
            app_id="explorer",
            display_name="File Explorer",
            window_keywords=["file explorer", "explorer", "desktop", "downloads", "documents"],
            trusted_for_exploration=True,
            risky_terms={"delete", "rename", "move", "share", "properties", "close", "new tab"},
        )

    def classify_node(self, node: ObservationNode) -> str:
        label_blob = self.label_blob(node)
        label = normalize_text(node.display_label())
        text = f"{label} {label_blob}".strip()
        _, ext = os.path.splitext(label)
        ext = ext.lower()

        if node.semantic_role == "list_header":
            return "table_header"
        if node.semantic_role == "list_row":
            if "file folder" in text or label.endswith(" folder") or "folder" in node.learned_concepts:
                return "folder"
            if "shortcut" in text or ext == ".lnk":
                return "shortcut"
            if "winrar zip archive" in text or ext in {".zip", ".rar", ".7z", ".tar", ".gz"}:
                return "archive"
            if "python file" in text or ext == ".py":
                return "python_file"
            if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}:
                return "image"
            if ext in {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".m4v"}:
                return "video"
            if ext in {".pdf", ".doc", ".docx", ".txt", ".ppt", ".pptx"}:
                return "document"
            if ext:
                return "file"
        if node.semantic_role == "menu_item" and ("desktop" in text or "downloads" in text or "documents" in text):
            return "navigation_item"
        if "search" in text and node.region == "top_menu":
            return "search_field"
        return node.entity_type

    def affordances_for(self, node: ObservationNode) -> list[str]:
        affordances = super().affordances_for(node)
        if node.entity_type in {"folder", "file", "image", "video", "archive", "shortcut", "document", "python_file"}:
            affordances.extend(["open", "inspect"])
        if node.entity_type in {"folder", "navigation_item"}:
            affordances.append("navigate")
        if node.entity_type == "search_field":
            affordances.extend(["focus", "type", "search"])
        return sorted(set(affordances))

    def state_tags_for(self, node: ObservationNode) -> list[str]:
        tags = super().state_tags_for(node)
        if node.semantic_role == "list_row":
            tags.append("table_surface")
        if node.entity_type in {"folder", "file", "image", "video", "archive", "shortcut", "document", "python_file"}:
            tags.append("content_item")
        return sorted(set(tags))

    def is_safe_target(self, node: ObservationNode) -> bool:
        if not super().is_safe_target(node):
            return False
        if node.entity_type in {"table_header"}:
            return False
        if node.entity_type in {"folder", "file", "image", "video", "archive", "shortcut", "document", "python_file", "navigation_item", "search_field"}:
            return True
        if node.region == "main_page" and node.semantic_role in {"list_row", "clickable_container"}:
            return True
        return False
