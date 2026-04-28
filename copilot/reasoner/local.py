from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import re
import urllib.error
import urllib.request

from copilot.profiles import AppProfileRegistry
from copilot.perception.target_ranking import TargetRankResult, rank_action_targets
from copilot.schemas import ObservationGraph, ObservationNode, SceneDelta, TaskSpec
from copilot.shortcuts import match_shortcut_intents


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


class LocalReasoner(ABC):
    @abstractmethod
    def interpret_scene(self, task: TaskSpec, observation: ObservationGraph | None, environment: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def decompose_task(self, task: TaskSpec, observation: ObservationGraph | None, environment: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def choose_action_target(
        self,
        filters: dict,
        observation: ObservationGraph | None,
        scene: dict | None = None,
    ) -> ObservationNode | None:
        raise NotImplementedError

    def rank_action_targets(
        self,
        filters: dict,
        observation: ObservationGraph | None,
        scene: dict | None = None,
    ) -> TargetRankResult:
        return rank_action_targets(filters, observation)

    @abstractmethod
    def resolve_ambiguity(self, candidates: list[ObservationNode], task_prompt: str, scene: dict | None = None) -> ObservationNode | None:
        raise NotImplementedError

    @abstractmethod
    def summarize_scene_change(
        self,
        before: ObservationGraph | None,
        after: ObservationGraph | None,
        step_id: str,
        expected_change: str = "",
    ) -> SceneDelta:
        raise NotImplementedError


class HybridLocalReasoner(LocalReasoner):
    def __init__(self, profile_registry: AppProfileRegistry) -> None:
        self.profile_registry = profile_registry
        self.endpoint_url = os.getenv("COPILOT_LOCAL_VLM_URL", "").strip()
        self.model_name = os.getenv("COPILOT_LOCAL_VLM_MODEL", "").strip()
        self.timeout = float(os.getenv("COPILOT_LOCAL_VLM_TIMEOUT", "1.5"))

    def _call_local_model(self, purpose: str, payload: dict) -> dict:
        if not self.endpoint_url:
            return {}
        body = {
            "model": self.model_name,
            "purpose": purpose,
            "payload": payload,
        }
        request = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, urllib.error.URLError):
            return {}

    def interpret_scene(self, task: TaskSpec, observation: ObservationGraph | None, environment: dict) -> dict:
        profile = self.profile_registry.detect(environment, observation)
        active_window = environment.get("windows", {}).get("active_window", {})
        nodes = observation.flatten() if observation else []
        readable = [node.display_label() for node in nodes if node.display_label() and not node.display_label().startswith("[")]
        scene = {
            "app_id": profile.app_id if profile else "",
            "app_name": profile.display_name if profile else "",
            "active_window_title": active_window.get("title", ""),
            "node_count": len(nodes),
            "salient_labels": readable[:10],
            "ambiguous": len(readable) < 3,
            "summary": "",
        }

        if observation and profile:
            scene["summary"] = profile.describe_scene(observation)
        elif readable:
            scene["summary"] = f"Scene with {len(nodes)} nodes; highlights: {', '.join(readable[:6])}"
        else:
            scene["summary"] = "Scene is sparse or unreadable."

        model_result = self._call_local_model(
            "interpret_scene",
            {
                "prompt": task.prompt,
                "scene": scene,
                "environment": environment,
            },
        )
        if isinstance(model_result, dict):
            for key in ("summary", "ambiguity", "subgoal", "notes"):
                if key in model_result:
                    scene[key] = model_result[key]
        return scene

    def decompose_task(self, task: TaskSpec, observation: ObservationGraph | None, environment: dict) -> dict:
        scene = self.interpret_scene(task, observation, environment)
        prompt = normalize_text(task.prompt)
        actions: list[dict] = []
        required_apps: list[str] = []
        subgoal = ""

        def add_action(
            verb: str,
            title: str,
            target_kind: str,
            target: str,
            parameters: dict | None,
            success: str,
            fallback_hint: str = "",
        ) -> None:
            actions.append(
                {
                    "verb": verb,
                    "title": title,
                    "target_kind": target_kind,
                    "target": target,
                    "parameters": parameters or {},
                    "success": success,
                    "fallback_hint": fallback_hint,
                }
            )

        def ensure_app(app_id: str, window_title: str) -> None:
            if app_id not in required_apps:
                required_apps.append(app_id)
            add_action(
                "ensure_app",
                f"Ensure {app_id} is focused",
                "application",
                app_id,
                {"app_id": app_id, "window_title": window_title},
                f"{app_id} should be visible and focused.",
                fallback_hint=f"Use Windows routing to bring {app_id} into focus.",
            )

        def current_target_app() -> str:
            return (
                required_apps[0]
                if required_apps
                else scene.get("app_id")
                or environment.get("windows", {}).get("active_app_guess", "")
                or "current_window"
            )

        chrome_requested = "chrome" in prompt or "youtube" in prompt or "browser" in prompt or "search" in prompt
        explorer_requested = any(term in prompt for term in ("explorer", "desktop", "file", "folder", "documents")) or (
            "downloads" in prompt and not any(term in prompt for term in ("chrome", "browser"))
        )
        requested_location = ""
        for candidate in ("downloads", "documents", "desktop"):
            if re.search(rf"\b{re.escape(candidate)}\b", prompt):
                requested_location = candidate
                break

        requested_entity_type = ""
        for token, entity_type in {
            "videos": "video",
            "video": "video",
            "images": "image",
            "image": "image",
            "folders": "folder",
            "folder": "folder",
            "archives": "archive",
            "archive": "archive",
            "documents": "document",
            "document": "document",
            "python": "python_file",
            "files": "file",
            "file": "file",
        }.items():
            if re.search(rf"\b{re.escape(token)}\b", prompt):
                requested_entity_type = entity_type
                break

        quoted_target = ""
        quoted_match = re.search(r"['\"]([^'\"]+)['\"]", task.prompt)
        if quoted_match:
            quoted_target = quoted_match.group(1).strip()

        file_match = re.search(r"\bopen\s+(?!explorer\b|chrome\b|browser\b)([A-Za-z0-9_. -]+\.[A-Za-z0-9]+)\b", task.prompt, flags=re.IGNORECASE)
        folder_match = re.search(r"\bopen\s+folder\s+([A-Za-z0-9_. -]+)\b", task.prompt, flags=re.IGNORECASE)
        open_target = quoted_target or (file_match.group(1).strip() if file_match else "") or (folder_match.group(1).strip() if folder_match else "")
        modifier_click_match = re.search(r"\b(ctrl|control|shift)\s*\+?\s*click\s+(?:the\s+)?(.+)$", task.prompt, flags=re.IGNORECASE)

        youtube_match = re.search(r"search youtube for\s+(.+?)(?:\s+and verify results page)?$", task.prompt, flags=re.IGNORECASE)
        chrome_search_match = re.search(
            r"(?:search chrome for|search in chrome for|search the web for|search for)\s+(.+?)(?:\s+and verify results page)?$",
            task.prompt,
            flags=re.IGNORECASE,
        )
        chrome_address_type_match = re.search(
            r"(?:address bar|omnibox|search field).*?\b(?:type|enter)\s+(.+?)(?:\s+and verify(?:\s+the)?(?:\s+text)?|\s+then|\s*$)",
            task.prompt,
            flags=re.IGNORECASE,
        )
        verify_results = "verify results page" in prompt or "results page" in prompt
        open_safe_result = bool(re.search(r"\b(open|click)\s+(?:the\s+)?(?:first|top|safe)?\s*(?:result|link)\b", prompt))
        search_query = ""
        search_scope = ""
        if youtube_match:
            search_query = youtube_match.group(1).strip()
            search_scope = "youtube"
        elif chrome_search_match and chrome_requested:
            search_query = chrome_search_match.group(1).strip()
            search_scope = "web"
        elif chrome_address_type_match and chrome_requested:
            search_query = chrome_address_type_match.group(1).strip()
            search_scope = "web"

        if explorer_requested:
            ensure_app("explorer", "File Explorer")
            if requested_location:
                add_action(
                    "navigate_sidebar",
                    f"Navigate Explorer to {requested_location}",
                    "location",
                    requested_location,
                    {"app_id": "explorer", "location": requested_location},
                    f"Explorer should show the {requested_location} location.",
                    fallback_hint=f"Retry Explorer sidebar navigation to {requested_location}.",
                )
            if "search explorer for" in prompt or "search files for" in prompt or "find " in prompt:
                query_match = re.search(r"(?:search explorer for|search files for|find)\s+(.+)", task.prompt, flags=re.IGNORECASE)
                if query_match:
                    query = query_match.group(1).strip()
                    add_action(
                        "focus_search",
                        "Focus Explorer search",
                        "application",
                        "explorer",
                        {"app_id": "explorer"},
                        "Explorer search should be focused.",
                    )
                    add_action(
                        "type_query",
                        f"Type Explorer search query: {query}",
                        "text",
                        query,
                        {"app_id": "explorer", "query": query},
                        "The Explorer search query should be entered.",
                    )
                    add_action(
                        "submit_query",
                        "Submit Explorer search",
                        "keyboard",
                        "enter",
                        {"app_id": "explorer", "query": query},
                        "Explorer search should execute.",
                    )
            if open_target and not modifier_click_match:
                add_action(
                    "open_visible_target",
                    f"Open {open_target}",
                    "label",
                    open_target,
                    {"app_id": "explorer", "label": open_target},
                    f"{open_target} should be opened from the current Explorer view.",
                    fallback_hint=f"Reparse Explorer and retry opening {open_target}.",
                )
            if any(term in prompt for term in ("identify", "detect", "which files are", "which is", "what is")):
                target_type = requested_entity_type or "file"
                add_action(
                    "classify_visible_items",
                    f"Classify visible Explorer items as {target_type}",
                    "entity_type",
                    target_type,
                    {"app_id": "explorer", "entity_type": target_type},
                    f"Explorer should classify visible items with entity_type '{target_type}'.",
                    fallback_hint="Reparse Explorer and teach ambiguous rows if needed.",
                )
            if any(term in prompt for term in ("summarize", "folder contents", "what is happening", "look around", "observe", "parse")):
                add_action(
                    "summarize_folder",
                    "Summarize current Explorer folder",
                    "ui_surface",
                    "explorer",
                    {"app_id": "explorer"},
                    "The current Explorer folder contents should be summarized.",
                )

        if modifier_click_match and (explorer_requested or scene.get("app_id") == "explorer"):
            modifier = "ctrl" if modifier_click_match.group(1).lower() in {"ctrl", "control"} else "shift"
            target_label = modifier_click_match.group(2).strip()
            add_action(
                "modified_click_node",
                f"{modifier.title()} click {target_label}",
                "label",
                target_label,
                {
                    "modifiers": [modifier],
                    "click_count": 1,
                    "settle_wait": 0.4,
                    "filters": {
                        "app_id": "explorer",
                        "region": "main_page",
                        "label_contains": target_label,
                        "affordance": "open",
                        "min_score": 0.45,
                    },
                },
                f"Explorer should apply {modifier}+click selection to {target_label}.",
                fallback_hint=f"Reparse Explorer and retry {modifier}+click only on the intended item.",
            )

        if chrome_requested or search_query:
            ensure_app("chrome", "Google Chrome")
            if search_query:
                subgoal = f"Search for {search_query}"
                add_action(
                    "browser_search",
                    f"Search Chrome for {search_query}",
                    "search_query",
                    search_query,
                    {
                        "app_id": "chrome",
                        "query": search_query,
                        "scope": search_scope or "web",
                        "verify_results": verify_results or bool(search_scope),
                    },
                    "Chrome should run the requested search and show a results page.",
                    fallback_hint="Retry via the omnibox or vision fallback if the selector path is unavailable.",
                )
            elif verify_results:
                add_action(
                    "verify_results_page",
                    "Verify Chrome results page",
                    "application",
                    "chrome",
                    {"app_id": "chrome"},
                    "Chrome should show a search results page.",
                )
            if open_safe_result:
                result_target = quoted_target or search_query or "result"
                add_action(
                    "open_safe_result",
                    f"Open a safe Chrome result for {result_target}",
                    "page_content",
                    result_target,
                    {"app_id": "chrome", "query": result_target},
                    f"Chrome should open a safe result matching {result_target}.",
                    fallback_hint="Retry with a safer DOM-ranked result target if page content is ambiguous.",
                )
            if "dismiss modal" in prompt or "close popup" in prompt:
                add_action(
                    "dismiss_modal",
                    "Dismiss a Chrome modal",
                    "application",
                    "chrome",
                    {"app_id": "chrome"},
                    "A simple Chrome modal should be dismissed safely.",
                )

        shortcut_apps: list[str] = []
        if chrome_requested:
            shortcut_apps.append("chrome")
        if explorer_requested:
            shortcut_apps.append("explorer")
        if not shortcut_apps:
            active_app = current_target_app()
            if active_app and active_app != "current_window":
                shortcut_apps.append(active_app)

        for shortcut_app in shortcut_apps:
            for shortcut in match_shortcut_intents(task.prompt, shortcut_app):
                add_action(
                    "press_shortcut",
                    shortcut.title,
                    "keyboard",
                    "+".join(shortcut.keys),
                    {
                        "app_id": shortcut_app,
                        "shortcut_id": shortcut.shortcut_id,
                        "keys": list(shortcut.keys),
                        "hotkey": shortcut.hotkey,
                        "expected_change": shortcut.expected_change,
                        "destructive": shortcut.destructive,
                        "requires_approval": shortcut.requires_approval,
                        "allow_destructive_shortcut": shortcut.destructive,
                    },
                    shortcut.expected_change,
                    fallback_hint=f"Refocus {shortcut_app} and retry {'+'.join(shortcut.keys)}.",
                )

        interaction_learn_match = re.search(r"\b(self[- ]?explor|interaction|reward|punish|clicking what opens|what opens what|click.+opens?)\b", prompt)
        if interaction_learn_match and "randomly" not in prompt:
            target_app = current_target_app()
            add_action(
                "interaction_learning",
                "Learn safe click outcomes",
                "application",
                target_app,
                {
                    "app_id": target_app,
                    "max_actions": 5,
                    "settle_wait": 0.8,
                    "recover_after_each": True,
                    "filters": {
                        "exclude_destructive": True,
                    },
                },
                "Safe clicks should produce rewarded or punished interaction edges.",
                fallback_hint="Reduce action count and retry if scene changes are unstable.",
            )

        hover_learn_match = re.search(r"\b(hover|tooltip|learn|inspect)\b", prompt)
        if hover_learn_match and not interaction_learn_match and "randomly" not in prompt:
            target_app = current_target_app()
            add_action(
                "learning_session",
                "Run safe learning session",
                "application",
                target_app,
                {"app_id": target_app, "max_nodes": 8, "settle_wait": 0.45},
                "Safe hover-only learning should harvest feedback and queue uncertain detections.",
                fallback_hint="Reduce target count and retry if hover feedback is unstable.",
            )

        explore_match = re.search(r"\b(explore|randomly|safe explore)\b", prompt)
        if explore_match:
            rounds_match = re.search(r"(\d+)\s*(?:rounds?|times?|clicks?)", prompt)
            rounds = int(rounds_match.group(1)) if rounds_match else 4
            target_app = current_target_app()
            add_action(
                "observe",
                "Parse before safe exploration",
                "ui_surface",
                target_app,
                {"app_id": target_app, "output_filename": "explore_seed.png"},
                "The current surface should be parsed before exploration.",
            )
            add_action(
                "explore_safe",
                f"Safely explore the current UI for {rounds} rounds",
                "application",
                target_app,
                {"rounds": max(1, min(rounds, 12)), "app_id": target_app},
                "Safe content exploration should complete without leaving the trusted surface.",
            )
            add_action(
                "observe",
                "Parse after safe exploration",
                "ui_surface",
                target_app,
                {"app_id": target_app, "output_filename": "explore_after.png"},
                "The post-exploration surface should be parsed.",
            )

        if not actions:
            add_action(
                "observe",
                "Parse current UI for planning context",
                "ui_surface",
                scene.get("app_id") or "current_window",
                {"output_filename": "panel_parse.png"},
                "The system should capture the current UI before acting.",
            )

        model_result = self._call_local_model(
            "decompose_task",
            {
                "prompt": task.prompt,
                "scene": scene,
                "actions": actions,
            },
        )
        if isinstance(model_result, dict) and isinstance(model_result.get("actions"), list):
            candidate_actions = [item for item in model_result["actions"] if isinstance(item, dict)]
            if candidate_actions:
                actions = candidate_actions
        return {
            "scene": scene,
            "actions": actions,
            "required_apps": required_apps,
            "subgoal": subgoal,
        }

    def _score_node(self, node: ObservationNode, filters: dict) -> float:
        if filters.get("exclude_destructive") and "destructive" in node.state_tags:
            return -1.0

        score = 0.0
        max_score = 0.0
        label = normalize_text(node.display_label())

        if filters.get("label_contains"):
            max_score += 2.0
            if normalize_text(filters["label_contains"]) in label:
                score += 2.0

        if filters.get("entity_type"):
            max_score += 1.5
            if node.entity_type == filters["entity_type"]:
                score += 1.5

        if filters.get("semantic_role"):
            max_score += 1.0
            if node.semantic_role == filters["semantic_role"]:
                score += 1.0

        if filters.get("region"):
            max_score += 1.0
            if node.region == filters["region"]:
                score += 1.0

        if filters.get("app_id"):
            max_score += 0.75
            if node.app_id == filters["app_id"]:
                score += 0.75

        if filters.get("affordance"):
            max_score += 1.0
            if filters["affordance"] in node.affordances:
                score += 1.0

        if filters.get("state_tag"):
            max_score += 0.75
            if filters["state_tag"] in node.state_tags:
                score += 0.75

        if filters.get("concept"):
            max_score += 0.75
            if filters["concept"] in node.learned_concepts:
                score += 0.75

        preferred_labels = [normalize_text(item) for item in filters.get("preferred_labels", []) if item]
        if preferred_labels:
            max_score += 1.0
            if any(pref in label or label in pref for pref in preferred_labels):
                score += 1.0

        avoid_labels = [normalize_text(item) for item in filters.get("avoid_labels", []) if item]
        if avoid_labels:
            max_score += 1.25
            if any(avoid in label or label in avoid for avoid in avoid_labels):
                score -= 1.25

        avoid_visual_ids = {str(item) for item in filters.get("avoid_visual_ids", []) if item}
        if avoid_visual_ids:
            max_score += 1.0
            if node.visual_id and node.visual_id in avoid_visual_ids:
                score -= 1.0

        if filters.get("y_max") is not None:
            max_score += 0.5
            if node.center.get("y", 99999) <= int(filters["y_max"]):
                score += 0.5

        if filters.get("x_max") is not None:
            max_score += 0.5
            if node.center.get("x", 99999) <= int(filters["x_max"]):
                score += 0.5

        if max_score <= 0:
            return 0.0
        return score / max_score

    def choose_action_target(
        self,
        filters: dict,
        observation: ObservationGraph | None,
        scene: dict | None = None,
    ) -> ObservationNode | None:
        return self.rank_action_targets(filters, observation, scene).selected_node

    def rank_action_targets(
        self,
        filters: dict,
        observation: ObservationGraph | None,
        scene: dict | None = None,
    ) -> TargetRankResult:
        return rank_action_targets(filters, observation)

    def resolve_ambiguity(self, candidates: list[ObservationNode], task_prompt: str, scene: dict | None = None) -> ObservationNode | None:
        if not candidates:
            return None
        prompt = normalize_text(task_prompt)
        scene_region = normalize_text(str((scene or {}).get("region", "")))

        def score_candidate(node: ObservationNode) -> tuple[float, int, int, int]:
            label = normalize_text(node.display_label())
            role = normalize_text(node.semantic_role)
            entity = normalize_text(node.entity_type)
            region = normalize_text(node.region)
            score = 0.0
            if label and label in prompt:
                score += 4.0
            elif label and any(part and part in prompt for part in label.split()):
                score += 1.5
            if role and role in prompt:
                score += 1.25
            if entity and entity in prompt:
                score += 1.5
            if region and (region in prompt or region == scene_region):
                score += 0.75
            if node.affordances:
                score += min(0.6, len(node.affordances) * 0.15)
            center_x = int(node.center.get("x", 99999) or 99999)
            center_y = int(node.center.get("y", 99999) or 99999)
            proximity = -abs(center_x) - abs(center_y)
            score += float(node.stability or 0.0)
            return (score, len(label), proximity, len(node.affordances))

        candidates.sort(
            key=score_candidate,
            reverse=True,
        )

        model_result = self._call_local_model(
            "resolve_ambiguity",
            {
                "task_prompt": task_prompt,
                "scene": scene or {},
                "candidates": [candidate.to_dict() for candidate in candidates[:4]],
            },
        )
        chosen_label = normalize_text(model_result.get("label", "")) if isinstance(model_result, dict) else ""
        if chosen_label:
            for candidate in candidates:
                if normalize_text(candidate.display_label()) == chosen_label:
                    return candidate
        return candidates[0]

    def summarize_scene_change(
        self,
        before: ObservationGraph | None,
        after: ObservationGraph | None,
        step_id: str,
        expected_change: str = "",
    ) -> SceneDelta:
        before_labels = {normalize_text(node.display_label()) for node in before.flatten()} if before else set()
        after_labels = {normalize_text(node.display_label()) for node in after.flatten()} if after else set()
        added = sorted(label for label in after_labels - before_labels if label)
        removed = sorted(label for label in before_labels - after_labels if label)

        before_summary = before.metadata.get("scene_summary", "") if before else "No prior scene."
        after_summary = after.metadata.get("scene_summary", "") if after else "No post-action scene."
        before_app = before.metadata.get("app_id", "") if before else ""
        after_app = after.metadata.get("app_id", "") if after else ""

        if added or removed:
            parts = []
            if added:
                parts.append(f"added: {', '.join(added[:5])}")
            if removed:
                parts.append(f"removed: {', '.join(removed[:5])}")
            actual = "; ".join(parts)
        elif after_summary != before_summary:
            actual = "Scene summary changed."
        else:
            actual = "No obvious scene change."

        changed = bool(added or removed or after_summary != before_summary or before_app != after_app)
        recovery_hint = "Reparse and refocus the current window before retrying." if not changed else ""

        model_result = self._call_local_model(
            "summarize_scene_change",
            {
                "expected_change": expected_change,
                "before_summary": before_summary,
                "after_summary": after_summary,
                "actual_change": actual,
                "added": added[:8],
                "removed": removed[:8],
            },
        )
        if isinstance(model_result, dict) and model_result.get("actual_change"):
            actual = str(model_result.get("actual_change", actual))
        if isinstance(model_result, dict) and model_result.get("recovery_hint"):
            recovery_hint = str(model_result.get("recovery_hint", recovery_hint))

        return SceneDelta(
            step_id=step_id,
            expected_change=expected_change,
            before_summary=before_summary,
            after_summary=after_summary,
            actual_change=actual,
            changed=changed,
            maintained_app_id=(not before_app or not after_app or before_app == after_app),
            added_labels=added[:8],
            removed_labels=removed[:8],
            recovery_hint=recovery_hint,
        )
