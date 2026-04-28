from __future__ import annotations

from typing import Any

from copilot.memory.store import MemoryStore
from copilot.schemas import PlanStep, PolicyDecision, RiskLevel, TaskSpec, TrustMode


class PolicyEngine:
    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    def _step_app_id(self, step: PlanStep) -> str:
        target_value = (step.target.value if step.target else "").lower()
        return str(step.parameters.get("app_id") or step.parameters.get("expected_app") or target_value or "").strip().lower()

    def _text_blob(self, task: TaskSpec, step: PlanStep) -> str:
        chunks = [
            task.prompt,
            task.goal,
            step.title,
            step.action_type,
            step.target.value if step.target else "",
            str(step.target.filters if step.target else {}),
            str(step.parameters),
            step.intent.description if step.intent else "",
            step.intent.semantic_target if step.intent else "",
            " ".join(step.intent.risk_tags) if step.intent else "",
        ]
        return " ".join(chunk for chunk in chunks if chunk).lower()

    def _classify_risk(self, task: TaskSpec, step: PlanStep) -> RiskLevel:
        text_blob = self._text_blob(task, step)
        high_terms = {
            "delete",
            "remove",
            "rm ",
            "rmdir",
            "del ",
            "erase",
            "format",
            "payment",
            "purchase",
            "checkout",
            "password",
            "credential",
            "secret",
            "token",
            "api key",
            "terminal",
            "command",
            "powershell",
            "cmd.exe",
            "sudo",
        }
        medium_terms = {"send", "message", "email", "edit", "write file", "rename", "move", "upload", "submit", "save"}
        low_terms = {"search", "open app", "open chrome", "read", "observe", "find", "navigate"}
        if step.action_type in {"legacy_command", "run_terminal", "terminal_command"}:
            return RiskLevel.HIGH
        if step.action_type == "press_key" and bool(step.parameters.get("destructive", False)):
            return RiskLevel.HIGH
        if any(term in text_blob for term in high_terms):
            return RiskLevel.HIGH
        if step.action_type in {"type_text", "edit_file", "send_message"} or any(term in text_blob for term in medium_terms):
            return RiskLevel.MEDIUM
        if any(term in text_blob for term in low_terms):
            return RiskLevel.LOW
        return step.risk_level

    def evaluate_step(self, task: TaskSpec, step: PlanStep) -> PolicyDecision:
        policy = self.memory_store.policy_memory
        blocked_apps = {app.lower() for app in policy.get("blocked_apps", [])}
        trusted_apps = {app.lower() for app in policy.get("trusted_apps", [])}
        high_risk_allowed_apps = {app.lower() for app in policy.get("high_risk_allowed_apps", [])}
        blocked_concepts = {concept.lower() for concept in policy.get("blocked_concepts", [])}
        allowlisted_paths = [path.lower() for path in policy.get("allowlisted_paths", [])]
        blocked_paths = [path.lower() for path in policy.get("blocked_paths", [])]
        target_value = (step.target.value if step.target else "").lower()
        text_blob = self._text_blob(task, step)
        classified_risk = self._classify_risk(task, step)
        effective_risk = classified_risk
        if step.risk_level == RiskLevel.CRITICAL:
            effective_risk = RiskLevel.CRITICAL
        elif step.risk_level == RiskLevel.HIGH or classified_risk == RiskLevel.HIGH:
            effective_risk = RiskLevel.HIGH
        elif step.risk_level == RiskLevel.MEDIUM or classified_risk == RiskLevel.MEDIUM:
            effective_risk = RiskLevel.MEDIUM

        if any(concept in text_blob for concept in blocked_concepts):
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason="Step references a blocked concept.",
                risk_level=RiskLevel.CRITICAL,
            )

        if target_value and target_value in blocked_apps:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason=f"Target '{target_value}' is blocked by policy.",
                risk_level=RiskLevel.CRITICAL,
            )

        path_candidates = []
        if step.target and step.target.kind in {"path", "file_path"}:
            path_candidates.append(step.target.value)
        for key in ("path", "source_path", "target_path", "url"):
            value = step.parameters.get(key)
            if isinstance(value, str) and value:
                path_candidates.append(value)

        normalized_paths = [candidate.lower() for candidate in path_candidates]
        for candidate in normalized_paths:
            if any(blocked in candidate for blocked in blocked_paths):
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason=f"Path '{candidate}' is blocked by policy.",
                    risk_level=RiskLevel.CRITICAL,
                )

        dangerous_terms = {"delete", "overwrite", "move", "upload", "download", "purchase", "submit", "payment", "security", "account", "password", "terminal", "command"}
        if step.action_type in {"click_point"}:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason="Raw point clicks are disabled by default.",
                risk_level=RiskLevel.CRITICAL,
            )

        keyboard_keys = []
        if step.action_type == "press_key":
            raw_keys = step.parameters.get("keys", [])
            if isinstance(raw_keys, str):
                keyboard_keys = [raw_keys.lower()]
            elif isinstance(raw_keys, list):
                keyboard_keys = [str(key).lower() for key in raw_keys]
        safe_search_submit = step.action_type == "press_key" and keyboard_keys == ["enter"] and any(
            token in text_blob for token in ("search", "query", "results", "omnibox", "explorer")
        )
        shortcut_id = str(step.parameters.get("shortcut_id", "")).strip()
        trusted_shortcut = step.action_type == "press_key" and bool(shortcut_id) and not bool(step.parameters.get("destructive", False))
        allowed_destructive_shortcut = (
            step.action_type == "press_key"
            and bool(shortcut_id)
            and bool(step.parameters.get("allow_destructive_shortcut", False))
        )

        if step.action_type in {"type_text", "press_key"} and "submit" in text_blob and not safe_search_submit and not trusted_shortcut:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason="Step appears to submit typed content without a safe-search context.",
                risk_level=RiskLevel.CRITICAL,
            )

        if task.trust_mode == TrustMode.ALWAYS_CONFIRM:
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                reason="Always-confirm mode requires explicit approval.",
                risk_level=effective_risk,
            )

        risky_actions = {"explore_safe", "interaction_learning", "recover", "route_window"}
        requires_approval = step.requires_approval or step.action_type in risky_actions
        if step.action_type == "press_key" and (bool(step.parameters.get("destructive", False)) or bool(step.parameters.get("requires_approval", False))):
            requires_approval = True
        if any(word in text_blob for word in dangerous_terms):
            requires_approval = True

        if effective_risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            requires_approval = True

        app_id = self._step_app_id(step)
        if effective_risk == RiskLevel.HIGH and app_id and app_id in high_risk_allowed_apps:
            requires_approval = False

        if step.action_type == "route_window":
            if app_id and app_id not in trusted_apps:
                requires_approval = True

        if step.action_type == "explore_safe":
            app_id = (step.parameters.get("app_id") or target_value or "").lower()
            if not app_id:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason="Safe exploration requires a target application.",
                    risk_level=RiskLevel.CRITICAL,
                )
            if app_id not in trusted_apps:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason=f"Exploration is only allowed inside trusted apps; '{app_id}' is not trusted.",
                    risk_level=RiskLevel.CRITICAL,
                )

        if step.action_type == "learning_session":
            app_id = (step.parameters.get("app_id") or target_value or "").lower()
            if not app_id:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason="Learning sessions require a target application.",
                    risk_level=RiskLevel.CRITICAL,
                )
            if app_id != "current_window" and app_id not in trusted_apps:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason=f"Learning sessions are only allowed inside trusted apps; '{app_id}' is not trusted.",
                    risk_level=RiskLevel.CRITICAL,
                )

        if step.action_type == "interaction_learning":
            app_id = (step.parameters.get("app_id") or target_value or "").lower()
            if not app_id:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason="Interaction learning requires a target application.",
                    risk_level=RiskLevel.CRITICAL,
                )
            if app_id == "current_window":
                return PolicyDecision(
                    allowed=True,
                    requires_approval=False,
                    reason="Current-window interaction learning is allowed with runtime safe-target filters.",
                    risk_level=RiskLevel.MEDIUM,
                )
            if app_id not in trusted_apps:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason=f"Interaction learning is only allowed inside trusted apps; '{app_id}' is not trusted.",
                    risk_level=RiskLevel.CRITICAL,
                )

        if normalized_paths and allowlisted_paths:
            if not any(any(allowed in candidate for allowed in allowlisted_paths) for candidate in normalized_paths):
                requires_approval = True

        if task.trust_mode == TrustMode.MOSTLY_AUTONOMOUS:
            if app_id in trusted_apps and effective_risk in {RiskLevel.LOW, RiskLevel.MEDIUM}:
                requires_approval = False

        reason = "Risk-gated step." if requires_approval else "Step is allowed."
        if effective_risk == RiskLevel.HIGH and requires_approval:
            reason = "High-risk action requires approval."
        elif effective_risk == RiskLevel.HIGH and app_id in high_risk_allowed_apps:
            reason = f"High-risk action allowed by app policy for '{app_id}'."

        return PolicyDecision(
            allowed=True,
            requires_approval=requires_approval,
            reason=reason,
            risk_level=effective_risk,
        )
