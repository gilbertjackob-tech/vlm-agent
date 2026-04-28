from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from copilot.schemas import ObservationGraph, ObservationNode
from copilot.runtime.target_identity import create_target_identity, detect_target_drift, ambiguous_identity_matches


FAILURE_TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
FAILURE_TARGET_AMBIGUOUS = "TARGET_AMBIGUOUS"
FAILURE_FOCUS_NOT_CONFIRMED = "FOCUS_NOT_CONFIRMED"
FAILURE_NO_STATE_CHANGE = "NO_STATE_CHANGE"
FAILURE_UNSAFE_COORDINATE = "UNSAFE_COORDINATE"
FAILURE_POLICY_BLOCKED = "POLICY_BLOCKED"
FAILURE_TIMEOUT = "TIMEOUT"


@dataclass
class ActionContract:
    step_id: str
    action_type: str
    intent: str
    target: str
    evidence: list[str] = field(default_factory=list)
    before_checks: dict[str, bool] = field(default_factory=dict)
    during_checks: dict[str, bool] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    evidence_score: float = 0.0
    evidence_grade: str = "none"
    failure_reason: str = ""
    recovery_strategy: list[str] = field(default_factory=list)
    target_identity: dict[str, Any] = field(default_factory=dict)
    target_ranking: dict[str, Any] = field(default_factory=dict)
    identity_drifted: bool = False
    identity_ambiguous: bool = False
    status: str = "planned"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def verified(self) -> bool:
        return bool(self.verification.get("verified"))


def score_evidence(evidence: list[str]) -> tuple[float, str]:
    score = 0.0
    for item in evidence:
        lower = item.lower()
        if lower.startswith("uia:") or lower.startswith("accessibility:"):
            score += 1.0
        elif lower.startswith("dom:") or lower.startswith("dom_selector:"):
            score += 0.85
        elif lower.startswith("ocr:") or lower.startswith("label:"):
            score += 0.55
        elif lower.startswith("visual:") or lower.startswith("screenshot:") or lower.startswith("region:"):
            score += 0.45
        elif lower.startswith("center:") or lower.startswith("coordinate:"):
            score += 0.2
        else:
            score += 0.25
    score = min(1.0, score)
    if score >= 0.85:
        return score, "strong"
    if score >= 0.55:
        return score, "medium"
    if score > 0:
        return score, "weak"
    return 0.0, "none"


def recovery_strategy_for(reason: str) -> list[str]:
    strategies = {
        FAILURE_TARGET_NOT_FOUND: ["retry_with_uia", "retry_with_ocr", "retry_with_vision", "ask_user_or_stop"],
        FAILURE_TARGET_AMBIGUOUS: ["resolve_by_role_text_region", "retry_with_uia", "ask_user_or_stop"],
        FAILURE_FOCUS_NOT_CONFIRMED: ["refocus_window", "retry_with_uia", "ask_user_or_stop"],
        FAILURE_NO_STATE_CHANGE: ["reparse_scene", "retry_with_ocr", "retry_with_vision", "ask_user_or_stop"],
        FAILURE_UNSAFE_COORDINATE: ["retry_with_node_target", "retry_with_uia", "ask_user_or_stop"],
        FAILURE_POLICY_BLOCKED: ["ask_user_or_stop"],
        FAILURE_TIMEOUT: ["wait_and_reparse", "refocus_window", "ask_user_or_stop"],
    }
    return strategies.get(reason, ["reparse_scene", "ask_user_or_stop"])


def finalize_contract(contract: ActionContract) -> ActionContract:
    score, grade = score_evidence(contract.evidence)
    contract.evidence_score = round(score, 3)
    contract.evidence_grade = grade
    if contract.failure_reason and not contract.recovery_strategy:
        contract.recovery_strategy = recovery_strategy_for(contract.failure_reason)
    return contract


def fail_contract(contract: ActionContract, reason: str, note: str) -> ActionContract:
    contract.status = "blocked" if contract.status == "planned" else contract.status
    contract.failure_reason = reason
    contract.recovery_strategy = recovery_strategy_for(reason)
    contract.notes.append(note)
    return finalize_contract(contract)


def node_evidence(node: ObservationNode, graph: ObservationGraph | None = None) -> list[str]:
    evidence = []
    label = node.display_label() or node.label
    if label:
        evidence.append(f"label:{label}")
    if node.semantic_role:
        evidence.append(f"role:{node.semantic_role}")
    if node.entity_type:
        evidence.append(f"entity:{node.entity_type}")
    if node.region:
        evidence.append(f"region:{node.region}")
    if node.affordances:
        evidence.append(f"affordances:{','.join(node.affordances)}")
    if node.visual_id:
        evidence.append(f"visual:{node.visual_id}")
    if node.center:
        evidence.append(f"center:{node.center.get('x', '')},{node.center.get('y', '')}")
    if node.label:
        evidence.append(f"ocr:{node.label}")
    if graph and graph.metadata.get("app_id"):
        evidence.append(f"app:{graph.metadata.get('app_id')}")
    if graph and graph.metadata.get("output_filename"):
        evidence.append(f"screenshot:{graph.metadata.get('output_filename')}")
    return evidence


def build_click_contract(
    step_id: str,
    intent: str,
    node: ObservationNode,
    graph: ObservationGraph | None,
    click_count: int,
    target_ranking: dict[str, Any] | None = None,
) -> ActionContract:
    label = node.display_label() or node.label or node.node_id
    center = node.center or {}
    box = node.box or {}
    affordances = {str(item).lower() for item in node.affordances}
    clickable_role = node.semantic_role in {"menu_item", "clickable_container", "list_row", "text_field"}
    clickable_type = node.node_type in {"button", "text_field", "container", "icon"}
    actionable_affordance = bool(affordances & {"click", "open", "focus", "navigate", "inspect"})
    has_center = center.get("x") is not None and center.get("y") is not None
    visible_box = int(box.get("width", 0) or 0) >= 0 and int(box.get("height", 0) or 0) >= 0

    contract = ActionContract(
        step_id=step_id,
        action_type="click_node",
        intent=intent,
        target=label,
        evidence=node_evidence(node, graph),
        target_identity=create_target_identity(node, graph).to_dict(),
        target_ranking=dict(target_ranking or {}),
        before_checks={
            "target_exists": bool(node.node_id or label),
            "target_visible": has_center and visible_box,
            "target_clickable": clickable_role or clickable_type or actionable_affordance,
            "safe_click_count": 1 <= int(click_count) <= 2,
        },
    )
    if target_ranking and bool(target_ranking.get("ambiguous")):
        contract.identity_ambiguous = True
        return fail_contract(contract, FAILURE_TARGET_AMBIGUOUS, "Target ranking score gap is too small for a safe click.")
    if graph and len(ambiguous_identity_matches(contract.target_identity, graph)) > 1:
        contract.identity_ambiguous = True
        return fail_contract(contract, FAILURE_TARGET_AMBIGUOUS, "Target identity is ambiguous in the current UI.")
    if not all(contract.before_checks.values()):
        return fail_contract(contract, FAILURE_TARGET_NOT_FOUND, "Pre-action contract checks failed.")
    return finalize_contract(contract)


def build_click_point_contract(
    step_id: str,
    intent: str,
    x: int,
    y: int,
    evidence: list[str],
    click_count: int = 1,
) -> ActionContract:
    contract = ActionContract(
        step_id=step_id,
        action_type="click_point",
        intent=intent,
        target=f"{x},{y}",
        evidence=evidence,
        before_checks={
            "coordinate_present": x >= 0 and y >= 0,
            "screenshot_evidence": any(item.startswith("screenshot:") for item in evidence),
            "region_evidence": any(item.startswith("region:") for item in evidence),
            "safe_coordinate_reason": any(item.startswith("reason:") for item in evidence),
            "safe_click_count": 1 <= int(click_count) <= 2,
        },
    )
    if not all(contract.before_checks.values()):
        return fail_contract(contract, FAILURE_UNSAFE_COORDINATE, "Point clicks require screenshot, region, and safe-coordinate reason evidence.")
    return finalize_contract(contract)


def build_dom_click_contract(
    step_id: str,
    intent: str,
    selectors: list[str],
    dom_available: bool,
) -> ActionContract:
    clean_selectors = [selector for selector in selectors if selector]
    contract = ActionContract(
        step_id=step_id,
        action_type="click_node",
        intent=intent,
        target=clean_selectors[0] if clean_selectors else "dom_selector",
        evidence=[f"dom_selector:{selector}" for selector in clean_selectors] + (["dom:available"] if dom_available else []),
        before_checks={
            "selectors_present": bool(clean_selectors),
            "dom_available": bool(dom_available),
        },
    )
    if not all(contract.before_checks.values()):
        return fail_contract(contract, FAILURE_TARGET_NOT_FOUND, "DOM click requires selector evidence and available browser DOM.")
    return finalize_contract(contract)


def build_type_text_contract(
    step_id: str,
    intent: str,
    text: str,
    selector: str = "",
    focused_target: str = "",
    dom_available: bool = False,
    active_window: str = "",
    expected_app: str = "",
    focus_confirmed: bool = False,
    active_focus_editable: bool | None = None,
    deterministic_focus: str = "",
) -> ActionContract:
    evidence = []
    if selector:
        evidence.append(f"dom_selector:{selector}")
    if focused_target:
        evidence.append(f"focused_target:{focused_target}")
    if dom_available:
        evidence.append("dom:available")
    if active_window:
        evidence.append(f"active_window:{active_window}")
    if expected_app:
        evidence.append(f"expected_app:{expected_app}")
    if deterministic_focus:
        evidence.append(f"deterministic_focus:{deterministic_focus}")
    editable_evidence = (
        bool(selector)
        or (active_focus_editable is True)
        or (bool(focused_target) and bool(focus_confirmed or deterministic_focus))
    )
    contract = ActionContract(
        step_id=step_id,
        action_type="type_text",
        intent=intent,
        target=selector or focused_target or "active_focus",
        evidence=evidence,
        before_checks={
            "text_present": bool(text),
            "active_window_matches": not expected_app or bool(focus_confirmed),
            "editable_target_exists": bool(editable_evidence or (dom_available and not expected_app)),
        },
    )
    if not all(contract.before_checks.values()):
        return fail_contract(contract, FAILURE_FOCUS_NOT_CONFIRMED, "Typing requires confirmed app focus and editable target evidence.")
    return finalize_contract(contract)


def build_press_key_contract(
    step_id: str,
    intent: str,
    keys: list[str],
    hotkey: bool,
    active_window: str = "",
    expected_change: str = "",
) -> ActionContract:
    normalized = [str(key).strip().lower() for key in keys if str(key).strip()]
    destructive_hotkeys = {
        ("alt", "f4"),
        ("ctrl", "w"),
        ("ctrl", "shift", "w"),
        ("ctrl", "q"),
        ("ctrl", "a"),
        ("delete",),
        ("shift", "delete"),
        ("backspace",),
    }
    key_tuple = tuple(normalized)
    evidence = [f"active_window:{active_window}"] if active_window else []
    if expected_change:
        evidence.append(f"expected_change:{expected_change}")
    contract = ActionContract(
        step_id=step_id,
        action_type="hotkey" if hotkey or len(normalized) > 1 else "press_key",
        intent=intent,
        target="+".join(normalized),
        evidence=evidence,
        before_checks={
            "keys_present": bool(normalized),
            "active_window_known": bool(active_window),
            "no_destructive_ambiguity": key_tuple not in destructive_hotkeys or bool(expected_change),
        },
    )
    if not all(contract.before_checks.values()):
        return fail_contract(contract, FAILURE_FOCUS_NOT_CONFIRMED, "Key press requires known focus and no destructive ambiguity.")
    return finalize_contract(contract)


def build_wait_contract(
    step_id: str,
    intent: str,
    seconds: float,
    expected_focus: str = "",
    timeout: float = 0.0,
) -> ActionContract:
    contract = ActionContract(
        step_id=step_id,
        action_type="wait_for",
        intent=intent,
        target=expected_focus or f"{seconds:.2f}s",
        evidence=[f"timeout:{timeout:.2f}"],
        before_checks={
            "bounded_wait": 0.0 <= seconds <= 30.0 and 0.0 <= timeout <= 120.0,
        },
    )
    if not all(contract.before_checks.values()):
        return fail_contract(contract, FAILURE_TIMEOUT, "Wait duration is outside contract bounds.")
    return finalize_contract(contract)


def scene_label_set(graph: ObservationGraph | None) -> set[str]:
    if not graph:
        return set()
    labels = set()
    for node in graph.flatten():
        label = str(node.display_label() or node.label or "").strip().lower()
        if label and not (label.startswith("[") and label.endswith("]")):
            labels.add(label)
    scene_summary = str(graph.metadata.get("scene_summary", "")).strip().lower()
    if scene_summary:
        labels.add(scene_summary)
    return labels


def verify_click_contract(
    contract: ActionContract,
    before: ObservationGraph | None,
    after: ObservationGraph | None,
    executor_ok: bool,
    target_node: ObservationNode,
) -> ActionContract:
    before_labels = scene_label_set(before)
    after_labels = scene_label_set(after)
    before_app = str(before.metadata.get("app_id", "") if before else "")
    after_app = str(after.metadata.get("app_id", "") if after else "")
    scene_changed = bool(before_labels != after_labels)
    app_maintained = not before_app or not after_app or before_app == after_app
    focus_candidate = target_node.semantic_role == "text_field" or target_node.entity_type in {"search_field", "omnibox"}
    identity_drifted = False
    if after and contract.target_identity:
        identity_drifted = detect_target_drift(contract.target_identity, target_node, before)
    verified = bool(executor_ok and after and app_maintained and (scene_changed or focus_candidate))
    if identity_drifted and not scene_changed:
        verified = False

    contract.during_checks["executor_accepted"] = bool(executor_ok)
    contract.identity_drifted = identity_drifted
    contract.verification = {
        "verified": verified,
        "scene_changed": scene_changed,
        "app_maintained": app_maintained,
        "focus_candidate": focus_candidate,
        "identity_drifted": identity_drifted,
        "before_label_count": len(before_labels),
        "after_label_count": len(after_labels),
        "screenshot_before": before.metadata.get("output_filename", "") if before else "",
        "screenshot_after": after.metadata.get("output_filename", "") if after else "",
    }
    contract.status = "verified" if verified else "failed"
    if not verified:
        reason = FAILURE_TARGET_AMBIGUOUS if contract.identity_ambiguous else (FAILURE_FOCUS_NOT_CONFIRMED if not app_maintained else FAILURE_NO_STATE_CHANGE)
        contract.failure_reason = reason
        contract.recovery_strategy = recovery_strategy_for(reason)
        contract.notes.append("Click did not produce a verified scene or focus signal.")
    return finalize_contract(contract)


def dom_text_blob(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return ""
    parts = [
        str(snapshot.get("title", "")),
        str(snapshot.get("url", "")),
        str(snapshot.get("focused_text", "")),
        str(snapshot.get("active_selector", "")),
        str(snapshot.get("stable_hash", "")),
    ]
    for item in snapshot.get("items", []):
        if isinstance(item, dict):
            parts.extend(
                [
                    str(item.get("text", "")),
                    str(item.get("aria_label", "")),
                    str(item.get("placeholder", "")),
                    str(item.get("value", "")),
                    str(item.get("selector", "")),
                    str(item.get("stable_hash", "")),
                    "focused" if item.get("focused") else "",
                ]
            )
    return " ".join(part.strip().lower() for part in parts if part.strip())


def verify_point_contract(
    contract: ActionContract,
    before: ObservationGraph | None,
    after: ObservationGraph | None,
    executor_ok: bool,
) -> ActionContract:
    before_labels = scene_label_set(before)
    after_labels = scene_label_set(after)
    scene_changed = bool(before_labels != after_labels)
    verified = bool(executor_ok and after and scene_changed)
    contract.during_checks["executor_accepted"] = bool(executor_ok)
    contract.verification = {
        "verified": verified,
        "scene_changed": scene_changed,
        "screenshot_before": before.metadata.get("output_filename", "") if before else "",
        "screenshot_after": after.metadata.get("output_filename", "") if after else "",
    }
    contract.status = "verified" if verified else "failed"
    if not verified:
        contract.failure_reason = FAILURE_NO_STATE_CHANGE
        contract.recovery_strategy = recovery_strategy_for(FAILURE_NO_STATE_CHANGE)
        contract.notes.append("Point click did not produce a verified scene change.")
    return finalize_contract(contract)


def verify_dom_click_contract(
    contract: ActionContract,
    chosen_selector: str,
    executor_ok: bool,
    dom_before: dict[str, Any] | None = None,
    dom_after: dict[str, Any] | None = None,
    after: ObservationGraph | None = None,
) -> ActionContract:
    dom_changed = bool(dom_after and dom_text_blob(dom_before) != dom_text_blob(dom_after))
    chosen_norm = str(chosen_selector or "").strip().lower()
    before_url = str((dom_before or {}).get("url", ""))
    after_url = str((dom_after or {}).get("url", ""))
    before_title = str((dom_before or {}).get("title", ""))
    after_title = str((dom_after or {}).get("title", ""))
    url_or_title_changed = bool(dom_after and (before_url != after_url or before_title != after_title))
    focus_confirmed = False
    if chosen_norm and isinstance(dom_after, dict):
        active_selector = str(dom_after.get("active_selector", "")).strip().lower()
        focus_confirmed = active_selector == chosen_norm
        for item in dom_after.get("items", []):
            if isinstance(item, dict) and item.get("focused"):
                selectors = [str(item.get("selector", ""))] + [str(selector) for selector in item.get("selectors", [])]
                if chosen_norm in {selector.strip().lower() for selector in selectors if selector.strip()}:
                    focus_confirmed = True
                    break
    observed_after = bool(after)
    verified = bool(executor_ok and chosen_selector and (dom_changed or url_or_title_changed or focus_confirmed or observed_after))
    contract.during_checks["executor_accepted"] = bool(executor_ok)
    contract.during_checks["selector_chosen"] = bool(chosen_selector)
    contract.verification = {
        "verified": verified,
        "chosen_selector": chosen_selector,
        "dom_changed": dom_changed,
        "url_or_title_changed": url_or_title_changed,
        "focus_confirmed": focus_confirmed,
        "observed_after": observed_after,
        "screenshot_after": after.metadata.get("output_filename", "") if after else "",
    }
    contract.status = "verified" if verified else "failed"
    if not verified:
        contract.failure_reason = FAILURE_TARGET_NOT_FOUND if not chosen_selector else FAILURE_NO_STATE_CHANGE
        contract.recovery_strategy = recovery_strategy_for(contract.failure_reason)
        contract.notes.append("DOM click did not produce a verified DOM or observed UI signal.")
    return finalize_contract(contract)


def verify_type_text_contract(
    contract: ActionContract,
    text: str,
    executor_ok: bool,
    dom_before: dict[str, Any] | None = None,
    dom_after: dict[str, Any] | None = None,
    after: ObservationGraph | None = None,
) -> ActionContract:
    typed = text.strip().lower()
    before_blob = dom_text_blob(dom_before)
    after_blob = dom_text_blob(dom_after)
    graph_blob = " ".join(scene_label_set(after))
    target_selector = str(contract.target or "").strip().lower()
    selector_value_verified = False
    if typed and target_selector and isinstance(dom_after, dict):
        for item in dom_after.get("items", []):
            if not isinstance(item, dict):
                continue
            selectors = [str(item.get("selector", ""))] + [str(selector) for selector in item.get("selectors", [])]
            selector_match = target_selector in {selector.strip().lower() for selector in selectors if selector.strip()}
            value = str(item.get("value") or item.get("text") or "").strip().lower()
            if selector_match and typed in value:
                selector_value_verified = True
                break
    value_appeared = bool(typed and (selector_value_verified or typed in after_blob or typed in graph_blob))
    content_changed = bool(after_blob and before_blob != after_blob)
    verified = bool(executor_ok and (value_appeared or content_changed))
    contract.during_checks["executor_accepted"] = bool(executor_ok)
    contract.verification = {
        "verified": verified,
        "typed_value_appeared": value_appeared,
        "selector_value_verified": selector_value_verified,
        "field_content_changed": content_changed,
        "screenshot_after": after.metadata.get("output_filename", "") if after else "",
    }
    contract.status = "verified" if verified else "failed"
    if not verified:
        contract.failure_reason = FAILURE_FOCUS_NOT_CONFIRMED if executor_ok else FAILURE_NO_STATE_CHANGE
        contract.recovery_strategy = recovery_strategy_for(contract.failure_reason)
        contract.notes.append("Typed value was not verified in DOM or observed UI state.")
    return finalize_contract(contract)


def verify_press_key_contract(
    contract: ActionContract,
    executor_ok: bool,
    dom_before: dict[str, Any] | None = None,
    dom_after: dict[str, Any] | None = None,
    before: ObservationGraph | None = None,
    after: ObservationGraph | None = None,
) -> ActionContract:
    dom_changed = bool(dom_after and dom_text_blob(dom_before) != dom_text_blob(dom_after))
    scene_changed = bool(after and scene_label_set(before) != scene_label_set(after))
    expected_change = any(str(item).startswith("expected_change:") for item in contract.evidence)
    focus_confirmed = bool(expected_change and isinstance(dom_after, dict) and str(dom_after.get("active_selector", "") or dom_after.get("focused_text", "")).strip())
    no_destructive_ambiguity = bool(contract.before_checks.get("no_destructive_ambiguity"))
    verified = bool(executor_ok and no_destructive_ambiguity and (dom_changed or scene_changed or focus_confirmed))
    contract.during_checks["executor_accepted"] = bool(executor_ok)
    contract.verification = {
        "verified": verified,
        "dom_changed": dom_changed,
        "scene_changed": scene_changed,
        "focus_confirmed": focus_confirmed,
        "no_destructive_ambiguity": no_destructive_ambiguity,
        "screenshot_before": before.metadata.get("output_filename", "") if before else "",
        "screenshot_after": after.metadata.get("output_filename", "") if after else "",
    }
    contract.status = "verified" if verified else "failed"
    if not verified:
        contract.failure_reason = FAILURE_NO_STATE_CHANGE
        contract.recovery_strategy = recovery_strategy_for(FAILURE_NO_STATE_CHANGE)
        contract.notes.append("Key press was not verified by state change or safety classification.")
    return finalize_contract(contract)


def verify_wait_contract(contract: ActionContract, executor_ok: bool) -> ActionContract:
    contract.during_checks["executor_accepted"] = bool(executor_ok)
    contract.verification = {"verified": bool(executor_ok), "traceable_wait": True}
    contract.status = "verified" if executor_ok else "failed"
    if not executor_ok:
        contract.failure_reason = FAILURE_TIMEOUT
        contract.recovery_strategy = recovery_strategy_for(FAILURE_TIMEOUT)
        contract.notes.append("Wait condition timed out.")
    return finalize_contract(contract)
