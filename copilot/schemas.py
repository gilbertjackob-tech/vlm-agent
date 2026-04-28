from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


class TrustMode(str, Enum):
    PLAN_AND_RISK_GATES = "plan_and_risk_gates"
    ALWAYS_CONFIRM = "always_confirm"
    MOSTLY_AUTONOMOUS = "mostly_autonomous"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ControlMode(str, Enum):
    NATIVE = "native"
    BROWSER_DOM = "browser_dom"
    HYBRID = "hybrid"
    VISION_FALLBACK = "vision_fallback"
    LEGACY = "legacy"


class RunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class MissionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    SUCCESS = "success"


@dataclass
class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskSpec(Serializable):
    prompt: str
    goal: str
    constraints: list[str] = field(default_factory=list)
    trust_mode: TrustMode = TrustMode.PLAN_AND_RISK_GATES
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    created_at: float = field(default_factory=time.time)


@dataclass
class ActionTarget(Serializable):
    kind: str
    value: str = ""
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionIntent(Serializable):
    verb: str
    semantic_target: str = ""
    description: str = ""
    preferred_mode: ControlMode = ControlMode.HYBRID
    fallback_modes: list[ControlMode] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)


@dataclass
class PlanStep(Serializable):
    step_id: str
    title: str
    action_type: str
    target: ActionTarget | None = None
    intent: ActionIntent | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    risk_level: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False
    success_criteria: str = ""
    fallback_hint: str = ""
    control_modes: list[ControlMode] = field(default_factory=list)


@dataclass
class ExecutionPlan(Serializable):
    task: TaskSpec
    steps: list[PlanStep]
    source: str = "heuristic_planner"
    summary: str = ""
    required_apps: list[str] = field(default_factory=list)
    success_conditions: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)


@dataclass
class PolicyDecision(Serializable):
    allowed: bool
    requires_approval: bool
    reason: str
    risk_level: RiskLevel


@dataclass
class ObservationNode(Serializable):
    node_id: str
    label: str
    learned_label: str = ""
    learned_concepts: list[str] = field(default_factory=list)
    node_type: str = "unknown"
    semantic_role: str = ""
    entity_type: str = ""
    app_id: str = ""
    region: str = ""
    state_tags: list[str] = field(default_factory=list)
    affordances: list[str] = field(default_factory=list)
    stability: float = 0.0
    source_frame_id: str = ""
    box: dict[str, int] = field(default_factory=dict)
    center: dict[str, int] = field(default_factory=dict)
    visual_id: str = ""
    visual_ids: list[str] = field(default_factory=list)
    tag: str = ""
    role: str = ""
    text: str = ""
    aria_label: str = ""
    accessible_name: str = ""
    placeholder: str = ""
    selector: str = ""
    frame_path: str = ""
    shadow_path: str = ""
    rect: dict[str, int] = field(default_factory=dict)
    visible: bool = True
    enabled: bool = True
    stable_hash: str = ""
    children: list["ObservationNode"] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "ObservationNode":
        return cls(
            node_id=str(raw.get("id", "")),
            label=str(raw.get("label", "")),
            learned_label=str(raw.get("learned_label", "")),
            learned_concepts=list(raw.get("learned_concepts", [])),
            node_type=str(raw.get("type", "unknown")),
            semantic_role=str(raw.get("semantic_role", "")),
            entity_type=str(raw.get("entity_type", "")),
            app_id=str(raw.get("app_id", "")),
            region=str(raw.get("region", "")),
            state_tags=list(raw.get("state_tags", [])),
            affordances=list(raw.get("affordances", [])),
            stability=float(raw.get("stability", 0.0) or 0.0),
            source_frame_id=str(raw.get("source_frame_id", "")),
            box=dict(raw.get("box", {})),
            center=dict(raw.get("center", {})),
            visual_id=str(raw.get("visual_id", "")),
            visual_ids=[str(item) for item in raw.get("visual_ids", []) if str(item)],
            tag=str(raw.get("tag", "")),
            role=str(raw.get("role", "")),
            text=str(raw.get("text", "")),
            aria_label=str(raw.get("aria_label", "")),
            accessible_name=str(raw.get("accessible_name", "")),
            placeholder=str(raw.get("placeholder", "")),
            selector=str(raw.get("selector", "")),
            frame_path=str(raw.get("frame_path", "")),
            shadow_path=str(raw.get("shadow_path", "")),
            rect=dict(raw.get("rect", {})),
            visible=bool(raw.get("visible", True)),
            enabled=bool(raw.get("enabled", True)),
            stable_hash=str(raw.get("stable_hash", "")),
            children=[cls.from_raw(child) for child in raw.get("children", [])],
        )

    def display_label(self) -> str:
        return self.learned_label or self.label

    def flatten(self) -> list["ObservationNode"]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.flatten())
        return nodes


@dataclass
class ObservationGraph(Serializable):
    nodes: list[ObservationNode]
    source: str = "vision"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw_nodes: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> "ObservationGraph":
        return cls(
            nodes=[ObservationNode.from_raw(node) for node in raw_nodes],
            metadata=metadata or {},
        )

    def flatten(self) -> list[ObservationNode]:
        items: list[ObservationNode] = []
        for node in self.nodes:
            items.extend(node.flatten())
        return items


@dataclass
class SceneDelta(Serializable):
    step_id: str = ""
    expected_change: str = ""
    before_summary: str = ""
    after_summary: str = ""
    actual_change: str = ""
    changed: bool = False
    maintained_app_id: bool = True
    added_labels: list[str] = field(default_factory=list)
    removed_labels: list[str] = field(default_factory=list)
    recovery_hint: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionOutcome(Serializable):
    step_id: str
    action_type: str
    ok: bool
    control_mode: str = ""
    target_label: str = ""
    target_node_id: str = ""
    focus_confirmed: bool = False
    notes: str = ""
    scene_delta: SceneDelta | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class MissionCheckpoint(Serializable):
    checkpoint_id: str
    title: str
    expected_scene: str = ""
    completion_rule: str = ""
    recovery_rule: str = ""
    status: str = "pending"
    notes: list[str] = field(default_factory=list)


@dataclass
class MissionState(Serializable):
    mission_id: str
    goal: str
    subgoal: str = ""
    app_id: str = ""
    status: MissionStatus = MissionStatus.PLANNED
    current_checkpoint_id: str = ""
    checkpoints: list[MissionCheckpoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class TraceEvent(Serializable):
    phase: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunTrace(Serializable):
    run_id: str
    task: TaskSpec
    plan: ExecutionPlan
    status: RunStatus = RunStatus.PLANNED
    events: list[TraceEvent] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    memory_hits: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    mission: MissionState | None = None
    scene_deltas: list[SceneDelta] = field(default_factory=list)
    action_outcomes: list[ActionOutcome] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def add_event(self, phase: str, message: str, **metadata: Any) -> None:
        self.events.append(TraceEvent(phase=phase, message=message, metadata=metadata))


@dataclass
class SkillDefinition(Serializable):
    skill_id: str
    name: str
    prompt_pattern: str
    steps: list[dict[str, Any]]
    required_apps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    promotion_state: str = "candidate"
    success_count: int = 0
    failure_count: int = 0


@dataclass
class MemoryRecord(Serializable):
    record_id: str
    kind: str
    confidence: float = 0.0
    provenance: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


@dataclass
class EntityMemoryRecord(MemoryRecord):
    entity_key: str = ""
    display_name: str = ""
    labels: dict[str, int] = field(default_factory=dict)
    visual_ids: dict[str, int] = field(default_factory=dict)
    process_name: str = ""
    window_titles: list[str] = field(default_factory=list)
    risk_level: str = ""


@dataclass
class VisualMemoryRecord(MemoryRecord):
    visual_id: str = ""
    label_counts: dict[str, int] = field(default_factory=dict)
    concept_counts: dict[str, int] = field(default_factory=dict)
    seen_count: int = 0


@dataclass
class ControlMemoryRecord(MemoryRecord):
    control_key: str = ""
    labels: dict[str, int] = field(default_factory=dict)
    concepts: dict[str, int] = field(default_factory=dict)
    affordances: dict[str, int] = field(default_factory=dict)
    node_type: str = ""
    semantic_role: str = ""
    entity_type: str = ""
    region: str = ""
    app_id: str = ""
    seen_count: int = 0
    risk_level: str = ""


@dataclass
class WorkflowMemoryRecord(MemoryRecord):
    workflow_id: str = ""
    name: str = ""
    prompt_pattern: str = ""
    normalized_prompt: str = ""
    plan_signature: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    required_apps: list[str] = field(default_factory=list)
    promotion_state: str = "candidate"
    success_count: int = 0
    failure_count: int = 0


@dataclass
class TransitionMemoryRecord(MemoryRecord):
    action_type: str = ""
    source_visual_id: str = ""
    target_visual_id: str = ""
    target_label: str = ""
    outcome: str = ""
    seen_count: int = 0
