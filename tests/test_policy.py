from __future__ import annotations

import tempfile
import unittest

from copilot.memory.store import MemoryStore
from copilot.runtime.policy import PolicyEngine
from copilot.schemas import ActionTarget, PlanStep, RiskLevel, TaskSpec, TrustMode


class PolicyEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.memory = MemoryStore(base_dir=self.tmp.name)
        self.policy = PolicyEngine(self.memory)

    def test_blocks_raw_point_clicks(self) -> None:
        decision = self.policy.evaluate_step(
            TaskSpec(prompt="click point", goal="click point"),
            PlanStep(step_id="1", title="Click raw point", action_type="click_point"),
        )
        self.assertFalse(decision.allowed)

    def test_blocks_exploration_for_untrusted_app(self) -> None:
        decision = self.policy.evaluate_step(
            TaskSpec(prompt="explore app", goal="explore app"),
            PlanStep(
                step_id="1",
                title="Explore unsafe app",
                action_type="explore_safe",
                target=ActionTarget(kind="application", value="unknown_app"),
                parameters={"app_id": "unknown_app"},
            ),
        )
        self.assertFalse(decision.allowed)

    def test_allows_hover_learning_for_unknown_current_window(self) -> None:
        decision = self.policy.evaluate_step(
            TaskSpec(prompt="learn current UI", goal="learn current UI"),
            PlanStep(
                step_id="1",
                title="Learn current UI",
                action_type="learning_session",
                target=ActionTarget(kind="application", value="current_window"),
                parameters={"app_id": "current_window"},
            ),
        )
        self.assertTrue(decision.allowed)

    def test_allows_interaction_learning_for_current_window_with_runtime_filters(self) -> None:
        decision = self.policy.evaluate_step(
            TaskSpec(prompt="learn clicks", goal="learn clicks"),
            PlanStep(
                step_id="1",
                title="Learn click outcomes",
                action_type="interaction_learning",
                target=ActionTarget(kind="application", value="current_window"),
                parameters={"app_id": "current_window"},
            ),
        )
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_approval)

    def test_blocks_dangerous_typed_actions(self) -> None:
        decision = self.policy.evaluate_step(
            TaskSpec(prompt="submit payment", goal="submit payment"),
            PlanStep(
                step_id="1",
                title="Type payment form",
                action_type="type_text",
                target=ActionTarget(kind="text", value="submit payment"),
                parameters={"text": "submit payment"},
                risk_level=RiskLevel.HIGH,
            ),
        )
        self.assertFalse(decision.allowed)

    def test_mostly_autonomous_trusted_route_skips_approval(self) -> None:
        decision = self.policy.evaluate_step(
            TaskSpec(prompt="open chrome", goal="open chrome", trust_mode=TrustMode.MOSTLY_AUTONOMOUS),
            PlanStep(
                step_id="1",
                title="Route to chrome",
                action_type="route_window",
                target=ActionTarget(kind="application", value="chrome"),
                parameters={"app_id": "chrome"},
                risk_level=RiskLevel.LOW,
            ),
        )
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_approval)

    def test_allows_trusted_downloads_shortcut_without_dangerous_download_block(self) -> None:
        decision = self.policy.evaluate_step(
            TaskSpec(prompt="open chrome downloads", goal="open chrome downloads"),
            PlanStep(
                step_id="1",
                title="Open Chrome downloads",
                action_type="press_key",
                target=ActionTarget(kind="keyboard", value="ctrl+j"),
                parameters={"keys": ["ctrl", "j"], "hotkey": True, "shortcut_id": "downloads"},
                risk_level=RiskLevel.LOW,
            ),
        )
        self.assertTrue(decision.allowed)

    def test_destructive_shortcut_requires_approval_but_is_not_hard_blocked(self) -> None:
        decision = self.policy.evaluate_step(
            TaskSpec(prompt="delete selected file", goal="delete selected file"),
            PlanStep(
                step_id="1",
                title="Delete Explorer selection",
                action_type="press_key",
                target=ActionTarget(kind="keyboard", value="delete"),
                parameters={
                    "keys": ["delete"],
                    "shortcut_id": "delete",
                    "destructive": True,
                    "requires_approval": True,
                    "allow_destructive_shortcut": True,
                },
                risk_level=RiskLevel.HIGH,
                requires_approval=True,
            ),
        )
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_approval)

    def test_high_risk_delete_requires_approval(self) -> None:
        decision = self.policy.evaluate_step(
            TaskSpec(prompt="delete selected file", goal="delete selected file"),
            PlanStep(
                step_id="1",
                title="Delete file",
                action_type="click_node",
                target=ActionTarget(kind="button", value="Delete"),
                parameters={"app_id": "explorer"},
            ),
        )

        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_approval)
        self.assertEqual(decision.risk_level, RiskLevel.HIGH)

    def test_always_allow_app_policy_skips_future_high_risk_approval(self) -> None:
        self.memory.allow_high_risk_for_app("explorer")

        decision = self.policy.evaluate_step(
            TaskSpec(prompt="delete selected file", goal="delete selected file"),
            PlanStep(
                step_id="1",
                title="Delete file",
                action_type="click_node",
                target=ActionTarget(kind="button", value="Delete"),
                parameters={"app_id": "explorer"},
            ),
        )

        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_approval)

    def test_medium_risk_send_message_is_risk_gated(self) -> None:
        decision = self.policy.evaluate_step(
            TaskSpec(prompt="send message", goal="send message"),
            PlanStep(
                step_id="1",
                title="Send message",
                action_type="send_message",
                target=ActionTarget(kind="button", value="Send"),
                parameters={"app_id": "chrome"},
            ),
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.risk_level, RiskLevel.MEDIUM)


if __name__ == "__main__":
    unittest.main()
