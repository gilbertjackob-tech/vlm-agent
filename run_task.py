import argparse
import os
import threading

# Force stable Paddle engine on Windows
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from copilot.runtime.engine import CopilotEngine
from copilot.schemas import TrustMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Developer harness for the Windows Copilot Robot runtime.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Open explorer and parse screen",
        help="Natural-language task prompt to execute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile the plan and trace it without taking actions.",
    )
    parser.add_argument(
        "--trust-mode",
        choices=[mode.value for mode in TrustMode],
        default=TrustMode.PLAN_AND_RISK_GATES.value,
        help="Runtime trust mode.",
    )
    parser.add_argument(
        "--save-skill",
        action="store_true",
        help="Save the compiled plan as a reusable workflow after the run.",
    )
    parser.add_argument(
        "--taskbar",
        action="store_true",
        help="Show the always-on-top Agent Live Feed bar while the task runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.taskbar:
        return run_with_taskbar(args)

    engine = CopilotEngine()
    trust_mode = TrustMode(args.trust_mode)

    print("=" * 60)
    print("WINDOWS COPILOT ROBOT DEV HARNESS")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Trust mode: {trust_mode.value}")
    print(f"Dry run: {args.dry_run}")
    print()

    plan = engine.plan_prompt(args.prompt, trust_mode=trust_mode)
    print(f"Plan source: {plan.source}")
    print(f"Summary: {plan.summary}")
    print(f"Required apps: {', '.join(plan.required_apps) or 'None'}")
    print("Steps:")
    for step in plan.steps:
        target = step.target.value if step.target else ""
        print(
            f"  - {step.step_id}: {step.title} | action={step.action_type} "
            f"| target={target} | risk={step.risk_level.value} | approval={step.requires_approval}"
        )
    print()

    def trace_callback(event: dict) -> None:
        phase = event.get("phase", "runtime")
        message = event.get("message", "")
        metadata = event.get("metadata", {})
        if metadata:
            print(f"[{phase}] {message} | {metadata}")
        else:
            print(f"[{phase}] {message}")

    trace = engine.execute_prompt(
        args.prompt,
        trust_mode=trust_mode,
        trace_callback=trace_callback,
        dry_run=args.dry_run,
    )

    print()
    print(f"Run status: {trace.status.value}")
    print(f"Trace path: {trace.outputs.get('trace_path', '')}")
    print(f"Memory summary: {engine.get_memory_summary()}")

    if args.save_skill:
        workflow = engine.save_current_plan_as_skill(args.prompt)
        if workflow:
            print(f"Saved workflow: {workflow.get('workflow_id', '')} ({workflow.get('name', '')})")


def run_with_taskbar(args: argparse.Namespace) -> None:
    from PySide6.QtCore import QObject, QTimer, Signal
    from PySide6.QtWidgets import QApplication

    from copilot.core import event_bus
    from copilot.ui.taskbar_bar import TaskbarBar, attach_event_bus

    class CompletionSignal(QObject):
        finished = Signal()

    app = QApplication.instance() or QApplication([])
    bar = TaskbarBar()
    attach_event_bus(bar)
    bar.push("Agent live feed ready.")
    bar.show()
    completion = CompletionSignal()
    completion.finished.connect(lambda: QTimer.singleShot(4000, app.quit))

    def worker() -> None:
        try:
            engine = CopilotEngine()
            trust_mode = TrustMode(args.trust_mode)
            event_bus.emit(
                {
                    "type": "status",
                    "msg": f"Starting task: {args.prompt}",
                    "phase": "status",
                    "message": f"Starting task: {args.prompt}",
                    "metadata": {},
                }
            )
            trace = engine.execute_prompt(
                args.prompt,
                trust_mode=trust_mode,
                trace_callback=None,
                dry_run=args.dry_run,
            )
            event_bus.emit(
                {
                    "type": "status",
                    "msg": f"Run completed with status: {trace.status.value}",
                    "phase": "done",
                    "message": f"Run completed with status: {trace.status.value}",
                    "metadata": {"status": trace.status.value, "trace_path": trace.outputs.get("trace_path", "")},
                }
            )
        except Exception as exc:
            event_bus.emit(
                {
                    "type": "status",
                    "msg": f"Run failed: {exc}",
                    "phase": "error",
                    "message": f"Run failed: {exc}",
                    "metadata": {"status": "failed"},
                }
            )
        finally:
            completion.finished.emit()

    threading.Thread(target=worker, name="copilot-taskbar-run", daemon=True).start()
    app.exec()


if __name__ == "__main__":
    main()
