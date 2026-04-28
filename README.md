# VLM Agent

VLM Agent is a verified Windows desktop operator. It turns natural-language goals into planned desktop actions, but the runtime is designed around deterministic perception first: Browser DOM for Chrome/Edge, Windows UI Automation for native apps, and OCR/vision only as fallback.

Core rule:

```text
No action without evidence. No success without verification.
```

The current runtime line is **v4.1 deterministic-speed**. It prioritizes low-latency DOM/UIA perception, persistent state comparison, direct action verification, bounded recovery, and benchmark evidence.

## Quick Start

Install dependencies:

```powershell
pip install -r req.txt
```

Run a single task from the CLI:

```powershell
python .\run_task.py "Open explorer and parse screen"
```

Run a task with the always-on-top Agent Live Feed bar:

```powershell
python .\run_task.py "Open explorer and parse screen" --taskbar
```

Compile and trace a task without taking desktop actions:

```powershell
python .\run_task.py "Open chrome and search youtube for lo-fi coding mix" --dry-run
```

Open the local GUI panel:

```powershell
python .\main.py
```

Run the local daemon:

```powershell
python .\run_daemon.py --host 127.0.0.1 --port 8765
```

Follow a daemon run with the overlay:

```powershell
python .\run_overlay.py --base-url http://127.0.0.1:8765 --run-id <run_id>
```

Run the benchmark in dry-run mode:

```powershell
python .\run_benchmark.py
```

## Running Styles

### CLI Task Runner

Use `run_task.py` for direct development runs:

```powershell
python .\run_task.py "Open explorer and open downloads"
python .\run_task.py "Open chrome and search for python documentation" --trust-mode plan_and_risk_gates
python .\run_task.py "Open explorer and parse screen" --dry-run
python .\run_task.py "Open chrome and search for OpenAI docs" --taskbar
```

Useful flags:

- `--dry-run`: compile and trace the plan without executing desktop actions.
- `--trust-mode`: choose one of the runtime trust modes from `copilot.schemas.TrustMode`.
- `--save-skill`: save the compiled plan as a reusable workflow after the run.
- `--taskbar`: show the always-on-top Agent Live Feed bar while the run emits live status events.

### Agent Live Feed Bar

Use the taskbar feed when you want a small always-on-top mission-control view of runtime events:

```powershell
python .\run_task.py "Open explorer and parse screen" --taskbar
```

The feed is implemented in `copilot/ui/taskbar_bar.py`, uses PySide6, and subscribes to the process-local event bus in `copilot/core/event_bus.py`. It is an always-on-top, frameless, taskbar-like control surface for live autonomous execution.

Structured feed events use this shape:

```python
event_bus.emit({
    "type": "action",          # thinking | seeing | action | verify | error | status
    "msg": "Typing query...",
    "confidence": 0.91,
    "step": 3,
    "step_total": 7,
    "metadata": {"app_id": "chrome", "risk_level": "low"},
})
```

The bar renders:

```text
Chrome | Step 3/7 | Confidence 91% | SAFE
Thinking: Found search box
Action: Typing query...
Verify: focus confirmed
```

Operational behavior:

- Colored event types: action/cyan, verify/green, error/red, thinking/yellow, seeing/purple.
- 150 ms UI debounce keeps high-frequency runtime events readable.
- `STOP` requests cancellation through the global event bus flag; the runtime checks it in the normal cancellation path.
- The agent runs in a worker thread when `--taskbar` is used, so the PySide UI remains responsive.
- The last 200 feed events are persisted to `logs/live_feed.log` for debugging and replay analysis.

### GUI Panel

Use `main.py` to launch the local panel:

```powershell
python .\main.py
```

The panel is useful for observing parsed UI nodes, teaching labels/concepts, and running tasks with visible feedback.

### Local Daemon

Use `run_daemon.py` when building external shells, UI clients, overlays, or integration tests:

```powershell
python .\run_daemon.py --host 127.0.0.1 --port 8765
```

Core daemon endpoints:

```text
POST /tasks
GET /runs
GET /runs/{id}
GET /runs/{id}/stream
POST /runs/{id}/cancel
GET /approvals?pending=true
POST /approvals/{id}
GET /skills
POST /skills
POST /benchmarks/run
GET /runs/{id}/xray
```

The daemon exposes run lifecycle, current step, confidence, approvals, cancellation state, benchmark runs, skill management, and X-Ray node overlays.

### Overlay Follower

Use `run_overlay.py` to follow a daemon run:

```powershell
python .\run_overlay.py --base-url http://127.0.0.1:8765 --run-id <run_id>
```

The overlay reads daemon state and can show current run status, detected nodes, confidence, and X-Ray boxes.

### Benchmarks

Dry-run benchmark:

```powershell
python .\run_benchmark.py
```

List benchmark missions:

```powershell
python .\run_benchmark.py --list
```

Validate benchmark design coverage without running tasks:

```powershell
python .\run_benchmark.py --design-only --output-dir benchmark_runs\live_design_readme_check
```

Small live smoke run:

```powershell
python .\run_benchmark.py --live --repeat 1 --auto-approve --max-missions 3 --output-dir benchmark_runs\live_probe_v41
```

Larger live run:

```powershell
python .\run_benchmark.py --live --repeat 1 --auto-approve --max-missions 10 --output-dir benchmark_runs\live_10_v41
```

Full live benchmark:

```powershell
python .\run_benchmark.py --live --repeat 5 --auto-approve --output-dir benchmark_runs\live_full_v41
```

### Voice Narration

Benchmark narration can be disabled, printed to console, or sent to TTS:

```powershell
python .\run_benchmark.py --live --voice off
python .\run_benchmark.py --live --voice console
python .\run_benchmark.py --live --voice tts
```

TTS configuration uses LocalAI-compatible speech settings:

```env
COPILOT_VOICE_MODE=tts
COPILOT_TTS_PROVIDER=localai
COPILOT_TTS_MODEL=voice-en-us-ryan-high
COPILOT_TTS_ENDPOINT=http://localhost:8080/v1/audio/speech
COPILOT_TTS_VOICE=voice-en-us-ryan-high
COPILOT_TTS_TIMEOUT=15
COPILOT_TTS_SPEAK_SENSITIVE=false
```

Narration is output-only. It never exposes hidden reasoning, and sensitive typed text is not spoken by default.

## v4.1 Deterministic-Speed Runtime

The v4.1 runtime is built to reduce latency and wrong actions by avoiding image/OCR perception when deterministic state is available.

### Chrome CDP Is Automatic

For Chrome/Edge tasks, the runtime:

```text
try existing Chrome DevTools Protocol endpoint
  -> if unavailable, auto-launch debug Chrome
  -> if still unavailable, fail safely
```

This is not hidden behind an opt-in environment flag. Browser tasks are hard-locked to DOM. If DOM cannot be obtained, the runtime raises `BrowserDOMUnavailable` instead of falling back to visual guessing.

### Browser DOM Snapshot

The browser adapter snapshots:

- main document
- same-origin iframes
- open shadow roots
- interactive elements and accessibility labels

DOM nodes include:

```text
tag
role
text
aria_label
accessible_name
placeholder
selector
frame_path
shadow_path
rect
visible
enabled
stable_hash
```

`stable_hash` is derived from deterministic DOM identity fields, including selector plus frame/shadow path context.

### Persistent Desktop State

The persistent state layer lives under `copilot/state/desktop_state.py` and records:

```text
active_app
active_window
dom_snapshot
uia_snapshot
last_action
last_verified_change
state_hash
```

The engine compares the current probed state with the previous state before parsing. If the state is unchanged, it reuses the last observation instead of reparsing.

The latest state is persisted to:

```text
memory/desktop_state.json
```

### Windows UIA

Windows UI Automation is the deterministic source for native Windows apps. The adapter collects UIA elements from the active window handle when possible instead of scanning the entire UIA root tree.

UIA elements include:

```text
name
automation_id
control_type
rectangle
click_point
enabled
visible
parent_window
stable_hash
```

The bridge preserves `clickable_point` as a compatibility alias, but `click_point` is the preferred deterministic center point.

### Direct Verification

The runtime avoids full post-action parses when direct verification is enough:

- DOM click: verify focus, URL, title, or DOM change.
- DOM type: verify target selector value/text.
- Browser Enter/hotkey: verify URL, title, focused element, or result DOM change.
- UIA click: verify focus, control, active window, or UIA snapshot delta.

Full parsing remains available when a task genuinely needs a fresh UI graph.

## Architecture

High-level loop:

```text
User Goal
  -> Prompt Compiler / Planner
  -> Perception Bridge
  -> Browser and Windows Adapters
  -> Desktop State Layer
  -> UI Graph / Target Resolver
  -> Policy and Risk Gate
  -> Action Contract
  -> Action Executor
  -> Direct Verifier
  -> Repair / Recovery Planner
  -> Memory and Trace Outputs
```

Primary modules:

- `copilot/planner/`: compiles user prompts into structured steps.
- `copilot/core/`: process-local runtime utilities, including the structured live event bus.
- `copilot/perception/`: builds observations from DOM, UIA, OCR, and legacy vision.
- `copilot/adapters/`: browser and Windows integration boundaries.
- `copilot/state/`: persistent desktop and DOM identity state.
- `copilot/runtime/`: contracts, execution, verification, recovery, task state, daemon.
- `copilot/memory/`: semantic memory, episodic traces, workflows, policies.
- `copilot/benchmark/`: mission catalog, benchmark runner, report metrics.
- `copilot/ui/`: local panel, overlay, and Agent Live Feed taskbar bar.

Root-level legacy files are isolated behind compatibility factories:

- `copilot/perception/legacy_agent.py`
- `copilot/planner/legacy.py`

New product code should live under `copilot/`.

## Evidence, State, and Verification

Target resolution prefers deterministic evidence before visual guesses:

```text
1. Browser DOM
2. Windows UIA / accessibility metadata
3. OCR text
4. Vision layout/object detection
5. CLIP/VLM semantic fallback
```

Browser DOM and Windows UIA are the primary sources for exact interaction. OCR and vision are fallback paths for unknown or unsupported screens.

Every runtime action should produce an action contract:

```text
intent
  -> target
  -> evidence
  -> before checks
  -> execution result
  -> verification
```

Contracted primitives include:

- `click_node`
- `click_point`
- `type_text`
- `press_key`
- `hotkey`
- `wait_for`
- DOM selector click

Failure reasons are normalized:

```text
TARGET_NOT_FOUND
TARGET_AMBIGUOUS
FOCUS_NOT_CONFIRMED
NO_STATE_CHANGE
UNSAFE_COORDINATE
POLICY_BLOCKED
TIMEOUT
```

Failed actions are not treated as success. They are classified and routed into bounded repair or recovery.

Important trace outputs:

```text
trace.outputs["action_contracts"]
trace.outputs["failure_recovery"]
trace.outputs["recovery_attempts"]
trace.outputs["state_snapshots"]
trace.outputs["state_diffs"]
trace.outputs["task_state_timeline"]
trace.outputs["plan_replacements"]
trace.outputs["parse_health"]
trace.outputs["voice_events"]
```

## Benchmarking

The default benchmark catalog contains 100 missions:

```text
30 Chrome / browser DOM missions
30 Explorer / Windows UIA missions
20 forms and input missions
10 recovery missions
10 safety-block missions
```

Benchmark metrics include:

```text
success_rate
stable_success_rate
wrong_action_count
wrong_click_count
verification_failure_rate
verification_failure_count
recovery_count
recovery_success_rate
repair_vs_replan_count
avg_steps_per_task
avg_latency_per_step
latency_per_step
dom_parse_count
uia_parse_count
ocr_fallback_count
ocr_call_count
ocr_timeout_count
parse_cache_hit_count
state_cache_hit_count
parse_count_avoided_by_state_cache
failure_reason_distribution
```

Benchmark outputs are written under:

```text
benchmark_runs\<run>\benchmark_report.json
benchmark_runs\<run>\failed_artifacts\
```

Use dry runs and design-only validation while developing. Use live benchmarks only when the desktop environment is prepared and the consequences of each mission are understood.

## Current Verified Result

Current local test-suite verification:

```text
Command: python -m pytest -q
Result:  217 passed in 10.01s
```

This is a local unit/integration test result. It is not a live desktop benchmark success claim.

To generate fresh results on your machine:

```powershell
python -m pytest -q
python .\run_benchmark.py --design-only --output-dir benchmark_runs\live_design_readme_check
python .\run_benchmark.py --live --repeat 1 --auto-approve --max-missions 3 --output-dir benchmark_runs\live_probe_v41
```

## Outputs and Artifacts

Runtime memory:

```text
memory/semantic_memory.json
memory/episodic_memory.json
memory/workflow_memory.json
memory/policy_memory.json
memory/desktop_state.json
```

Debug and trace artifacts:

```text
logs/live_feed.log
debug_steps/traces/*.json
debug_steps/learning_sessions/*.json
benchmark_runs/.../benchmark_report.json
benchmark_runs/.../failed_artifacts/
```

Primary structured trace data:

```text
trace.outputs["action_contracts"]
trace.outputs["parse_health"]
trace.outputs["state_snapshots"]
trace.outputs["recovery_attempts"]
trace.outputs["voice_events"]
```

## Safety Rules and Limitations

Hard safety rules:

```text
No blind coordinate retries.
No destructive hotkeys without expected change or approval.
No browser visual fallback when Chrome/Edge DOM is unavailable.
No success without verification.
No failed action without a classified reason.
No retry without a recovery strategy.
No unbounded waits.
```

Browser safety:

- Chrome/Edge tasks require DOM.
- If CDP cannot be reached after automatic debug launch, the runtime fails safely.
- Frame and shadow DOM identity fields are captured.
- Full action routing into nested frame/shadow contexts is a known next hardening step.

Windows safety:

- UIA is preferred for native app interaction.
- Raw coordinate clicks require strong evidence and are not blindly retried.
- UIA collection is scoped to the active window handle when possible for speed.

Live feed safety:

- The taskbar `STOP` button requests runtime cancellation and is checked by the execution loop.
- The feed is non-blocking when launched through `run_task.py --taskbar`; the agent runs in a worker thread and the UI stays in the Qt event loop.
- `logs/live_feed.log` is a local debug artifact and is ignored by Git.

Live benchmark safety:

- Start with `--max-missions 3`.
- Use disposable browser tabs, test forms, and non-sensitive windows.
- Avoid live destructive workflows unless the policy gate and approval path are explicitly tested.

## Developer Notes

Recommended validation before committing runtime changes:

```powershell
python -m pytest -q
python .\run_benchmark.py --list
python .\run_benchmark.py --design-only --output-dir benchmark_runs\live_design_readme_check
```

Paddle/OCR stability flags are set by the runner scripts:

```text
FLAGS_enable_pir_api=0
FLAGS_use_mkldnn=0
```

The product direction is deterministic first:

```text
DOM/UIA exactness
  -> persistent state comparison
  -> direct verification
  -> bounded repair/recovery
  -> OCR/vision only when deterministic sources cannot cover the screen
```
