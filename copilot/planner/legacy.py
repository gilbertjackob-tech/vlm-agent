from __future__ import annotations

from importlib import import_module
from typing import Any


def create_legacy_task_planner() -> Any:
    legacy_module = import_module("planner")
    return legacy_module.TaskPlanner()
