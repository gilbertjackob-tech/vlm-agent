from __future__ import annotations

from importlib import import_module
from typing import Any


def create_legacy_vlm_agent() -> Any:
    legacy_module = import_module("agent")
    return legacy_module.VLMAgent()
