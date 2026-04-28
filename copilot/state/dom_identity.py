from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import time


@dataclass
class DOMIdentityRecord:
    node_id: str
    stable_hash: str
    last_seen: float
    last_rect: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DOMIdentityTracker:
    """Tracks DOM node identities across observations for re-identification."""

    def __init__(self) -> None:
        self._by_node_id: dict[str, DOMIdentityRecord] = {}
        self._node_id_by_hash: dict[str, str] = {}

    def track(
        self,
        node_id: str,
        stable_hash: str,
        rect: dict[str, Any] | None = None,
        *,
        seen_at: float | None = None,
    ) -> DOMIdentityRecord:
        stable_hash = str(stable_hash or "")
        node_id = str(node_id or self.node_id_for_hash(stable_hash) or "")
        if not node_id:
            raise ValueError("DOM identity tracking requires node_id or stable_hash")
        normalized_rect = self._normalize_rect(rect or {})
        record = DOMIdentityRecord(
            node_id=node_id,
            stable_hash=stable_hash,
            last_seen=float(seen_at if seen_at is not None else time.time()),
            last_rect=normalized_rect,
        )
        self._by_node_id[node_id] = record
        if stable_hash:
            self._node_id_by_hash[stable_hash] = node_id
        return record

    def node_id_for_hash(self, stable_hash: str) -> str:
        return self._node_id_by_hash.get(str(stable_hash or ""), "")

    def reidentify(self, stable_hash: str) -> DOMIdentityRecord | None:
        node_id = self.node_id_for_hash(stable_hash)
        if not node_id:
            return None
        return self._by_node_id.get(node_id)

    def verify(self, node_id: str, stable_hash: str = "", rect: dict[str, Any] | None = None) -> bool:
        record = self._by_node_id.get(str(node_id or ""))
        if record is None:
            return False
        if stable_hash and record.stable_hash != str(stable_hash):
            return False
        if rect is None:
            return True
        return self._rect_close(record.last_rect, self._normalize_rect(rect))

    def records(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._by_node_id.values()]

    @staticmethod
    def _normalize_rect(rect: dict[str, Any]) -> dict[str, int]:
        return {
            "x": int(rect.get("x", 0) or 0),
            "y": int(rect.get("y", 0) or 0),
            "w": int(rect.get("w", rect.get("width", 0)) or 0),
            "h": int(rect.get("h", rect.get("height", 0)) or 0),
        }

    @staticmethod
    def _rect_close(a: dict[str, int], b: dict[str, int], tolerance: int = 16) -> bool:
        return (
            abs(int(a.get("x", 0)) - int(b.get("x", 0))) <= tolerance
            and abs(int(a.get("y", 0)) - int(b.get("y", 0))) <= tolerance
            and abs(int(a.get("w", 0)) - int(b.get("w", 0))) <= tolerance
            and abs(int(a.get("h", 0)) - int(b.get("h", 0))) <= tolerance
        )
