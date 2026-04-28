from __future__ import annotations

from typing import Any
from hashlib import sha256
import ctypes
import json
import os
import subprocess
import time

import pygetwindow as gw
import pyautogui

try:
    import comtypes.client  # type: ignore
except Exception:  # pragma: no cover - optional Windows UIA dependency
    comtypes = None  # type: ignore


class WindowsAdapter:
    def __init__(self) -> None:
        self.app_aliases = {
            "explorer": [
                "file explorer",
                "explorer",
                "desktop",
                "downloads",
                "documents",
                "pictures",
                "music",
                "videos",
                "this pc",
                "home",
                "quick access",
            ],
            "chrome": [
                "google chrome",
                "chrome",
                "microsoft edge",
                "edge",
                "mozilla firefox",
                "firefox",
                "brave",
                "opera",
            ],
            "edge": ["microsoft edge", "edge"],
        }

    def _window_payload(self, win: Any) -> dict[str, Any]:
        return {
            "title": getattr(win, "title", ""),
            "left": getattr(win, "left", 0),
            "top": getattr(win, "top", 0),
            "width": getattr(win, "width", 0),
            "height": getattr(win, "height", 0),
            "hwnd": int(getattr(win, "_hWnd", 0) or getattr(win, "_hwnd", 0) or 0),
        }

    def _stable_hash(self, payload: Any) -> str:
        try:
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        except TypeError:
            blob = str(payload)
        return sha256(blob.encode("utf-8")).hexdigest()[:16]

    def get_active_window(self) -> dict[str, Any]:
        try:
            win = gw.getActiveWindow()
            if not win:
                return {}
            return self._window_payload(win)
        except Exception:
            return {}

    def list_windows(self) -> list[dict[str, Any]]:
        results = []
        try:
            for win in gw.getAllWindows():
                title = getattr(win, "title", "")
                if not title:
                    continue
                results.append(self._window_payload(win))
        except Exception:
            return []
        return results

    def _match_keywords(self, title: str, keywords: list[str]) -> bool:
        lowered = title.lower()
        return any(keyword.lower() in lowered for keyword in keywords if keyword)

    def find_window(self, title_contains: str = "", app_id: str = "") -> dict[str, Any]:
        title_query = title_contains.lower().strip()
        keywords = list(self.app_aliases.get(app_id.lower().strip(), [])) if app_id else []
        try:
            for win in gw.getAllWindows():
                title = getattr(win, "title", "")
                if not title:
                    continue
                lowered = title.lower()
                if title_query and title_query in lowered:
                    return self._window_payload(win)
                if keywords and self._match_keywords(title, keywords):
                    return self._window_payload(win)
        except Exception:
            return {}
        return {}

    def focus_window(self, title_contains: str = "", app_id: str = "", timeout: float = 2.0) -> bool:
        title_query = title_contains.lower().strip()
        keywords = list(self.app_aliases.get(app_id.lower().strip(), [])) if app_id else []
        deadline = time.time() + max(0.2, timeout)
        while time.time() < deadline:
            try:
                for win in gw.getAllWindows():
                    title = getattr(win, "title", "")
                    if not title:
                        continue
                    lowered = title.lower()
                    if title_query and title_query not in lowered:
                        if not keywords or not self._match_keywords(title, keywords):
                            continue
                    elif not title_query and keywords and not self._match_keywords(title, keywords):
                        continue

                    try:
                        if getattr(win, "isMinimized", False):
                            win.restore()
                            time.sleep(0.1)
                    except Exception:
                        pass
                    try:
                        win.activate()
                        time.sleep(0.15)
                    except Exception:
                        pass
                    self._force_foreground(win)
                    time.sleep(0.1)
                    if self.confirm_focus(title_contains=title_contains, app_id=app_id):
                        return True
            except Exception:
                return False
            time.sleep(0.1)
        return False

    def _force_foreground(self, win: Any) -> bool:
        hwnd = int(getattr(win, "_hWnd", 0) or getattr(win, "_hwnd", 0) or 0)
        if not hwnd:
            return False
        try:
            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            return True
        except Exception:
            return False

    def confirm_focus(self, title_contains: str = "", app_id: str = "") -> bool:
        active = self.get_active_window()
        title = active.get("title", "").lower()
        if not title:
            return False
        if title_contains and title_contains.lower().strip() in title:
            return True
        if app_id:
            return self._match_keywords(title, self.app_aliases.get(app_id.lower().strip(), []))
        return False

    def route_to_application(
        self,
        app_id: str,
        window_title: str = "",
        launch_callback: Any | None = None,
        timeout: float = 4.0,
    ) -> dict[str, Any]:
        if self.focus_window(title_contains=window_title, app_id=app_id, timeout=1.2):
            return {"ok": True, "launched": False, "window": self.get_active_window()}

        native_launched = self.launch_application(app_id)
        if native_launched and self.focus_window(title_contains=window_title, app_id=app_id, timeout=timeout):
            return {"ok": True, "launched": True, "window": self.get_active_window(), "launch_mode": "native"}

        if launch_callback:
            launched = bool(launch_callback())
            if launched and self.focus_window(title_contains=window_title, app_id=app_id, timeout=timeout):
                return {"ok": True, "launched": True, "window": self.get_active_window()}
            return {"ok": False, "launched": launched, "window": self.get_active_window()}

        return {"ok": False, "launched": False, "window": self.get_active_window()}

    def launch_application(self, app_id: str) -> bool:
        app_id = app_id.lower().strip()
        if app_id == "explorer":
            try:
                pyautogui.hotkey("win", "e")
                time.sleep(0.2)
                return True
            except Exception:
                pass
        if self._launch_from_taskbar(app_id):
            return True
        commands = {
            "explorer": [["explorer.exe"]],
            "chrome": [["cmd.exe", "/c", "start", "", "chrome"]],
            "edge": [["cmd.exe", "/c", "start", "", "msedge"]],
        }
        for command in commands.get(app_id, []):
            try:
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except OSError:
                continue
        if app_id == "chrome" and self._launch_default_browser_url("https://www.google.com/"):
            return True
        return self._launch_via_start_menu_search(app_id)

    def _taskbar_shortcuts_path(self) -> str:
        app_data = os.environ.get("APPDATA", "").strip()
        if not app_data:
            return ""
        return os.path.join(
            app_data,
            "Microsoft",
            "Internet Explorer",
            "Quick Launch",
            "User Pinned",
            "TaskBar",
        )

    def _taskbar_pinned_entries(self) -> list[str]:
        taskbar_path = self._taskbar_shortcuts_path()
        if not taskbar_path or not os.path.isdir(taskbar_path):
            return []
        try:
            entries = [name for name in os.listdir(taskbar_path) if name.lower().endswith(".lnk")]
        except OSError:
            return []
        entries.sort(key=lambda value: os.path.getmtime(os.path.join(taskbar_path, value)))
        return entries

    def _launch_from_taskbar(self, app_id: str) -> bool:
        keywords = [keyword for keyword in self.app_aliases.get(app_id, []) if keyword]
        if not keywords:
            return False
        entries = self._taskbar_pinned_entries()
        for index, entry in enumerate(entries[:9], start=1):
            label = entry[:-4].lower()
            if not any(keyword in label for keyword in keywords):
                continue
            try:
                pyautogui.hotkey("win", str(index))
                time.sleep(0.3)
                return True
            except Exception:
                return False
        return False

    def _launch_default_browser_url(self, url: str) -> bool:
        command = ["cmd.exe", "/c", "start", "", url]
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            return False

    def _launch_via_start_menu_search(self, app_id: str) -> bool:
        query = app_id.strip()
        if not query:
            return False
        try:
            pyautogui.press("win")
            time.sleep(1.0)
            pyautogui.write(query, interval=0.03)
            time.sleep(0.5)
            pyautogui.press("enter")
            time.sleep(0.3)
            return True
        except Exception:
            return False

    def is_same_app_family(self, app_id: str, window: dict[str, Any] | None = None) -> bool:
        payload = window or self.get_active_window()
        title = payload.get("title", "")
        if not title:
            return False
        return self._match_keywords(title, self.app_aliases.get(app_id.lower().strip(), []))

    def clamp_point_to_active_window(self, x: int, y: int) -> tuple[int, int]:
        active = self.get_active_window()
        if not active:
            return x, y
        left = int(active.get("left", 0))
        top = int(active.get("top", 0))
        width = int(active.get("width", 0))
        height = int(active.get("height", 0))
        if width <= 0 or height <= 0:
            return x, y
        clamped_x = min(max(x, left + 4), left + width - 4)
        clamped_y = min(max(y, top + 4), top + height - 4)
        return clamped_x, clamped_y

    def observe(self) -> dict[str, Any]:
        active_window = self.get_active_window()
        return {
            "active_window": active_window,
            "open_windows": self.list_windows()[:20],
            "active_app_guess": self._guess_app(active_window.get("title", "")),
            "adapter_mode": "native_window_metadata",
            "uia_elements": self._collect_uia_elements(active_window),
        }

    def _collect_uia_elements(self, active_window: dict[str, Any], limit: int = 160) -> list[dict[str, Any]]:
        if not active_window:
            return []
        try:
            import comtypes.client as cc  # type: ignore
        except Exception:
            return []
        try:
            uia = cc.CreateObject("UIAutomationClient.CUIAutomation")
            hwnd = int(active_window.get("hwnd", 0) or 0)
            root = uia.ElementFromHandle(hwnd) if hwnd and hasattr(uia, "ElementFromHandle") else uia.GetRootElement()
            if root is None:
                return []
            true_condition = uia.CreateTrueCondition()
            elements = root.FindAll(4, true_condition)  # TreeScope_Descendants
        except Exception:
            return []

        active_rect = {
            "left": int(active_window.get("left", 0) or 0),
            "top": int(active_window.get("top", 0) or 0),
            "right": int(active_window.get("left", 0) or 0) + int(active_window.get("width", 0) or 0),
            "bottom": int(active_window.get("top", 0) or 0) + int(active_window.get("height", 0) or 0),
        }
        parent_title = str(active_window.get("title", ""))
        results: list[dict[str, Any]] = []
        try:
            count = int(elements.Length)
        except Exception:
            return []
        for index in range(min(count, limit * 4)):
            if len(results) >= limit:
                break
            try:
                el = elements.GetElement(index)
                rect = self._element_rectangle(el)
                left = int(rect.get("left", 0) or 0)
                top = int(rect.get("top", 0) or 0)
                right = int(rect.get("right", 0) or 0)
                bottom = int(rect.get("bottom", 0) or 0)
                width = max(0, right - left)
                height = max(0, bottom - top)
                if width <= 0 or height <= 0:
                    continue
                if right < active_rect["left"] or left > active_rect["right"] or bottom < active_rect["top"] or top > active_rect["bottom"]:
                    continue
                name = self._element_name(el)
                automation_id = self._element_automation_id(el)
                control_type = self._element_control_type(el)
                if not (name or automation_id or control_type):
                    continue
                click_point = {"x": left + width // 2, "y": top + height // 2}
                payload = {
                    "name": name,
                    "automation_id": automation_id,
                    "control_type": control_type,
                    "rectangle": {"x": left, "y": top, "width": width, "height": height},
                    "click_point": click_point,
                    "clickable_point": click_point,
                    "enabled": self._element_is_enabled(el),
                    "visible": self._element_is_visible(el),
                    "parent_window": parent_title,
                }
                payload["stable_hash"] = self._stable_hash(
                    {
                        "name": name,
                        "automation_id": automation_id,
                        "control_type": control_type,
                        "parent_window": parent_title,
                    }
                )
                results.append(payload)
            except Exception:
                continue
        return results

    def _element_name(self, element: Any) -> str:
        for attr in ("window_text", "texts"):
            method = getattr(element, attr, None)
            if not callable(method):
                continue
            try:
                value = method()
                if isinstance(value, list):
                    return str(value[0] if value else "")
                return str(value or "")
            except Exception:
                continue
        return str(getattr(element, "CurrentName", "") or "")

    def _element_rectangle(self, element: Any) -> dict[str, int]:
        method = getattr(element, "rectangle", None)
        if callable(method):
            rect = method()
            left = int(getattr(rect, "left", 0) if not callable(getattr(rect, "left", None)) else rect.left())
            top = int(getattr(rect, "top", 0) if not callable(getattr(rect, "top", None)) else rect.top())
            right = int(getattr(rect, "right", 0) if not callable(getattr(rect, "right", None)) else rect.right())
            bottom = int(getattr(rect, "bottom", 0) if not callable(getattr(rect, "bottom", None)) else rect.bottom())
            return {"left": left, "top": top, "right": right, "bottom": bottom}
        rect = getattr(element, "CurrentBoundingRectangle", None)
        return {
            "left": int(getattr(rect, "left", 0) or 0),
            "top": int(getattr(rect, "top", 0) or 0),
            "right": int(getattr(rect, "right", 0) or 0),
            "bottom": int(getattr(rect, "bottom", 0) or 0),
        }

    def _element_control_type(self, element: Any) -> str:
        method = getattr(element, "control_type", None)
        if callable(method):
            try:
                return str(method() or "")
            except Exception:
                return ""
        return str(getattr(element, "CurrentLocalizedControlType", "") or getattr(element, "CurrentControlType", "") or "")

    def _element_automation_id(self, element: Any) -> str:
        method = getattr(element, "automation_id", None)
        if callable(method):
            try:
                return str(method() or "")
            except Exception:
                return ""
        return str(getattr(element, "CurrentAutomationId", "") or "")

    def _element_is_enabled(self, element: Any) -> bool:
        method = getattr(element, "is_enabled", None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
        return bool(getattr(element, "CurrentIsEnabled", False))

    def _element_is_visible(self, element: Any) -> bool:
        method = getattr(element, "is_visible", None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
        return not bool(getattr(element, "CurrentIsOffscreen", True))

    def _guess_app(self, title: str) -> str:
        lowered = title.lower()
        for app_id, keywords in self.app_aliases.items():
            if any(keyword in lowered for keyword in keywords):
                return app_id
        return ""
