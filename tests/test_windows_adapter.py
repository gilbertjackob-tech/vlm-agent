from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest

from copilot.adapters import windows as windows_module
from copilot.adapters.windows import WindowsAdapter


class FakeWindow:
    def __init__(self, title: str) -> None:
        self.title = title
        self.left = 0
        self.top = 0
        self.width = 800
        self.height = 600


class FakeRect:
    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class FakeUIAElement:
    CurrentName = "Downloads"

    def rectangle(self):
        return FakeRect(10, 20, 110, 60)

    def control_type(self):
        return "Button"

    def automation_id(self):
        return "DownloadsButton"

    def is_enabled(self):
        return True

    def is_visible(self):
        return True


class FakeUIACollection:
    Length = 1

    def GetElement(self, _index):
        return FakeUIAElement()


class FakeUIARoot:
    def FindAll(self, _scope, _condition):
        return FakeUIACollection()


class FakeUIAutomation:
    def GetRootElement(self):
        return FakeUIARoot()

    def CreateTrueCondition(self):
        return object()


class WindowsAdapterTests(unittest.TestCase):
    def test_explorer_common_folder_titles_match_existing_window(self) -> None:
        adapter = WindowsAdapter()
        original_get_active = windows_module.gw.getActiveWindow
        original_get_all = windows_module.gw.getAllWindows
        downloads = FakeWindow("Downloads")

        windows_module.gw.getActiveWindow = lambda: downloads
        windows_module.gw.getAllWindows = lambda: [downloads]
        try:
            self.assertTrue(adapter.confirm_focus(app_id="explorer"))
            self.assertEqual(adapter.find_window(app_id="explorer")["title"], "Downloads")
        finally:
            windows_module.gw.getActiveWindow = original_get_active
            windows_module.gw.getAllWindows = original_get_all

    def test_launch_chrome_prefers_pinned_taskbar_entry(self) -> None:
        adapter = WindowsAdapter()
        original_appdata = os.environ.get("APPDATA")
        original_hotkey = windows_module.pyautogui.hotkey
        original_sleep = windows_module.time.sleep
        hotkeys: list[tuple[str, ...]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            taskbar_dir = os.path.join(
                tmpdir,
                "Microsoft",
                "Internet Explorer",
                "Quick Launch",
                "User Pinned",
                "TaskBar",
            )
            os.makedirs(taskbar_dir, exist_ok=True)
            open(os.path.join(taskbar_dir, "Google Chrome.lnk"), "w", encoding="utf-8").close()
            os.environ["APPDATA"] = tmpdir
            windows_module.pyautogui.hotkey = lambda *keys: hotkeys.append(tuple(keys))
            windows_module.time.sleep = lambda _seconds: None
            try:
                self.assertTrue(adapter.launch_application("chrome"))
            finally:
                windows_module.pyautogui.hotkey = original_hotkey
                windows_module.time.sleep = original_sleep
                if original_appdata is None:
                    os.environ.pop("APPDATA", None)
                else:
                    os.environ["APPDATA"] = original_appdata

        self.assertIn(("win", "1"), hotkeys)

    def test_launch_chrome_falls_back_to_default_browser_url(self) -> None:
        adapter = WindowsAdapter()
        original_popen = windows_module.subprocess.Popen
        original_listdir = windows_module.os.listdir
        original_isdir = windows_module.os.path.isdir
        popen_calls: list[list[str]] = []

        def fake_popen(command, stdout=None, stderr=None):
            popen_calls.append(list(command))
            if command[:4] == ["cmd.exe", "/c", "start", ""] and command[-1] == "chrome":
                raise OSError("chrome unavailable")
            return object()

        windows_module.subprocess.Popen = fake_popen
        windows_module.os.listdir = lambda _path: []
        windows_module.os.path.isdir = lambda _path: False
        try:
            self.assertTrue(adapter.launch_application("chrome"))
        finally:
            windows_module.subprocess.Popen = original_popen
            windows_module.os.listdir = original_listdir
            windows_module.os.path.isdir = original_isdir

        self.assertEqual(popen_calls[-1], ["cmd.exe", "/c", "start", "", "https://www.google.com/"])

    def test_launch_unknown_app_uses_windows_search_fallback(self) -> None:
        adapter = WindowsAdapter()
        original_press = windows_module.pyautogui.press
        original_write = windows_module.pyautogui.write
        original_sleep = windows_module.time.sleep
        events: list[tuple[str, str]] = []

        windows_module.pyautogui.press = lambda key: events.append(("press", str(key)))
        windows_module.pyautogui.write = lambda text, interval=0.0: events.append(("write", str(text)))
        windows_module.time.sleep = lambda _seconds: None
        try:
            self.assertTrue(adapter.launch_application("notepad"))
        finally:
            windows_module.pyautogui.press = original_press
            windows_module.pyautogui.write = original_write
            windows_module.time.sleep = original_sleep

        self.assertEqual(events[0], ("press", "win"))
        self.assertEqual(events[1], ("write", "notepad"))
        self.assertEqual(events[-1], ("press", "enter"))

    def test_collect_uia_elements_uses_real_element_methods_and_click_point(self) -> None:
        adapter = WindowsAdapter()
        original_comtypes = sys.modules.get("comtypes")
        original_client = sys.modules.get("comtypes.client")
        fake_comtypes = types.ModuleType("comtypes")
        fake_client = types.ModuleType("comtypes.client")
        fake_client.CreateObject = lambda _name: FakeUIAutomation()  # type: ignore[attr-defined]
        fake_comtypes.client = fake_client  # type: ignore[attr-defined]
        sys.modules["comtypes"] = fake_comtypes
        sys.modules["comtypes.client"] = fake_client
        try:
            elements = adapter._collect_uia_elements({"title": "Downloads", "left": 0, "top": 0, "width": 800, "height": 600})
        finally:
            if original_comtypes is None:
                sys.modules.pop("comtypes", None)
            else:
                sys.modules["comtypes"] = original_comtypes
            if original_client is None:
                sys.modules.pop("comtypes.client", None)
            else:
                sys.modules["comtypes.client"] = original_client

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["control_type"], "Button")
        self.assertEqual(elements[0]["automation_id"], "DownloadsButton")
        self.assertTrue(elements[0]["enabled"])
        self.assertTrue(elements[0]["visible"])
        self.assertEqual(elements[0]["rectangle"], {"x": 10, "y": 20, "width": 100, "height": 40})
        self.assertEqual(elements[0]["click_point"], {"x": 60, "y": 40})
        self.assertEqual(elements[0]["clickable_point"], {"x": 60, "y": 40})


if __name__ == "__main__":
    unittest.main()
