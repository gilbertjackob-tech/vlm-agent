from __future__ import annotations

from dataclasses import dataclass
import re


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


@dataclass(frozen=True)
class ShortcutIntent:
    shortcut_id: str
    app_id: str
    title: str
    keys: tuple[str, ...]
    aliases: tuple[str, ...]
    expected_change: str
    destructive: bool = False
    requires_approval: bool = False

    @property
    def hotkey(self) -> bool:
        return len(self.keys) > 1


SHORTCUTS: tuple[ShortcutIntent, ...] = (
    ShortcutIntent("focus_address_bar", "common", "Focus address/search bar", ("ctrl", "l"), ("focus address bar", "focus search bar", "focus address search bar", "ctrl l", "ctrl+l"), "The address or search bar should be focused."),
    ShortcutIntent("new_window", "common", "Open new window", ("ctrl", "n"), ("new window", "open new window", "ctrl n", "ctrl+n"), "A new window should open."),
    ShortcutIntent("next_tab", "common", "Switch to next tab", ("ctrl", "tab"), ("next tab", "switch to next tab", "ctrl tab", "ctrl+tab"), "The next tab should become active."),
    ShortcutIntent("previous_tab", "common", "Switch to previous tab", ("ctrl", "shift", "tab"), ("previous tab", "prev tab", "switch to previous tab", "ctrl shift tab", "ctrl+shift+tab"), "The previous tab should become active."),
    ShortcutIntent("go_back", "common", "Go back", ("alt", "left"), ("go back", "navigate back", "back one page", "alt left", "alt+left"), "The current view should go back."),
    ShortcutIntent("go_forward", "common", "Go forward", ("alt", "right"), ("go forward", "navigate forward", "alt right", "alt+right"), "The current view should go forward."),
    ShortcutIntent("close_current", "common", "Close current tab or window", ("ctrl", "w"), ("close current tab", "close tab", "close current window", "ctrl w", "ctrl+w"), "The current tab or window should close.", destructive=True, requires_approval=True),
    ShortcutIntent("new_tab", "chrome", "Open new Chrome tab", ("ctrl", "t"), ("new tab", "open new tab", "chrome new tab", "ctrl t", "ctrl+t"), "A new Chrome tab should open."),
    ShortcutIntent("reopen_closed_tab", "chrome", "Reopen closed Chrome tab", ("ctrl", "shift", "t"), ("reopen closed tab", "restore closed tab", "ctrl shift t", "ctrl+shift+t"), "The most recently closed Chrome tab should reopen."),
    ShortcutIntent("last_tab", "chrome", "Jump to last Chrome tab", ("ctrl", "9"), ("last tab", "jump to last tab", "ctrl 9", "ctrl+9"), "Chrome should activate the last tab."),
    ShortcutIntent("close_window", "chrome", "Close Chrome window", ("ctrl", "shift", "w"), ("close chrome window", "close window", "ctrl shift w", "ctrl+shift+w"), "The current Chrome window should close.", destructive=True, requires_approval=True),
    ShortcutIntent("refresh", "chrome", "Refresh Chrome page", ("ctrl", "r"), ("refresh page", "reload page", "chrome refresh", "ctrl r", "ctrl+r", "f5"), "The Chrome page should refresh."),
    ShortcutIntent("hard_refresh", "chrome", "Hard refresh Chrome page", ("ctrl", "shift", "r"), ("hard refresh", "force refresh", "ctrl shift r", "ctrl+shift+r"), "The Chrome page should hard refresh."),
    ShortcutIntent("bookmark_page", "chrome", "Bookmark Chrome page", ("ctrl", "d"), ("bookmark page", "add bookmark", "ctrl d", "ctrl+d"), "Chrome should open the bookmark action."),
    ShortcutIntent("history", "chrome", "Open Chrome history", ("ctrl", "h"), ("open history", "chrome history", "history page", "ctrl h", "ctrl+h"), "Chrome history should open."),
    ShortcutIntent("downloads", "chrome", "Open Chrome downloads", ("ctrl", "j"), ("open chrome downloads", "chrome downloads", "downloads page", "ctrl j", "ctrl+j"), "Chrome downloads should open."),
    ShortcutIntent("find_in_page", "chrome", "Find in Chrome page", ("ctrl", "f"), ("find in page", "find on page", "page search", "ctrl f", "ctrl+f"), "Chrome find-in-page should be focused."),
    ShortcutIntent("zoom_in", "chrome", "Zoom in Chrome page", ("ctrl", "+"), ("zoom in", "increase zoom", "ctrl plus", "ctrl+plus", "ctrl +"), "Chrome zoom should increase."),
    ShortcutIntent("zoom_out", "chrome", "Zoom out Chrome page", ("ctrl", "-"), ("zoom out", "decrease zoom", "ctrl minus", "ctrl+minus", "ctrl -"), "Chrome zoom should decrease."),
    ShortcutIntent("reset_zoom", "chrome", "Reset Chrome zoom", ("ctrl", "0"), ("reset zoom", "default zoom", "ctrl 0", "ctrl+0"), "Chrome zoom should reset."),
    ShortcutIntent("incognito", "chrome", "Open Chrome incognito window", ("ctrl", "shift", "n"), ("incognito", "new incognito", "private window", "ctrl shift n", "ctrl+shift+n"), "A Chrome incognito window should open."),
    ShortcutIntent("new_folder", "explorer", "Create new Explorer folder", ("ctrl", "shift", "n"), ("new folder", "create folder", "make folder", "ctrl shift n", "ctrl+shift+n"), "A new folder should be created in Explorer.", requires_approval=True),
    ShortcutIntent("copy", "explorer", "Copy Explorer selection", ("ctrl", "c"), ("copy selected", "copy selection", "copy file", "copy folder", "ctrl c", "ctrl+c"), "The Explorer selection should be copied."),
    ShortcutIntent("cut", "explorer", "Cut Explorer selection", ("ctrl", "x"), ("cut selected", "cut selection", "cut file", "cut folder", "ctrl x", "ctrl+x"), "The Explorer selection should be cut.", requires_approval=True),
    ShortcutIntent("paste", "explorer", "Paste in Explorer", ("ctrl", "v"), ("paste here", "paste in explorer", "paste file", "paste folder", "ctrl v", "ctrl+v"), "Clipboard contents should be pasted in Explorer.", requires_approval=True),
    ShortcutIntent("delete", "explorer", "Delete Explorer selection", ("delete",), ("delete selected", "delete file", "delete folder", "delete selection"), "The selected Explorer item should be deleted.", destructive=True, requires_approval=True),
    ShortcutIntent("permanent_delete", "explorer", "Permanently delete Explorer selection", ("shift", "delete"), ("permanent delete", "permanently delete", "shift delete", "shift+delete"), "The selected Explorer item should be permanently deleted.", destructive=True, requires_approval=True),
    ShortcutIntent("parent_folder", "explorer", "Go to parent Explorer folder", ("alt", "up"), ("parent folder", "go to parent", "up one folder", "alt up", "alt+up"), "Explorer should navigate to the parent folder."),
    ShortcutIntent("properties", "explorer", "Open Explorer properties", ("alt", "enter"), ("properties", "file properties", "folder properties", "alt enter", "alt+enter"), "Explorer properties should open."),
    ShortcutIntent("preview_pane", "explorer", "Toggle Explorer preview pane", ("alt", "p"), ("preview pane", "toggle preview pane", "alt p", "alt+p"), "Explorer preview pane should toggle."),
    ShortcutIntent("details_pane", "explorer", "Toggle Explorer details pane", ("alt", "shift", "p"), ("details pane", "toggle details pane", "alt shift p", "alt+shift+p"), "Explorer details pane should toggle."),
    ShortcutIntent("explorer_back", "explorer", "Go back in Explorer", ("backspace",), ("backspace", "explorer back"), "Explorer should go back."),
    ShortcutIntent("explorer_refresh", "explorer", "Refresh Explorer", ("f5",), ("refresh explorer", "refresh folder", "f5"), "Explorer should refresh."),
    ShortcutIntent("select_all", "explorer", "Select all Explorer items", ("ctrl", "a"), ("select all", "select everything", "ctrl a", "ctrl+a"), "All Explorer items should be selected."),
    ShortcutIntent("rename", "explorer", "Rename Explorer selection", ("f2",), ("rename", "rename file", "rename folder", "f2"), "Explorer rename should start.", requires_approval=True),
    ShortcutIntent("search", "explorer", "Focus Explorer search", ("f3",), ("search explorer", "explorer search", "f3"), "Explorer search should be focused."),
    ShortcutIntent("focus_address_bar_alt", "explorer", "Focus Explorer address bar", ("alt", "d"), ("explorer address bar", "focus explorer address", "alt d", "alt+d"), "Explorer address bar should be focused."),
    ShortcutIntent("open_explorer", "explorer", "Open Explorer", ("win", "e"), ("open explorer shortcut", "windows e", "win e", "win+e", "windows+e"), "Explorer should open or become visible."),
    ShortcutIntent("view_extra_large_icons", "explorer", "Set Explorer extra large icons view", ("ctrl", "shift", "1"), ("extra large icons", "extra large icon view", "view 1", "ctrl shift 1", "ctrl+shift+1"), "Explorer should switch to extra large icons view."),
    ShortcutIntent("view_large_icons", "explorer", "Set Explorer large icons view", ("ctrl", "shift", "2"), ("large icons", "large icon view", "view 2", "ctrl shift 2", "ctrl+shift+2"), "Explorer should switch to large icons view."),
    ShortcutIntent("view_medium_icons", "explorer", "Set Explorer medium icons view", ("ctrl", "shift", "3"), ("medium icons", "medium icon view", "view 3", "ctrl shift 3", "ctrl+shift+3"), "Explorer should switch to medium icons view."),
    ShortcutIntent("view_small_icons", "explorer", "Set Explorer small icons view", ("ctrl", "shift", "4"), ("small icons", "small icon view", "view 4", "ctrl shift 4", "ctrl+shift+4"), "Explorer should switch to small icons view."),
    ShortcutIntent("view_list", "explorer", "Set Explorer list view", ("ctrl", "shift", "5"), ("list view", "view 5", "ctrl shift 5", "ctrl+shift+5"), "Explorer should switch to list view."),
    ShortcutIntent("view_stable_details", "explorer", "Set Explorer stable details view", ("ctrl", "shift", "6"), ("details view", "stable view", "stable details view", "view 6", "ctrl shift 6", "ctrl+shift+6"), "Explorer should switch to the stable details view."),
    ShortcutIntent("view_tiles", "explorer", "Set Explorer tiles view", ("ctrl", "shift", "7"), ("tiles view", "tile view", "view 7", "ctrl shift 7", "ctrl+shift+7"), "Explorer should switch to tiles view."),
    ShortcutIntent("view_content", "explorer", "Set Explorer content view", ("ctrl", "shift", "8"), ("content view", "view 8", "ctrl shift 8", "ctrl+shift+8"), "Explorer should switch to content view."),
)


def shortcuts_for_app(app_id: str) -> list[ShortcutIntent]:
    normalized_app = normalize_text(app_id)
    return [shortcut for shortcut in SHORTCUTS if shortcut.app_id == normalized_app or shortcut.app_id == "common"]


def match_shortcut_intents(prompt: str, app_id: str) -> list[ShortcutIntent]:
    text = normalize_text(prompt).replace("+", " + ")
    text = normalize_text(text)
    matches: list[ShortcutIntent] = []
    for shortcut in shortcuts_for_app(app_id):
        if any(_alias_matches(text, alias) for alias in shortcut.aliases):
            matches.append(shortcut)
    deduped: list[ShortcutIntent] = []
    seen: set[str] = set()
    for shortcut in matches:
        if shortcut.shortcut_id in seen:
            continue
        seen.add(shortcut.shortcut_id)
        deduped.append(shortcut)
    return deduped[:3]


def _alias_matches(text: str, alias: str) -> bool:
    normalized_alias = normalize_text(alias).replace("+", " + ")
    normalized_alias = normalize_text(normalized_alias)
    if not normalized_alias:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized_alias)}(?!\w)", text) is not None
