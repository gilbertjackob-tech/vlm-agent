from __future__ import annotations

from typing import Any
from hashlib import sha256
import json
import re
import urllib.error
import urllib.request

try:
    import websocket  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    websocket = None


def _normalize(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _stable_hash(payload: Any) -> str:
    try:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        blob = str(payload)
    return sha256(blob.encode("utf-8")).hexdigest()[:16]


class BrowserAdapter:
    def __init__(self, debug_url: str = "http://127.0.0.1:9222/json") -> None:
        self.debug_url = debug_url.rstrip("/")
        self._message_id = 0

    def _fetch_tabs(self) -> list[dict[str, Any]]:
        try:
            with urllib.request.urlopen(self.debug_url, timeout=0.5) as response:
                data = json.loads(response.read().decode("utf-8"))
            return [tab for tab in data if isinstance(tab, dict)]
        except (OSError, ValueError, urllib.error.URLError):
            return []

    def _page_tabs(self) -> list[dict[str, Any]]:
        return [tab for tab in self._fetch_tabs() if tab.get("type") == "page"]

    def _primary_tab(self) -> dict[str, Any]:
        tabs = self._page_tabs()
        return tabs[0] if tabs else {}

    def _send_cdp(self, method: str, params: dict[str, Any] | None = None, tab: dict[str, Any] | None = None) -> dict[str, Any]:
        page = tab or self._primary_tab()
        ws_url = page.get("webSocketDebuggerUrl", "")
        if not ws_url or websocket is None:
            return {"ok": False, "reason": "cdp_unavailable"}
        try:
            sock = websocket.create_connection(ws_url, timeout=1.0)
            try:
                self._message_id += 1
                sock.send(json.dumps({"id": self._message_id, "method": method, "params": params or {}}))
                while True:
                    message = json.loads(sock.recv())
                    if message.get("id") == self._message_id:
                        if "result" in message:
                            return {"ok": True, "result": message["result"]}
                        return {"ok": False, "reason": message.get("error", {})}
            finally:
                sock.close()
        except Exception as exc:  # pragma: no cover - depends on external browser
            return {"ok": False, "reason": str(exc)}

    def observe(self) -> dict[str, Any]:
        tabs = self._page_tabs()
        primary = tabs[0] if tabs else {}
        cdp_available = False
        if tabs and websocket is not None:
            probe = self._send_cdp(
                "Runtime.evaluate",
                {"expression": "1", "returnByValue": True},
                tab=primary,
            )
            cdp_available = bool(probe.get("ok"))
        return {
            "adapter_mode": "browser_dom_metadata",
            "available": bool(tabs),
            "cdp_available": cdp_available,
            "tab_count": len(tabs),
            "tab_titles": [tab.get("title", "") for tab in tabs[:10]],
            "active_title": primary.get("title", ""),
            "active_url": primary.get("url", ""),
        }

    def navigate(self, url: str) -> bool:
        result = self._send_cdp("Page.navigate", {"url": url})
        return bool(result.get("ok"))

    def evaluate(self, expression: str) -> dict[str, Any]:
        result = self._send_cdp(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        if not result.get("ok"):
            return result
        value = result.get("result", {}).get("result", {}).get("value")
        return {"ok": True, "value": value}

    def snapshot_dom(self) -> dict[str, Any]:
        expression = """
        (() => {
          const escCss = (value) => {
            const raw = String(value || '');
            if (window.CSS && CSS.escape) return CSS.escape(raw);
            return raw.replace(/["\\\\#.:\\[\\]\\s]/g, '\\\\$&');
          };
          const escAttr = (value) => String(value || '').replace(/\\\\/g, '\\\\\\\\').replace(/"/g, '\\"');
          const isVisible = (el, rect) => {
            const style = window.getComputedStyle(el);
            return !!(
              rect.width > 0 &&
              rect.height > 0 &&
              style &&
              style.visibility !== 'hidden' &&
              style.display !== 'none' &&
              Number(style.opacity || '1') > 0
            );
          };
          const nthSelector = (el) => {
            const parts = [];
            let node = el;
            while (node && node.nodeType === Node.ELEMENT_NODE && parts.length < 5) {
              const tag = node.tagName.toLowerCase();
              if (node.id) {
                parts.unshift(`${tag}#${escCss(node.id)}`);
                break;
              }
              let index = 1;
              let sib = node;
              while ((sib = sib.previousElementSibling)) {
                if (sib.tagName === node.tagName) index++;
              }
              parts.unshift(`${tag}:nth-of-type(${index})`);
              node = node.parentElement;
            }
            return parts.join(' > ');
          };
          const selectorCandidates = (el) => {
            const tag = el.tagName.toLowerCase();
            const items = [];
            if (el.id) items.push(`#${escCss(el.id)}`);
            const attrs = [
              ['name', el.getAttribute('name')],
              ['aria-label', el.getAttribute('aria-label')],
              ['placeholder', el.getAttribute('placeholder')],
              ['role', el.getAttribute('role')],
              ['type', el.getAttribute('type')],
              ['href', el.getAttribute('href')],
            ];
            for (const [key, value] of attrs) {
              if (value) items.push(`${tag}[${key}="${escAttr(value)}"]`);
            }
            const stable = nthSelector(el);
            if (stable) items.push(stable);
            items.push(tag);
            return [...new Set(items)].slice(0, 8);
          };
          const items = [];
          const selectors = [
            'input', 'textarea', 'button', 'a', 'select', 'summary',
            '[role]', '[aria-label]', '[placeholder]', '[contenteditable="true"]'
          ];
          const accessibleName = (el) => (
            el.getAttribute('aria-label') ||
            el.getAttribute('alt') ||
            el.getAttribute('title') ||
            el.innerText ||
            el.value ||
            el.getAttribute('placeholder') ||
            ''
          ).slice(0, 240);
          const addElement = (el, query, context) => {
            const rect = el.getBoundingClientRect();
            const selectorsForEl = selectorCandidates(el);
            const text = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || '').slice(0, 240);
            const href = el.getAttribute('href') || '';
            const tag = el.tagName.toLowerCase();
            const role = el.getAttribute('role') || '';
            const disabled = !!(el.disabled || el.getAttribute('aria-disabled') === 'true');
            const visible = isVisible(el, rect);
            const x = Math.round(rect.left + (context.offsetX || 0));
            const y = Math.round(rect.top + (context.offsetY || 0));
            const selector = selectorsForEl[0] || query;
            items.push({
              tag,
              role,
              text,
              accessible_name: accessibleName(el),
              value: ('value' in el ? String(el.value || '').slice(0, 240) : ''),
              id: el.id || '',
              name: el.getAttribute('name') || '',
              placeholder: el.getAttribute('placeholder') || '',
              aria_label: el.getAttribute('aria-label') || '',
              type: el.getAttribute('type') || '',
              href,
              selector,
              selectors: selectorsForEl,
              frame_path: context.framePath || '',
              shadow_path: context.shadowPath || '',
              visible,
              enabled: !disabled,
              focused: context.doc.activeElement === el,
              rect: {x, y, w: Math.round(rect.width), h: Math.round(rect.height)},
              box: {x, y, width: Math.round(rect.width), height: Math.round(rect.height)},
            });
          };
          const walkRoot = (root, context) => {
            const seen = new Set();
            for (const query of selectors) {
              for (const el of root.querySelectorAll(query)) {
                if (seen.has(el)) continue;
                seen.add(el);
                addElement(el, query, context);
                if (items.length >= 240) return;
              }
            }
            for (const host of root.querySelectorAll('*')) {
              if (items.length >= 240) return;
              if (host.shadowRoot) {
                const hostSelector = selectorCandidates(host)[0] || nthSelector(host);
                walkRoot(host.shadowRoot, {
                  ...context,
                  shadowPath: [context.shadowPath, hostSelector].filter(Boolean).join(' >> shadow ')
                });
              }
            }
            for (const frame of root.querySelectorAll('iframe,frame')) {
              if (items.length >= 240) return;
              try {
                const doc = frame.contentDocument;
                if (!doc) continue;
                const frameRect = frame.getBoundingClientRect();
                const frameSelector = selectorCandidates(frame)[0] || nthSelector(frame);
                walkRoot(doc, {
                  doc,
                  offsetX: (context.offsetX || 0) + frameRect.left,
                  offsetY: (context.offsetY || 0) + frameRect.top,
                  framePath: [context.framePath, frameSelector].filter(Boolean).join(' > '),
                  shadowPath: ''
                });
              } catch (_err) {
                continue;
              }
            }
          };
          const seen = new Set();
          walkRoot(document, {doc: document, offsetX: 0, offsetY: 0, framePath: '', shadowPath: ''});
          return {
            title: document.title,
            url: location.href,
            ready_state: document.readyState,
            active_selector: document.activeElement ? selectorCandidates(document.activeElement)[0] || '' : '',
            items,
          };
        })()
        """
        result = self.evaluate(expression)
        snapshot = result.get("value", {}) if result.get("ok") else {}
        if not isinstance(snapshot, dict):
            return {}
        for item in snapshot.get("items", []):
            if not isinstance(item, dict):
                continue
            item.setdefault("selector", (item.get("selectors") or [""])[0])
            item["stable_hash"] = _stable_hash(
                {
                    "tag": item.get("tag", ""),
                    "role": item.get("role", ""),
                    "text": item.get("text", ""),
                    "selector": item.get("selector", ""),
                    "frame_path": item.get("frame_path", ""),
                    "shadow_path": item.get("shadow_path", ""),
                }
            )
        snapshot["stable_hash"] = _stable_hash(
            {
                "title": snapshot.get("title", ""),
                "url": snapshot.get("url", ""),
                "items": [item.get("stable_hash", "") for item in snapshot.get("items", []) if isinstance(item, dict)],
            }
        )
        return snapshot

    def rank_selector_candidates(
        self,
        snapshot: dict[str, Any],
        purpose: str,
        query: str = "",
        blocked_terms: list[str] | None = None,
    ) -> list[str]:
        blocked_terms = [_normalize(term) for term in (blocked_terms or []) if term]
        query_tokens = [token for token in re.split(r"\W+", _normalize(query)) if len(token) >= 2][:8]
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()

        for item in snapshot.get("items", []):
            if not isinstance(item, dict):
                continue
            selectors = [selector for selector in item.get("selectors", []) if isinstance(selector, str) and selector]
            if not selectors:
                continue

            blob = " ".join(
                _normalize(str(item.get(key, "")))
                for key in ("text", "tag", "id", "name", "placeholder", "aria_label", "role", "type", "href")
            ).strip()
            if not blob:
                continue

            penalty = 0.0
            if any(term and term in blob for term in blocked_terms):
                penalty += 8.0

            score = 0.0
            tag = _normalize(item.get("tag", ""))
            input_like = tag in {"input", "textarea"}
            button_like = tag in {"button", "a"} or _normalize(item.get("role", "")) == "button"
            type_value = _normalize(item.get("type", ""))
            role = _normalize(item.get("role", ""))
            name = _normalize(item.get("name", ""))
            item_id = _normalize(item.get("id", ""))
            placeholder = _normalize(item.get("placeholder", ""))
            aria_label = _normalize(item.get("aria_label", ""))
            href = _normalize(item.get("href", ""))
            selector_blob = " ".join(_normalize(selector) for selector in selectors)
            omnibox_blob = " ".join(
                part
                for part in (
                    name,
                    item_id,
                    placeholder,
                    aria_label,
                    role,
                    type_value,
                    selector_blob,
                )
                if part
            )

            if purpose == "omnibox":
                if input_like:
                    score += 4.0
                if role in {"combobox", "textbox"}:
                    score += 2.5
                if type_value in {"search", "text", "url"}:
                    score += 2.0
                if any(
                    token in omnibox_blob
                    for token in (
                        "address",
                        "url",
                        "omnibox",
                        "search google or type a url",
                        "search or type web address",
                    )
                ):
                    score += 7.0
                if any(token in omnibox_blob for token in ("toolbar", "location", "address bar")):
                    score += 3.0
                if name == "q":
                    score += 2.0
                if href or button_like:
                    score -= 3.5
                if "search_query" in omnibox_blob or "youtube search" in omnibox_blob:
                    score -= 2.5
                if any(token in blob for token in ("youtube", "results", "video")):
                    score -= 0.75
            elif purpose == "search_field":
                if input_like:
                    score += 4.0
                if type_value in {"search", "text", "url"}:
                    score += 2.0
                if any(token in blob for token in ("search", "find", "query", "q")):
                    score += 5.0
                if any(token in omnibox_blob for token in ("search_query", "search", "find")):
                    score += 2.0
                if "youtube" in blob:
                    score += 1.0
            elif purpose == "link":
                if tag == "a" or href:
                    score += 4.0
                for token in query_tokens:
                    if token in blob:
                        score += 1.5
            elif purpose in {"button", "page_content", "safe_result"}:
                if button_like:
                    score += 3.0
                if href:
                    score += 2.0
                if input_like:
                    score -= 2.0
                for token in query_tokens:
                    if token in blob:
                        score += 1.2
            elif purpose == "modal_dismiss":
                if tag == "button" or role == "button":
                    score += 2.0
                if any(token in blob for token in ("close", "dismiss", "cancel", "ok", "accept")):
                    score += 5.0
                if tag == "a" and role != "button":
                    score -= 2.5

            if query_tokens and purpose in {"safe_result", "button", "page_content", "link"}:
                matched = sum(1 for token in query_tokens if token in blob)
                score += min(4.0, matched * 1.25)

            score -= penalty
            if score <= 0:
                continue

            for selector in selectors:
                if selector in seen:
                    continue
                seen.add(selector)
                scored.append((score, selector))

        scored.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
        return [selector for _, selector in scored[:8]]

    def best_selector(
        self,
        snapshot: dict[str, Any],
        purpose: str,
        query: str = "",
        blocked_terms: list[str] | None = None,
    ) -> str:
        ranked = self.rank_selector_candidates(snapshot, purpose=purpose, query=query, blocked_terms=blocked_terms)
        return ranked[0] if ranked else ""

    def query_selector(self, selector: str) -> dict[str, Any]:
        safe_selector = json.dumps(selector)
        expression = f"""
        (() => {{
          const el = document.querySelector({safe_selector});
          if (!el) return {{ ok: false }};
          const rect = el.getBoundingClientRect();
          return {{
            ok: true,
            text: (el.innerText || el.value || el.getAttribute('aria-label') || '').slice(0, 120),
            value: ('value' in el ? String(el.value || '') : '').slice(0, 160),
            tag: el.tagName.toLowerCase(),
            role: el.getAttribute('role') || '',
            aria_label: el.getAttribute('aria-label') || '',
            placeholder: el.getAttribute('placeholder') || '',
            enabled: !(el.disabled || el.getAttribute('aria-disabled') === 'true'),
            visible: rect.width > 0 && rect.height > 0,
            focused: document.activeElement === el,
            left: rect.left,
            top: rect.top,
            width: rect.width,
            height: rect.height
          }};
        }})()
        """
        result = self.evaluate(expression)
        return result.get("value", {}) if result.get("ok") else {"ok": False}

    def focused_element_info(self, selector: str = "") -> dict[str, Any]:
        safe_selector = json.dumps(selector)
        expression = f"""
        (() => {{
          const el = {safe_selector} ? document.querySelector({safe_selector}) : document.activeElement;
          if (!el) return {{ ok: false, editable: false }};
          const tag = el.tagName.toLowerCase();
          const role = (el.getAttribute('role') || '').toLowerCase();
          const type = (el.getAttribute('type') || '').toLowerCase();
          const editable = (
            tag === 'textarea' ||
            tag === 'input' ||
            role === 'textbox' ||
            role === 'combobox' ||
            el.isContentEditable === true
          );
          return {{
            ok: true,
            editable,
            tag,
            role,
            type,
            id: el.id || '',
            name: el.getAttribute('name') || '',
            aria_label: el.getAttribute('aria-label') || '',
            placeholder: el.getAttribute('placeholder') || '',
            value: ('value' in el ? String(el.value || '') : '').slice(0, 160)
          }};
        }})()
        """
        result = self.evaluate(expression)
        return result.get("value", {}) if result.get("ok") else {"ok": False, "editable": False}

    def click_selector(self, selector: str) -> bool:
        safe_selector = json.dumps(selector)
        expression = f"""
        (() => {{
          const el = document.querySelector({safe_selector});
          if (!el) return false;
          el.scrollIntoView({{block: 'center', inline: 'center'}});
          if (el.disabled || el.getAttribute('aria-disabled') === 'true') return false;
          el.click();
          return {{
            clicked: true,
            focused: document.activeElement === el,
            title: document.title,
            url: location.href,
            text: (el.innerText || el.value || el.getAttribute('aria-label') || '').slice(0, 160)
          }};
        }})()
        """
        result = self.evaluate(expression)
        value = result.get("value")
        return bool(result.get("ok") and (value is True or (isinstance(value, dict) and value.get("clicked"))))

    def type_text(self, text: str, selector: str = "", clear_first: bool = False) -> bool:
        if selector:
            safe_selector = json.dumps(selector)
            safe_text = json.dumps(text)
            expression = f"""
            (() => {{
              const el = document.querySelector({safe_selector});
              if (!el) return false;
              el.focus();
              if ({str(clear_first).lower()} && 'value' in el) {{
                el.value = '';
              }}
              if ('value' in el) {{
                el.value = {safe_text};
                el.dispatchEvent(new Event('input', {{bubbles: true}}));
                el.dispatchEvent(new Event('change', {{bubbles: true}}));
                return true;
              }}
              return false;
            }})()
            """
            result = self.evaluate(expression)
            return bool(result.get("ok") and result.get("value") is True)

        safe_text = json.dumps(text)
        expression = f"""
        (() => {{
          const el = document.activeElement;
          if (!el) return false;
          if ('value' in el) {{
            if ({str(clear_first).lower()}) {{
              el.value = '';
            }}
            el.value = {safe_text};
            el.dispatchEvent(new Event('input', {{bubbles: true}}));
            el.dispatchEvent(new Event('change', {{bubbles: true}}));
            return true;
          }}
          return false;
        }})()
        """
        result = self.evaluate(expression)
        return bool(result.get("ok") and result.get("value") is True)

    def press_enter(self) -> bool:
        expression = """
        (() => {
          const el = document.activeElement;
          if (!el) return false;
          el.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', bubbles: true}));
          el.dispatchEvent(new KeyboardEvent('keyup', {key: 'Enter', code: 'Enter', bubbles: true}));
          return true;
        })()
        """
        result = self.evaluate(expression)
        return bool(result.get("ok") and result.get("value") is True)
