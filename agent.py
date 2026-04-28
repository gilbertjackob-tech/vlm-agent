import os
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
import sys
import cv2
import numpy as np
import mss
import torch
import torch.nn.functional as F
import pyautogui
import time
import hashlib
import re
import difflib
import json
import random
import queue
import threading
from datetime import datetime
import pygetwindow as gw
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from paddleocr import PaddleOCR
import logging

# Suppress PaddleOCR's noisy debug logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

class VLMAgent:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        print("🧠 Initializing High-Speed Cognitive Agent (CPU)...")
        self.device = "cpu"
        
        local_model_path = "./clip-local"
        if not os.path.exists(local_model_path):
            print("⏳ Downloading CLIP model ONCE...")
            CLIPModel.from_pretrained(model_id).save_pretrained(local_model_path)
            CLIPProcessor.from_pretrained(model_id).save_pretrained(local_model_path)
            
        self.model = CLIPModel.from_pretrained(local_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(local_model_path)
        
        print("⏳ Initializing PaddleOCR (Lightweight Model)...")
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en')

        # --- CACHES & MEMORY ---
        self.crop_cache = {}    
        self.ocr_cache = {}     
        self.text_embed_cache = {} 
        self.frame_embed_cache = {}      
        self.ocr_call_timeout_seconds = float(os.environ.get("COPILOT_OCR_CALL_TIMEOUT_SECONDS", "2.0"))
        self.ocr_step_budget_seconds = float(os.environ.get("COPILOT_OCR_STEP_BUDGET_SECONDS", "6.0"))
        self.max_parse_ocr_candidates = int(os.environ.get("COPILOT_MAX_PARSE_OCR_CANDIDATES", "40"))
        self.ocr_disabled_until = 0.0
        self.last_parse_health = {}
        
        self.learning_memory = {}       
        self.negative_memory = {} 
        self.spatial_memory = {}  
        self.temporal_scores = {}
        
        # --- TEMPORAL TRACKER ---
        self.active_track = None
        self.focused_window_bbox = None
        self.frame_id = 0
        
        # --- TELEMETRY & SAFETY ---
        self.stats = {
            'track_hits': 0, 'foveated_hits': 0, 'full_hits': 0, 
            'failures': 0, 'total_latency': 0.0, 'frames_processed': 0,
            'time_to_first_detect': None, 'recovery_times': [],
            'max_failure_streak': 0
        }
        self.consecutive_failures = 0
        self.loss_timestamp = None
        
        pyautogui.FAILSAFE = True 
        pyautogui.PAUSE = 0.1
        self.sct = None
        self.monitor = self._detect_primary_monitor()
        self.memory_dir = "memory"
        self.semantic_memory_path = os.path.join(self.memory_dir, "semantic_memory.json")
        self.managed_semantic_memory = False
        self.semantic_memory = self._load_semantic_memory()
        self.last_memory_save = 0.0
        print("✅ Optimized CPU Agent ready.")

    # ==========================================
    # 🛠️ CORE UTILITIES
    # ==========================================
    def _iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]), min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0
        boxAArea, boxBArea = boxA[2] * boxA[3], boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea)

    def _hash_crop(self, crop_array):
        small = cv2.resize(crop_array, (16, 16))
        return hashlib.md5(small.tobytes()).hexdigest()

    def _hash_frame(self, frame_array):
        if frame_array is None or frame_array.size <= 0:
            return ""
        small = cv2.resize(frame_array, (32, 32))
        return hashlib.md5(small.tobytes()).hexdigest()

    def _run_ocr_with_timeout(self, crop_arr, timeout_seconds):
        result_queue = queue.Queue(maxsize=1)

        def worker():
            try:
                result_queue.put(("ok", self.ocr.ocr(crop_arr)))
            except Exception as exc:
                result_queue.put(("error", str(exc)))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            return result_queue.get(timeout=max(0.05, timeout_seconds))
        except queue.Empty:
            self.ocr_disabled_until = time.time() + 30.0
            return "timeout", None

    def _ocr_crop_cached(self, crop_arr, crop_hash, deadline):
        now = time.time()
        if crop_hash in self.ocr_cache:
            cached = self.ocr_cache[crop_hash]
            return str(cached.get("text", "")), "cache_hit", 0.0
        if now < self.ocr_disabled_until:
            return "", "disabled_after_timeout", 0.0
        remaining = deadline - now
        if remaining <= 0:
            return "", "step_budget_exhausted", 0.0

        timeout = min(self.ocr_call_timeout_seconds, remaining)
        started = time.time()
        status, result = self._run_ocr_with_timeout(crop_arr, timeout)
        elapsed = time.time() - started
        if status == "timeout":
            return "", "timeout", elapsed
        if status == "error":
            return "", "error", elapsed

        text_label = ""
        if result and result[0]:
            text_label = " ".join([line[1][0] for line in result[0]]).strip()
        self.ocr_cache[crop_hash] = {"text": text_label, "conf": 1.0}
        return text_label, "ran", elapsed

    def _detect_primary_monitor(self):
        try:
            with mss.mss() as capture:
                monitors = list(getattr(capture, "monitors", []) or [])
                if len(monitors) > 1:
                    return dict(monitors[1])
                if monitors:
                    return dict(monitors[0])
        except Exception:
            pass

        width, height = pyautogui.size()
        return {"left": 0, "top": 0, "width": int(width), "height": int(height)}

    def capture_screen(self):
        monitor = dict(getattr(self, "monitor", None) or self._detect_primary_monitor())
        try:
            with mss.mss() as capture:
                screenshot = np.array(capture.grab(monitor))
        except Exception:
            self.monitor = self._detect_primary_monitor()
            with mss.mss() as capture:
                screenshot = np.array(capture.grab(self.monitor))
        return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    def _fuzzy_match(self, detected_text, target_texts):
        if not detected_text or not target_texts: return 0.0
        det = detected_text.lower()
        best_score = 0.0
        for tgt in target_texts:
            tgt = tgt.lower()
            if tgt in det or det in tgt: return 1.0 
            ratio = difflib.SequenceMatcher(None, tgt, det).ratio()
            if ratio > best_score: best_score = ratio
        return best_score

    def _memory_template(self):
        return {
            "version": 1,
            "labels": {},
            "visuals": {},
            "concepts": {},
            "transitions": [],
        }

    def _load_semantic_memory(self):
        os.makedirs(self.memory_dir, exist_ok=True)
        if not os.path.exists(self.semantic_memory_path):
            return self._memory_template()

        try:
            with open(self.semantic_memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            template = self._memory_template()
            if isinstance(data, dict):
                template.update(data)
            return template
        except Exception as e:
            print(f"⚠️ Memory load warning: {e}")
            return self._memory_template()

    def _save_semantic_memory(self):
        if getattr(self, "managed_semantic_memory", False):
            return
        try:
            os.makedirs(self.memory_dir, exist_ok=True)
            with open(self.semantic_memory_path, "w", encoding="utf-8") as f:
                json.dump(self.semantic_memory, f, indent=2, ensure_ascii=False)
            self.last_memory_save = time.time()
        except Exception as e:
            print(f"⚠️ Memory save warning: {e}")

    def _increment_counter(self, mapping, key, amount=1):
        if not key:
            return
        mapping[key] = mapping.get(key, 0) + amount

    def _normalize_memory_label(self, label):
        text = re.sub(r"\s+", " ", str(label or "").strip().lower())
        return text

    def _best_count_key(self, mapping):
        if not mapping:
            return None
        return max(mapping.items(), key=lambda item: (item[1], item[0]))[0]

    def _collect_node_labels(self, node):
        labels = []

        def visit(current):
            label = str(current.get("label", "")).strip()
            if label and not self._is_placeholder_label(label):
                labels.append(label)
            for child in current.get("children", []):
                visit(child)

        visit(node)
        return labels

    def _collect_visual_ids(self, node):
        visual_ids = []

        def visit(current):
            visual_id = current.get("visual_id")
            if visual_id:
                visual_ids.append(visual_id)
            for child in current.get("children", []):
                visit(child)

        visit(node)
        return visual_ids

    def _infer_concepts_from_labels(self, labels):
        concepts = set()
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".m4v"}
        archive_exts = {".zip", ".rar", ".7z", ".tar", ".gz"}
        document_exts = {".pdf", ".doc", ".docx", ".txt", ".ppt", ".pptx"}
        code_exts = {".py", ".js", ".ts", ".html", ".css", ".json", ".bat", ".ps1"}
        spreadsheet_exts = {".csv", ".xls", ".xlsx"}

        for raw_label in labels:
            label = self._normalize_memory_label(raw_label)
            if not label:
                continue

            _, ext = os.path.splitext(label)
            if ext:
                concepts.add("file")

            if ext in image_exts:
                concepts.update({"image", "file"})
            if ext in video_exts:
                concepts.update({"video", "file"})
            if ext in archive_exts:
                concepts.update({"archive", "file"})
            if ext in document_exts:
                concepts.update({"document", "file"})
            if ext in code_exts:
                concepts.update({"code", "file"})
            if ext in spreadsheet_exts:
                concepts.update({"spreadsheet", "file"})
            if ext == ".py" or "python file" in label:
                concepts.update({"python_file", "code", "file"})
            if ext == ".json":
                concepts.update({"data_file", "file"})
            if "file folder" in label or label == "folder":
                concepts.update({"folder", "container"})
            if "shortcut" in label:
                concepts.update({"shortcut", "file"})
            if "image" in label and "before ui parse" not in label:
                concepts.add("image")
            if "video" in label:
                concepts.add("video")
            if "chrome" in label or "google chrome" in label:
                concepts.update({"chrome", "browser", "application"})
            if "browser" in label:
                concepts.update({"browser", "application"})
            if "file explorer" in label or label == "desktop":
                concepts.update({"folder", "container"})
            if "winrar zip archive" in label:
                concepts.update({"archive", "file"})
            if "pdf" in label:
                concepts.update({"pdf", "document", "file"})

        return sorted(concepts)

    def _infer_concepts_from_node(self, node):
        labels = self._collect_node_labels(node)
        concepts = set(self._infer_concepts_from_labels(labels))

        role = node.get("semantic_role")
        if role == "list_row":
            concepts.add("list_item")
        elif role == "clickable_container":
            concepts.add("tile")
        elif role == "menu_item":
            concepts.add("control")

        if node.get("region") == "main_page":
            concepts.add("content")

        return sorted(concepts)

    def _best_label_for_visual_id(self, visual_id):
        if not visual_id:
            return None
        record = self.semantic_memory.get("visuals", {}).get(visual_id, {})
        label_counts = record.get("label_counts", {})
        return self._best_count_key(label_counts)

    def _node_text(self, node):
        learned_label = str(node.get("learned_label", "")).strip()
        raw_label = str(node.get("label", "")).strip()
        if learned_label and self._is_placeholder_label(raw_label):
            return learned_label
        return raw_label

    def _lookup_learned_memory(self, node):
        concept_votes = {}
        label_votes = {}

        for raw_label in self._collect_node_labels(node):
            norm = self._normalize_memory_label(raw_label)
            if not norm:
                continue
            record = self.semantic_memory.get("labels", {}).get(norm)
            if not record:
                continue
            for concept, count in record.get("concept_counts", {}).items():
                concept_votes[concept] = concept_votes.get(concept, 0) + count
            best_label = record.get("display_label")
            if best_label:
                label_votes[best_label] = label_votes.get(best_label, 0) + record.get("seen_count", 1)

        for visual_id in self._collect_visual_ids(node):
            record = self.semantic_memory.get("visuals", {}).get(visual_id)
            if not record:
                continue
            for concept, count in record.get("concept_counts", {}).items():
                concept_votes[concept] = concept_votes.get(concept, 0) + count
            best_label = self._best_count_key(record.get("label_counts", {}))
            if best_label:
                label_votes[best_label] = label_votes.get(best_label, 0) + record.get("seen_count", 1)

        learned_concepts = [concept for concept, _ in sorted(concept_votes.items(), key=lambda item: (-item[1], item[0]))[:6]]
        learned_label = self._best_count_key(label_votes)
        return learned_label, learned_concepts, sum(concept_votes.values())

    def _apply_memory_to_graph(self, graph):
        def enrich(node):
            learned_label, learned_concepts, memory_votes = self._lookup_learned_memory(node)
            if learned_label and self._is_placeholder_label(node.get("label", "")):
                node["learned_label"] = learned_label
            if learned_concepts:
                node["learned_concepts"] = learned_concepts
                node["memory_votes"] = memory_votes
            for child in node.get("children", []):
                enrich(child)

        for node in graph:
            enrich(node)
        return graph

    def _learn_from_graph(self, graph, source="parse"):
        changed = False

        for node in self._iter_ui_nodes(graph):
            concepts = self._infer_concepts_from_node(node)
            labels = [label for label in self._collect_node_labels(node) if label]
            visual_ids = self._collect_visual_ids(node)

            if not concepts and not labels and not visual_ids:
                continue

            for raw_label in labels:
                norm = self._normalize_memory_label(raw_label)
                if not norm:
                    continue
                record = self.semantic_memory["labels"].setdefault(norm, {
                    "display_label": raw_label,
                    "seen_count": 0,
                    "concept_counts": {},
                    "visual_ids": {},
                    "last_seen": None,
                })
                record["display_label"] = raw_label
                record["seen_count"] += 1
                record["last_seen"] = time.time()
                for concept in concepts:
                    self._increment_counter(record["concept_counts"], concept)
                for visual_id in visual_ids:
                    self._increment_counter(record["visual_ids"], visual_id)
                changed = True

            for visual_id in visual_ids:
                record = self.semantic_memory["visuals"].setdefault(visual_id, {
                    "seen_count": 0,
                    "concept_counts": {},
                    "label_counts": {},
                    "last_seen": None,
                })
                record["seen_count"] += 1
                record["last_seen"] = time.time()
                for concept in concepts:
                    self._increment_counter(record["concept_counts"], concept)
                for raw_label in labels:
                    if not self._is_placeholder_label(raw_label):
                        self._increment_counter(record["label_counts"], raw_label)
                changed = True

            for concept in concepts:
                concept_record = self.semantic_memory["concepts"].setdefault(concept, {
                    "seen_count": 0,
                    "labels": {},
                    "last_seen": None,
                })
                concept_record["seen_count"] += 1
                concept_record["last_seen"] = time.time()
                for raw_label in labels:
                    if not self._is_placeholder_label(raw_label):
                        self._increment_counter(concept_record["labels"], raw_label)
                changed = True

        if changed:
            self._save_semantic_memory()
        return graph

    def _remember_transition(self, clicked_node, graph_after):
        transition = {
            "timestamp": time.time(),
            "clicked_label": self._node_text(clicked_node),
            "clicked_role": clicked_node.get("semantic_role"),
            "clicked_region": clicked_node.get("region"),
            "clicked_concepts": clicked_node.get("learned_concepts", []),
            "after_regions": [node.get("semantic_role") for node in graph_after],
            "after_labels": [self._node_text(node) for node in graph_after[:8]],
        }
        history = self.semantic_memory.setdefault("transitions", [])
        history.append(transition)
        self.semantic_memory["transitions"] = history[-400:]
        self._save_semantic_memory()
    
    def focus_window(self, window_title):
        """Locks the agent's vision to a specific application window."""
        try:
            title_query = window_title.lower().strip()
            windows = [
                win for win in gw.getAllWindows()
                if getattr(win, "title", None) and title_query in win.title.lower()
            ]
            if windows:
                win = windows[0]
                if getattr(win, "isMinimized", False):
                    try:
                        win.restore()
                        time.sleep(0.2)
                    except Exception as restore_error:
                        print(f"âš ï¸ Window restore warning: {restore_error}")
                try:
                    win.activate()
                    time.sleep(0.2)
                except Exception as activate_error:
                    print(f"âš ï¸ Window activation warning: {activate_error}")
                self.focused_window_bbox = (win.left, win.top, win.width, win.height)
                print(f"🪟 Focused window: '{win.title}' at {self.focused_window_bbox}")
                return True
            else:
                print(f"⚠️ Could not find window containing '{window_title}'")
                self.focused_window_bbox = None
                return False
        except Exception as e:
            print(f"⚠️ Window focus failed: {e}")
            self.focused_window_bbox = None
            return False

    def clear_app_context(self):
        """Returns vision to full-screen mode."""
        self.focused_window_bbox = None
        print("👁️ Vision returned to full-screen monitor.")

    # ==========================================
    # 🧠 SEMANTIC PARSING & MEMORY
    # ==========================================
    def infer_target_type(self, prompts):
        text = " ".join(prompts).lower()
        if "icon" in text or "logo" in text: return "icon"
        if "search" in text or "bar" in text or "field" in text or "input" in text: return "text_field"
        if "button" in text or "submit" in text: return "button"
        if "container" in text or "window" in text or "panel" in text: return "container"
        return "any"

    def classify_region(self, w, h):
        aspect = w / float(h + 1e-5)
        if w < 120 and h < 120 and 0.5 < aspect < 2.0: return "icon"
        elif aspect >= 3.0 and w >= 150: return "text_field"
        elif 1.0 <= aspect < 4.0 and 40 <= w < 300: return "button"
        elif w >= 300 and h >= 150: return "container"
        else: return "unknown"

    def _build_list_rows(self, children, container_box):
        """Groups horizontally aligned cells into list rows for dense table-like UIs."""
        if not children:
            return [], children

        left = container_box["x"]
        top = container_box["y"]
        width = container_box["width"]
        height = container_box["height"]
        bottom = top + height

        eligible = []
        leftovers = []
        for child in children:
            box = child.get("box", {})
            center = child.get("center", {})
            cx = center.get("x", 0)
            cy = center.get("y", 0)
            cw = box.get("width", 0)
            ch = box.get("height", 0)

            is_scrollbar = cw <= 18 and ch >= 120
            is_too_large = cw >= width * 0.85 and ch >= height * 0.55
            is_top_chrome = cy <= top + 18
            if is_scrollbar or is_too_large or is_top_chrome:
                leftovers.append(child)
                continue
            eligible.append(child)

        if len(eligible) < 4:
            return [], children

        eligible.sort(key=lambda node: node["center"]["y"])
        median_height = int(np.median([max(12, node["box"]["height"]) for node in eligible]))
        y_tolerance = max(14, min(28, int(median_height * 0.75)))

        rows = []
        current = [eligible[0]]
        for node in eligible[1:]:
            if abs(node["center"]["y"] - current[-1]["center"]["y"]) <= y_tolerance:
                current.append(node)
            else:
                rows.append(current)
                current = [node]
        rows.append(current)

        header_tokens = {"name", "status", "date modified", "type", "size"}
        grouped_ids = set()
        row_nodes = []

        for idx, row in enumerate(rows):
            row = sorted(row, key=lambda node: node["center"]["x"])
            x_values = [node["center"]["x"] for node in row]
            y_values = [node["center"]["y"] for node in row]
            min_x = min(node["box"]["x"] for node in row)
            min_y = min(node["box"]["y"] for node in row)
            max_x = max(node["box"]["x"] + node["box"]["width"] for node in row)
            max_y = max(node["box"]["y"] + node["box"]["height"] for node in row)
            span_x = max_x - min_x
            span_y = max_y - min_y

            textish = [node for node in row if node.get("label") and node["label"] != f"[{node['type'].upper()}]"]
            if len(row) < 3 or span_x < max(260, int(width * 0.22)) or span_y > 90:
                leftovers.extend(row)
                continue

            labels = [node["label"].strip().lower() for node in textish]
            is_header = sum(1 for label in labels if label in header_tokens) >= 2

            label_source = next(
                (
                    node["label"] for node in row
                    if node["type"] != "icon" and not re.fullmatch(r"\d{1,2}:\d{2}", node["label"].strip())
                ),
                row[0]["label"]
            )
            semantic_role = "list_header" if is_header else "list_row"
            row_label = f"Header: {label_source}" if is_header else f"Row: {label_source}"
            visual_ids = self._visual_ids_for_children(row)

            row_nodes.append({
                "id": f"row_{idx}",
                "label": row_label[:120],
                "box": {
                    "x": max(left, min_x - 12),
                    "y": max(top, min_y - 6),
                    "width": min(width, max_x - min_x + 24),
                    "height": min(bottom - max(top, min_y - 6), max_y - min_y + 12),
                },
                "type": "container",
                "center": {
                    "x": int(sum(x_values) / len(x_values)),
                    "y": int(sum(y_values) / len(y_values)),
                },
                "semantic_role": semantic_role,
                "visual_id": visual_ids[0] if visual_ids else "",
                "visual_ids": visual_ids,
                "children": row,
            })
            grouped_ids.update(node["id"] for node in row)

        for child in eligible:
            if child["id"] not in grouped_ids:
                leftovers.append(child)

        row_nodes.sort(key=lambda node: node["center"]["y"])
        leftovers.sort(key=lambda node: (node["center"]["y"], node["center"]["x"]))
        return row_nodes, leftovers

    def _is_placeholder_label(self, label):
        text = str(label or "").strip()
        return not text or (text.startswith("[") and text.endswith("]"))

    def _node_box_tuple(self, node):
        box = node.get("box", {})
        return (
            int(box.get("x", 0)),
            int(box.get("y", 0)),
            int(box.get("width", 0)),
            int(box.get("height", 0)),
        )

    def _node_bounds(self, node):
        x, y, w, h = self._node_box_tuple(node)
        return x, y, x + w, y + h

    def _node_area(self, node):
        _, _, w, h = self._node_box_tuple(node)
        return max(0, w) * max(0, h)

    def _node_center_tuple(self, node):
        center = node.get("center", {})
        return int(center.get("x", 0)), int(center.get("y", 0))

    def _merge_nodes_box(self, nodes, pad_x=0, pad_y=0, clamp_box=None):
        min_x = min(node["box"]["x"] for node in nodes) - pad_x
        min_y = min(node["box"]["y"] for node in nodes) - pad_y
        max_x = max(node["box"]["x"] + node["box"]["width"] for node in nodes) + pad_x
        max_y = max(node["box"]["y"] + node["box"]["height"] for node in nodes) + pad_y

        if clamp_box:
            clamp_x = clamp_box["x"]
            clamp_y = clamp_box["y"]
            clamp_right = clamp_box["x"] + clamp_box["width"]
            clamp_bottom = clamp_box["y"] + clamp_box["height"]
            min_x = max(clamp_x, min_x)
            min_y = max(clamp_y, min_y)
            max_x = min(clamp_right, max_x)
            max_y = min(clamp_bottom, max_y)

        return {
            "x": int(min_x),
            "y": int(min_y),
            "width": int(max(1, max_x - min_x)),
            "height": int(max(1, max_y - min_y)),
        }

    def _label_for_children(self, children, default_label="Item"):
        candidates = []
        for child in children:
            label = str(child.get("label", "")).strip()
            if self._is_placeholder_label(label):
                label = self._best_label_for_visual_id(child.get("visual_id")) or ""
                if not label:
                    continue
            if re.fullmatch(r"\d{1,2}:\d{2}", label):
                continue
            candidates.append(label)

        if candidates:
            candidates.sort(key=lambda text: (-len(text), text))
            return candidates[0]
        if any(child.get("type") == "icon" for child in children):
            return "[ICON]"
        return default_label

    def _visual_ids_for_children(self, children):
        visual_ids = []
        seen = set()
        for child in children:
            child_ids = []
            if child.get("visual_id"):
                child_ids.append(child.get("visual_id"))
            child_ids.extend(child.get("visual_ids", []) or [])
            for visual_id in child_ids:
                if visual_id and visual_id not in seen:
                    seen.add(visual_id)
                    visual_ids.append(visual_id)
        return visual_ids

    def _make_container_node(self, node_id, semantic_role, children, region, clamp_box, label=None, pad_x=8, pad_y=6):
        ordered_children = sorted(children, key=lambda node: (node["center"]["y"], node["center"]["x"]))
        box = self._merge_nodes_box(ordered_children, pad_x=pad_x, pad_y=pad_y, clamp_box=clamp_box)
        center_x = int(sum(child["center"]["x"] for child in ordered_children) / len(ordered_children))
        center_y = int(sum(child["center"]["y"] for child in ordered_children) / len(ordered_children))
        visual_ids = self._visual_ids_for_children(ordered_children)
        return {
            "id": node_id,
            "label": label or self._label_for_children(ordered_children),
            "box": box,
            "type": "container",
            "center": {"x": center_x, "y": center_y},
            "semantic_role": semantic_role,
            "region": region,
            "visual_id": visual_ids[0] if visual_ids else "",
            "visual_ids": visual_ids,
            "children": ordered_children,
        }

    def _node_is_noise(self, node, region_box=None):
        x, y, w, h = self._node_box_tuple(node)
        if w <= 4 or h <= 4:
            return True
        if w <= 18 and h >= 120:
            return True
        if self._is_probable_tooltip(node):
            return True
        if region_box:
            region_area = max(1, region_box["width"] * region_box["height"])
            if (w * h) >= region_area * 0.70:
                return True
        return False

    def _is_probable_tooltip(self, node):
        label = str(node.get("label", "")).strip()
        x, y, w, h = self._node_box_tuple(node)
        if self._is_placeholder_label(label):
            return False
        if len(label) < 18:
            return False
        word_count = len(label.split())
        looks_sentence = word_count >= 5 or "." in label
        return looks_sentence and 180 <= w <= 700 and 30 <= h <= 180

    def _build_connected_groups(self, nodes, should_link):
        groups = []
        visited = set()

        for idx, node in enumerate(nodes):
            if idx in visited:
                continue
            stack = [idx]
            visited.add(idx)
            component = []

            while stack:
                current_idx = stack.pop()
                current = nodes[current_idx]
                component.append(current)

                for other_idx, other in enumerate(nodes):
                    if other_idx in visited:
                        continue
                    if should_link(current, other):
                        visited.add(other_idx)
                        stack.append(other_idx)

            groups.append(component)

        return groups

    def _build_vertical_menu_items(self, nodes, region_box, region_name):
        usable = [node for node in nodes if not self._node_is_noise(node, region_box)]
        if not usable:
            return []
        return self._build_menu_items(usable, region_box, region_name, orientation="vertical")

    def _build_horizontal_menu_items(self, nodes, region_box, region_name):
        usable = [node for node in nodes if not self._node_is_noise(node, region_box)]
        if not usable:
            return []
        return self._build_menu_items(usable, region_box, region_name, orientation="horizontal")

    def _build_menu_items(self, nodes, region_box, region_name, orientation="horizontal"):
        def is_anchor(node):
            label = str(node.get("label", "")).strip()
            if not self._is_placeholder_label(label):
                return True
            if node.get("type") in {"button", "text_field"}:
                return True
            _, _, w, h = self._node_box_tuple(node)
            return node.get("type") == "icon" and w >= 20 and h >= 20

        anchors = [node for node in nodes if is_anchor(node)]
        if not anchors:
            anchors = nodes[:]

        if orientation == "vertical":
            anchors.sort(key=lambda node: (node["center"]["y"], node["center"]["x"]))
        else:
            anchors.sort(key=lambda node: (node["center"]["y"], node["center"]["x"]))

        groups = [{"anchor": anchor, "children": [anchor]} for anchor in anchors]
        assigned_ids = {anchor["id"] for anchor in anchors}

        def attachment_score(anchor, node):
            ax1, ay1, ax2, ay2 = self._node_bounds(anchor)
            bx1, by1, bx2, by2 = self._node_bounds(node)
            a_w, a_h = max(1, ax2 - ax1), max(1, ay2 - ay1)
            b_w, b_h = max(1, bx2 - bx1), max(1, by2 - by1)
            horizontal_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
            vertical_overlap = max(0, min(ay2, by2) - max(ay1, by1))
            gap_x = max(0, max(ax1 - bx2, bx1 - ax2))
            gap_y = max(0, max(ay1 - by2, by1 - ay2))

            if orientation == "vertical":
                same_row = abs(anchor["center"]["y"] - node["center"]["y"]) <= max(20, min(a_h, b_h) + 10)
                side_attachment = same_row and gap_x <= 34 and vertical_overlap >= min(a_h, b_h) * 0.35
                near_row = same_row and gap_x <= 90
                if side_attachment:
                    return gap_x + gap_y * 0.2
                if near_row and node.get("type") == "icon":
                    return gap_x + 12
                return None

            same_band = abs(anchor["center"]["y"] - node["center"]["y"]) <= max(22, min(a_h, b_h) + 8)
            side_attachment = same_band and gap_x <= 30 and vertical_overlap >= min(a_h, b_h) * 0.35
            tight_overlap = gap_x <= 10 and gap_y <= 10
            if side_attachment:
                return gap_x + gap_y * 0.2
            if tight_overlap and (node.get("type") == "icon" or self._is_placeholder_label(node.get("label"))):
                return 5
            return None

        for node in nodes:
            if node["id"] in assigned_ids:
                continue

            best_idx = None
            best_score = None
            for idx, group in enumerate(groups):
                score = attachment_score(group["anchor"], node)
                if score is None:
                    continue
                if best_score is None or score < best_score:
                    best_idx = idx
                    best_score = score

            if best_idx is not None:
                groups[best_idx]["children"].append(node)
                assigned_ids.add(node["id"])

        items = []
        for idx, group in enumerate(groups):
            children = sorted(group["children"], key=lambda node: (node["center"]["y"], node["center"]["x"]))
            anchor = group["anchor"]
            label = anchor.get("label") if not self._is_placeholder_label(anchor.get("label")) else self._label_for_children(children, default_label=f"{region_name} item")
            items.append(
                self._make_container_node(
                    f"{region_name}_item_{idx}",
                    "menu_item",
                    children,
                    region_name,
                    region_box,
                    label=label,
                    pad_x=10,
                    pad_y=8 if orientation == "horizontal" else 6,
                )
            )

        for node in nodes:
            if node["id"] in assigned_ids:
                continue
            items.append(
                self._make_container_node(
                    f"{region_name}_loose_{node['id']}",
                    "menu_item",
                    [node],
                    region_name,
                    region_box,
                    label=self._label_for_children([node], default_label=f"{region_name} item"),
                    pad_x=8,
                    pad_y=6,
                )
            )

        items.sort(key=lambda node: (node["center"]["y"], node["center"]["x"]))
        return items

    def _build_icon_tiles(self, nodes, region_box, region_name):
        usable = [node for node in nodes if not self._node_is_noise(node, region_box)]
        if not usable:
            return []

        def is_anchor(node):
            x, y, w, h = self._node_box_tuple(node)
            if node.get("type") == "icon":
                return True
            if self._is_placeholder_label(node.get("label")):
                return True
            if h >= 46 and w >= 46:
                return True
            if w * h >= 3200:
                return True
            return False

        anchors = [node for node in usable if is_anchor(node)]
        if not anchors:
            anchors = usable[:]

        anchors.sort(key=lambda node: (node["center"]["y"], node["center"]["x"]))
        groups = [{"anchor": anchor, "children": [anchor]} for anchor in anchors]
        assigned_ids = {anchor["id"] for anchor in anchors}

        def attachment_score(anchor, node):
            ax1, ay1, ax2, ay2 = self._node_bounds(anchor)
            bx1, by1, bx2, by2 = self._node_bounds(node)
            a_w, a_h = max(1, ax2 - ax1), max(1, ay2 - ay1)
            b_w, b_h = max(1, bx2 - bx1), max(1, by2 - by1)
            horizontal_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
            vertical_overlap = max(0, min(ay2, by2) - max(ay1, by1))
            gap_x = max(0, max(ax1 - bx2, bx1 - ax2))
            gap_y = max(0, max(ay1 - by2, by1 - ay2))

            below_anchor = by1 >= ay1 and horizontal_overlap >= min(a_w, b_w) * 0.25 and gap_y <= 95 and gap_x <= 36
            right_of_anchor = bx1 >= ax1 and vertical_overlap >= min(a_h, b_h) * 0.40 and gap_x <= 28 and a_w <= 160 and a_h <= 160
            overlapping = horizontal_overlap >= min(a_w, b_w) * 0.25 and vertical_overlap >= min(a_h, b_h) * 0.25

            if below_anchor:
                return gap_y + gap_x * 0.5
            if right_of_anchor:
                return gap_x + gap_y * 0.5 + 5
            if overlapping:
                return 2
            return None

        for node in usable:
            if node["id"] in assigned_ids:
                continue

            best_idx = None
            best_score = None
            for idx, group in enumerate(groups):
                score = attachment_score(group["anchor"], node)
                if score is None:
                    continue
                if best_score is None or score < best_score:
                    best_idx = idx
                    best_score = score

            if best_idx is not None:
                groups[best_idx]["children"].append(node)
                assigned_ids.add(node["id"])

        tiles = []
        for idx, group in enumerate(groups):
            children = sorted(group["children"], key=lambda node: (node["center"]["y"], node["center"]["x"]))
            label = self._label_for_children(children, default_label="tile")
            tiles.append(
                self._make_container_node(
                    f"{region_name}_tile_{idx}",
                    "clickable_container",
                    children,
                    region_name,
                    region_box,
                    label=label,
                    pad_x=12,
                    pad_y=10,
                )
            )

        for node in usable:
            if node["id"] in assigned_ids:
                continue
            tiles.append(
                self._make_container_node(
                    f"{region_name}_loose_{node['id']}",
                    "clickable_container",
                    [node],
                    region_name,
                    region_box,
                    label=self._label_for_children([node], default_label="item"),
                    pad_x=8,
                    pad_y=6,
                )
            )

        tiles.sort(key=lambda node: (node["center"]["y"], node["center"]["x"]))
        return tiles

    def _looks_like_list_layout(self, row_nodes, region_box, members):
        if len(row_nodes) < 3:
            return False

        if any(row.get("semantic_role") == "list_header" for row in row_nodes):
            return True

        region_width = max(1, region_box["width"])
        median_row_width = float(np.median([row["box"]["width"] for row in row_nodes]))
        median_row_height = float(np.median([row["box"]["height"] for row in row_nodes]))
        textish_count = sum(1 for node in members if not self._is_placeholder_label(node.get("label")))
        textish_ratio = textish_count / max(1, len(members))

        return median_row_width >= region_width * 0.72 and median_row_height <= 58 and textish_ratio >= 0.55

    def _looks_like_desktop_layout(self, flat_elements):
        if not flat_elements:
            return False

        usable = [node for node in flat_elements if not self._is_probable_tooltip(node)]
        if not usable:
            usable = flat_elements

        screen_w = max(node["box"]["x"] + node["box"]["width"] for node in usable)
        screen_h = max(node["box"]["y"] + node["box"]["height"] for node in usable)

        wide_top_bars = [
            node for node in usable
            if node["box"]["y"] <= screen_h * 0.20
            and node["box"]["width"] >= screen_w * 0.45
            and node["box"]["height"] <= 120
            and len(str(node.get("label", "")).split()) <= 4
        ]

        tall_sidebars = [
            node for node in usable
            if node["box"]["x"] <= screen_w * 0.14
            and node["box"]["height"] >= screen_h * 0.35
            and node["box"]["width"] >= 140
        ]

        top_left_icons = [
            node for node in usable
            if node["center"]["x"] <= screen_w * 0.26 and node["center"]["y"] <= screen_h * 0.45
        ]

        broad_rows = [
            node for node in usable
            if node["box"]["width"] >= 280 and 60 <= node["box"]["height"] <= 160 and node["center"]["x"] <= screen_w * 0.28
        ]

        return len(wide_top_bars) == 0 and len(tall_sidebars) == 0 and len(top_left_icons) >= 8 and len(broad_rows) <= 2

    def _estimate_layout_regions(self, flat_elements):
        usable = [node for node in flat_elements if not self._is_probable_tooltip(node)]
        if not usable:
            usable = flat_elements

        screen_w = max(node["box"]["x"] + node["box"]["width"] for node in usable)
        screen_h = max(node["box"]["y"] + node["box"]["height"] for node in usable)

        if self._looks_like_desktop_layout(usable):
            return {
                "main_page": {"x": 0, "y": 0, "width": screen_w, "height": screen_h},
            }

        top_nodes = [node for node in usable if node["center"]["y"] < screen_h * 0.32 and node["box"]["height"] <= 140]
        top_cut = int(np.percentile([node["box"]["y"] + node["box"]["height"] for node in top_nodes], 82)) if top_nodes else int(screen_h * 0.18)
        top_cut = max(120, min(220, top_cut))

        bottom_nodes = [node for node in usable if node["center"]["y"] > screen_h * 0.94 and node["box"]["height"] <= 120]
        bottom_start = int(np.percentile([node["box"]["y"] for node in bottom_nodes], 25)) if bottom_nodes else screen_h
        bottom_start = max(screen_h - 70, min(screen_h, bottom_start))

        middle_nodes = [node for node in usable if top_cut <= node["center"]["y"] < bottom_start]

        left_nodes = [node for node in middle_nodes if node["center"]["x"] < screen_w * 0.24]
        left_end = int(np.percentile([node["box"]["x"] + node["box"]["width"] for node in left_nodes], 82)) if left_nodes else int(screen_w * 0.16)
        left_end = max(180, min(360, left_end))

        right_nodes = [node for node in middle_nodes if node["center"]["x"] > screen_w * 0.82]
        has_right_panel = len(right_nodes) >= 4
        right_start = int(np.percentile([node["box"]["x"] for node in right_nodes], 18)) if has_right_panel else screen_w
        if has_right_panel:
            right_start = max(int(screen_w * 0.72), min(screen_w - 120, right_start))

        regions = {
            "top_menu": {"x": 0, "y": 0, "width": screen_w, "height": top_cut},
            "left_menu": {"x": 0, "y": top_cut, "width": left_end, "height": max(1, bottom_start - top_cut)},
            "main_page": {"x": left_end, "y": top_cut, "width": max(1, right_start - left_end), "height": max(1, bottom_start - top_cut)},
        }
        if bottom_start < screen_h:
            regions["bottom_menu"] = {"x": 0, "y": bottom_start, "width": screen_w, "height": screen_h - bottom_start}
        if has_right_panel and right_start < screen_w:
            regions["right_menu"] = {"x": right_start, "y": top_cut, "width": screen_w - right_start, "height": max(1, bottom_start - top_cut)}
            regions["main_page"]["width"] = max(1, right_start - left_end)

        return regions

    def _get_heatmap(self, target_id, grid_size=20):
        if target_id not in self.spatial_memory:
            self.spatial_memory[target_id] = np.zeros((grid_size, grid_size), dtype=np.float32)
        return self.spatial_memory[target_id]

    def update_spatial_memory(self, target_id, center, screen_w, screen_h, grid_size=20):
        heatmap = self._get_heatmap(target_id, grid_size)
        gx = max(0, min(int(center[0] / screen_w * grid_size), grid_size - 1))
        gy = max(0, min(int(center[1] / screen_h * grid_size), grid_size - 1))
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    weight = 1.0 if dx == 0 and dy == 0 else 0.5
                    heatmap[ny, nx] += weight

    def _decay_spatial_memory(self, decay_rate=0.98):
        for target in self.spatial_memory:
            self.spatial_memory[target] *= decay_rate
            self.spatial_memory[target][self.spatial_memory[target] < 0.05] = 0.0

    def get_text_embedding(self, text):
        if text in self.text_embed_cache: return self.text_embed_cache[text]
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_img_inputs = self.processor(images=[dummy_image], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=dummy_img_inputs.pixel_values)
            emb = outputs.text_embeds
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            
        self.text_embed_cache[text] = emb
        return emb

    # ==========================================
    # 👁️ VISION ENGINE (EXTRACTION & FUSION)
    # ==========================================
    def extract_candidates(self, bgr_frame, target_type="any", max_boxes=80, scale=0.5):
        small_frame = cv2.resize(bgr_frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 20, 80)
        dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            
            if w < 5 or h < 5: continue
            region_type = self.classify_region(w, h)
            
            raw_candidates.append({'box': (x, y, w, h), 'type': region_type, 'area': w * h})
                
        raw_candidates.sort(key=lambda c: c['area'], reverse=True)
        candidates = []
        for cand in raw_candidates:
            keep = True
            for kept_cand in candidates:
                if self._iou(cand['box'], kept_cand['box']) > 0.5:
                    keep = False
                    break
            if keep:
                candidates.append(cand)
                if len(candidates) >= max_boxes: break
        return candidates

    def find_ui_element(self, bgr_frame, prompts, target_id, zone_bias=None, threshold=0.55, provided_candidates=None):
        dynamic_max_boxes = 50 if target_id in self.learning_memory else 80
        target_type = self.infer_target_type(prompts)
        candidates = provided_candidates if provided_candidates else self.extract_candidates(bgr_frame, target_type=target_type, max_boxes=dynamic_max_boxes)
        
        if not candidates: return None, 0.0, None, None, None, [], {}
        screen_h, screen_w = bgr_frame.shape[:2]

        max_heat_val, grid_size, heatmap = 0.0, 20, None
        if target_id in self.spatial_memory:
            heatmap = self._get_heatmap(target_id)
            max_heat_val = heatmap.max() + 1e-5
            grid_size = heatmap.shape[0]

        for cand in candidates:
            x, y, w, h = cand['box']
            center = (x + w//2, y + h//2)
            
            heat_score = 0.0
            if heatmap is not None:
                gx = min(int(center[0] / screen_w * grid_size), grid_size - 1)
                gy = min(int(center[1] / screen_h * grid_size), grid_size - 1)
                heat_score = heatmap[gy, gx] / max_heat_val

            area_score = min(0.1, cand['area'] / (screen_w * screen_h))

            type_score = 0.0
            if target_type != "any":
                if cand['type'] == target_type: type_score = 0.15
                elif cand['type'] == "unknown": type_score = 0.05
                else: type_score = -0.10

            cand['cheap_score'] = (heat_score * 0.5) + (area_score * 0.3) + (type_score * 0.2)

        candidates.sort(key=lambda c: c['cheap_score'], reverse=True)
        K_limit = 8 if (heatmap is not None and heatmap.max() > 0.7) else 15
        candidates = candidates[:K_limit]

        keys_to_delete = [k for k in self.frame_embed_cache if self.frame_embed_cache[k]['ttl'] <= 0]
        for k in keys_to_delete: del self.frame_embed_cache[k]
        for k in self.frame_embed_cache: self.frame_embed_cache[k]['ttl'] -= 1

        pos_v = [p for p in prompts if not p.startswith("text:") and not p.startswith("not:")]
        neg_v = [p[4:] for p in prompts if p.startswith("not:") and not p.startswith("not:text:")]
        pos_t = [p[5:] for p in prompts if p.startswith("text:")]
        neg_t = [p[9:] for p in prompts if p.startswith("not:text:")]

        all_texts = pos_v + neg_v
        text_embeds = {t: self.get_text_embedding(t) for t in all_texts}
        use_clip = bool(pos_v or neg_v or target_id in self.learning_memory)

        crops_to_embed, crop_hashes, candidate_embeddings = [], [], []
        pil_image = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        
        for cand in candidates:
            x, y, w, h = cand['box']
            pad = 5
            crop_arr = bgr_frame[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
            c_hash = self._hash_crop(crop_arr) if crop_arr.size > 0 else "empty"
            crop_hashes.append(c_hash)
            
            approx_key = (round((x+w/2)/5)*5, round((y+h/2)/5)*5, w, h)
            
            if c_hash in self.crop_cache:
                candidate_embeddings.append(self.crop_cache[c_hash])
                self.frame_embed_cache[approx_key] = {'embed': self.crop_cache[c_hash], 'ttl': 3}
            elif approx_key in self.frame_embed_cache:
                candidate_embeddings.append(self.frame_embed_cache[approx_key]['embed'])
                self.frame_embed_cache[approx_key]['ttl'] = 3
            elif c_hash != "empty":
                candidate_embeddings.append(None)
                if use_clip: crops_to_embed.append((c_hash, approx_key, pil_image.crop((x-pad, y-pad, x+w+pad, y+h+pad))))
            else:
                candidate_embeddings.append(None)

        if use_clip and crops_to_embed:
            new_hashes, new_approx_keys, new_images = zip(*crops_to_embed)
            inputs = self.processor(images=list(new_images), return_tensors="pt").to(self.device)
            dummy_texts = ["a"] * len(new_images)
            dummy_t_inputs = self.processor(text=dummy_texts, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(pixel_values=inputs.pixel_values, input_ids=dummy_t_inputs.input_ids, attention_mask=dummy_t_inputs.attention_mask)
                new_embeds = outputs.image_embeds
                new_embeds = new_embeds / new_embeds.norm(p=2, dim=-1, keepdim=True)
            
            new_idx = 0
            for i, emb in enumerate(candidate_embeddings):
                if emb is None and crop_hashes[i] in new_hashes:
                    candidate_embeddings[i] = new_embeds[new_idx].unsqueeze(0)
                    self.crop_cache[new_hashes[new_idx]] = candidate_embeddings[i]
                    self.frame_embed_cache[new_approx_keys[new_idx]] = {'embed': candidate_embeddings[i], 'ttl': 3}
                    new_idx += 1

        prelim_scores = []
        image_embeds_tensor = torch.cat([e for e in candidate_embeddings if e is not None]) if use_clip and any(e is not None for e in candidate_embeddings) else None
        
        valid_idx = 0
        for i, cand in enumerate(candidates):
            img_score, penalty = 0.0, 0.0
            if use_clip and image_embeds_tensor is not None and candidate_embeddings[i] is not None:
                if pos_v:
                    pos_stack = torch.cat([text_embeds[t] for t in pos_v])
                    img_score = torch.matmul(pos_stack, image_embeds_tensor[valid_idx:valid_idx+1].t()).max().item()
                if neg_v:
                    neg_stack = torch.cat([text_embeds[t] for t in neg_v])
                    penalty = torch.matmul(neg_stack, image_embeds_tensor[valid_idx:valid_idx+1].t()).max().item()
                valid_idx += 1
            norm_img = max(0.0, min((img_score - penalty - 0.18) / 0.12, 1.0)) if pos_v else 0.0
            prelim_scores.append({'idx': i, 'norm_img': norm_img})

        prelim_scores.sort(key=lambda item: item['norm_img'], reverse=True)
        ocr_candidates = []
        for item in prelim_scores:
            if not pos_v and len(ocr_candidates) < 8: ocr_candidates.append(item['idx'])
            elif 0.30 <= item['norm_img'] <= 0.70 and len(ocr_candidates) < 3: ocr_candidates.append(item['idx'])

        scored_candidates = []
        for item in prelim_scores:
            i = item['idx']
            cand = candidates[i]
            x, y, w, h = cand['box']
            c_hash, c_embed, norm_img = crop_hashes[i], candidate_embeddings[i], item['norm_img']
            
            txt_score, ran_ocr = 0.0, False
            if (pos_t or neg_t) and i in ocr_candidates:
                ran_ocr = True
                if c_hash not in self.ocr_cache:
                    pad = 5
                    crop_arr = bgr_frame[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
                    
                    full_text, conf_sum, w_count = "", 0.0, 0
                    if crop_arr.size > 0:
                        result = self.ocr.ocr(crop_arr) 
                        if result and result[0]:
                            for line in result[0]:
                                full_text += line[1][0] + " "
                                conf_sum += line[1][1] 
                                w_count += 1
                                
                    avg_conf = (conf_sum / w_count) if w_count > 0 else 0.0
                    self.ocr_cache[c_hash] = {"text": full_text.strip(), "conf": avg_conf}
                
                cached_ocr = self.ocr_cache[c_hash]
                pos_txt_match, neg_txt_match = self._fuzzy_match(cached_ocr["text"], pos_t), self._fuzzy_match(cached_ocr["text"], neg_t)
                txt_score = max(0.0, (pos_txt_match - neg_txt_match)) * cached_ocr["conf"]

            type_score = 0.15 if cand['type'] == target_type else (0.05 if cand['type'] == "unknown" else -0.10) if target_type != "any" else 0.0
            area_score = min(0.1, cand['area'] / (screen_w * screen_h))
            center = (x + w//2, y + h//2)
            
            heat_score = 0.0
            if target_id in self.spatial_memory:
                heatmap = self._get_heatmap(target_id)
                gx = min(int(center[0] / screen_w * heatmap.shape[0]), heatmap.shape[0] - 1)
                gy = min(int(center[1] / screen_h * heatmap.shape[0]), heatmap.shape[0] - 1)
                heat_score = heatmap[gy, gx] / (heatmap.max() + 1e-5)
                
            pos_score = 0.0
            if zone_bias == "taskbar":
                if center[1] > screen_h * 0.85: pos_score += 0.3
                else: pos_score -= 0.2
            elif zone_bias == "topbar":
                if center[1] < screen_h * 0.25: pos_score += 0.3 
                elif center[1] > screen_h * 0.80: pos_score -= 0.5 
            elif zone_bias == "center":
                if screen_h * 0.20 <= center[1] <= screen_h * 0.80: pos_score += 0.2

            mem_score, neg_penalty = 0.0, 0.0
            if target_id in self.learning_memory and c_embed is not None:
                for exp in self.learning_memory[target_id]:
                    sim = F.cosine_similarity(c_embed, exp['embedding']).item()
                    dist = np.sqrt((center[0]-exp['center'][0])**2 + (center[1]-exp['center'][1])**2)
                    exp_score = (0.15 * sim + 0.05 * max(0.0, 1.0 - (dist / 800))) * exp['weight']
                    if exp_score > mem_score: mem_score = exp_score

            w_txt, w_img, w_pos, w_mem = 0.0, 0.0, 0.0, 0.0
            if pos_v and pos_t:
                if txt_score > 0.7: w_txt, w_img = 0.55, 0.25
                elif norm_img > 0.7: w_txt, w_img = 0.25, 0.55
                else: w_txt, w_img = 0.40, 0.40
            elif pos_v: w_img = 0.8
            elif pos_t: w_txt = 0.8
            if zone_bias: w_pos = 0.2
            if mem_score > 0: w_mem = 1.0 

            raw_final = max(0.0, min(1.0, (w_txt * txt_score) + (w_img * norm_img) + (w_pos * pos_score) + mem_score - neg_penalty + type_score + area_score + (0.20 * heat_score)))
            self.temporal_scores.setdefault(c_hash, []).append(raw_final)
            self.temporal_scores[c_hash] = self.temporal_scores[c_hash][-3:] 
            
            scored_candidates.append({
                'box': (x, y, w, h), 'center': center, 'hash': c_hash, 'embed': c_embed,
                'final': sum(self.temporal_scores[c_hash]) / len(self.temporal_scores[c_hash]), 'ran_ocr': ran_ocr
            })

        scored_candidates.sort(key=lambda c: c['final'], reverse=True)
        best = scored_candidates[0] if scored_candidates else None
        info = {'status': 'Below Threshold', 'gap': 0.0}
        
        if best:
            info['gap'] = (best['final'] - scored_candidates[1]['final']) if len(scored_candidates) > 1 else best['final']
            if len(scored_candidates) > 1 and info['gap'] < 0.05 and best['final'] < 0.80:
                info['status'] = 'Uncertain (Gap too small)'
                return None, best['final'], None, None, None, scored_candidates, info
            if best['final'] >= threshold:
                info['status'] = 'Confident Match'
                return best['center'], best['final'], best['box'], best['hash'], best['embed'], scored_candidates, info

        return None, (best['final'] if best else 0.0), None, None, None, scored_candidates, info

    # ==========================================
    # ⚡ SMART LOOP (CONTROLLER)
    # ==========================================
    def smart_find(self, target, zone_bias=None, timeout=10, debug=False, log_failures=True, continuous=False):
        prompts = target if isinstance(target, list) else [target]
        target_id = str(prompts) 
        start_time = time.time()
        d_frame = None
        
        self.temporal_scores.clear() 
        stability_tracker = {'hash': None, 'frames': 0}
        early_exit_frames = 0
        self.consecutive_failures = 0
        
        print(f"⏳ Fast Adaptive Search for '{prompts[0]}'...")

        while time.time() - start_time < timeout:
            loop_start = time.time() 
            self.frame_id += 1
            self._decay_spatial_memory(decay_rate=0.99)
            
            clean_frame = self.capture_screen()
            
            offset_x, offset_y = 0, 0
            if getattr(self, 'focused_window_bbox', None):
                wx, wy, ww, wh = self.focused_window_bbox
                x1, y1 = max(0, wx), max(0, wy)
                x2, y2 = min(clean_frame.shape[1], wx + ww), min(clean_frame.shape[0], wy + wh)
                
                clean_frame = clean_frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1 
                
            process_frame = clean_frame.copy()
            frame_h, frame_w = process_frame.shape[:2]
            
            hud_w, hud_h = int(self.monitor["width"] * 0.25), int(self.monitor["height"] * 0.25)
            if debug and not getattr(self, 'focused_window_bbox', None): 
                process_frame[0:hud_h, frame_w - hud_w:frame_w] = 0
            
            coords, score, bbox, b_hash, b_embed = None, 0.0, None, None, None
            all_cands, info = [], {'status': 'Init', 'gap': 0.0}
            current_mode = "SEARCHING"

            if self.consecutive_failures > 8:
                if debug: print("⚠️ Safety Reset! Dropping track and cooling heatmaps.")
                self.active_track = None
                self._decay_spatial_memory(decay_rate=0.50) 
                self.consecutive_failures = 0

            if self.active_track and self.active_track.get('target_id') != target_id: 
                self.active_track = None

            if self.active_track and not coords:
                if self.frame_id - self.active_track['last_full_detect'] < 10:
                    tx, ty, tw, th = self.active_track['bbox']
                    sx1, sy1 = max(0, tx - tw), max(0, ty - th)
                    sx2, sy2 = min(frame_w, tx + 2*tw), min(frame_h, ty + 2*th)
                    search_region = process_frame[sy1:sy2, sx1:sx2]
                    
                    if search_region.shape[0] >= th and search_region.shape[1] >= tw:
                        res = cv2.matchTemplate(search_region, self.active_track['template'], cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        
                        if max_val >= 0.65:
                            nx, ny = sx1 + max_loc[0], sy1 + max_loc[1]
                            track_accepted, sim = False, 0.0
                            
                            if self.frame_id % 3 == 0:
                                pil_crop = Image.fromarray(cv2.cvtColor(process_frame[ny:ny+th, nx:nx+tw], cv2.COLOR_BGR2RGB))
                                inputs = self.processor(images=[pil_crop], return_tensors="pt").to(self.device)
                                dummy_t = self.processor(text=["a"], return_tensors="pt", padding=True).to(self.device)
                                with torch.no_grad():
                                    outputs = self.model(pixel_values=inputs.pixel_values, input_ids=dummy_t.input_ids, attention_mask=dummy_t.attention_mask)
                                    new_embed = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
                                
                                sim = F.cosine_similarity(new_embed, self.active_track['embed']).item()
                                if (max_val > 0.85 and sim >= 0.70) or (max_val >= 0.65 and sim >= 0.80):
                                    track_accepted = True
                                    
                                    # --- THE FIX: Correctly normalize the blended embedding tensor ---
                                    updated_embed = (0.8 * self.active_track['embed']) + (0.2 * new_embed)
                                    self.active_track['embed'] = updated_embed / updated_embed.norm(p=2, dim=-1, keepdim=True)
                            else:
                                track_accepted = True
                                
                            if track_accepted:
                                coords, score, bbox = (nx + tw//2, ny + th//2), max_val, (nx, ny, tw, th)
                                b_hash, b_embed = self.active_track['hash'], self.active_track['embed']
                                info['status'] = f'Hybrid Lock ({max_val:.2f})'
                                self.active_track['bbox'], self.active_track['center'], self.active_track['last_seen'] = bbox, coords, self.frame_id
                                current_mode = "TRACKING"

            if not coords and target_id in self.learning_memory:
                mem_cands = [{'box': (max(0, int(cx-bw/2)-10), max(0, int(cy-bh/2)-10), bw+20, bh+20), 'type': 'unknown', 'area': (bw+20)*(bh+20)} for exp in self.learning_memory[target_id] for cx, cy in [exp['center']] for bw, bh in [exp.get('size', (60, 60))]]
                coords, score, bbox, b_hash, b_embed, all_cands, info = self.find_ui_element(process_frame, prompts, target_id, zone_bias, provided_candidates=mem_cands)
                if score < 0.85: coords = None 

            if not coords and target_id in self.spatial_memory:
                heatmap = self._get_heatmap(target_id)
                if heatmap.max() > 0.2:
                    gy, gx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    hx, hy = int((gx + 0.5) * (frame_w / heatmap.shape[0])), int((gy + 0.5) * (frame_h / heatmap.shape[0]))
                    cw, ch = int(frame_w * (0.25 if heatmap.max() > 0.85 else 0.40)), int(frame_h * (0.25 if heatmap.max() > 0.85 else 0.40))
                    x1, y1 = max(0, hx - cw // 2), max(0, hy - ch // 2)
                    x2, y2 = min(frame_w, hx + cw // 2), min(frame_h, hy + ch // 2)
                    
                    f_c, f_s, f_b, f_h, f_e, f_cands, f_i = self.find_ui_element(process_frame[y1:y2, x1:x2], prompts, target_id, zone_bias)
                    if f_c and f_s >= 0.55:
                        coords, score, bbox, b_hash, b_embed, info = (f_c[0] + x1, f_c[1] + y1), f_s, (f_b[0] + x1, f_b[1] + y1, f_b[2], f_b[3]), f_h, f_e, f_i
                        info['status'] = f"Foveated Lock ({info['status']})"
                        all_cands = [{'box': (c['box'][0] + x1, c['box'][1] + y1, c['box'][2], c['box'][3]), **{k:v for k,v in c.items() if k != 'box'}} for c in f_cands]
                        current_mode = "FOVEATED"

            if not coords:
                coords, score, bbox, b_hash, b_embed, all_cands, info = self.find_ui_element(process_frame, prompts, target_id, zone_bias)
                if coords: current_mode = "FULL SCAN"

            loop_latency = time.time() - loop_start
            self.stats['frames_processed'] += 1
            self.stats['total_latency'] += loop_latency
            
            if coords:
                if self.stats['time_to_first_detect'] is None: self.stats['time_to_first_detect'] = time.time() - start_time
                if self.loss_timestamp is not None:
                    self.stats['recovery_times'].append(time.time() - self.loss_timestamp)
                    self.loss_timestamp = None
                self.consecutive_failures = 0
                if current_mode == "TRACKING": self.stats['track_hits'] += 1
                elif current_mode == "FOVEATED": self.stats['foveated_hits'] += 1
                elif current_mode == "FULL SCAN": self.stats['full_hits'] += 1
            else:
                if self.consecutive_failures == 0 and self.stats['time_to_first_detect'] is not None: self.loss_timestamp = time.time()
                self.consecutive_failures += 1
                self.stats['failures'] += 1
                if self.consecutive_failures > self.stats['max_failure_streak']: self.stats['max_failure_streak'] = self.consecutive_failures

            if debug and all_cands:
                d_frame = clean_frame.copy()
                for c in all_cands: cv2.rectangle(d_frame, (c['box'][0], c['box'][1]), (c['box'][0]+c['box'][2], c['box'][1]+c['box'][3]), (0, 255, 0) if bbox and c['box'] == bbox else (255, 0, 0), 2 if bbox and c['box'] == bbox else 1)
                
                if target_id in self.spatial_memory:
                    hm = self._get_heatmap(target_id)
                    gy, gx = np.unravel_index(np.argmax(hm), hm.shape)
                    hx, hy = int((gx + 0.5) * (frame_w / hm.shape[0])), int((gy + 0.5) * (frame_h / hm.shape[0]))
                    cv2.circle(d_frame, (hx, hy), 40, (0, 165, 255), 2) 

                cv2.rectangle(d_frame, (10, 10), (400, 130), (0, 0, 0), -1)
                avg_lat = (self.stats['total_latency'] / max(1, self.stats['frames_processed'])) * 1000
                total_hits = max(1, self.stats['track_hits'] + self.stats['foveated_hits'] + self.stats['full_hits'])
                cv2.putText(d_frame, f"MODE: {current_mode}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if current_mode == "TRACKING" else (0, 255, 255) if current_mode == "FOVEATED" else (0, 0, 255), 2)
                cv2.putText(d_frame, f"CONF: {score:.2f} | FAIL STREAK: {self.consecutive_failures}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(d_frame, f"LATENCY: {loop_latency*1000:.1f}ms (Avg: {avg_lat:.1f}ms)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(d_frame, f"PIPE: Trk {int(self.stats['track_hits']/total_hits*100)}% | Fov {int(self.stats['foveated_hits']/total_hits*100)}% | Full {int(self.stats['full_hits']/total_hits*100)}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

                cv2.namedWindow("Cognitive Fusion Debug", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Cognitive Fusion Debug", hud_w, hud_h)
                cv2.moveWindow("Cognitive Fusion Debug", self.monitor["left"] + self.monitor["width"] - hud_w, self.monitor["top"])
                cv2.imshow("Cognitive Fusion Debug", cv2.resize(d_frame, (hud_w, hud_h)))
                cv2.waitKey(1)

            if coords and (info['status'].startswith('Confident') or info['status'].startswith('Memory Early Exit') or info['status'].startswith('Stable') or info['status'].startswith('Foveated') or info['status'].startswith('Hybrid')):
                if b_hash == stability_tracker['hash']: stability_tracker['frames'] += 1
                else: stability_tracker['hash'], stability_tracker['frames'] = b_hash, 1

                if (stability_tracker['frames'] >= 3 and score > 0.85) or early_exit_frames >= 2:
                    experiences = self.learning_memory.setdefault(target_id, [])
                    for exp in experiences: exp['weight'] *= 0.85 
                    experiences.append({'embedding': b_embed.cpu().clone(), 'center': coords, 'size': (bbox[2], bbox[3]), 'weight': 1.0})
                    experiences.sort(key=lambda x: x['weight'], reverse=True)
                    self.learning_memory[target_id] = experiences[:5]
                    self.update_spatial_memory(target_id, coords, frame_w, frame_h)

                if not info['status'].startswith('Hybrid Lock'):
                    self.active_track = {
                        'target_id': target_id, 'bbox': bbox, 'center': coords, 'hash': b_hash, 'embed': b_embed,
                        'template': process_frame[max(0, bbox[1]):bbox[1]+bbox[3], max(0, bbox[0]):bbox[0]+bbox[2]].copy(),
                        'last_seen': self.frame_id, 'last_full_detect': self.frame_id, 'confidence': score
                    }

                if not continuous:
                    if debug: cv2.destroyAllWindows()
                    return (coords[0] + offset_x, coords[1] + offset_y)
                    
            time.sleep(0.05) 

        if not continuous:
            if debug: cv2.destroyAllWindows()
            print("❌ Target not found.")
            return None

    # ==========================================
    # 🔍 UI MAPPING
    # ==========================================
    def _build_ui_hierarchy(self, flat_elements):
        if not flat_elements:
            return []

        regions = self._estimate_layout_regions(flat_elements)
        region_nodes = []

        for region_name, region_box in regions.items():
            x1 = region_box["x"]
            y1 = region_box["y"]
            x2 = x1 + region_box["width"]
            y2 = y1 + region_box["height"]

            members = [
                node for node in flat_elements
                if x1 <= node["center"]["x"] <= x2 and y1 <= node["center"]["y"] <= y2
            ]
            if not members:
                continue

            children = []
            if region_name in {"left_menu", "right_menu"}:
                children = self._build_vertical_menu_items(members, region_box, region_name)
            elif region_name in {"top_menu", "bottom_menu"}:
                children = self._build_horizontal_menu_items(members, region_box, region_name)
            else:
                row_nodes, leftovers = self._build_list_rows(members, region_box)
                if self._looks_like_list_layout(row_nodes, region_box, members):
                    filtered_leftovers = []
                    for leftover in leftovers:
                        if leftover["box"]["width"] >= region_box["width"] * 0.45 and 24 <= leftover["box"]["height"] <= 90:
                            continue
                        lx1, ly1, lx2, ly2 = self._node_bounds(leftover)
                        l_area = max(1, (lx2 - lx1) * (ly2 - ly1))
                        overlaps_row = False
                        for row in row_nodes:
                            rx1, ry1, rx2, ry2 = self._node_bounds(row)
                            inter_w = max(0, min(lx2, rx2) - max(lx1, rx1))
                            inter_h = max(0, min(ly2, ry2) - max(ly1, ry1))
                            inter_area = inter_w * inter_h
                            if inter_area / l_area >= 0.55:
                                overlaps_row = True
                                break
                        if not overlaps_row:
                            filtered_leftovers.append(leftover)

                    list_container = self._make_container_node(
                        "main_list",
                        "list_container",
                        row_nodes,
                        region_name,
                        region_box,
                        label=f"List: {row_nodes[0]['label'].replace('Row: ', '').replace('Header: ', '')[:40]}",
                        pad_x=14,
                        pad_y=10,
                    )
                    children.append(list_container)

                    leftover_tiles = self._build_icon_tiles(filtered_leftovers, region_box, region_name)
                    children.extend(leftover_tiles)
                else:
                    children = self._build_icon_tiles(members, region_box, region_name)

            region_label = region_name.replace("_", " ").title()
            region_nodes.append({
                "id": region_name,
                "label": region_label,
                "box": region_box,
                "type": "container",
                "center": {
                    "x": region_box["x"] + region_box["width"] // 2,
                    "y": region_box["y"] + region_box["height"] // 2,
                },
                "semantic_role": region_name,
                "region": region_name,
                "children": children,
            })

        return region_nodes

    def parse_interface(self, output_filename="ui_parsed_map.png"):
        print("\n🔍 Scanning and parsing current interface...")
        
        parse_started = time.time()
        clean_frame = self.capture_screen()
        if getattr(self, 'focused_window_bbox', None):
            wx, wy, ww, wh = self.focused_window_bbox
            x1, y1 = max(0, wx), max(0, wy)
            x2, y2 = min(clean_frame.shape[1], wx + ww), min(clean_frame.shape[0], wy + wh)
            clean_frame = clean_frame[y1:y2, x1:x2]
            
        process_frame = clean_frame.copy()
        raw_debug_frame = clean_frame.copy()
        screen_hash = self._hash_frame(clean_frame)
        candidates = self.extract_candidates(process_frame, target_type="any", max_boxes=150)
        ocr_deadline = time.time() + max(0.0, self.ocr_step_budget_seconds)
        ocr_health = {
            "screen_hash": screen_hash,
            "candidate_count": len(candidates),
            "ocr_candidate_limit": self.max_parse_ocr_candidates,
            "ocr_calls": 0,
            "ocr_cache_hits": 0,
            "ocr_skipped": 0,
            "ocr_timeouts": 0,
            "ocr_errors": 0,
            "ocr_elapsed_seconds": 0.0,
            "ocr_budget_seconds": self.ocr_step_budget_seconds,
        }
        
        if not candidates:
            print("⚠️ No UI elements found.")
            self.last_parse_health = {
                **ocr_health,
                "parse_elapsed_seconds": round(time.time() - parse_started, 6),
                "element_count": 0,
            }
            return []

        parsed_elements = []
        for i, cand in enumerate(candidates):
            x, y, w, h = cand['box']
            ui_type = cand['type']
            
            pad = 5
            crop_arr = process_frame[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
            crop_hash = self._hash_crop(crop_arr) if crop_arr.size > 0 else ""
            
            text_label = ""
            if crop_arr.size > 0:
                if i < self.max_parse_ocr_candidates:
                    text_label, ocr_status, ocr_elapsed = self._ocr_crop_cached(crop_arr, crop_hash, ocr_deadline)
                    if ocr_status == "cache_hit":
                        ocr_health["ocr_cache_hits"] += 1
                    elif ocr_status == "ran":
                        ocr_health["ocr_calls"] += 1
                    elif ocr_status == "timeout":
                        ocr_health["ocr_timeouts"] += 1
                    elif ocr_status == "error":
                        ocr_health["ocr_errors"] += 1
                    else:
                        ocr_health["ocr_skipped"] += 1
                    ocr_health["ocr_elapsed_seconds"] += ocr_elapsed
                else:
                    ocr_health["ocr_skipped"] += 1
            
            display_label = text_label if len(text_label) > 1 else f"[{ui_type.upper()}]"
            
            parsed_elements.append({
                "id": i, 
                "label": display_label, 
                "box": {"x": x, "y": y, "width": w, "height": h}, 
                "type": ui_type,
                "center": {"x": x + w//2, "y": y + h//2},
                "visual_id": crop_hash,
            })
            
            color = (0, 255, 255) if ui_type == "text_field" else (255, 0, 255) if ui_type == "button" else (0, 255, 0)
            cv2.rectangle(raw_debug_frame, (x, y), (x+w, y+h), color, 2)
            
            (text_w, text_h), _ = cv2.getTextSize(f"{i}: {display_label[:15]}", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(raw_debug_frame, (x, y - text_h - 4), (x + text_w, y), (0, 0, 0), -1)
            cv2.putText(raw_debug_frame, f"{i}: {display_label[:15]}", (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        structured_ui = self._build_ui_hierarchy(parsed_elements)
        structured_ui = self.clean_ui_graph(structured_ui)
        structured_ui = self._apply_memory_to_graph(structured_ui)
        structured_ui = self._learn_from_graph(structured_ui, source="parse")
        structured_ui = self._apply_memory_to_graph(structured_ui)

        structured_frame = clean_frame.copy()

        def draw_structured_nodes(nodes, frame):
            color_map = {
                "top_menu": (255, 0, 0),
                "left_menu": (0, 255, 0),
                "right_menu": (0, 180, 255),
                "bottom_menu": (180, 0, 255),
                "main_page": (0, 255, 255),
                "list_container": (0, 255, 255),
                "list_header": (255, 200, 0),
                "list_row": (0, 200, 255),
                "menu_item": (120, 255, 120),
                "clickable_container": (255, 180, 0),
            }

            for node in nodes:
                role = node.get("semantic_role")
                box = node.get("box", {})
                if role:
                    x = box.get("x", 0)
                    y = box.get("y", 0)
                    w = box.get("width", 0)
                    h = box.get("height", 0)
                    color = color_map.get(role, (200, 200, 200))
                    thickness = 3 if role in {"top_menu", "left_menu", "right_menu", "bottom_menu", "main_page", "list_container", "list_header", "list_row"} else 2
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                    label_text = self._node_text(node)
                    if node.get("learned_concepts"):
                        label_text = f"{label_text} [{'|'.join(node['learned_concepts'][:2])}]"
                    label = f"{role}: {label_text[:32]}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    text_y = max(text_h + 4, y)
                    cv2.rectangle(frame, (x, text_y - text_h - 4), (x + text_w, text_y), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                child_nodes = [
                    child for child in node.get("children", [])
                    if child.get("semantic_role") or child.get("type") == "container"
                ]
                if child_nodes:
                    draw_structured_nodes(child_nodes, frame)

        draw_structured_nodes(structured_ui, structured_frame)

        os.makedirs("debug_steps", exist_ok=True)
        save_path_img = os.path.join("debug_steps", output_filename)
        raw_save_path = os.path.join("debug_steps", output_filename.replace(".png", "_raw.png").replace(".jpg", "_raw.jpg"))
        cv2.imwrite(raw_save_path, raw_debug_frame)
        cv2.imwrite(save_path_img, structured_frame)
        
        json_filename = output_filename.replace(".png", ".json").replace(".jpg", ".json")
        save_path_json = os.path.join("debug_steps", json_filename)
        
        state_dump = {
            "timestamp": time.time(),
            "context_window": self.focused_window_bbox if getattr(self, 'focused_window_bbox', None) else "Full Screen",
            "element_count": len(parsed_elements),
            "parse_health": {
                **ocr_health,
                "ocr_elapsed_seconds": round(ocr_health["ocr_elapsed_seconds"], 6),
                "parse_elapsed_seconds": round(time.time() - parse_started, 6),
            },
            "memory_summary": {
                "known_labels": len(self.semantic_memory.get("labels", {})),
                "known_visuals": len(self.semantic_memory.get("visuals", {})),
                "known_concepts": len(self.semantic_memory.get("concepts", {})),
                "transition_count": len(self.semantic_memory.get("transitions", [])),
            },
            "ui_graph": structured_ui  
        }
        
        with open(save_path_json, 'w', encoding='utf-8') as f:
            json.dump(state_dump, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Found {len(candidates)} elements. Saved map to {save_path_img}")
        print(f"💾 Saved Structured UI Graph to {save_path_json}")
        self.last_parse_health = state_dump["parse_health"]
        
        return structured_ui

    # ==========================================
    # 🧠 DETERMINISTIC QUERY ENGINE
    # ==========================================
    def clean_ui_graph(self, graph, inherited_region=None):
        """Annotates the graph with stable regions while preserving the hierarchy."""
        cleaned_graph = []
        for node in graph:
            node_region = node.get('region') or inherited_region
            if not node_region:
                x = node.get('center', {}).get('x', 0)
                y = node.get('center', {}).get('y', 0)
                if y < 130:
                    node_region = 'top_menu'
                elif x < 320:
                    node_region = 'left_menu'
                elif y > 940:
                    node_region = 'bottom_menu'
                else:
                    node_region = 'main_page'

            node['region'] = node_region
            if 'children' in node and node['children']:
                node['children'] = self.clean_ui_graph(node['children'], inherited_region=node_region)
            cleaned_graph.append(node)
        return cleaned_graph

    def query_ui(self, graph, filters):
        """Scores and filters the UI graph based on structured constraints."""
        candidates = []
        
        def traverse(nodes):
            for node in nodes:
                score = 0.0
                max_score = 0.0
                node_label = self._node_text(node)
                
                if 'type' in filters:
                    max_score += 1.0
                    if node.get('type') == filters['type']: score += 1.0
                        
                if 'label_contains' in filters:
                    max_score += 2.0 
                    if filters['label_contains'].lower() in node_label.lower(): 
                        score += 2.0

                if 'concept' in filters:
                    max_score += 1.0
                    if filters['concept'] in node.get('learned_concepts', []):
                        score += 1.0
                        
                if 'semantic_role' in filters:
                    max_score += 1.0
                    if node.get('semantic_role') == filters['semantic_role']: score += 1.0
                        
                if 'region' in filters:
                    max_score += 1.0
                    if node.get('region') == filters['region']: score += 1.0
                        
                if 'y_max' in filters:
                    max_score += 1.0
                    if node.get('center', {}).get('y', 9999) <= filters['y_max']: score += 1.0
                        
                if 'x_max' in filters:
                    max_score += 1.0
                    if node.get('center', {}).get('x', 9999) <= filters['x_max']: score += 1.0
                
                if max_score > 0:
                    node['confidence'] = score / max_score
                    if node['confidence'] > 0.5: 
                        candidates.append(node)
                
                if 'children' in node and node['children']:
                    traverse(node['children'])

        traverse(graph)
        candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return candidates

    def _iter_ui_nodes(self, nodes):
        for node in nodes:
            yield node
            for child in node.get("children", []):
                yield from self._iter_ui_nodes([child])

    def _node_signature(self, node):
        center = node.get("center", {})
        return (
            node.get("semantic_role", ""),
            node.get("region", ""),
            self._node_text(node).strip().lower(),
            int(center.get("x", 0) // 10),
            int(center.get("y", 0) // 10),
        )

    def _is_safe_random_target(self, node, frame_w, frame_h):
        role = node.get("semantic_role")
        label = self._node_text(node).strip().lower()
        region = node.get("region")
        box = node.get("box", {})
        center = node.get("center", {})

        if role not in {"menu_item", "clickable_container", "list_row"}:
            return False

        if box.get("width", 0) < 24 or box.get("height", 0) < 18:
            return False
        if self._is_probable_tooltip(node):
            return False

        banned_terms = {
            "close", "minimize", "maximize", "restore", "exit", "x",
            "back", "forward", "refresh", "more", "sort", "view", "details"
        }
        if label in banned_terms:
            return False

        if region == "top_menu":
            if center.get("y", 0) <= max(48, int(frame_h * 0.08)):
                return False
            if center.get("x", 0) >= frame_w * 0.86:
                return False
            if self._is_placeholder_label(label):
                return False

        if self._is_placeholder_label(label) and region != "main_page":
            return False

        return True

    def collect_random_click_candidates(self, graph):
        if getattr(self, "focused_window_bbox", None):
            _, _, frame_w, frame_h = self.focused_window_bbox
        else:
            frame_w = self.monitor["width"]
            frame_h = self.monitor["height"]

        nodes = list(self._iter_ui_nodes(graph))
        candidates = [node for node in nodes if self._is_safe_random_target(node, frame_w, frame_h)]

        def priority(node):
            region_priority = {
                "main_page": 0,
                "left_menu": 1,
                "right_menu": 2,
                "bottom_menu": 3,
                "top_menu": 4,
            }
            role_priority = {
                "clickable_container": 0,
                "list_row": 1,
                "menu_item": 2,
            }
            return (
                region_priority.get(node.get("region"), 9),
                role_priority.get(node.get("semantic_role"), 9),
                -node.get("box", {}).get("width", 0) * node.get("box", {}).get("height", 0),
            )

        candidates.sort(key=priority)
        return candidates

    def click_ui_node(self, node):
        cx = int(node["center"]["x"])
        cy = int(node["center"]["y"])

        if getattr(self, 'focused_window_bbox', None):
            wx, wy, _, _ = self.focused_window_bbox
            cx += max(0, wx)
            cy += max(0, wy)

        pyautogui.moveTo(cx, cy, duration=0.25)
        pyautogui.click()
        print(f"🖱️ Random Click -> '{self._node_text(node)}' | Region: {node.get('region')} | Target: ({cx}, {cy})")
        return True

    def run_random_ui_exploration(self, rounds=5, initial_wait=3.0, settle_wait=2.0):
        print(f"\n🎲 Starting random UI exploration for {rounds} rounds...")
        print(f"⏳ Waiting {initial_wait:.1f}s before the first parse...")
        time.sleep(initial_wait)

        visited_signatures = set()
        graph = self._apply_memory_to_graph(self.clean_ui_graph(self.parse_interface(output_filename="random_round_0_parse.png")))
        if not graph:
            print("❌ No UI graph available for random exploration.")
            return False

        for round_idx in range(1, rounds + 1):
            candidates = self.collect_random_click_candidates(graph)
            fresh_candidates = [node for node in candidates if self._node_signature(node) not in visited_signatures]
            pool = fresh_candidates if fresh_candidates else candidates

            if not pool:
                print("⚠️ No safe clickable targets were detected.")
                return False

            chosen = random.choice(pool[: min(len(pool), 12)])
            visited_signatures.add(self._node_signature(chosen))

            print(f"\n▶ Random Round {round_idx}/{rounds}")
            print(f"   -> Chosen: '{self._node_text(chosen)}' | Role: {chosen.get('semantic_role')} | Region: {chosen.get('region')}")
            self.click_ui_node(chosen)

            print(f"⏳ Waiting {settle_wait:.1f}s for UI to react...")
            time.sleep(settle_wait)

            graph = self._apply_memory_to_graph(self.clean_ui_graph(self.parse_interface(output_filename=f"random_round_{round_idx}_parse.png")))
            if not graph:
                print("⚠️ Parsing returned no graph after the click.")
                return False
            self._remember_transition(chosen, graph)

        print("\n✅ Random UI exploration completed.")
        return True

    def click_structured(self, filters, fallback_prompts=None, zone_bias=None, debug=True):
        """Executes a parse-query-act cycle, with an optional raw visual fallback."""
        import pyautogui
        print(f"\n⚙️ Executing Structured Query: {filters}")
        
        # 1. Parse & Clean (Ensure parse_interface returns the graph)
        raw_graph = self.parse_interface(output_filename="temp_query_map.png")
        if not raw_graph:
            print("⚠️ Structured execution failed: No UI elements parsed.")
            raw_graph = []
            
        clean_graph = self._apply_memory_to_graph(self.clean_ui_graph(raw_graph))
        
        def count_nodes(nodes):
            return sum(1 + count_nodes(n.get('children', [])) for n in nodes)
            
        total_nodes = count_nodes(clean_graph)
        matches = self.query_ui(clean_graph, filters)
        
        if debug:
            print(f"📊 Total Nodes Parsed (Recursive): {total_nodes}")
            print(f"🏆 Top Query Matches:")
            for i, m in enumerate(matches[:5]):
                print(f"  #{i+1} [Conf: {m.get('confidence', 0):.2f}] -> ID: {m.get('id')} | Label: '{self._node_text(m)}' | Type: {m.get('type')}")
        
        if matches:
            winner = matches[0]
            cx, cy = winner['center']['x'], winner['center']['y']
            
            # --- THE FIX: Convert window-relative coordinates back to absolute screen coordinates ---
            if getattr(self, 'focused_window_bbox', None):
                wx, wy, ww, wh = self.focused_window_bbox
                cx += max(0, wx)
                cy += max(0, wy)
            
            print(f"\n✅ WINNER CHOSEN:")
            print(f"   -> ID: {winner.get('id')}")
            print(f"   -> Label: '{self._node_text(winner)}'")
            print(f"   -> Region: {winner.get('region')}")
            print(f"   -> Absolute Target: ({cx}, {cy})")
            
            pyautogui.moveTo(cx, cy, duration=0.25)
            pyautogui.click()
            return True
        else:
            print("\n⚠️ Structured query failed. No nodes matched constraints.")
            
            if fallback_prompts:
                print(f"🔄 Triggering Vision Fallback for: {fallback_prompts} (Zone: {zone_bias})")
                coords = self.smart_find(fallback_prompts, zone_bias=zone_bias, debug=debug)
                if coords:
                    pyautogui.moveTo(coords[0], coords[1], duration=0.25)
                    pyautogui.click()
                    print(f"🖱️ Fallback Visual Clicked at {coords}")
                    return True
                    
            print("❌ FATAL: Both structured and fallback queries failed.")
            return False
    
    # ==========================================
    # 🚀 EXECUTION LOOP
    # ==========================================
    def execute_plan(self, plan, debug=True):
        print("\n🚀 Starting Fast Cognitive Automation...")
        for idx, step in enumerate(plan):
            action = step[0]
            print(f"\n▶ Step {idx+1}/{len(plan)}: {step}")
            
            if action == "click":
                coords = self.smart_find(step[1], zone_bias=step[3] if len(step)>3 else None, timeout=step[2] if len(step)>2 else 10, debug=debug)
                if coords:
                    pyautogui.moveTo(coords[0], coords[1], duration=0.25)
                    pyautogui.click() 
                    print(f"🖱️ Clicked at {coords}")
                else:
                    print(f"❌ FATAL: Step {idx+1} failed.")
                    return False
                    
            elif action == "click_structured":
                filters = step[1]
                fallback = step[2] if len(step) > 2 else None
                zone = step[3] if len(step) > 3 else None
                
                success = self.click_structured(filters, fallback_prompts=fallback, zone_bias=zone, debug=debug)
                if not success:
                    print(f"❌ FATAL: Step {idx+1} failed.")
                    return False

            elif action == "focus_window":
                success = self.focus_window(step[1])
                if not success:
                    print(f"❌ FATAL: Step {idx+1} failed to focus window '{step[1]}'.")
                    return False
                time.sleep(1.0)

            elif action == "type": 
                pyautogui.write(step[1], interval=0.03)
            elif action == "press": 
                pyautogui.press(step[1])
            elif action == "hotkey": 
                pyautogui.hotkey(*step[1])
            elif action == "wait": 
                time.sleep(step[1])
                
            elif action == "clear_window":
                self.clear_app_context()
            elif action == "parse_ui":
                self.parse_interface()
                
            else:
                print(f"⚠️ Unknown action type: {action}")
                
            time.sleep(0.5) 
            
        print("\n✅ Task completed successfully!")
        return True

    def run_benchmark(self, target, scenario_type, duration=10):
        print(f"\n{'='*60}\n🚀 INITIATING BENCHMARK: {scenario_type.upper()}\n🎯 Target: {target}\n⏱️  Duration: {duration} seconds\n👉 INSTRUCTION: Please perform the '{scenario_type}' action on screen NOW.\n{'='*60}\n")
        
        self.stats = {'track_hits': 0, 'foveated_hits': 0, 'full_hits': 0, 'failures': 0, 'total_latency': 0.0, 'frames_processed': 0, 'time_to_first_detect': None, 'recovery_times': [], 'max_failure_streak': 0}
        self.consecutive_failures = 0
        self.loss_timestamp, self.active_track = None, None
        self.spatial_memory.clear()
        
        self.smart_find(target, timeout=duration, debug=True, continuous=True, log_failures=False)
        cv2.destroyAllWindows()
        
        total_frames = max(1, self.stats['frames_processed'])
        success_frames = total_frames - self.stats['failures']
        success_rate = (success_frames / total_frames) * 100
        avg_lat = (self.stats['total_latency'] / total_frames) * 1000
        track_pct = (self.stats['track_hits'] / max(1, success_frames)) * 100
        fov_pct = (self.stats['foveated_hits'] / max(1, success_frames)) * 100
        full_pct = (self.stats['full_hits'] / max(1, success_frames)) * 100
        avg_recovery = sum(self.stats['recovery_times']) / len(self.stats['recovery_times']) if self.stats['recovery_times'] else 0.0
        tfd = self.stats['time_to_first_detect'] or 0.0
        
        suggestions = []
        if full_pct > 40: suggestions.append("⚠️ Tracking threshold too STRICT. Agent is abandoning tracking and falling back to full scans too often.")
        if self.stats['failures'] > 0 and track_pct > 95 and success_rate < 60: suggestions.append("⚠️ Tracking threshold too LOOSE. Agent is locking onto false positives.")
        if avg_recovery > 1.5: suggestions.append("⚠️ Recovery is SLOW. Foveated crop might be too small.")
        if not suggestions: suggestions.append("✅ System is dialed in. Thresholds appear optimal.")

        print(f"\n📊 --- BENCHMARK RESULTS: {scenario_type.upper()} ---")
        print(f"Avg Latency:           {avg_lat:.1f} ms / frame\nDetection Success Rate: {success_rate:.1f}%\nTime to First Detect:   {tfd:.2f} sec\nAvg Recovery Speed:     {avg_recovery:.2f} sec (across {len(self.stats['recovery_times'])} losses)\nMax Failure Streak:     {self.stats['max_failure_streak']} frames\nPipeline Distribution:  [Track: {track_pct:.1f}% | Foveal: {fov_pct:.1f}% | Full: {full_pct:.1f}%]")
        print("\n🛠️ --- AUTO-TUNING INSIGHTS ---")
        for s in suggestions: print(s)
        
        export_data = {"timestamp": time.time(), "scenario": scenario_type, "target": target, "duration_sec": duration, "metrics": {"avg_latency_ms": round(avg_lat, 2), "success_rate_pct": round(success_rate, 2), "time_to_first_detect_sec": round(tfd, 2), "avg_recovery_sec": round(avg_recovery, 2), "pipeline_pct": {"track": round(track_pct, 2), "foveated": round(fov_pct, 2), "full": round(full_pct, 2)}}, "tuning_suggestions": suggestions}
        os.makedirs("benchmarks", exist_ok=True)
        file_path = f"benchmarks/benchmark_{scenario_type.replace(' ', '_').lower()}.json"
        with open(file_path, "w") as f: json.dump(export_data, f, indent=4)
        print(f"\n💾 Log saved to {file_path}")
