# apps/vacuum_monitor/wizard/ui_cv2.py
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Helpers
# ----------------------------
def clamp_rect(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), max(0, W - 1)))
    y = max(0, min(int(y), max(0, H - 1)))
    w = max(1, min(int(w), max(1, W - x)))
    h = max(1, min(int(h), max(1, H - y)))
    return x, y, w, h


def crop_frame(frame: np.ndarray, rect: Optional[List[int]]) -> np.ndarray:
    if rect is None:
        return frame
    x, y, w, h = rect
    H, W = frame.shape[:2]
    x, y, w, h = clamp_rect(x, y, w, h, W, H)
    return frame[y : y + h, x : x + w].copy()


def _safe_get_window_wh(win: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    try:
        r = cv2.getWindowImageRect(win)
        w, h = int(r[2]), int(r[3])
        if w > 10 and h > 10:
            return w, h
    except Exception:
        pass
    return fallback


def _compute_fit_scale(src_w: int, src_h: int, dst_w: int, dst_h: int) -> float:
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return 1.0
    return min(dst_w / src_w, dst_h / src_h)


def _draw_rect(img: np.ndarray, rect: List[int], color: Tuple[int, int, int], thickness: int = 2) -> None:
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def _text_size(text: str, font, font_scale: float, thickness: int) -> Tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    return int(w), int(h)


def _wrap_text_to_width(text: str, max_w: int, font, font_scale: float, thickness: int) -> List[str]:
    text = text.rstrip("\n")
    if not text:
        return [""]

    raw_lines = text.splitlines() if "\n" in text else [text]
    out: List[str] = []

    for raw in raw_lines:
        s = raw.strip("\r")
        if not s:
            out.append("")
            continue

        words = s.split(" ")
        if len(words) >= 2:
            cur = ""
            for w in words:
                cand = (cur + " " + w).strip() if cur else w
                tw, _ = _text_size(cand, font, font_scale, thickness)
                if tw <= max_w:
                    cur = cand
                else:
                    if cur:
                        out.append(cur)
                    tw2, _ = _text_size(w, font, font_scale, thickness)
                    if tw2 <= max_w:
                        cur = w
                    else:
                        piece = ""
                        for ch in w:
                            cand2 = piece + ch
                            tw3, _ = _text_size(cand2, font, font_scale, thickness)
                            if tw3 <= max_w:
                                piece = cand2
                            else:
                                if piece:
                                    out.append(piece)
                                piece = ch
                        cur = piece
            if cur:
                out.append(cur)
            continue

        piece = ""
        for ch in s:
            cand = piece + ch
            tw, _ = _text_size(cand, font, font_scale, thickness)
            if tw <= max_w:
                piece = cand
            else:
                if piece:
                    out.append(piece)
                piece = ch
        if piece:
            out.append(piece)

    return out


def _draw_text_panel(
    canvas: np.ndarray,
    panel_x0: int,
    panel_w: int,
    lines: List[str],
    scroll: int,
    title: str = "VACUUM MONITOR WIZARD",
) -> Tuple[int, int]:
    H, W = canvas.shape[:2]
    x0 = panel_x0
    x1 = min(W, x0 + panel_w)

    cv2.rectangle(canvas, (x0, 0), (x1, H), (25, 25, 25), -1)

    pad = 12
    inner_w = max(10, panel_w - 2 * pad)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.60
    body_scale = 0.52
    thick = 1

    y = 28
    cv2.putText(canvas, title, (x0 + pad, y), font, title_scale, (240, 240, 240), 1, cv2.LINE_AA)
    y += 18
    cv2.line(canvas, (x0 + pad, y), (x1 - pad, y), (70, 70, 70), 1)
    y += 16

    wrapped: List[str] = []
    for ln in lines:
        wrapped.extend(_wrap_text_to_width(ln, inner_w, font, body_scale, thick))

    _, line_h = _text_size("Ag", font, body_scale, thick)
    line_h = int(line_h + 6)
    max_lines_fit = max(1, (H - y - pad) // line_h)

    total = len(wrapped)
    max_scroll = max(0, total - max_lines_fit)
    scroll = max(0, min(int(scroll), int(max_scroll)))

    show_lines = wrapped[scroll : scroll + max_lines_fit]

    for ln in show_lines:
        cv2.putText(canvas, ln, (x0 + pad, y), font, body_scale, (230, 230, 230), 1, cv2.LINE_AA)
        y += line_h

    if total > max_lines_fit:
        hint = f"[J/K] scroll  {scroll}/{max_scroll}  (lines {total})"
        cv2.putText(canvas, hint, (x0 + pad, H - 10), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return max_scroll, total


@dataclass
class _DragState:
    dragging: bool = False
    x0: int = 0
    y0: int = 0
    x1: int = 0
    y1: int = 0


class SetupWizardCV2:
    def __init__(self, sources: Dict[str, object], cam_specs: List[dict], email_cfg: dict, window_name: str = "vacuum_monitor_setup"):
        self.win = window_name
        self.sources_map = sources
        self.cameras = copy.deepcopy(cam_specs)
        
        # Email Config
        self.email_cfg = email_cfg
        if "to" not in self.email_cfg:
            self.email_cfg["to"] = []
        if not isinstance(self.email_cfg["to"], list):
             self.email_cfg["to"] = []

        # 내부 로직용
        self.rois_by_cam: Dict[str, List[Dict[str, Any]]] = {}
        self.deleted_names_pool: Dict[str, List[str]] = {}

        for cam in self.cameras:
            cid = str(cam.get("id", "unknown"))
            rois_list = list(cam.get("rois", []) or [])
            self.rois_by_cam[cid] = rois_list
            self.deleted_names_pool[cid] = []

        self.cam_i = 0
        self.mode = "crop"  # crop | roi | email
        self.drag = _DragState()
        self.pending_rect: Optional[List[int]] = None

        self.text_scroll = 0
        self.show_help = True

        self._last_scale: float = 1.0
        self._last_disp_wh: Tuple[int, int] = (1, 1)
        self._last_img_xy: Tuple[int, int] = (0, 0)
        self._last_img_wh: Tuple[int, int] = (1, 1)
        self._panel_w: int = 360

        self.selected_roi_i: int = 0
        
        # Email Input State
        self.email_input_buf = ""
        self.email_selected_idx = 0

    def _get_current_cam(self) -> Dict[str, Any]:
        if not self.cameras:
            raise RuntimeError("No cameras available")
        self.cam_i = max(0, min(self.cam_i, len(self.cameras) - 1))
        return self.cameras[self.cam_i]

    def _get_current_rois(self) -> List[Dict[str, Any]]:
        cam = self._get_current_cam()
        cam_id = str(cam.get("id", "cam"))
        if cam_id not in self.rois_by_cam:
            self.rois_by_cam[cam_id] = []
        return self.rois_by_cam[cam_id]

    def _read_frame(self) -> np.ndarray:
        cam = self._get_current_cam()
        cap_key = str(cam.get("cap_key", ""))
        source = self.sources_map.get(cap_key)
        
        if source is None:
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(dummy, f"Source not found: {cap_key}", (50, 360), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return dummy

        try:
            frame = source.read()
            if frame is None:
                raise ValueError("Frame is None")
            return frame
        except Exception as e:
            err_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            msg = f"Read Error ({cap_key}): {str(e)}"
            cv2.putText(err_img, msg, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return err_img

    def _mouse_to_disp_xy(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        img_x0, img_y0 = self._last_img_xy
        img_w, img_h = self._last_img_wh

        if x < img_x0 or y < img_y0 or x >= img_x0 + img_w or y >= img_y0 + img_h:
            return None

        sx = x - img_x0
        sy = y - img_y0

        scale = self._last_scale
        if scale <= 1e-9:
            return None

        dx = int(sx / scale)
        dy = int(sy / scale)

        disp_w, disp_h = self._last_disp_wh
        if dx < 0 or dy < 0 or dx >= disp_w or dy >= disp_h:
            return None
        return dx, dy

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if self.mode == "email":
            return 

        pt = self._mouse_to_disp_xy(x, y)
        if pt is None:
            return
        dx, dy = pt

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag.dragging = True
            self.drag.x0, self.drag.y0 = dx, dy
            self.drag.x1, self.drag.y1 = dx, dy

        elif event == cv2.EVENT_MOUSEMOVE and self.drag.dragging:
            self.drag.x1, self.drag.y1 = dx, dy

        elif event == cv2.EVENT_LBUTTONUP and self.drag.dragging:
            self.drag.dragging = False
            self.drag.x1, self.drag.y1 = dx, dy

            x0, y0 = self.drag.x0, self.drag.y0
            x1, y1 = self.drag.x1, self.drag.y1
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])
            w = max(1, x_max - x_min)
            h = max(1, y_max - y_min)

            self.pending_rect = [int(x_min), int(y_min), int(w), int(h)]
            self._apply_pending_rect()

    def _apply_pending_rect(self) -> None:
        if self.pending_rect is None:
            return
        cam = self._get_current_cam()

        if self.mode == "crop":
            cam["crop"] = list(self.pending_rect)
        else:
            rois = self._get_current_rois()
            if not rois:
                self._add_roi(use_pending_rect=True)
            else:
                self.selected_roi_i = max(0, min(self.selected_roi_i, len(rois) - 1))
                rois[self.selected_roi_i]["rect"] = list(self.pending_rect)

        self.pending_rect = None

    def _build_help_lines(self) -> List[str]:
        lines: List[str] = []
        
        # 1. EMAIL MODE HELP
        if self.mode == "email":
            lines.append("=== EMAIL SETTINGS ===")
            lines.append("")
            lines.append("Current Recipients:")
            tos = self.email_cfg.get("to", [])
            if not tos:
                lines.append("  (None)")
            else:
                for i, email in enumerate(tos):
                    mark = ">>" if i == self.email_selected_idx else "  "
                    lines.append(f"{mark} {email}")
            
            lines.append("")
            lines.append("New Email Input:")
            lines.append(f"> {self.email_input_buf}_")
            lines.append("")
            lines.append("Controls:")
            lines.append(" [Typing] Enter email address")
            lines.append(" [Enter] Add to list")
            lines.append(" [Backsp] Delete char")
            lines.append(" [Shift+X] Delete selected item")
            lines.append(" [TAB] Exit Email Mode (Auto-add)")
            lines.append("")
            lines.append(" ** Press TAB to go back")
            lines.append(" ** and then press S to Save")
            return lines

        # 2. CAMERA MODE HELP
        cam = self._get_current_cam()
        cam_id = str(cam.get("id", "cam"))
        cap_key = str(cam.get("cap_key", "unknown"))
        crop = cam.get("crop", None)

        rois = self._get_current_rois()
        self.selected_roi_i = max(0, min(self.selected_roi_i, max(0, len(rois) - 1)))

        lines.append(f"Camera: {self.cam_i+1}/{len(self.cameras)}")
        lines.append(f"  ID={cam_id}")
        lines.append(f"  Source={cap_key}")
        lines.append(f"Mode: {self.mode.upper()}")

        if crop is None:
            lines.append("Crop: (none) -> Draw to set")
        else:
            lines.append(f"Crop: {crop}")

        if self.mode == "roi":
            if crop is None:
                lines.append("!! Set CROP first !!")
            else:
                if not rois:
                    lines.append("ROI: (none) -> Press [N] to add")
                else:
                    sel = rois[self.selected_roi_i]
                    lines.append(f"ROI [{self.selected_roi_i+1}/{len(rois)}]: {sel.get('name')}")
                    lines.append(f"  Rect: {sel.get('rect')}")

        lines.append("")
        lines.append("[Controls]")
        lines.append(" [TAB] Switch Mode / Cam")
        lines.append(" [1] Crop Mode")
        lines.append(" [2] ROI Mode")
        lines.append(" [3] Email Settings")
        lines.append(" Mouse Drag: Set Area")
        lines.append(" [N] Add ROI (RoiMode)")
        lines.append(" [X] Del ROI (RoiMode)")
        lines.append(" [U/M] Prev/Next ROI")
        lines.append(" [J/K] Scroll Text")
        lines.append(" [S] Save & Exit")
        lines.append(" [Q] Cancel")
        return lines

    def _render(self, raw: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        win_w, win_h = _safe_get_window_wh(self.win, fallback=(1280, 720))
        panel_w = int(max(260, min(520, win_w * 0.30)))
        self._panel_w = panel_w

        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        if self.show_help:
            lines = self._build_help_lines()
        else:
            lines = ["(Help hidden) Press H."]
        max_scroll, total_lines = _draw_text_panel(
            canvas, 0, panel_w, lines, self.text_scroll
        )
        self.text_scroll = max(0, min(self.text_scroll, max_scroll))

        img_area_w = max(1, win_w - panel_w)
        img_area_h = max(1, win_h)
        
        if self.mode == "email":
            cx = panel_w + img_area_w // 2
            cy = img_area_h // 2
            
            msgs = [
                "EMAIL CONFIGURATION MODE",
                "------------------------",
                f"Recipients: {len(self.email_cfg.get('to', []))}",
                "",
                "Type email and press ENTER.",
                "Press TAB to finish editing.",
                "",
                f"Input: {self.email_input_buf}"
            ]
            y_start = cy - (len(msgs) * 20)
            for i, m in enumerate(msgs):
                tsize, _ = cv2.getTextSize(m, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx = int(cx - tsize[0] / 2)
                ty = int(y_start + i * 40)
                col = (0, 255, 255) if m.startswith("Input") else (200, 200, 200)
                cv2.putText(canvas, m, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                
            return canvas, (0,0,0,0)

        cam = self._get_current_cam()
        crop = cam.get("crop", None)

        if self.mode == "crop":
            disp = raw.copy()
        else:
            disp = crop_frame(raw, crop).copy()

        disp_h, disp_w = disp.shape[:2]
        self._last_disp_wh = (disp_w, disp_h)

        if self.mode == "crop":
            if crop is not None:
                _draw_rect(disp, crop, (0, 255, 255), 2)
        else:
            rois = self._get_current_rois()
            for i, r in enumerate(rois):
                rect = r.get("rect", None)
                if rect is None: continue
                col = (0, 0, 255) if (i == self.selected_roi_i) else (0, 255, 0)
                _draw_rect(disp, rect, col, 2)
                name = str(r.get("name", f"R{i}"))
                cv2.putText(disp, name, (rect[0], max(0, rect[1]-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        if self.drag.dragging:
            x0, y0, x1, y1 = self.drag.x0, self.drag.y0, self.drag.x1, self.drag.y1
            xm, xM = sorted([x0, x1])
            ym, yM = sorted([y0, y1])
            tmp_rect = [xm, ym, max(1, xM - xm), max(1, yM - ym)]
            col = (255, 200, 0) if self.mode == "crop" else (255, 100, 0)
            _draw_rect(disp, tmp_rect, col, 2)

        scale = _compute_fit_scale(disp_w, disp_h, img_area_w, img_area_h)
        self._last_scale = float(scale)

        show_w = max(1, int(disp_w * scale))
        show_h = max(1, int(disp_h * scale))

        if abs(scale - 1.0) < 1e-5:
            show = disp
        else:
            show = cv2.resize(disp, (show_w, show_h), interpolation=cv2.INTER_AREA)

        x0 = panel_w + max(0, (img_area_w - show_w) // 2)
        y0 = max(0, (img_area_h - show_h) // 2)
        x1 = min(win_w, x0 + show_w)
        y1 = min(win_h, y0 + show_h)

        self._last_img_xy = (x0, y0)
        self._last_img_wh = (x1 - x0, y1 - y0)

        h_s, w_s = y1 - y0, x1 - x0
        if h_s > 0 and w_s > 0:
            canvas[y0:y1, x0:x1] = show[0:h_s, 0:w_s]

        return canvas, (x0, y0, w_s, h_s)

    def _add_roi(self, use_pending_rect: bool = False) -> None:
        rois = self._get_current_rois()
        cam = self._get_current_cam()
        cid = str(cam.get("id", "cam"))

        pool = self.deleted_names_pool.get(cid, [])
        if pool:
            name = pool.pop()
        else:
            n = len(rois) + 1
            name = f"ROI_{n:02d}"

        if use_pending_rect and self.pending_rect is not None:
            rect = list(self.pending_rect)
        else:
            rect = [10, 10, 50, 50]

        rois.append({"name": name, "rect": rect})
        self.selected_roi_i = len(rois) - 1

    def _delete_roi(self) -> None:
        rois = self._get_current_rois()
        if not rois: return
        self.selected_roi_i = max(0, min(self.selected_roi_i, len(rois) - 1))
        removed_roi = rois.pop(self.selected_roi_i)
        removed_name = str(removed_roi.get("name", "unknown"))

        cam = self._get_current_cam()
        cid = str(cam.get("id", "cam"))
        if cid not in self.deleted_names_pool:
            self.deleted_names_pool[cid] = []
        self.deleted_names_pool[cid].append(removed_name)
        
        self.selected_roi_i = max(0, min(self.selected_roi_i, max(0, len(rois) - 1)))

    def _handle_email_input(self, k: int) -> None:
        """이메일 모드일 때 키 입력 처리"""
        # Backspace
        if k == 8:
            if len(self.email_input_buf) > 0:
                self.email_input_buf = self.email_input_buf[:-1]
            return

        # Enter
        if k in (10, 13):
            val = self.email_input_buf.strip()
            if val:
                tos = self.email_cfg.setdefault("to", [])
                if val not in tos:
                    tos.append(val)
                    self.email_cfg["enabled"] = True
                self.email_input_buf = ""
            return

        # Delete Selected: 대문자 X (Shift + x)
        if k == ord('X'):
            tos = self.email_cfg.get("to", [])
            if tos:
                idx = max(0, min(self.email_selected_idx, len(tos)-1))
                tos.pop(idx)
                self.email_selected_idx = max(0, min(idx, len(tos)-1))
            return
            
        # Navigation (List selection)
        # Note: u/m are just characters in email mode, so we use ARROW KEYS if possible,
        # or just rely on 'Shift+X' to delete the 'selected' one.
        # Let's support 'Shift+U' and 'Shift+M' for navigation in email mode to avoid conflict.
        if k in (ord('U'),): # Shift + u
            self.email_selected_idx = max(0, self.email_selected_idx - 1)
            return
        if k in (ord('M'),): # Shift + m
            tos = self.email_cfg.get("to", [])
            self.email_selected_idx = min(len(tos)-1, self.email_selected_idx + 1)
            return

        # Typing (Ascii printable)
        # 32(Space) ~ 126(~)
        if 32 <= k <= 126:
            char = chr(k)
            self.email_input_buf += char

    def _commit_email_buffer(self):
        """Buffer에 남아있는 텍스트를 리스트에 강제 추가 (Tab 등으로 나갈 때)"""
        val = self.email_input_buf.strip()
        if val:
            tos = self.email_cfg.setdefault("to", [])
            if val not in tos:
                tos.append(val)
                self.email_cfg["enabled"] = True
            self.email_input_buf = ""

    def run(self) -> Tuple[List[dict], bool]:
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 1280, 720)
        cv2.setMouseCallback(self.win, self._on_mouse)

        saved_ok = False

        while True:
            try:
                raw = self._read_frame()
                canvas, _ = self._render(raw)
                cv2.imshow(self.win, canvas)
            except Exception as e:
                print(f"Render Error: {e}")
                time.sleep(1)
                continue

            k = cv2.waitKey(10) & 0xFF
            if k == 255: continue

            # === LOGIC SPLIT ===
            
            if self.mode == "email":
                # [EMAIL MODE]
                # S, Q, 1, 2 etc. are treated as TEXT INPUT here.
                # Only TAB is used to exit.
                
                if k == 9: # TAB
                    self._commit_email_buffer() # Auto-save buffer
                    self.mode = "crop" # Exit to camera
                else:
                    self._handle_email_input(k)

            else:
                # [CAMERA MODE] (Crop/ROI)
                
                # Global Controls (only active in Camera mode)
                if k in (27, ord('q'), ord('Q')):
                    saved_ok = False
                    break
                if k in (ord('s'), ord('S')):
                    saved_ok = True
                    break
                
                # Navigation
                if k == 9: # TAB
                    self.cam_i = (self.cam_i + 1) % len(self.cameras)
                    self.selected_roi_i = 0

                # Scroll Help
                if k in (ord('j'), ord('J')): self.text_scroll += 1
                if k in (ord('k'), ord('K')): self.text_scroll = max(0, self.text_scroll - 1)
                if k in (ord('h'), ord('H')): self.show_help = not self.show_help

                # Mode Switch
                if k == ord('1'): self.mode = "crop"
                if k == ord('2'): self.mode = "roi"
                if k == ord('3'): self.mode = "email"

                # ROI Controls
                if self.mode == "roi":
                    if k in (ord('n'), ord('N')): self._add_roi()
                    if k in (ord('x'), ord('X')): self._delete_roi()
                    if k in (ord('u'), ord('U')): self.selected_roi_i = max(0, self.selected_roi_i - 1)
                    if k in (ord('m'), ord('M')): 
                        rois = self._get_current_rois()
                        self.selected_roi_i = min(len(rois)-1, self.selected_roi_i + 1)

        cv2.destroyWindow(self.win)

        if saved_ok:
            for cam in self.cameras:
                cid = str(cam.get("id", "cam"))
                if cid in self.rois_by_cam:
                    cam["rois"] = self.rois_by_cam[cid]
            return self.cameras, True
        else:
            return [], False