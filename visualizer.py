"""
Tactical Visualization Module — Segmentation-Only Edition (Phase 1 MVP).

Renders:
  • Annotated broadcast frame: translucent zone masks + zone labels (name + side)
    + tracked object bounding boxes with zone tags.
  • 2D Tactical Pitch (Radar View) with player dots from pure segmentation positions.
  • Composite side-by-side dashboard.

No keypoint dots (magenta / green), no homography inlier counts.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    PITCH_LENGTH, PITCH_WIDTH,
    HALF_LINE_X, PITCH_CENTER_Y,
    CENTER_CIRCLE_RADIUS,
    PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN, PENALTY_BOX_Y_MAX,
    GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN, GOAL_BOX_Y_MAX,
    PENALTY_SPOT_DIST,
    COLOR_PLAYER, COLOR_GOALKEEPER, COLOR_REFEREE,
    RADAR_CANVAS_WIDTH,
)
from zone_validator import DetectedZone
from tracker import TrackedObject


# Short display labels for zones
_ZONE_SHORT = {
    "18Yard":                    "18Yd",
    "18Yard Circle":             "18Yd\u25cb",
    "5Yard":                     "5Yd",
    "First Half Central Circle": "CC-L",
    "Second Half Central Circle":"CC-R",
    "First Half Field":          "HalfF-L",
    "Second Half Field":         "HalfF-R",
}

_ZONE_COLORS = {
    "18Yard":                    (0, 165, 255),
    "18Yard Circle":             (255, 100, 0),
    "5Yard":                     (0, 220, 255),
    "First Half Central Circle": (0, 255, 100),
    "Second Half Central Circle":(100, 255, 0),
    "First Half Field":          (50, 180, 50),
    "Second Half Field":         (50, 150, 50),
}


class TacticalVisualizer:
    """Renders broadcast overlays and 2D tactical radar pitch."""

    def __init__(self, radar_width: int = RADAR_CANVAS_WIDTH):
        self.radar_width = radar_width
        self.radar_margin = 25
        self.pitch_pixel_w = self.radar_width - 2 * self.radar_margin
        self.pitch_pixel_h = int(self.pitch_pixel_w * (PITCH_WIDTH / PITCH_LENGTH))
        self.radar_height = self.pitch_pixel_h + 2 * self.radar_margin

        self.scale_x = self.pitch_pixel_w / PITCH_LENGTH
        self.scale_y = self.pitch_pixel_h / PITCH_WIDTH

        self.turf_color  = (36, 128, 40)
        self.stripe_color = (42, 145, 46)
        self.line_color  = (255, 255, 255)
        self.panel_bg    = (24, 28, 32)

    # ------------------------------------------------------------------
    def m2px(self, xm: float, ym: float) -> Tuple[int, int]:
        """Pitch metres → radar canvas pixels."""
        px = int(self.radar_margin + xm * self.scale_x)
        py = int(self.radar_margin + ym * self.scale_y)
        return px, py

    # ------------------------------------------------------------------
    def draw_tactical_pitch(self, tracked_objects: List[TrackedObject]) -> np.ndarray:
        """Draw 2D radar with FIFA markings and player dots."""
        canvas = np.full((self.radar_height, self.radar_width, 3),
                         self.turf_color, dtype=np.uint8)

        # Alternating grass stripes
        stripe_m = PITCH_LENGTH / 10.0
        for i in range(0, 10, 2):
            x1, y1 = self.m2px(i * stripe_m, 0.0)
            x2, y2 = self.m2px((i + 1) * stripe_m, PITCH_WIDTH)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), self.stripe_color, -1)

        # Outer boundary
        cv2.rectangle(canvas, self.m2px(0.0, 0.0),
                      self.m2px(PITCH_LENGTH, PITCH_WIDTH), self.line_color, 2)

        # Halfway line
        cv2.line(canvas, self.m2px(HALF_LINE_X, 0.0),
                 self.m2px(HALF_LINE_X, PITCH_WIDTH), self.line_color, 2)

        # Centre circle & spot
        cc = self.m2px(HALF_LINE_X, PITCH_CENTER_Y)
        cv2.circle(canvas, cc, int(CENTER_CIRCLE_RADIUS * self.scale_x), self.line_color, 2)
        cv2.circle(canvas, cc, 3, self.line_color, -1)

        # Left penalty area & goal box
        cv2.rectangle(canvas, self.m2px(0.0, PENALTY_BOX_Y_MIN),
                      self.m2px(PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX), self.line_color, 2)
        cv2.rectangle(canvas, self.m2px(0.0, GOAL_BOX_Y_MIN),
                      self.m2px(GOAL_BOX_DEPTH, GOAL_BOX_Y_MAX), self.line_color, 2)
        lspot = self.m2px(PENALTY_SPOT_DIST, PITCH_CENTER_Y)
        cv2.circle(canvas, lspot, 3, self.line_color, -1)
        cv2.ellipse(canvas, lspot,
                    (int(CENTER_CIRCLE_RADIUS * self.scale_x),
                     int(CENTER_CIRCLE_RADIUS * self.scale_y)),
                    0, -53, 53, self.line_color, 2)

        # Right penalty area & goal box
        cv2.rectangle(canvas,
                      self.m2px(PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN),
                      self.m2px(PITCH_LENGTH, PENALTY_BOX_Y_MAX), self.line_color, 2)
        cv2.rectangle(canvas,
                      self.m2px(PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN),
                      self.m2px(PITCH_LENGTH, GOAL_BOX_Y_MAX), self.line_color, 2)
        rspot = self.m2px(PITCH_LENGTH - PENALTY_SPOT_DIST, PITCH_CENTER_Y)
        cv2.circle(canvas, rspot, 3, self.line_color, -1)
        cv2.ellipse(canvas, rspot,
                    (int(CENTER_CIRCLE_RADIUS * self.scale_x),
                     int(CENTER_CIRCLE_RADIUS * self.scale_y)),
                    0, 127, 233, self.line_color, 2)

        # Goals
        goal_w = 7.32
        gy_min = PITCH_CENTER_Y - goal_w / 2.0
        gy_max = PITCH_CENTER_Y + goal_w / 2.0
        cv2.rectangle(canvas, self.m2px(-2.0, gy_min),
                      self.m2px(0.0, gy_max), (200, 200, 200), 2)
        cv2.rectangle(canvas, self.m2px(PITCH_LENGTH, gy_min),
                      self.m2px(PITCH_LENGTH + 2.0, gy_max), (200, 200, 200), 2)

        # Player dots
        for obj in tracked_objects:
            if obj.pitch_pt is None:
                continue
            xm, ym = obj.pitch_pt
            if np.isnan(xm) or np.isnan(ym):
                continue

            ppx, ppy = self.m2px(xm, ym)
            color = (COLOR_GOALKEEPER if obj.class_name == "goalkeeper"
                     else COLOR_REFEREE if obj.class_name == "referee"
                     else COLOR_PLAYER)

            cv2.circle(canvas, (ppx, ppy), 9, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(canvas, (ppx, ppy), 7, color, -1, cv2.LINE_AA)

            label = str(obj.track_id) if obj.track_id > 0 else ""
            if label:
                cv2.putText(canvas, label,
                            (ppx - 4 if len(label) == 1 else ppx - 8, ppy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas

    # ------------------------------------------------------------------
    def draw_broadcast_frame(
        self,
        frame: np.ndarray,
        zones: List[DetectedZone],
        matches: list,                  # ignored (keypoints disabled)
        tracked_objects: List[TrackedObject],
    ) -> np.ndarray:
        """
        Draw zone masks + zone text labels + player bounding boxes.
        NO keypoint / synthetic-anchor dots drawn.
        """
        vis     = frame.copy()
        overlay = frame.copy()

        # 1. Translucent zone fills
        for z in zones:
            color = _ZONE_COLORS.get(z.name, (0, 200, 0))
            cv2.fillPoly(overlay, [z.poly], color)
            cv2.polylines(vis, [z.poly], True, color, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.20, vis, 0.80, 0, vis)

        # 2. Zone name + side labels (centroid position)
        for z in zones:
            color = _ZONE_COLORS.get(z.name, (0, 200, 0))
            M = cv2.moments(z.poly)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = int(z.poly[:, 0].mean())
                cy = int(z.poly[:, 1].mean())

            short = _ZONE_SHORT.get(z.name, z.name[:6])
            side_tag = f"[{z.side.upper()[:1]}]" if z.side not in ("unknown", "center") else ""
            label = f"{short}{side_tag}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Small background rectangle for readability
            cv2.rectangle(vis,
                          (cx - tw // 2 - 3, cy - th - 4),
                          (cx + tw // 2 + 3, cy + 4),
                          (20, 20, 20), -1)
            cv2.putText(vis, label, (cx - tw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 3. Tracked objects: bbox + zone tag label
        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            fx, fy = int(obj.foot_pt[0]), int(obj.foot_pt[1])

            color = (COLOR_GOALKEEPER if obj.class_name == "goalkeeper"
                     else COLOR_REFEREE if obj.class_name == "referee"
                     else COLOR_PLAYER)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.circle(vis, (fx, fy), 5, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (fx, fy), 7, (255, 255, 255), 1, cv2.LINE_AA)

            # Build label: #ID ROLE [ZoneShort-Side]
            id_str   = f"#{obj.track_id} " if obj.track_id > 0 else ""
            role_str = {"player": "PLAY", "goalkeeper": "GK", "referee": "REF"}.get(
                obj.class_name, obj.class_name[:4].upper())

            if obj.zones:
                top_zone = obj.zones[0]
                zs = _ZONE_SHORT.get(top_zone, top_zone[:5])
                # Determine side tag from the zone's side attribute (look it up)
                side_char = ""
                for z in zones:
                    if z.name == top_zone and z.side not in ("unknown", "center"):
                        side_char = z.side[0].upper()
                        break
                zone_tag = f" [{zs}-{side_char}]" if side_char else f" [{zs}]"
            else:
                zone_tag = ""

            label = f"{id_str}{role_str}{zone_tag}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(vis,
                          (x1, max(0, y1 - 18)),
                          (x1 + tw + 6, max(18, y1)),
                          color, -1)
            cv2.putText(vis, label, (x1 + 3, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    # ------------------------------------------------------------------
    def create_composite_dashboard(
        self,
        broadcast_img: np.ndarray,
        radar_img: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        pitch_side: str,
        tracked_objects: List[TrackedObject],
        zone_counts: Optional[dict] = None,
        inlier_count: int = 0,          # kept for API compat, not displayed
    ) -> np.ndarray:
        """
        Composite dashboard: broadcast left, telemetry panel right.
        Prominently shows the locked pitch_side so we can verify it never flips.
        """
        target_h = 720
        bh, bw = broadcast_img.shape[:2]
        scaled_bw = int(bw * (target_h / bh))
        scaled_broadcast = cv2.resize(broadcast_img, (scaled_bw, target_h),
                                      interpolation=cv2.INTER_AREA)

        panel_w = 560
        panel = np.full((target_h, panel_w, 3), self.panel_bg, dtype=np.uint8)

        # Header
        cv2.putText(panel, "FOOTBALL TACTICAL ANALYZER", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "PHASE 1 — PURE SEGMENTATION PIPELINE", (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(panel, (20, 75), (panel_w - 20, 75), (60, 65, 75), 1)

        # Frame + time
        time_sec = frame_idx / fps if fps > 0 else 0.0
        cv2.putText(panel, f"Frame: {frame_idx:04d}/{total_frames:04d}  ({time_sec:.1f}s)",
                    (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        # Pitch side — prominent display
        side_color = (0, 255, 120) if pitch_side == "left" else (0, 180, 255)
        cv2.putText(panel, f"PITCH SIDE LOCKED: {pitch_side.upper()}",
                    (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.50, side_color, 2, cv2.LINE_AA)

        # Player counts
        players_cnt = sum(1 for o in tracked_objects
                          if o.class_name == "player" and o.pitch_pt is not None)
        gk_cnt      = sum(1 for o in tracked_objects
                          if o.class_name == "goalkeeper" and o.pitch_pt is not None)
        ref_cnt     = sum(1 for o in tracked_objects
                          if o.class_name == "referee" and o.pitch_pt is not None)
        cv2.putText(panel, f"Tracking: {players_cnt} Players | {gk_cnt} GK | {ref_cnt} Ref",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        # Zone counts table
        if zone_counts:
            y_c = 175
            cv2.putText(panel, "Zone Counts:", (20, y_c),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
            y_c += 18
            short_map = {
                "5Yard": "5Yd", "18Yard": "18Yd", "18Yard Circle": "18Yd-Arc",
                "First Half Central Circle": "CC-L", "Second Half Central Circle": "CC-R",
                "First Half Field": "HF-L", "Second Half Field": "HF-R",
            }
            for zn, cnt in sorted(zone_counts.items(), key=lambda x: -x[1]):
                cv2.putText(panel, f"  {short_map.get(zn, zn)}: {cnt}",
                            (20, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (200, 220, 200), 1, cv2.LINE_AA)
                y_c += 16
            radar_y_start = y_c + 10
        else:
            radar_y_start = 175

        # Radar
        rh, rw = radar_img.shape[:2]
        radar_target_w = panel_w - 30
        radar_target_h = int(rh * (radar_target_w / rw))
        scaled_radar = cv2.resize(radar_img, (radar_target_w, radar_target_h),
                                  interpolation=cv2.INTER_AREA)

        end_y = radar_y_start + radar_target_h
        if end_y < target_h:
            panel[radar_y_start:end_y, 15:15 + radar_target_w] = scaled_radar

        # Legend
        leg_y = min(end_y + 18, target_h - 80)
        cv2.line(panel, (20, leg_y - 8), (panel_w - 20, leg_y - 8), (60, 65, 75), 1)
        items = [
            ("Players", COLOR_PLAYER),
            ("Goalkeepers", COLOR_GOALKEEPER),
            ("Referees", COLOR_REFEREE),
        ]
        for i, (lbl, col) in enumerate(items):
            iy = leg_y + 15 + i * 22
            if iy > target_h - 5:
                break
            cv2.circle(panel, (30, iy - 4), 6, col, -1, cv2.LINE_AA)
            cv2.putText(panel, lbl, (48, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        composite = np.hstack([scaled_broadcast, panel])
        return composite
