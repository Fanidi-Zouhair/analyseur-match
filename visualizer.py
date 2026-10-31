"""
Tactical Visualization Module.
Renders both the annotated Broadcast View (with zones, snapped keypoints, and tracks)
and the 2D Tactical Pitch (Radar View) with real-time player positions.
Composites both views into a unified side-by-side broadcast telemetry display.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    HALF_LINE_X,
    PITCH_CENTER_Y,
    CENTER_CIRCLE_RADIUS,
    PENALTY_BOX_DEPTH,
    PENALTY_BOX_Y_MIN,
    PENALTY_BOX_Y_MAX,
    GOAL_BOX_DEPTH,
    GOAL_BOX_Y_MIN,
    GOAL_BOX_Y_MAX,
    PENALTY_SPOT_DIST,
    COLOR_PLAYER,
    COLOR_GOALKEEPER,
    COLOR_REFEREE,
    COLOR_KEYPOINT_RAW,
    COLOR_KEYPOINT_SNAPPED,
    COLOR_SYNTHETIC_ANCHOR,
    RADAR_CANVAS_WIDTH,
    RADAR_CANVAS_HEIGHT
)
from zone_validator import DetectedZone, KeypointMatch
from tracker import TrackedObject


class TacticalVisualizer:
    """
    Renders broadcast overlays, 2D tactical radar pitch, and composite video frames.
    """

    def __init__(self, radar_width: int = RADAR_CANVAS_WIDTH):
        self.radar_width = radar_width
        self.radar_margin = 25
        self.pitch_pixel_w = self.radar_width - 2 * self.radar_margin
        self.pitch_pixel_h = int(self.pitch_pixel_w * (PITCH_WIDTH / PITCH_LENGTH))
        self.radar_height = self.pitch_pixel_h + 2 * self.radar_margin

        self.scale_x = self.pitch_pixel_w / PITCH_LENGTH
        self.scale_y = self.pitch_pixel_h / PITCH_WIDTH

        # Colors
        self.turf_color = (36, 128, 40)       # Dark grass green
        self.stripe_color = (42, 145, 46)     # Light grass green
        self.line_color = (255, 255, 255)     # Pitch lines white
        self.panel_bg = (24, 28, 32)          # Dark panel background

    def m2px(self, xm: float, ym: float) -> Tuple[int, int]:
        """Converts pitch metric coordinates (meters) to radar pixel coordinates."""
        px = int(self.radar_margin + xm * self.scale_x)
        py = int(self.radar_margin + ym * self.scale_y)
        return px, py

    def draw_tactical_pitch(self, tracked_objects: List[TrackedObject]) -> np.ndarray:
        """
        Draws the 2D tactical pitch canvas with FIFA markings and projected participants.
        """
        canvas = np.full((self.radar_height, self.radar_width, 3), self.turf_color, dtype=np.uint8)

        # Alternating vertical grass stripes
        stripe_width_m = PITCH_LENGTH / 10.0
        for i in range(0, 10, 2):
            x1, y1 = self.m2px(i * stripe_width_m, 0.0)
            x2, y2 = self.m2px((i + 1) * stripe_width_m, PITCH_WIDTH)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), self.stripe_color, -1)

        # Outer pitch boundary
        cv2.rectangle(canvas, self.m2px(0.0, 0.0), self.m2px(PITCH_LENGTH, PITCH_WIDTH), self.line_color, 2)

        # Halfway line
        cv2.line(canvas, self.m2px(HALF_LINE_X, 0.0), self.m2px(HALF_LINE_X, PITCH_WIDTH), self.line_color, 2)

        # Center circle & spot
        cc_px = self.m2px(HALF_LINE_X, PITCH_CENTER_Y)
        cv2.circle(canvas, cc_px, int(CENTER_CIRCLE_RADIUS * self.scale_x), self.line_color, 2)
        cv2.circle(canvas, cc_px, 3, self.line_color, -1)

        # Left penalty area & 6-yard box
        cv2.rectangle(canvas, self.m2px(0.0, PENALTY_BOX_Y_MIN),
                      self.m2px(PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX), self.line_color, 2)
        cv2.rectangle(canvas, self.m2px(0.0, GOAL_BOX_Y_MIN),
                      self.m2px(GOAL_BOX_DEPTH, GOAL_BOX_Y_MAX), self.line_color, 2)

        # Left penalty spot & arc (D-arc)
        left_spot = self.m2px(PENALTY_SPOT_DIST, PITCH_CENTER_Y)
        cv2.circle(canvas, left_spot, 3, self.line_color, -1)
        cv2.ellipse(canvas, left_spot, (int(CENTER_CIRCLE_RADIUS * self.scale_x), int(CENTER_CIRCLE_RADIUS * self.scale_y)),
                    0, -53, 53, self.line_color, 2)

        # Right penalty area & 6-yard box
        cv2.rectangle(canvas, self.m2px(PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN),
                      self.m2px(PITCH_LENGTH, PENALTY_BOX_Y_MAX), self.line_color, 2)
        cv2.rectangle(canvas, self.m2px(PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN),
                      self.m2px(PITCH_LENGTH, GOAL_BOX_Y_MAX), self.line_color, 2)

        # Right penalty spot & arc
        right_spot = self.m2px(PITCH_LENGTH - PENALTY_SPOT_DIST, PITCH_CENTER_Y)
        cv2.circle(canvas, right_spot, 3, self.line_color, -1)
        cv2.ellipse(canvas, right_spot, (int(CENTER_CIRCLE_RADIUS * self.scale_x), int(CENTER_CIRCLE_RADIUS * self.scale_y)),
                    0, 127, 233, self.line_color, 2)

        # Goals (outer rectangles)
        goal_w = 7.32
        g_ymin, g_ymax = PITCH_CENTER_Y - goal_w / 2.0, PITCH_CENTER_Y + goal_w / 2.0
        cv2.rectangle(canvas, self.m2px(-2.0, g_ymin), self.m2px(0.0, g_ymax), (200, 200, 200), 2)
        cv2.rectangle(canvas, self.m2px(PITCH_LENGTH, g_ymin), self.m2px(PITCH_LENGTH + 2.0, g_ymax), (200, 200, 200), 2)

        # Plot Tracked Objects
        for obj in tracked_objects:
            if obj.pitch_pt is None:
                continue

            xm, ym = obj.pitch_pt
            if np.isnan(xm) or np.isnan(ym):
                continue

            px, py = self.m2px(xm, ym)

            # Assign color by role
            if obj.class_name == "player":
                color = COLOR_PLAYER
            elif obj.class_name == "goalkeeper":
                color = COLOR_GOALKEEPER
            else: # referee
                color = COLOR_REFEREE

            # Draw outer ring & center dot
            cv2.circle(canvas, (px, py), 9, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 7, color, -1, cv2.LINE_AA)

            # Draw track ID inside or beside dot
            label = str(obj.track_id) if obj.track_id > 0 else ""
            if label:
                cv2.putText(canvas, label, (px - 4 if len(label) == 1 else px - 8, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas

    def draw_broadcast_frame(self, frame: np.ndarray,
                             zones: List[DetectedZone],
                             matches: List[KeypointMatch],
                             tracked_objects: List[TrackedObject]) -> np.ndarray:
        """
        Draws translucent zone masks, snapped/synthetic keypoints, and player bounding boxes.
        """
        vis = frame.copy()
        overlay = frame.copy()

        # 1. Translucent Zone Polygons
        zone_colors = {
            "18Yard": (0, 165, 255),
            "18Yard Circle": (255, 100, 0),
            "5Yard": (0, 220, 255),
            "First Half Central Circle": (0, 255, 100),
            "Second Half Central Circle": (100, 255, 0),
            "First Half Field": (50, 180, 50),
            "Second Half Field": (50, 150, 50),
        }

        for z in zones:
            color = zone_colors.get(z.name, (0, 200, 0))
            cv2.fillPoly(overlay, [z.poly], color)
            cv2.polylines(vis, [z.poly], True, color, 2, cv2.LINE_AA)

        # Blend translucent zones at 20% opacity
        cv2.addWeighted(overlay, 0.20, vis, 0.80, 0, vis)

        # 2. Keypoints & Snapping Vectors
        for m in matches:
            rx, ry = int(m.raw_pt[0]), int(m.raw_pt[1])
            sx, sy = int(m.snapped_pt[0]), int(m.snapped_pt[1])

            if m.is_synthetic:
                # Synthetic geometric anchor (Magenta)
                cv2.circle(vis, (sx, sy), 7, COLOR_SYNTHETIC_ANCHOR, -1, cv2.LINE_AA)
                cv2.circle(vis, (sx, sy), 9, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(vis, m.name.replace("synthetic_", ""), (sx + 8, sy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_SYNTHETIC_ANCHOR, 1, cv2.LINE_AA)
            else:
                # Raw detection (Red) -> Snapped point (Green)
                if abs(rx - sx) > 1 or abs(ry - sy) > 1:
                    cv2.line(vis, (rx, ry), (sx, sy), (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(vis, (rx, ry), 4, COLOR_KEYPOINT_RAW, -1, cv2.LINE_AA)

                cv2.circle(vis, (sx, sy), 6, COLOR_KEYPOINT_SNAPPED, -1, cv2.LINE_AA)
                cv2.circle(vis, (sx, sy), 8, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vis, m.name, (sx + 8, sy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # 3. Tracked Objects (Bounding Box + Foot Marker)
        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            fx, fy = int(obj.foot_pt[0]), int(obj.foot_pt[1])

            if obj.class_name == "player":
                box_color = COLOR_PLAYER
            elif obj.class_name == "goalkeeper":
                box_color = COLOR_GOALKEEPER
            else:
                box_color = COLOR_REFEREE

            # Draw bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Foot contact circle
            cv2.circle(vis, (fx, fy), 5, box_color, -1, cv2.LINE_AA)
            cv2.circle(vis, (fx, fy), 7, (255, 255, 255), 1, cv2.LINE_AA)

            # Label banner
            id_str = f"#{obj.track_id} " if obj.track_id > 0 else ""
            label = f"{id_str}{obj.class_name[:4].upper()}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - 18)), (x1 + tw + 6, max(18, y1)), box_color, -1)
            cv2.putText(vis, label, (x1 + 3, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    def create_composite_dashboard(self, broadcast_img: np.ndarray,
                                   radar_img: np.ndarray,
                                   frame_idx: int,
                                   total_frames: int,
                                   fps: float,
                                   inlier_count: int,
                                   tracked_objects: List[TrackedObject]) -> np.ndarray:
        """
        Builds a broadcast-grade side-by-side display:
        Left: Broadcast Video View with zone anchors and bounding boxes.
        Right: Telemetry Panel containing 2D Tactical Pitch Radar and match metrics.
        """
        target_h = 720
        # Scale broadcast frame to height 720 while maintaining aspect ratio
        bh, bw = broadcast_img.shape[:2]
        scaled_bw = int(bw * (target_h / bh))
        scaled_broadcast = cv2.resize(broadcast_img, (scaled_bw, target_h), interpolation=cv2.INTER_AREA)

        # Right Panel dimensions
        panel_w = 560
        panel = np.full((target_h, panel_w, 3), self.panel_bg, dtype=np.uint8)

        # Header Title
        cv2.putText(panel, "FOOTBALL TACTICAL ANALYZER", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "PHASE 1 MVP - 2D RADAR TRACKING", (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(panel, (20, 75), (panel_w - 20, 75), (60, 65, 75), 1)

        # Telemetry metrics row
        time_sec = frame_idx / fps if fps > 0 else 0.0
        cv2.putText(panel, f"Frame: {frame_idx:04d}/{total_frames:04d} ({time_sec:.1f}s)",
                    (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(panel, f"H Inliers: {inlier_count} pts (EMA Smoothed)",
                    (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

        # Count participants
        players_cnt = sum(1 for o in tracked_objects if o.class_name == "player" and o.pitch_pt is not None)
        gk_cnt = sum(1 for o in tracked_objects if o.class_name == "goalkeeper" and o.pitch_pt is not None)
        ref_cnt = sum(1 for o in tracked_objects if o.class_name == "referee" and o.pitch_pt is not None)

        cv2.putText(panel, f"Tracking: {players_cnt} Players | {gk_cnt} GK | {ref_cnt} Ref",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        # Position Tactical Pitch Radar on right panel
        rh, rw = radar_img.shape[:2]
        radar_target_w = panel_w - 30
        radar_target_h = int(rh * (radar_target_w / rw))
        scaled_radar = cv2.resize(radar_img, (radar_target_w, radar_target_h), interpolation=cv2.INTER_AREA)

        radar_y_start = 175
        panel[radar_y_start:radar_y_start + radar_target_h, 15:15 + radar_target_w] = scaled_radar

        # Legend Panel
        leg_y = radar_y_start + radar_target_h + 30
        cv2.line(panel, (20, leg_y - 15), (panel_w - 20, leg_y - 15), (60, 65, 75), 1)
        cv2.putText(panel, "TACTICAL RADAR LEGEND:", (20, leg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        items = [
            ("Players", COLOR_PLAYER),
            ("Goalkeepers", COLOR_GOALKEEPER),
            ("Referees", COLOR_REFEREE),
            ("Synthetic Anchors (99.3%)", COLOR_SYNTHETIC_ANCHOR),
            ("Snapped Keypoints", COLOR_KEYPOINT_SNAPPED),
        ]

        for idx, (label, color) in enumerate(items):
            iy = leg_y + 25 + (idx * 24)
            cv2.circle(panel, (30, iy - 4), 6, color, -1, cv2.LINE_AA)
            cv2.putText(panel, label, (48, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        # Composite side-by-side
        composite = np.hstack([scaled_broadcast, panel])
        return composite

