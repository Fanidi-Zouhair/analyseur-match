"""
Tactical Visualization Module with TVCalib-Style Wireframe Back-Projection.
Renders:
1. Broadcast Video Frame with:
   - Translucent zone masks + labels.
   - Back-projected 2D pitch template grid wireframe (thin cyan line locked to grass markings).
   - Tracked player bounding boxes + role/zone tags.
2. 2D Tactical Pitch (Radar View) with real-time player positions.
3. Unified broadcast telemetry dashboard.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
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
    RADAR_CANVAS_WIDTH
)
from zone_validator import DetectedZone, KeypointMatch
from tracker import TrackedObject


_ZONE_SHORT = {
    "18Yard": "18Yd",
    "18Yard Circle": "18Yd-Arc",
    "5Yard": "5Yd",
    "First Half Central Circle": "CC-L",
    "Second Half Central Circle": "CC-R",
    "First Half Field": "HalfF-L",
    "Second Half Field": "HalfF-R",
}

_ZONE_COLORS = {
    "18Yard": (0, 165, 255),
    "18Yard Circle": (255, 100, 0),
    "5Yard": (0, 220, 255),
    "First Half Central Circle": (0, 255, 100),
    "Second Half Central Circle": (100, 255, 0),
    "First Half Field": (50, 180, 50),
    "Second Half Field": (50, 150, 50),
}


class TacticalVisualizer:
    """Renders broadcast overlays, pitch wireframe projections, and 2D radar pitch."""

    def __init__(self, radar_width: int = RADAR_CANVAS_WIDTH):
        self.radar_width = radar_width
        self.radar_margin = 25
        self.pitch_pixel_w = self.radar_width - 2 * self.radar_margin
        self.pitch_pixel_h = int(self.pitch_pixel_w * (PITCH_WIDTH / PITCH_LENGTH))
        self.radar_height = self.pitch_pixel_h + 2 * self.radar_margin

        self.scale_x = self.pitch_pixel_w / PITCH_LENGTH
        self.scale_y = self.pitch_pixel_h / PITCH_WIDTH

        self.turf_color = (36, 128, 40)
        self.stripe_color = (42, 145, 46)
        self.line_color = (255, 255, 255)
        self.panel_bg = (24, 28, 32)

        # Pre-compile 2D pitch metric lines for wireframe back-projection
        self.pitch_template_lines = self._compile_pitch_template_lines()

    def _compile_pitch_template_lines(self) -> List[np.ndarray]:
        """Pre-computes real-world pitch lines in metric coordinates."""
        lines = []

        # Halfway line
        lines.append(np.array([[HALF_LINE_X, 0.0], [HALF_LINE_X, PITCH_WIDTH]], dtype=np.float32))

        # Touchlines
        lines.append(np.array([[0.0, 0.0], [PITCH_LENGTH, 0.0]], dtype=np.float32))
        lines.append(np.array([[0.0, PITCH_WIDTH], [PITCH_LENGTH, PITCH_WIDTH]], dtype=np.float32))

        # Goal lines
        lines.append(np.array([[0.0, 0.0], [0.0, PITCH_WIDTH]], dtype=np.float32))
        lines.append(np.array([[PITCH_LENGTH, 0.0], [PITCH_LENGTH, PITCH_WIDTH]], dtype=np.float32))

        # Center circle
        theta = np.linspace(0, 2 * np.pi, 64)
        cc_x = HALF_LINE_X + CENTER_CIRCLE_RADIUS * np.cos(theta)
        cc_y = PITCH_CENTER_Y + CENTER_CIRCLE_RADIUS * np.sin(theta)
        lines.append(np.column_stack([cc_x, cc_y]).astype(np.float32))

        # Left Penalty Box (18-yard)
        lines.append(np.array([
            [0.0, PENALTY_BOX_Y_MIN],
            [PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN],
            [PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX],
            [0.0, PENALTY_BOX_Y_MAX]
        ], dtype=np.float32))

        # Left 6-yard Box (5-yard)
        lines.append(np.array([
            [0.0, GOAL_BOX_Y_MIN],
            [GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN],
            [GOAL_BOX_DEPTH, GOAL_BOX_Y_MAX],
            [0.0, GOAL_BOX_Y_MAX]
        ], dtype=np.float32))

        # Left D-arc
        d_theta = np.linspace(-0.925, 0.925, 32)
        arc_lx = PENALTY_SPOT_DIST + CENTER_CIRCLE_RADIUS * np.cos(d_theta)
        arc_ly = PITCH_CENTER_Y + CENTER_CIRCLE_RADIUS * np.sin(d_theta)
        lines.append(np.column_stack([arc_lx, arc_ly]).astype(np.float32))

        # Right Penalty Box (18-yard)
        lines.append(np.array([
            [PITCH_LENGTH, PENALTY_BOX_Y_MIN],
            [PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN],
            [PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX],
            [PITCH_LENGTH, PENALTY_BOX_Y_MAX]
        ], dtype=np.float32))

        # Right 6-yard Box
        lines.append(np.array([
            [PITCH_LENGTH, GOAL_BOX_Y_MIN],
            [PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN],
            [PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_MAX],
            [PITCH_LENGTH, GOAL_BOX_Y_MAX]
        ], dtype=np.float32))

        # Right D-arc
        d_theta_r = np.linspace(np.pi - 0.925, np.pi + 0.925, 32)
        arc_rx = (PITCH_LENGTH - PENALTY_SPOT_DIST) + CENTER_CIRCLE_RADIUS * np.cos(d_theta_r)
        arc_ry = PITCH_CENTER_Y + CENTER_CIRCLE_RADIUS * np.sin(d_theta_r)
        lines.append(np.column_stack([arc_rx, arc_ry]).astype(np.float32))

        return lines

    def m2px(self, xm: float, ym: float) -> Tuple[int, int]:
        px = int(self.radar_margin + xm * self.scale_x)
        py = int(self.radar_margin + ym * self.scale_y)
        return px, py

    def draw_tactical_pitch(self, tracked_objects: List[TrackedObject]) -> np.ndarray:
        canvas = np.full((self.radar_height, self.radar_width, 3), self.turf_color, dtype=np.uint8)

        # Grass stripes
        stripe_m = PITCH_LENGTH / 10.0
        for i in range(0, 10, 2):
            x1, y1 = self.m2px(i * stripe_m, 0.0)
            x2, y2 = self.m2px((i + 1) * stripe_m, PITCH_WIDTH)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), self.stripe_color, -1)

        # Pitch Boundary
        cv2.rectangle(canvas, self.m2px(0.0, 0.0), self.m2px(PITCH_LENGTH, PITCH_WIDTH), self.line_color, 2)

        # Halfway line
        cv2.line(canvas, self.m2px(HALF_LINE_X, 0.0), self.m2px(HALF_LINE_X, PITCH_WIDTH), self.line_color, 2)

        # Centre circle & spot
        cc = self.m2px(HALF_LINE_X, PITCH_CENTER_Y)
        cv2.circle(canvas, cc, int(CENTER_CIRCLE_RADIUS * self.scale_x), self.line_color, 2)
        cv2.circle(canvas, cc, 3, self.line_color, -1)

        # Left Penalty Box & Goal Box
        cv2.rectangle(canvas, self.m2px(0.0, PENALTY_BOX_Y_MIN),
                      self.m2px(PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX), self.line_color, 2)
        cv2.rectangle(canvas, self.m2px(0.0, GOAL_BOX_Y_MIN),
                      self.m2px(GOAL_BOX_DEPTH, GOAL_BOX_Y_MAX), self.line_color, 2)
        lspot = self.m2px(PENALTY_SPOT_DIST, PITCH_CENTER_Y)
        cv2.circle(canvas, lspot, 3, self.line_color, -1)
        cv2.ellipse(canvas, lspot,
                    (int(CENTER_CIRCLE_RADIUS * self.scale_x), int(CENTER_CIRCLE_RADIUS * self.scale_y)),
                    0, -53, 53, self.line_color, 2)

        # Right Penalty Box & Goal Box
        cv2.rectangle(canvas, self.m2px(PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN),
                      self.m2px(PITCH_LENGTH, PENALTY_BOX_Y_MAX), self.line_color, 2)
        cv2.rectangle(canvas, self.m2px(PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN),
                      self.m2px(PITCH_LENGTH, GOAL_BOX_Y_MAX), self.line_color, 2)
        rspot = self.m2px(PITCH_LENGTH - PENALTY_SPOT_DIST, PITCH_CENTER_Y)
        cv2.circle(canvas, rspot, 3, self.line_color, -1)
        cv2.ellipse(canvas, rspot,
                    (int(CENTER_CIRCLE_RADIUS * self.scale_x), int(CENTER_CIRCLE_RADIUS * self.scale_y)),
                    0, 127, 233, self.line_color, 2)

        # Goals
        goal_w = 7.32
        gy_min = PITCH_CENTER_Y - goal_w / 2.0
        gy_max = PITCH_CENTER_Y + goal_w / 2.0
        cv2.rectangle(canvas, self.m2px(-2.0, gy_min), self.m2px(0.0, gy_max), (200, 200, 200), 2)
        cv2.rectangle(canvas, self.m2px(PITCH_LENGTH, gy_min), self.m2px(PITCH_LENGTH + 2.0, gy_max), (200, 200, 200), 2)

        # Draw players on radar
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

    def draw_broadcast_frame(
        self,
        frame: np.ndarray,
        zones: List[DetectedZone],
        tracked_objects: List[TrackedObject],
        H: Optional[np.ndarray] = None,
        draw_wireframe: bool = True
    ) -> np.ndarray:
        """
        Draws translucent zone masks, back-projected pitch wireframe, and player tracks.
        """
        vis = frame.copy()
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # 1. Translucent Zone Fills
        for z in zones:
            color = _ZONE_COLORS.get(z.name, (0, 200, 0))
            cv2.fillPoly(overlay, [z.poly], color)
            cv2.polylines(vis, [z.poly], True, color, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.20, vis, 0.80, 0, vis)

        # 2. TVCalib-Style Wireframe Back-Projection (Thin Cyan Grid Lines)
        if draw_wireframe and H is not None:
            try:
                H_inv = np.linalg.inv(H)
                wireframe_color = (0, 255, 255) # Bright Yellow / Cyan (BGR)
                for line in self.pitch_template_lines:
                    pts = line.reshape(-1, 1, 2)
                    cam_pts = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
                    # Filter points that are within view
                    valid = (cam_pts[:, 0] > -300) & (cam_pts[:, 0] < w + 300) & \
                            (cam_pts[:, 1] > -300) & (cam_pts[:, 1] < h + 300)
                    if np.all(valid):
                        pts_int = cam_pts.astype(np.int32)
                        cv2.polylines(vis, [pts_int], False, wireframe_color, 2, cv2.LINE_AA)
            except Exception:
                pass

        # 3. Zone name labels at centroids
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
            cv2.rectangle(vis, (cx - tw // 2 - 3, cy - th - 4), (cx + tw // 2 + 3, cy + 4), (20, 20, 20), -1)
            cv2.putText(vis, label, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 4. Tracked Objects (BBox + Role + Zone Tag)
        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            fx, fy = int(obj.foot_pt[0]), int(obj.foot_pt[1])

            color = (COLOR_GOALKEEPER if obj.class_name == "goalkeeper"
                     else COLOR_REFEREE if obj.class_name == "referee"
                     else COLOR_PLAYER)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.circle(vis, (fx, fy), 5, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (fx, fy), 7, (255, 255, 255), 1, cv2.LINE_AA)

            id_str = f"#{obj.track_id} " if obj.track_id > 0 else ""
            role_str = {"player": "PLAY", "goalkeeper": "GK", "referee": "REF"}.get(
                obj.class_name, obj.class_name[:4].upper()
            )

            if obj.zones:
                top_zone = obj.zones[0]
                zs = _ZONE_SHORT.get(top_zone, top_zone[:5])
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
            cv2.rectangle(vis, (x1, max(0, y1 - 18)), (x1 + tw + 6, max(18, y1)), color, -1)
            cv2.putText(vis, label, (x1 + 3, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    def create_composite_dashboard(
        self,
        broadcast_img: np.ndarray,
        radar_img: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        pitch_side: str,
        tracked_objects: List[TrackedObject],
        inlier_count: int = 0,
        mask_iou: float = 0.0,
        zone_counts: Optional[dict] = None
    ) -> np.ndarray:
        """
        Creates side-by-side dashboard with live telemetry.
        """
        target_h = 720
        bh, bw = broadcast_img.shape[:2]
        scaled_bw = int(bw * (target_h / bh))
        scaled_broadcast = cv2.resize(broadcast_img, (scaled_bw, target_h), interpolation=cv2.INTER_AREA)

        panel_w = 560
        panel = np.full((target_h, panel_w, 3), self.panel_bg, dtype=np.uint8)

        # Header Title
        cv2.putText(panel, "FOOTBALL TACTICAL ANALYZER", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "TVCALIB MASK-IOU & OPTICAL FLOW TRACKER", (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(panel, (20, 75), (panel_w - 20, 75), (60, 65, 75), 1)

        # Frame + Time
        time_sec = frame_idx / fps if fps > 0 else 0.0
        cv2.putText(panel, f"Frame: {frame_idx:04d}/{total_frames:04d}  ({time_sec:.1f}s)",
                    (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        # Pitch Side Locked
        side_color = (0, 255, 120) if pitch_side == "left" else (0, 180, 255)
        cv2.putText(panel, f"Pitch Side: {pitch_side.upper()}",
                    (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.48, side_color, 2, cv2.LINE_AA)

        # IoU & Camera Motion Inliers
        iou_pct = int(mask_iou * 100)
        cv2.putText(panel, f"Template IoU: {iou_pct}% | H Inliers: {inlier_count} pts",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Participant counts
        players_cnt = sum(1 for o in tracked_objects if o.class_name == "player" and o.pitch_pt is not None)
        gk_cnt = sum(1 for o in tracked_objects if o.class_name == "goalkeeper" and o.pitch_pt is not None)
        ref_cnt = sum(1 for o in tracked_objects if o.class_name == "referee" and o.pitch_pt is not None)
        cv2.putText(panel, f"Tracking: {players_cnt} Players | {gk_cnt} GK | {ref_cnt} Ref",
                    (20, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        # Tactical Pitch Radar placement
        rh, rw = radar_img.shape[:2]
        radar_target_w = panel_w - 30
        radar_target_h = int(rh * (radar_target_w / rw))
        scaled_radar = cv2.resize(radar_img, (radar_target_w, radar_target_h), interpolation=cv2.INTER_AREA)

        radar_y_start = 195
        end_y = radar_y_start + radar_target_h
        if end_y < target_h:
            panel[radar_y_start:end_y, 15:15 + radar_target_w] = scaled_radar

        # Legend Panel
        leg_y = end_y + 25
        if leg_y + 60 < target_h:
            cv2.line(panel, (20, leg_y - 10), (panel_w - 20, leg_y - 10), (60, 65, 75), 1)
            items = [
                ("Players", COLOR_PLAYER),
                ("Goalkeepers", COLOR_GOALKEEPER),
                ("Referees", COLOR_REFEREE),
                ("2D Pitch Wireframe", (0, 255, 255)),
            ]
            for i, (lbl, col) in enumerate(items):
                iy = leg_y + 12 + i * 20
                if iy > target_h - 5:
                    break
                cv2.circle(panel, (30, iy - 4), 5, col, -1, cv2.LINE_AA)
                cv2.putText(panel, lbl, (48, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        composite = np.hstack([scaled_broadcast, panel])
        return composite
