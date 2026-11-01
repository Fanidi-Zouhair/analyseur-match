"""
seg_mapper.py — 100% Segmentation-Only 2D Position Mapper (Phase 1 MVP).

Computes every player's (X, Y) position in real-world metres using ONLY the
football_pitch.pt segmentation masks (99.3% precision).  No homography, no
keypoint model, no cv2.findHomography.

Zone priority (most-specific first):
  5Yard  >  18Yard  >  18Yard Circle  >  Central Circle (either half)  >  Open Field

Side detection uses ONLY mask centroid positions, never the central-circle
count heuristic that caused the Frame-138 flip.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    PITCH_LENGTH, PITCH_WIDTH,
    HALF_LINE_X, PITCH_CENTER_Y,
    CENTER_CIRCLE_RADIUS,
    PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN, PENALTY_BOX_Y_MAX,
    GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN, GOAL_BOX_Y_MAX,
)

# ── Pitch real-world bounding boxes for each zone ────────────────────────────
# Keyed by (zone_name, side) → (x_min, x_max, y_min, y_max) in metres.
_ZONE_BOUNDS: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {
    ("5Yard",            "left"):  (0.0,             GOAL_BOX_DEPTH,           GOAL_BOX_Y_MIN,      GOAL_BOX_Y_MAX),
    ("5Yard",            "right"): (PITCH_LENGTH - GOAL_BOX_DEPTH, PITCH_LENGTH, GOAL_BOX_Y_MIN,    GOAL_BOX_Y_MAX),
    ("18Yard",           "left"):  (0.0,             PENALTY_BOX_DEPTH,        PENALTY_BOX_Y_MIN,   PENALTY_BOX_Y_MAX),
    ("18Yard",           "right"): (PITCH_LENGTH - PENALTY_BOX_DEPTH, PITCH_LENGTH, PENALTY_BOX_Y_MIN, PENALTY_BOX_Y_MAX),
    # 18Yard Circle (D-arc) tight bbox
    ("18Yard Circle",    "left"):  (PENALTY_BOX_DEPTH, PENALTY_BOX_DEPTH + CENTER_CIRCLE_RADIUS,
                                    PITCH_CENTER_Y - CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y + CENTER_CIRCLE_RADIUS),
    ("18Yard Circle",    "right"): (PITCH_LENGTH - PENALTY_BOX_DEPTH - CENTER_CIRCLE_RADIUS,
                                    PITCH_LENGTH - PENALTY_BOX_DEPTH,
                                    PITCH_CENTER_Y - CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y + CENTER_CIRCLE_RADIUS),
    # Central circles (semicircles on each side of the halfway line)
    ("First Half Central Circle",  "left"):  (HALF_LINE_X - CENTER_CIRCLE_RADIUS, HALF_LINE_X,
                                              PITCH_CENTER_Y - CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y + CENTER_CIRCLE_RADIUS),
    ("Second Half Central Circle", "right"): (HALF_LINE_X, HALF_LINE_X + CENTER_CIRCLE_RADIUS,
                                              PITCH_CENTER_Y - CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y + CENTER_CIRCLE_RADIUS),
}

# For open-field interpolation: ordered X anchors (image-space pixel X → pitch metres)
_OPEN_FIELD_PRIORITY = [
    "5Yard", "18Yard", "18Yard Circle",
    "First Half Central Circle", "Second Half Central Circle",
    "First Half Field", "Second Half Field",
]


class SegMapper:
    """
    100% segmentation-only 2D position mapper.  No homography involved.
    """

    # EMA smoothing per tracker_id
    _ema_alpha: float = 0.30
    _ema_state: Dict[int, Tuple[float, float]]

    def __init__(self) -> None:
        self._ema_state = {}
        # Locked side: set once we have a confident read and only flipped with
        # a strong centroid signal (≥ 80 px from mid-frame).
        self.last_known_side: str = "left"
        self._side_confidence: float = 0.0   # how many px from mid-frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_side(self, zone_dict: Dict[str, "DetectedZone"], frame_width: int) -> str:  # noqa: F821
        """
        Determine left / right penalty side using ONLY mask X-centroids.

        Priority:
        1. If 5Yard AND 18Yard both visible: cx(5Yard) < cx(18Yard) → 5Yard is
           closer to goal line → left.  And vice-versa.
        2. If 18Yard AND 18Yard Circle both visible: cx(18Yard) < cx(18Yard Circle)
           → left.
        3. If only one penalty zone visible: compare cx vs frame midpoint.
           Accept only if centroid is > 50 px from midpoint (avoid noisy midline flips).
        4. If any penalty zone touches left border (<20 px) → left;
           right border (>frame_w-20) → right.
        5. Stick with last_known_side.
        """
        pz_names = ["5Yard", "18Yard", "18Yard Circle"]
        cx: Dict[str, float] = {}
        for name in pz_names:
            z = zone_dict.get(name)
            if z is not None and len(z.poly) >= 3:
                M = cv2.moments(z.poly)
                if M["m00"] > 0:
                    cx[name] = M["m10"] / M["m00"]

        mid = frame_width / 2.0

        # Rule 1
        if "5Yard" in cx and "18Yard" in cx:
            side = "left" if cx["5Yard"] < cx["18Yard"] else "right"
            self.last_known_side = side
            return side

        # Rule 2
        if "18Yard" in cx and "18Yard Circle" in cx:
            side = "left" if cx["18Yard"] < cx["18Yard Circle"] else "right"
            self.last_known_side = side
            return side

        # Rule 3 — single zone centroid, require confidence
        for name in pz_names:
            if name in cx:
                dist = cx[name] - mid
                if abs(dist) > 50:
                    side = "left" if dist < 0 else "right"
                    self.last_known_side = side
                    return side

        # Rule 4 — border clipping
        margin = 20
        for name in pz_names:
            z = zone_dict.get(name)
            if z is not None and len(z.poly) > 0:
                xs = z.poly[:, 0]
                if xs.min() < margin:
                    self.last_known_side = "left"
                    return "left"
                if xs.max() > frame_width - margin:
                    self.last_known_side = "right"
                    return "right"

        # Rule 5 — keep last known
        return self.last_known_side

    def map_player(
        self,
        foot_px: Tuple[float, float],
        zone_dict: Dict[str, "DetectedZone"],  # noqa: F821
        player_zones: List[str],
        side: str,
        frame_shape: Tuple[int, int],
    ) -> Tuple[float, float]:
        """
        Map a single player's pixel foot-point to (X_m, Y_m) in real metres.

        Args:
            foot_px: (px, py) foot contact in image pixels.
            zone_dict: dict of zone_name → DetectedZone (best instance per name).
            player_zones: list of zone names whose mask contains the foot point.
            side: 'left' or 'right' (penalty side).
            frame_shape: (height, width) of the video frame.
        """
        px, py = float(foot_px[0]), float(foot_px[1])

        # Priority ordered zone checks
        for zone_name in ["5Yard", "18Yard", "18Yard Circle",
                           "First Half Central Circle", "Second Half Central Circle"]:
            if zone_name not in player_zones:
                continue

            # Determine the side key for this zone
            if zone_name in ("5Yard", "18Yard", "18Yard Circle"):
                zone_side = side
            elif zone_name == "First Half Central Circle":
                zone_side = "left"
            else:  # Second Half Central Circle
                zone_side = "right"

            zone = zone_dict.get(zone_name)
            if zone is None:
                continue

            result = self._map_in_zone(px, py, zone, zone_name, zone_side)
            if result is not None:
                return result

        # Open-field fallback (Case E)
        result = self._map_open_field(px, py, zone_dict, side, player_zones, frame_shape)
        return result

    def apply_ema(self, track_id: int, xy: Tuple[float, float]) -> Tuple[float, float]:
        """Apply per-track EMA smoothing to the 2D position."""
        if track_id not in self._ema_state:
            self._ema_state[track_id] = xy
            return xy
        prev = self._ema_state[track_id]
        smoothed = (
            self._ema_alpha * xy[0] + (1 - self._ema_alpha) * prev[0],
            self._ema_alpha * xy[1] + (1 - self._ema_alpha) * prev[1],
        )
        self._ema_state[track_id] = smoothed
        return smoothed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_in_zone(
        self,
        px: float, py: float,
        zone: "DetectedZone",  # noqa: F821
        zone_name: str,
        side: str,
    ) -> Optional[Tuple[float, float]]:
        """
        Map pixel (px, py) inside a detected zone to real-world coords.
        Returns None if the bounding box is degenerate.
        """
        bounds = _ZONE_BOUNDS.get((zone_name, side))
        if bounds is None:
            return None

        xm_min, xm_max, ym_min, ym_max = bounds

        # Use the mask's bounding box as the image-space reference
        pts = zone.poly.astype(np.float32)
        img_x_min = float(pts[:, 0].min())
        img_x_max = float(pts[:, 0].max())
        img_y_min = float(pts[:, 1].min())
        img_y_max = float(pts[:, 1].max())

        if img_x_max <= img_x_min or img_y_max <= img_y_min:
            return None

        # Normalised [0,1] position inside the mask bbox
        u = np.clip((px - img_x_min) / (img_x_max - img_x_min), 0.0, 1.0)
        v = np.clip((py - img_y_min) / (img_y_max - img_y_min), 0.0, 1.0)

        # Flip u direction for right-side zones so goal line remains at X=105
        if side == "right":
            u = 1.0 - u

        xm = xm_min + u * (xm_max - xm_min)
        ym = ym_min + v * (ym_max - ym_min)

        return float(np.clip(xm, 0.0, PITCH_LENGTH)), float(np.clip(ym, 0.0, PITCH_WIDTH))

    def _map_open_field(
        self,
        px: float, py: float,
        zone_dict: Dict[str, "DetectedZone"],  # noqa: F821
        side: str,
        player_zones: List[str],
        frame_shape: Tuple[int, int],
    ) -> Tuple[float, float]:
        """
        Case E: piecewise linear ruler using visible zone edges as anchors.
        Builds a list of (image_x, pitch_x) pairs and interpolates.
        """
        frame_h, frame_w = frame_shape

        # Build X ruler from visible zone boundaries
        x_anchors: List[Tuple[float, float]] = []  # (img_x, pitch_x)

        for zone_name, zone in zone_dict.items():
            pts = zone.poly.astype(np.float32)
            if len(pts) < 3:
                continue
            img_x_min = float(pts[:, 0].min())
            img_x_max = float(pts[:, 0].max())
            img_y_min = float(pts[:, 1].min())
            img_y_max = float(pts[:, 1].max())

            z_side = side if zone_name in ("5Yard", "18Yard", "18Yard Circle") else None
            if zone_name == "First Half Central Circle":
                z_side = "left"
            elif zone_name == "Second Half Central Circle":
                z_side = "right"
            elif zone_name == "First Half Field":
                z_side = "left"
            elif zone_name == "Second Half Field":
                z_side = "right"

            bounds = _ZONE_BOUNDS.get((zone_name, z_side))
            if bounds is None:
                # For half-fields, use approximation
                if zone_name == "First Half Field":
                    # right edge ≈ halfway line
                    x_anchors.append((img_x_max, HALF_LINE_X))
                    x_anchors.append((img_x_min, 0.0))
                elif zone_name == "Second Half Field":
                    x_anchors.append((img_x_min, HALF_LINE_X))
                    x_anchors.append((img_x_max, PITCH_LENGTH))
                continue

            xm_min, xm_max, _, _ = bounds
            if side == "right":
                # right edge of mask → goal line (X=105), left edge → inner line
                x_anchors.append((img_x_max, PITCH_LENGTH))
                x_anchors.append((img_x_min, xm_min))
            else:
                x_anchors.append((img_x_min, xm_min))
                x_anchors.append((img_x_max, xm_max))

        # Build Y ruler: top and bottom edges of visible zones
        y_anchors: List[Tuple[float, float]] = []  # (img_y, pitch_y)
        for zone_name, zone in zone_dict.items():
            pts = zone.poly.astype(np.float32)
            if len(pts) < 3:
                continue
            img_y_min = float(pts[:, 1].min())
            img_y_max = float(pts[:, 1].max())

            z_side = side if zone_name in ("5Yard", "18Yard", "18Yard Circle") else None
            if zone_name in ("First Half Central Circle", "Second Half Central Circle"):
                z_side = "left" if zone_name == "First Half Central Circle" else "right"
            elif zone_name in ("First Half Field", "Second Half Field"):
                z_side = "left" if zone_name == "First Half Field" else "right"

            bounds = _ZONE_BOUNDS.get((zone_name, z_side))
            if bounds is None:
                if zone_name in ("First Half Field", "Second Half Field"):
                    y_anchors.append((img_y_min, 0.0))
                    y_anchors.append((img_y_max, PITCH_WIDTH))
                continue
            _, _, ym_min, ym_max = bounds
            y_anchors.append((img_y_min, ym_min))
            y_anchors.append((img_y_max, ym_max))

        # Interpolate X
        if len(x_anchors) >= 2:
            x_anchors_sorted = sorted(set(x_anchors), key=lambda t: t[0])
            img_xs = [t[0] for t in x_anchors_sorted]
            pitch_xs = [t[1] for t in x_anchors_sorted]
            xm = float(np.interp(px, img_xs, pitch_xs))
        else:
            # Coarse fallback: linear scale across full frame
            xm = (px / frame_w) * PITCH_LENGTH
        xm = float(np.clip(xm, 0.0, PITCH_LENGTH))

        # Interpolate Y
        if len(y_anchors) >= 2:
            y_anchors_sorted = sorted(set(y_anchors), key=lambda t: t[0])
            img_ys = [t[0] for t in y_anchors_sorted]
            pitch_ys = [t[1] for t in y_anchors_sorted]
            ym = float(np.interp(py, img_ys, pitch_ys))
        else:
            ym = (py / frame_h) * PITCH_WIDTH
        ym = float(np.clip(ym, 1.0, 67.0))

        return xm, ym

