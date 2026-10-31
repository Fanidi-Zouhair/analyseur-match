"""
Homography Estimation & Stabilization Module.
Computes projective transformation from camera pixels to 2D tactical pitch metric coordinates.
Applies RANSAC outlier rejection, Exponential Moving Average (EMA) temporal smoothing,
and zone-consistency spatial constraints on projected player coordinates.
"""

from collections import deque
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    HALF_LINE_X,
    PENALTY_BOX_DEPTH,
    PENALTY_BOX_Y_MIN,
    PENALTY_BOX_Y_MAX,
    GOAL_BOX_DEPTH,
    GOAL_BOX_Y_MIN,
    GOAL_BOX_Y_MAX,
    RANSAC_REPROJ_THRESHOLD,
    HOMOGRAPHY_EMA_ALPHA,
    MAX_HOMOGRAPHY_CACHE_AGE
)
from zone_validator import KeypointMatch


class HomographyEstimator:
    """
    Computes, smooths, and applies camera-to-pitch Homography (H).
    """

    def __init__(self,
                 ema_alpha: float = HOMOGRAPHY_EMA_ALPHA,
                 ransac_thresh: float = RANSAC_REPROJ_THRESHOLD,
                 max_cache_age: int = MAX_HOMOGRAPHY_CACHE_AGE):
        self.ema_alpha = ema_alpha
        self.ransac_thresh = ransac_thresh
        self.max_cache_age = max_cache_age

        self.H_smooth: Optional[np.ndarray] = None
        self.last_valid_H: Optional[np.ndarray] = None
        self.cache_age: int = 0
        self.inlier_count: int = 0
        self.history = deque(maxlen=15)

    def update(self, matches: List[KeypointMatch]) -> Optional[np.ndarray]:
        """
        Calculates homography H from matched keypoints and synthetic anchors.
        Smooths H across frames using Exponential Moving Average (EMA).
        """
        if len(matches) < 4:
            self.cache_age += 1
            if self.H_smooth is not None and self.cache_age <= self.max_cache_age:
                return self.H_smooth
            return None

        src_pts = np.array([m.snapped_pt for m in matches], dtype=np.float32)
        dst_pts = np.array([m.pitch_pt for m in matches], dtype=np.float32)

        # Compute Homography via RANSAC
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)

        if H is not None and self._is_homography_valid(H):
            # Normalize so H[2, 2] == 1.0
            if abs(H[2, 2]) > 1e-8:
                H = H / H[2, 2]

            self.inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0

            # Temporal Smoothing (EMA)
            if self.H_smooth is None:
                self.H_smooth = H.copy()
            else:
                self.H_smooth = (self.ema_alpha * H) + ((1.0 - self.ema_alpha) * self.H_smooth)
                if abs(self.H_smooth[2, 2]) > 1e-8:
                    self.H_smooth = self.H_smooth / self.H_smooth[2, 2]

            self.last_valid_H = self.H_smooth.copy()
            self.history.append(self.H_smooth.copy())
            self.cache_age = 0
            return self.H_smooth
        else:
            # Fallback to smoothed cache if recent
            self.cache_age += 1
            if self.H_smooth is not None and self.cache_age <= self.max_cache_age:
                return self.H_smooth
            return None

    def _is_homography_valid(self, H: np.ndarray) -> bool:
        """
        Validates condition number, non-zero determinant, and projective plausibility.
        """
        if np.isnan(H).any() or np.isinf(H).any():
            return False

        det = np.linalg.det(H)
        if abs(det) < 1e-10:
            return False

        # Condition number test
        cond = np.linalg.cond(H)
        if cond > 1e9:
            return False

        return True

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """
        Projects (N, 2) image pixel points onto (N, 2) 2D pitch coordinates in meters.
        Returns array of shape (N, 2). Points that cannot be projected return NaNs.
        """
        if self.H_smooth is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        try:
            proj = cv2.perspectiveTransform(pts, self.H_smooth).reshape(-1, 2)
            return proj
        except Exception:
            return np.full((len(points), 2), np.nan, dtype=np.float32)

    def enforce_zone_consistency(self, proj_pts: np.ndarray,
                                 zone_associations: List[List[str]],
                                 pitch_side: str = "left") -> np.ndarray:
        """
        Enforces geometric constraints between camera zone detections and 2D radar positions.
        If a player's feet are inside a zone in the camera image (99.3% precision anchor),
        their 2D pitch projection is verified and clamped to remain faithful to that zone.
        """
        if len(proj_pts) == 0:
            return proj_pts

        corrected = proj_pts.copy()

        for i, (pt, zones) in enumerate(zip(corrected, zone_associations)):
            x, y = pt[0], pt[1]
            if np.isnan(x) or np.isnan(y):
                continue

            # 18-Yard Box Constraint
            if "18Yard" in zones:
                if pitch_side == "left":
                    x = np.clip(x, 0.0, PENALTY_BOX_DEPTH)
                    y = np.clip(y, PENALTY_BOX_Y_MIN, PENALTY_BOX_Y_MAX)
                else:
                    x = np.clip(x, PITCH_LENGTH - PENALTY_BOX_DEPTH, PITCH_LENGTH)
                    y = np.clip(y, PENALTY_BOX_Y_MIN, PENALTY_BOX_Y_MAX)

            # 5-Yard (Goal Area) Box Constraint
            if "5Yard" in zones:
                if pitch_side == "left":
                    x = np.clip(x, 0.0, GOAL_BOX_DEPTH)
                    y = np.clip(y, GOAL_BOX_Y_MIN, GOAL_BOX_Y_MAX)
                else:
                    x = np.clip(x, PITCH_LENGTH - GOAL_BOX_DEPTH, PITCH_LENGTH)
                    y = np.clip(y, GOAL_BOX_Y_MIN, GOAL_BOX_Y_MAX)

            # Half Field Boundary Consistency
            if "First Half Field" in zones and "Second Half Field" not in zones:
                x = min(x, HALF_LINE_X)
            elif "Second Half Field" in zones and "First Half Field" not in zones:
                x = max(x, HALF_LINE_X)

            # Global Pitch Bounds Clamping (with 2m margin for touchline out-of-bounds)
            x = np.clip(x, -2.0, PITCH_LENGTH + 2.0)
            y = np.clip(y, -2.0, PITCH_WIDTH + 2.0)

            corrected[i] = [x, y]

        return corrected

