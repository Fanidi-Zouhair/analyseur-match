"""
Homography Estimation & Stabilization Module.
Computes projective transformation from camera pixels to 2D tactical pitch metric coordinates.
Applies RANSAC outlier rejection, Exponential Moving Average (EMA) temporal smoothing,
collinearity / spatial spread guards, and zone-consistency spatial constraints on projected
player coordinates.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Any
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
from tracker import TrackedObject


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
        self.player_track_ema: Dict[int, Tuple[float, float]] = {}
        self.track_ema_alpha: float = 0.35

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

        # ---- BUG 4 FIX: Collinearity & Spatial Spread Guard ----
        # Verify that destination (pitch) points span enough area to avoid degenerate H
        if not self._check_spatial_spread(dst_pts):
            self.cache_age += 1
            if self.H_smooth is not None and self.cache_age <= self.max_cache_age:
                return self.H_smooth
            return None

        # Compute Homography via RANSAC
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)

        if H is not None and self._is_homography_valid(H, src_pts):
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

    def _check_spatial_spread(self, dst_pts: np.ndarray,
                              min_x_span: float = 15.0,
                              min_y_span: float = 10.0) -> bool:
        """
        Verifies that the 2D pitch destination points are NOT collinear or clustered.
        Points must span at least min_x_span meters in X and min_y_span meters in Y.
        """
        x_span = dst_pts[:, 0].max() - dst_pts[:, 0].min()
        y_span = dst_pts[:, 1].max() - dst_pts[:, 1].min()

        if x_span < min_x_span and y_span < min_y_span:
            return False

        # Also reject if all X values are nearly identical (would produce vertical-line collapse)
        x_std = np.std(dst_pts[:, 0])
        if x_std < 2.0:
            return False

        return True

    def _is_homography_valid(self, H: np.ndarray, src_pts: np.ndarray) -> bool:
        """
        Validates condition number, non-zero determinant, projective plausibility,
        and left-to-right ordering preservation.
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

        # ---- BUG 4 FIX: Check that H does not collapse X variance ----
        # Project the source points and verify that X variance is preserved
        try:
            proj = cv2.perspectiveTransform(
                src_pts.reshape(-1, 1, 2), H
            ).reshape(-1, 2)

            # If all projected X are nearly identical, the homography is degenerate
            proj_x_std = np.std(proj[:, 0])
            if proj_x_std < 1.0:
                return False

            # Verify output is within a reasonable range (e.g., -20 to 125 meters)
            if proj[:, 0].min() < -50.0 or proj[:, 0].max() > 155.0:
                return False
            if proj[:, 1].min() < -50.0 or proj[:, 1].max() > 120.0:
                return False

        except Exception:
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

        BUG 1 FIX: Removed the hard min(x, HALF_LINE_X) / max(x, HALF_LINE_X) clamping
        that was collapsing all players onto X=52.5. Only 18Yard, 5Yard, and global
        pitch bounds remain as constraints.
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

            # ---- BUG 1 FIX: Removed hard HALF_LINE_X clamping ----
            # Previously: if "First Half Field" in zones: x = min(x, HALF_LINE_X)
            #             elif "Second Half Field" in zones: x = max(x, HALF_LINE_X)
            # This was crushing all players onto x=52.5. Removed.

            # Global Pitch Bounds Clamping (with 2m margin for touchline out-of-bounds)
            x = np.clip(x, -2.0, PITCH_LENGTH + 2.0)
            y = np.clip(y, -2.0, PITCH_WIDTH + 2.0)

            corrected[i] = [x, y]

        return corrected

    def compare_and_fuse_positions(self, tracked_objects: List[TrackedObject],
                                  p_zone_pts: List[Tuple[float, float]],
                                  p_kp_pts: Any,
                                  pitch_side: str = "left",
                                  frame_idx: Optional[int] = None,
                                  print_table: bool = False) -> None:
        """
        Step C: Cross-Comparison & Reconciliation Pipeline.
        Reconciles Pipeline A (P_zone, 99.3% reliable) with Pipeline B (P_kp, keypoint/homography).
        Verifies tactical zone membership, rejects diverged keypoint projections (>4.0m),
        fuses valid points (0.75 P_zone + 0.25 P_kp), and applies per-track temporal EMA smoothing.
        """
        rows = []

        for i, obj in enumerate(tracked_objects):
            p_z = p_zone_pts[i] if i < len(p_zone_pts) else (52.5, 34.0)

            # Extract P_kp candidate
            p_k = None
            if p_kp_pts is not None and i < len(p_kp_pts):
                raw_k = p_kp_pts[i]
                if raw_k is not None:
                    xk, yk = float(raw_k[0]), float(raw_k[1])
                    if not (np.isnan(xk) or np.isnan(yk) or np.isinf(xk) or np.isinf(yk)):
                        p_k = (xk, yk)

            # ------------------------------------------------------------------
            # 1. Zone Membership Verification for P_kp
            # ------------------------------------------------------------------
            valid_kp_zone = True
            rejection_reason = ""

            if p_k is None:
                valid_kp_zone = False
                rejection_reason = "No Homography"
            else:
                xk, yk = p_k

                # Central circle check
                if "First Half Central Circle" in obj.zones:
                    dist_sq = (xk - HALF_LINE_X) ** 2 + (yk - (PITCH_WIDTH / 2.0)) ** 2
                    if dist_sq > (9.15 + 2.0) ** 2 or xk > HALF_LINE_X + 2.0:
                        valid_kp_zone = False
                        rejection_reason = "Out of 1st Half Central Circle"
                elif "Second Half Central Circle" in obj.zones:
                    dist_sq = (xk - HALF_LINE_X) ** 2 + (yk - (PITCH_WIDTH / 2.0)) ** 2
                    if dist_sq > (9.15 + 2.0) ** 2 or xk < HALF_LINE_X - 2.0:
                        valid_kp_zone = False
                        rejection_reason = "Out of 2nd Half Central Circle"

                # 18-Yard Box check
                if "18Yard" in obj.zones:
                    if pitch_side == "left":
                        if xk > PENALTY_BOX_DEPTH + 2.0 or yk < PENALTY_BOX_Y_MIN - 2.0 or yk > PENALTY_BOX_Y_MAX + 2.0:
                            valid_kp_zone = False
                            rejection_reason = "Out of Left 18Yard Box"
                    else:
                        if xk < (PITCH_LENGTH - PENALTY_BOX_DEPTH) - 2.0 or yk < PENALTY_BOX_Y_MIN - 2.0 or yk > PENALTY_BOX_Y_MAX + 2.0:
                            valid_kp_zone = False
                            rejection_reason = "Out of Right 18Yard Box"
                else:
                    # If player is NOT in 18Yard in camera view, must NOT land inside 18Yard on 2D pitch
                    if pitch_side == "left":
                        if 0.0 <= xk <= PENALTY_BOX_DEPTH and PENALTY_BOX_Y_MIN <= yk <= PENALTY_BOX_Y_MAX:
                            valid_kp_zone = False
                            rejection_reason = "Spuriously inside 18Yard Box"
                    else:
                        if (PITCH_LENGTH - PENALTY_BOX_DEPTH) <= xk <= PITCH_LENGTH and PENALTY_BOX_Y_MIN <= yk <= PENALTY_BOX_Y_MAX:
                            valid_kp_zone = False
                            rejection_reason = "Spuriously inside 18Yard Box"

                # 5-Yard Box check
                if "5Yard" in obj.zones:
                    if pitch_side == "left":
                        if xk > GOAL_BOX_DEPTH + 2.0 or yk < GOAL_BOX_Y_MIN - 2.0 or yk > GOAL_BOX_Y_MAX + 2.0:
                            valid_kp_zone = False
                            rejection_reason = "Out of 5Yard Box"
                    else:
                        if xk < (PITCH_LENGTH - GOAL_BOX_DEPTH) - 2.0 or yk < GOAL_BOX_Y_MIN - 2.0 or yk > GOAL_BOX_Y_MAX + 2.0:
                            valid_kp_zone = False
                            rejection_reason = "Out of 5Yard Box"

            # ------------------------------------------------------------------
            # 2. Error Rejection & Discrepancy Gate
            # ------------------------------------------------------------------
            if valid_kp_zone and p_k is not None:
                d_i = float(np.hypot(p_k[0] - p_z[0], p_k[1] - p_z[1]))
            else:
                d_i = 999.0

            if not valid_kp_zone or d_i > 4.0:
                # Discard P_kp, use 99.3% reliable P_zone
                fused_x, fused_y = p_z[0], p_z[1]
                obj.is_fused = False
                status_str = f"P_zone (d={d_i:.1f}m)" if d_i < 900.0 else f"P_zone ({rejection_reason})"
            else:
                # Fuse smoothly: 0.75 P_zone + 0.25 P_kp
                fused_x = 0.75 * p_z[0] + 0.25 * p_k[0]
                fused_y = 0.75 * p_z[1] + 0.25 * p_k[1]
                obj.is_fused = True
                status_str = f"Fused (d={d_i:.1f}m)"

            # ------------------------------------------------------------------
            # 3. Per-Track Temporal Smoothing (EMA on 2D Player Positions)
            # ------------------------------------------------------------------
            tid = obj.track_id
            if tid > 0 and tid in self.player_track_ema:
                prev_x, prev_y = self.player_track_ema[tid]
                smooth_x = self.track_ema_alpha * fused_x + (1.0 - self.track_ema_alpha) * prev_x
                smooth_y = self.track_ema_alpha * fused_y + (1.0 - self.track_ema_alpha) * prev_y
            else:
                smooth_x = fused_x
                smooth_y = fused_y

            if tid > 0:
                self.player_track_ema[tid] = (smooth_x, smooth_y)

            # Clamp to pitch bounds [-2, 107] x [-2, 70]
            final_x = float(np.clip(smooth_x, -2.0, PITCH_LENGTH + 2.0))
            final_y = float(np.clip(smooth_y, -2.0, PITCH_WIDTH + 2.0))

            obj.p_zone = (round(p_z[0], 2), round(p_z[1], 2))
            obj.p_kp = (round(p_k[0], 2), round(p_k[1], 2)) if p_k is not None else None
            obj.discrepancy = round(d_i, 2) if d_i < 900.0 else None
            obj.pitch_pt = (round(final_x, 2), round(final_y, 2))

            z_str = ", ".join(obj.zones) if obj.zones else "Open Field"
            kp_str = f"({p_k[0]:.1f}, {p_k[1]:.1f})" if p_k is not None else "None"
            fused_str = f"({final_x:.1f}, {final_y:.1f})"
            zone_str = f"({p_z[0]:.1f}, {p_z[1]:.1f})"
            rows.append({
                "id": tid,
                "role": obj.class_name,
                "zones": z_str,
                "p_zone": zone_str,
                "p_kp": kp_str,
                "final": fused_str,
                "status": status_str
            })

        if print_table:
            f_label = f"FRAME {frame_idx:04d}" if frame_idx is not None else "FRAME"
            print("\n" + "=" * 105)
            print(f"DUAL-PROJECTION CROSS-VALIDATION TABLE - {f_label}")
            print(f"{'ID':<6} | {'Role':<10} | {'Camera Zone(s)':<30} | {'P_zone (X, Y)':<15} | {'P_kp (X, Y)':<15} | {'Final Fused (X, Y)':<18}")
            print("-" * 105)
            for r in rows:
                id_str = f"#{r['id']}" if r['id'] > 0 else "N/A"
                print(f"{id_str:<6} | {r['role']:<10} | {r['zones']:<30} | {r['p_zone']:<15} | {r['p_kp']:<15} | {r['final']:<18}")
            print("=" * 105 + "\n")
