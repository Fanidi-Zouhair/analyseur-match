"""
Homography & Camera Calibration Module.
Implements:
1. ViewTransformer (Roboflow sports / tvcalib style coordinate transformations).
2. Optical Flow Camera Motion Tracking (PyrLK + RANSAC homography chaining).
3. TVCalib-Style Mask-to-Template IoU Evaluation & Optimization (refine_homography_with_masks).
4. Zone Consistency Enforcement & Per-Track EMA smoothing.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from scipy.optimize import minimize

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
    RANSAC_REPROJ_THRESHOLD,
    HOMOGRAPHY_EMA_ALPHA,
    MAX_HOMOGRAPHY_CACHE_AGE
)
from zone_validator import KeypointMatch, DetectedZone
from tracker import TrackedObject


# ==============================================================================
# 1. ViewTransformer (Roboflow sports style)
# ==============================================================================

class ViewTransformer:
    """
    Projective coordinate transformer between camera pixels and 2D pitch metric coordinates.
    """

    def __init__(self, H: np.ndarray):
        self.H = H
        try:
            self.H_inv = np.linalg.inv(H)
        except Exception:
            self.H_inv = None

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms (N, 2) camera image pixel points to (N, 2) 2D pitch coordinates in meters.
        """
        if self.H is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        try:
            transformed = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
            return transformed
        except Exception:
            return np.full((len(points), 2), np.nan, dtype=np.float32)

    def project_to_camera(self, pitch_points: np.ndarray) -> np.ndarray:
        """
        Transforms (N, 2) 2D pitch metric coordinates to (N, 2) camera image pixels.
        """
        if self.H_inv is None or len(pitch_points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.asarray(pitch_points, dtype=np.float32).reshape(-1, 1, 2)
        try:
            transformed = cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)
            return transformed
        except Exception:
            return np.full((len(pitch_points), 2), np.nan, dtype=np.float32)


# ==============================================================================
# 2. Camera Motion Tracker (Optical Flow Homography Chaining)
# ==============================================================================

class CameraMotionTracker:
    """
    Tracks frame-to-frame broadcast camera motion using Lucas-Kanade optical flow
    on pitch background features (excluding players) to propagate homography seamlessly.
    """

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_H: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray,
               pitch_mask: Optional[np.ndarray],
               player_boxes: List[Tuple[float, float, float, float]],
               current_H: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
        """
        Tracks camera motion from prev_frame to curr_frame.
        Returns:
            H_pred: Predicted homography propagated via optical flow.
            inlier_ratio: Proportion of tracked background points supporting Delta_H.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        H_pred = None
        inlier_ratio = 0.0

        if self.prev_gray is not None and self.prev_H is not None:
            # Mask out moving player bounding boxes from optical flow tracking
            h, w = frame.shape[:2]
            flow_mask = pitch_mask.copy() if pitch_mask is not None else np.full((h, w), 255, dtype=np.uint8)

            for box in player_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                # Expand box slightly to eliminate player contours
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 10)
                x2 = min(w - 1, x2 + 10)
                y2 = min(h - 1, y2 + 10)
                flow_mask[y1:y2, x1:x2] = 0

            # Detect feature points on grass pitch background
            pts_prev = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=300,
                qualityLevel=0.01,
                minDistance=15,
                mask=flow_mask
            )

            if pts_prev is not None and len(pts_prev) >= 20:
                pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, curr_gray, pts_prev, None,
                    winSize=(21, 21), maxLevel=3
                )
                valid = (status.flatten() == 1)
                p0 = pts_prev[valid]
                p1 = pts_curr[valid]

                if len(p0) >= 15:
                    delta_H, inliers = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
                    if delta_H is not None:
                        inlier_cnt = int(inliers.sum()) if inliers is not None else 0
                        inlier_ratio = inlier_cnt / len(p0)
                        det = np.linalg.det(delta_H)
                        if abs(det) > 1e-4 and inlier_cnt >= 15:
                            try:
                                delta_H_inv = np.linalg.inv(delta_H)
                                # x_pitch = H_prev * x_prev = H_prev * delta_H_inv * x_curr
                                H_pred = self.prev_H @ delta_H_inv
                                if abs(H_pred[2, 2]) > 1e-8:
                                    H_pred = H_pred / H_pred[2, 2]
                            except Exception:
                                H_pred = None

        # Update state for next frame
        self.prev_gray = curr_gray
        if current_H is not None:
            self.prev_H = current_H.copy()
        elif H_pred is not None:
            self.prev_H = H_pred.copy()

        return H_pred, inlier_ratio


# ==============================================================================
# 3. TVCalib-Style Mask-to-Template IoU & Refinement
# ==============================================================================

class SemanticPitchTemplate:
    """
    Ground-truth FIFA 2D pitch template (1050 x 680, 10px = 1m).
    Evaluates multi-class mask IoU and optimizes homography candidates.
    """

    SCALE: float = 10.0 # 10 px per meter
    DS: int = 4         # Downsample factor for fast IoU evaluation (480 x 270)

    def __init__(self):
        self.tw = int(PITCH_LENGTH * self.SCALE) # 1050
        self.th = int(PITCH_WIDTH * self.SCALE)  # 680
        self.template_masks = self._build_template_masks()

        # Matrix mapping pitch metric coords (meters) to template map coords (pixels)
        self.S_metric2template = np.array([
            [self.SCALE, 0.0, 0.0],
            [0.0, self.SCALE, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def _m2t(self, xm: float, ym: float) -> Tuple[int, int]:
        return int(round(xm * self.SCALE)), int(round(ym * self.SCALE))

    def _build_template_masks(self) -> Dict[str, np.ndarray]:
        masks = {}

        # 1. First Half Field
        m_fh = np.zeros((self.th, self.tw), dtype=np.uint8)
        cv2.rectangle(m_fh, self._m2t(0.0, 0.0), self._m2t(HALF_LINE_X, PITCH_WIDTH), 255, -1)
        masks["First Half Field"] = m_fh

        # 2. Second Half Field
        m_sh = np.zeros((self.th, self.tw), dtype=np.uint8)
        cv2.rectangle(m_sh, self._m2t(HALF_LINE_X, 0.0), self._m2t(PITCH_LENGTH, PITCH_WIDTH), 255, -1)
        masks["Second Half Field"] = m_sh

        # 3. First Half Central Circle
        m_cc1 = np.zeros((self.th, self.tw), dtype=np.uint8)
        cv2.circle(m_cc1, self._m2t(HALF_LINE_X, PITCH_CENTER_Y), int(CENTER_CIRCLE_RADIUS * self.SCALE), 255, -1)
        m_cc1[:, int(HALF_LINE_X * self.SCALE):] = 0
        masks["First Half Central Circle"] = m_cc1

        # 4. Second Half Central Circle
        m_cc2 = np.zeros((self.th, self.tw), dtype=np.uint8)
        cv2.circle(m_cc2, self._m2t(HALF_LINE_X, PITCH_CENTER_Y), int(CENTER_CIRCLE_RADIUS * self.SCALE), 255, -1)
        m_cc2[:, :int(HALF_LINE_X * self.SCALE)] = 0
        masks["Second Half Central Circle"] = m_cc2

        # 5. 5Yard Left & Right
        m_5y_l = np.zeros((self.th, self.tw), dtype=np.uint8)
        cv2.rectangle(m_5y_l, self._m2t(0.0, GOAL_BOX_Y_MIN), self._m2t(GOAL_BOX_DEPTH, GOAL_BOX_Y_MAX), 255, -1)
        masks["5Yard_left"] = m_5y_l

        m_5y_r = np.zeros((self.th, self.tw), dtype=np.uint8)
        cv2.rectangle(m_5y_r, self._m2t(PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN), self._m2t(PITCH_LENGTH, GOAL_BOX_Y_MAX), 255, -1)
        masks["5Yard_right"] = m_5y_r

        # 6. 18Yard Left & Right
        m_18y_l = np.zeros((self.th, self.tw), dtype=np.uint8)
        cv2.rectangle(m_18y_l, self._m2t(0.0, PENALTY_BOX_Y_MIN), self._m2t(PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX), 255, -1)
        masks["18Yard_left"] = m_18y_l

        m_18y_r = np.zeros((self.th, self.tw), dtype=np.uint8)
        cv2.rectangle(m_18y_r, self._m2t(PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN), self._m2t(PITCH_LENGTH, PENALTY_BOX_Y_MAX), 255, -1)
        masks["18Yard_right"] = m_18y_r

        # 7. 18Yard Circle Left & Right (D-arc)
        m_18yc_l = np.zeros((self.th, self.tw), dtype=np.uint8)
        cv2.circle(m_18yc_l, self._m2t(PENALTY_SPOT_DIST, PITCH_CENTER_Y), int(CENTER_CIRCLE_RADIUS * self.SCALE), 255, -1)
        m_18yc_l[:, :int(PENALTY_BOX_DEPTH * self.SCALE)] = 0
        masks["18Yard Circle_left"] = m_18yc_l

        m_18yc_r = np.zeros((self.th, self.tw), dtype=np.uint8)
        cv2.circle(m_18yc_r, self._m2t(PITCH_LENGTH - PENALTY_SPOT_DIST, PITCH_CENTER_Y), int(CENTER_CIRCLE_RADIUS * self.SCALE), 255, -1)
        m_18yc_r[:, int((PITCH_LENGTH - PENALTY_BOX_DEPTH) * self.SCALE):] = 0
        masks["18Yard Circle_right"] = m_18yc_r

        return masks

    def evaluate_iou(self, H: np.ndarray,
                     zone_dict: Dict[str, DetectedZone],
                     frame_shape: Tuple[int, int],
                     pitch_side: str) -> float:
        """
        Computes multi-zone Mask IoU score between warped template and detected masks.
        """
        if H is None:
            return 0.0

        h, w = frame_shape
        w_ds, h_ds = w // self.DS, h // self.DS
        S_ds = np.array([[self.DS, 0, 0], [0, self.DS, 0], [0, 0, 1]], dtype=np.float32)

        try:
            H_c2t = self.S_metric2template @ H
            H_ds2t = H_c2t @ S_ds
            H_t2ds = np.linalg.inv(H_ds2t)
        except Exception:
            return 0.0

        total_inter = 0
        total_union = 0

        for z_name, z in zone_dict.items():
            t_key = z_name
            if z_name in ["18Yard", "5Yard", "18Yard Circle"]:
                t_key = f"{z_name}_{pitch_side}"

            if t_key not in self.template_masks:
                continue

            t_mask = self.template_masks[t_key]
            warped = cv2.warpPerspective(t_mask, H_t2ds, (w_ds, h_ds))
            det_ds = cv2.resize(z.mask, (w_ds, h_ds), interpolation=cv2.INTER_NEAREST)

            inter = np.logical_and(warped > 128, det_ds > 128).sum()
            union = np.logical_or(warped > 128, det_ds > 128).sum()

            total_inter += inter
            total_union += union

        if total_union == 0:
            return 0.0
        return float(total_inter / total_union)

    def refine_homography_with_masks(self, H_init: np.ndarray,
                                     zone_dict: Dict[str, DetectedZone],
                                     frame_shape: Tuple[int, int],
                                     pitch_side: str) -> Tuple[np.ndarray, float]:
        """
        Refines homography candidate by optimizing Mask IoU using fast Nelder-Mead search.
        """
        if H_init is None:
            return H_init, 0.0

        initial_iou = self.evaluate_iou(H_init, zone_dict, frame_shape, pitch_side)
        if initial_iou >= 0.85:
            return H_init, initial_iou

        h, w = frame_shape
        w_ds, h_ds = w // self.DS, h // self.DS
        S_ds = np.array([[self.DS, 0, 0], [0, self.DS, 0], [0, 0, 1]], dtype=np.float32)

        # Pre-downscale masks for fast evaluation
        downscaled_det: Dict[str, np.ndarray] = {}
        for z_name, z in zone_dict.items():
            t_key = z_name
            if z_name in ["18Yard", "5Yard", "18Yard Circle"]:
                t_key = f"{z_name}_{pitch_side}"
            if t_key in self.template_masks:
                downscaled_det[t_key] = cv2.resize(z.mask, (w_ds, h_ds), interpolation=cv2.INTER_NEAREST)

        def loss_func(params):
            dp = np.array([
                [1.0 + params[0], params[1], params[2]],
                [params[3], 1.0 + params[4], params[5]],
                [params[6] * 1e-4, params[7] * 1e-4, 1.0]
            ], dtype=np.float32)
            H_cand = dp @ H_init
            try:
                H_c2t = self.S_metric2template @ H_cand
                H_ds2t = H_c2t @ S_ds
                H_t2ds = np.linalg.inv(H_ds2t)
            except Exception:
                return 1.0

            inter_sum = 0
            union_sum = 0
            for t_key, det_mask in downscaled_det.items():
                t_mask = self.template_masks[t_key]
                warped = cv2.warpPerspective(t_mask, H_t2ds, (w_ds, h_ds))
                inter = np.logical_and(warped > 128, det_mask > 128).sum()
                union = np.logical_or(warped > 128, det_mask > 128).sum()
                inter_sum += inter
                union_sum += union

            if union_sum == 0:
                return 1.0
            return 1.0 - (inter_sum / union_sum)

        p0 = np.zeros(8, dtype=np.float32)
        try:
            res = minimize(
                loss_func, p0, method='Nelder-Mead',
                options={'maxiter': 20, 'xatol': 1e-3, 'fatol': 1e-3}
            )
            opt_iou = 1.0 - res.fun
            if opt_iou > initial_iou:
                dp = np.array([
                    [1.0 + res.x[0], res.x[1], res.x[2]],
                    [res.x[3], 1.0 + res.x[4], res.x[5]],
                    [res.x[6] * 1e-4, res.x[7] * 1e-4, 1.0]
                ], dtype=np.float32)
                H_opt = dp @ H_init
                if abs(H_opt[2, 2]) > 1e-8:
                    H_opt = H_opt / H_opt[2, 2]
                return H_opt, opt_iou
        except Exception:
            pass

        return H_init, initial_iou


# ==============================================================================
# 4. HomographyEstimator Orchestrator
# ==============================================================================

class HomographyEstimator:
    """
    Coordinates observation homography, optical flow camera motion prediction,
    and TVCalib-style Mask IoU optimization.
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
        self.current_iou: float = 0.0

        # Components
        self.camera_tracker = CameraMotionTracker()
        self.template_matcher = SemanticPitchTemplate()
        self.view_transformer: Optional[ViewTransformer] = None
        self.player_track_ema: Dict[int, Tuple[float, float]] = {}
        self.track_ema_alpha: float = 0.35

    def update(self,
               matches: List[KeypointMatch],
               frame: np.ndarray,
               zone_dict: Dict[str, DetectedZone],
               player_boxes: List[Tuple[float, float, float, float]],
               pitch_side: str) -> Optional[np.ndarray]:
        """
        Calculates or propagates homography H:
        1. Predicts H_pred via optical flow camera motion tracking.
        2. Estimates H_obs from unclipped shared boundary landmarks.
        3. Evaluates and selects candidate via Mask IoU scoring.
        4. Refines selected H using Mask-to-Template IoU optimization.
        5. Stabilizes H temporally via EMA.
        """
        h, w = frame.shape[:2]

        # Union field mask for optical flow
        field_mask = np.zeros((h, w), dtype=np.uint8)
        for z in zone_dict.values():
            field_mask = cv2.bitwise_or(field_mask, z.mask)

        # 1. Optical Flow Camera Motion Prediction
        H_pred, flow_ratio = self.camera_tracker.update(
            frame=frame,
            pitch_mask=field_mask,
            player_boxes=player_boxes,
            current_H=self.H_smooth
        )

        # 2. Observation Homography from Shared Boundary Landmarks
        H_obs = None
        inliers_obs = 0
        if len(matches) >= 4:
            src_pts = np.array([m.snapped_pt for m in matches], dtype=np.float32)
            dst_pts = np.array([m.pitch_pt for m in matches], dtype=np.float32)

            if self._check_spatial_spread(dst_pts):
                H_cand, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
                if H_cand is not None and self._is_homography_valid(H_cand, src_pts):
                    if abs(H_cand[2, 2]) > 1e-8:
                        H_cand = H_cand / H_cand[2, 2]
                    H_obs = H_cand
                    inliers_obs = int(inlier_mask.sum()) if inlier_mask is not None else 0

        # 3. Candidate Selection via Mask IoU
        iou_obs = self.template_matcher.evaluate_iou(H_obs, zone_dict, (h, w), pitch_side) if H_obs is not None else 0.0
        iou_pred = self.template_matcher.evaluate_iou(H_pred, zone_dict, (h, w), pitch_side) if H_pred is not None else 0.0

        selected_H = None
        if H_obs is not None and iou_obs >= 0.50:
            selected_H = H_obs
            self.inlier_count = inliers_obs
            self.current_iou = iou_obs
        elif H_pred is not None and iou_pred >= 0.45:
            selected_H = H_pred
            self.inlier_count = int(flow_ratio * 100)
            self.current_iou = iou_pred
        elif H_obs is not None:
            selected_H = H_obs
            self.inlier_count = inliers_obs
            self.current_iou = iou_obs
        elif self.H_smooth is not None and self.cache_age < self.max_cache_age:
            selected_H = self.H_smooth
            self.cache_age += 1

        if selected_H is None:
            self.view_transformer = None
            return None

        # 4. TVCalib Mask IoU Refinement
        refined_H, refined_iou = self.template_matcher.refine_homography_with_masks(
            selected_H, zone_dict, (h, w), pitch_side
        )
        self.current_iou = refined_iou

        # 5. Temporal EMA Smoothing
        if self.H_smooth is None or self.cache_age > 10:
            self.H_smooth = refined_H.copy()
        else:
            self.H_smooth = (self.ema_alpha * refined_H) + ((1.0 - self.ema_alpha) * self.H_smooth)
            if abs(self.H_smooth[2, 2]) > 1e-8:
                self.H_smooth = self.H_smooth / self.H_smooth[2, 2]

        self.last_valid_H = self.H_smooth.copy()
        self.cache_age = 0
        self.view_transformer = ViewTransformer(self.H_smooth)

        return self.H_smooth

    def project_and_constrain_players(self,
                                      tracked_objects: List[TrackedObject],
                                      zone_dict: Dict[str, DetectedZone],
                                      pitch_side: str) -> None:
        """
        Projects player foot points using ViewTransformer, enforces zone consistency,
        and applies per-track EMA smoothing.
        """
        if self.view_transformer is None or len(tracked_objects) == 0:
            return

        foot_pts = np.array([o.foot_pt for o in tracked_objects], dtype=np.float32)
        proj_pts = self.view_transformer.transform_points(foot_pts)

        for i, obj in enumerate(tracked_objects):
            if i >= len(proj_pts):
                continue
            xm, ym = float(proj_pts[i, 0]), float(proj_pts[i, 1])

            if np.isnan(xm) or np.isnan(ym):
                continue

            # Zone Membership Guarantee
            if "5Yard" in obj.zones:
                if pitch_side == "left":
                    xm = np.clip(xm, 0.0, GOAL_BOX_DEPTH)
                    ym = np.clip(ym, GOAL_BOX_Y_MIN, GOAL_BOX_Y_MAX)
                else:
                    xm = np.clip(xm, PITCH_LENGTH - GOAL_BOX_DEPTH, PITCH_LENGTH)
                    ym = np.clip(ym, GOAL_BOX_Y_MIN, GOAL_BOX_Y_MAX)
            elif "18Yard" in obj.zones:
                if pitch_side == "left":
                    xm = np.clip(xm, 0.0, PENALTY_BOX_DEPTH)
                    ym = np.clip(ym, PENALTY_BOX_Y_MIN, PENALTY_BOX_Y_MAX)
                else:
                    xm = np.clip(xm, PITCH_LENGTH - PENALTY_BOX_DEPTH, PITCH_LENGTH)
                    ym = np.clip(ym, PENALTY_BOX_Y_MIN, PENALTY_BOX_Y_MAX)
            elif "18Yard Circle" in obj.zones:
                if pitch_side == "left":
                    xm = np.clip(xm, PENALTY_BOX_DEPTH, PENALTY_BOX_DEPTH + CENTER_CIRCLE_RADIUS)
                else:
                    xm = np.clip(xm, PITCH_LENGTH - PENALTY_BOX_DEPTH - CENTER_CIRCLE_RADIUS, PITCH_LENGTH - PENALTY_BOX_DEPTH)

            # Global Pitch Clamping
            xm = float(np.clip(xm, -1.0, PITCH_LENGTH + 1.0))
            ym = float(np.clip(ym, -1.0, PITCH_WIDTH + 1.0))

            # Per-track EMA smoothing
            tid = obj.track_id
            if tid in self.player_track_ema:
                prev_x, prev_y = self.player_track_ema[tid]
                smoothed_x = self.track_ema_alpha * xm + (1.0 - self.track_ema_alpha) * prev_x
                smoothed_y = self.track_ema_alpha * ym + (1.0 - self.track_ema_alpha) * prev_y
            else:
                smoothed_x, smoothed_y = xm, ym

            self.player_track_ema[tid] = (smoothed_x, smoothed_y)
            obj.pitch_pt = (smoothed_x, smoothed_y)

    def _check_spatial_spread(self, dst_pts: np.ndarray,
                              min_x_span: float = 5.0,
                              min_y_span: float = 5.0) -> bool:
        x_span = dst_pts[:, 0].max() - dst_pts[:, 0].min()
        y_span = dst_pts[:, 1].max() - dst_pts[:, 1].min()
        if x_span < min_x_span or y_span < min_y_span:
            return False
        return True

    def _is_homography_valid(self, H: np.ndarray, src_pts: np.ndarray) -> bool:
        if np.isnan(H).any() or np.isinf(H).any():
            return False
        det = np.linalg.det(H)
        if abs(det) < 1e-12:
            return False
        cond = np.linalg.cond(H)
        if cond > 1e10:
            return False

        # Verify that H does NOT collapse horizontal span into a vertical line
        test_pts = np.array([[[100.0, 300.0], [500.0, 300.0], [1000.0, 300.0], [1500.0, 300.0]]], dtype=np.float32)
        try:
            proj = cv2.perspectiveTransform(test_pts, H).reshape(-1, 2)
            proj_x_std = float(np.std(proj[:, 0]))
            if proj_x_std < 2.5: # Degenerate vertical-line collapse
                return False
            # Check projection range is reasonable
            if proj[:, 0].min() < -60.0 or proj[:, 0].max() > 165.0:
                return False
            if proj[:, 1].min() < -50.0 or proj[:, 1].max() > 120.0:
                return False
        except Exception:
            return False

        return True
