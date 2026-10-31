"""
Zone Validator Module: Zone-First Guided Keypoint Validation & Synthetic Anchor Extraction.
Uses football_pitch.pt (99.3% precision) to anchor the geometry, extract synthetic landmarks,
validate and snap football_keypoints.pt (41.8% precision) detections, and associate players to zones.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    MODEL_PITCH_PATH,
    MODEL_KEYPOINTS_PATH,
    CONF_ZONE,
    CONF_KEYPOINT,
    KEYPOINT_MAX_ZONE_DIST_PX,
    KEYPOINT_PITCH_COORDS,
    ZONE_CLASSES,
    HALF_LINE_X,
    PITCH_CENTER_Y,
    CENTER_CIRCLE_RADIUS,
    PENALTY_BOX_DEPTH,
    PENALTY_BOX_Y_MIN,
    PENALTY_BOX_Y_MAX,
    GOAL_BOX_DEPTH,
    GOAL_BOX_Y_MIN,
    GOAL_BOX_Y_MAX,
    PENALTY_ARC_LEFT_APEX,
    PENALTY_ARC_RIGHT_APEX
)


@dataclass
class KeypointMatch:
    name: str
    raw_pt: Tuple[float, float]
    snapped_pt: Tuple[float, float]
    pitch_pt: Tuple[float, float]
    conf: float
    is_synthetic: bool = False


@dataclass
class DetectedZone:
    cls_id: int
    name: str
    conf: float
    poly: np.ndarray          # (N, 2) array of contour points
    mask: np.ndarray          # 2D binary uint8 mask
    side: str                 # 'left', 'right', 'center', or 'unknown'


class ZoneValidator:
    """
    Implements Zone-First spatial reasoning for robust camera-to-pitch calibration.
    """

    def __init__(self, pitch_model_path: str = MODEL_PITCH_PATH,
                 keypoints_model_path: str = MODEL_KEYPOINTS_PATH):
        self.pitch_model = YOLO(pitch_model_path)
        self.keypoints_model = YOLO(keypoints_model_path)
        self.last_known_side: str = "left"

    def process_frame(self, frame: np.ndarray) -> Tuple[List[DetectedZone], List[KeypointMatch]]:
        """
        Processes a single frame:
        1. Runs zone segmentation (football_pitch.pt).
        2. Disambiguates pitch half (Left vs Right).
        3. Extracts high-confidence synthetic anchors from zone intersections and polygon corners.
        4. Runs keypoint detection (football_keypoints.pt), filters outliers, and snaps to zone boundaries.
        5. Returns structured zones and combined point pairs for homography estimation.
        """
        h, w = frame.shape[:2]

        # ----------------------------------------------------------------------
        # Step 1: Detect Zones with football_pitch.pt (99.3% precision)
        # ----------------------------------------------------------------------
        res_pitch = self.pitch_model(frame, conf=CONF_ZONE, verbose=False)[0]
        detected_zones: List[DetectedZone] = []

        if res_pitch.boxes is not None and res_pitch.masks is not None:
            for box, mask_data in zip(res_pitch.boxes, res_pitch.masks):
                cls_id = int(box.cls[0])
                name = self.pitch_model.names[cls_id]
                conf = float(box.conf[0])
                poly = mask_data.xy[0].astype(np.int32)
                if len(poly) < 3:
                    continue

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)

                detected_zones.append(DetectedZone(
                    cls_id=cls_id,
                    name=name,
                    conf=conf,
                    poly=poly,
                    mask=mask,
                    side="unknown"
                ))

        # ----------------------------------------------------------------------
        # Step 2: Disambiguate Pitch Half (Left vs Right)
        # ----------------------------------------------------------------------
        pitch_side = self._disambiguate_pitch_half(detected_zones, w)
        self.last_known_side = pitch_side

        for zone in detected_zones:
            if zone.name in ["18Yard", "18Yard Circle", "5Yard"]:
                zone.side = pitch_side
            elif "First Half" in zone.name:
                zone.side = "left"
            elif "Second Half" in zone.name:
                zone.side = "right"
            else:
                zone.side = "center"

        # ----------------------------------------------------------------------
        # Step 3: Extract Synthetic Anchors from Zone Geometry
        # ----------------------------------------------------------------------
        synthetic_anchors = self._extract_synthetic_anchors(detected_zones, pitch_side, h, w)

        # ----------------------------------------------------------------------
        # Step 4: Detect, Validate, Snap Keypoints from football_keypoints.pt
        # ----------------------------------------------------------------------
        res_kp = self.keypoints_model(frame, conf=CONF_KEYPOINT, verbose=False)[0]
        validated_keypoints = self._validate_and_snap_keypoints(res_kp, detected_zones, pitch_side)

        # ----------------------------------------------------------------------
        # Step 5: Merge Synthetic Anchors & Validated Keypoints
        # ----------------------------------------------------------------------
        all_matches = self._merge_landmarks(synthetic_anchors, validated_keypoints)

        return detected_zones, all_matches

    def _disambiguate_pitch_half(self, zones: List[DetectedZone], frame_width: int) -> str:
        """
        Determines whether the penalty area in view is Left or Right side of the field.
        """
        halfway_x = None

        # Check central circle or halfway line position
        for z in zones:
            if "Central Circle" in z.name:
                M = cv2.moments(z.poly)
                if M["m00"] > 0:
                    halfway_x = M["m10"] / M["m00"]
                    break

        # Check 18Yard centroid relative to halfway line
        for z in zones:
            if z.name in ["18Yard", "18Yard Circle", "5Yard"]:
                M = cv2.moments(z.poly)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    if halfway_x is not None:
                        return "left" if cx < halfway_x else "right"
                    return "left" if cx < (frame_width / 2.0) else "right"

        # Fallback to dominant field mask
        first_half_cnt = sum(1 for z in zones if "First Half" in z.name)
        second_half_cnt = sum(1 for z in zones if "Second Half" in z.name)
        if first_half_cnt > second_half_cnt:
            return "left"
        elif second_half_cnt > first_half_cnt:
            return "right"

        return self.last_known_side

    def _extract_synthetic_anchors(self, zones: List[DetectedZone], side: str,
                                   h: int, w: int) -> List[KeypointMatch]:
        """
        Extracts robust geometric landmarks directly from high-confidence zone masks.
        """
        anchors: List[KeypointMatch] = []
        zone_dict: Dict[str, DetectedZone] = {z.name: z for z in zones}

        # ------------------------------------------------------------------
        # A. Central Circle & Halfway Line Intersections
        # ------------------------------------------------------------------
        cc_first = zone_dict.get("First Half Central Circle")
        cc_second = zone_dict.get("Second Half Central Circle")

        if cc_first is not None and cc_second is not None:
            kernel = np.ones((5, 5), np.uint8)
            d1 = cv2.dilate(cc_first.mask, kernel, iterations=2)
            d2 = cv2.dilate(cc_second.mask, kernel, iterations=2)
            inter = cv2.bitwise_and(d1, d2)

            pts = np.column_stack(np.where(inter > 0)) # y, x
            if len(pts) > 10:
                pts_xy = pts[:, [1, 0]]
                y_min_idx = np.argmin(pts_xy[:, 1])
                y_max_idx = np.argmax(pts_xy[:, 1])

                top_pt = (float(pts_xy[y_min_idx, 0]), float(pts_xy[y_min_idx, 1]))
                bot_pt = (float(pts_xy[y_max_idx, 0]), float(pts_xy[y_max_idx, 1]))

                # Center of circle
                center_pt = ((top_pt[0] + bot_pt[0]) / 2.0, (top_pt[1] + bot_pt[1]) / 2.0)

                # Central circle top extreme on halfway line: (52.5, 24.85)
                anchors.append(KeypointMatch(
                    name="synthetic_cc_top",
                    raw_pt=top_pt,
                    snapped_pt=top_pt,
                    pitch_pt=(HALF_LINE_X, PITCH_CENTER_Y - CENTER_CIRCLE_RADIUS),
                    conf=0.95,
                    is_synthetic=True
                ))

                # Central circle bottom extreme on halfway line: (52.5, 43.15)
                anchors.append(KeypointMatch(
                    name="synthetic_cc_bottom",
                    raw_pt=bot_pt,
                    snapped_pt=bot_pt,
                    pitch_pt=(HALF_LINE_X, PITCH_CENTER_Y + CENTER_CIRCLE_RADIUS),
                    conf=0.95,
                    is_synthetic=True
                ))

                # Center spot: (52.5, 34.0)
                anchors.append(KeypointMatch(
                    name="synthetic_center_spot",
                    raw_pt=center_pt,
                    snapped_pt=center_pt,
                    pitch_pt=(HALF_LINE_X, PITCH_CENTER_Y),
                    conf=0.95,
                    is_synthetic=True
                ))

        # Central Circle Left / Right extremes
        if cc_first is not None:
            # Left extreme 'cl'
            min_x_idx = np.argmin(cc_first.poly[:, 0])
            cl_pt = (float(cc_first.poly[min_x_idx, 0]), float(cc_first.poly[min_x_idx, 1]))
            anchors.append(KeypointMatch(
                name="synthetic_cl",
                raw_pt=cl_pt,
                snapped_pt=cl_pt,
                pitch_pt=(HALF_LINE_X - CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y),
                conf=0.93,
                is_synthetic=True
            ))

        if cc_second is not None:
            # Right extreme 'cr'
            max_x_idx = np.argmax(cc_second.poly[:, 0])
            cr_pt = (float(cc_second.poly[max_x_idx, 0]), float(cc_second.poly[max_x_idx, 1]))
            anchors.append(KeypointMatch(
                name="synthetic_cr",
                raw_pt=cr_pt,
                snapped_pt=cr_pt,
                pitch_pt=(HALF_LINE_X + CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y),
                conf=0.93,
                is_synthetic=True
            ))

        # ------------------------------------------------------------------
        # B. 18-Yard Circle Apex (Penalty Arc 'D')
        # ------------------------------------------------------------------
        box_circle = zone_dict.get("18Yard Circle")
        if box_circle is not None and len(box_circle.poly) > 10:
            if side == "left":
                # Apex extends furthest right towards center circle
                idx = np.argmax(box_circle.poly[:, 0])
                apex_pt = (float(box_circle.poly[idx, 0]), float(box_circle.poly[idx, 1]))
                anchors.append(KeypointMatch(
                    name="synthetic_arc_left_apex",
                    raw_pt=apex_pt,
                    snapped_pt=apex_pt,
                    pitch_pt=PENALTY_ARC_LEFT_APEX,
                    conf=0.92,
                    is_synthetic=True
                ))
            else:
                # Right side apex extends furthest left towards center circle
                idx = np.argmin(box_circle.poly[:, 0])
                apex_pt = (float(box_circle.poly[idx, 0]), float(box_circle.poly[idx, 1]))
                anchors.append(KeypointMatch(
                    name="synthetic_arc_right_apex",
                    raw_pt=apex_pt,
                    snapped_pt=apex_pt,
                    pitch_pt=PENALTY_ARC_RIGHT_APEX,
                    conf=0.92,
                    is_synthetic=True
                ))

        # ------------------------------------------------------------------
        # C. 18-Yard Box Polygon Corners
        # ------------------------------------------------------------------
        box_18 = zone_dict.get("18Yard")
        if box_18 is not None:
            poly_f = box_18.poly.astype(np.float32)
            peri = cv2.arcLength(poly_f, True)
            approx = cv2.approxPolyDP(poly_f, 0.03 * peri, True)

            if len(approx) == 4:
                pts4 = approx.reshape(4, 2)
                # Sort by X: 2 goal line points vs 2 infield points
                sorted_by_x = pts4[np.argsort(pts4[:, 0])]

                if side == "left":
                    goal_line_pts = sorted_by_x[:2]
                    infield_pts = sorted_by_x[2:]
                    # Sort infield points by Y: top vs bottom
                    infield_top = infield_pts[np.argmin(infield_pts[:, 1])]
                    infield_bot = infield_pts[np.argmax(infield_pts[:, 1])]
                    goal_top = goal_line_pts[np.argmin(goal_line_pts[:, 1])]
                    goal_bot = goal_line_pts[np.argmax(goal_line_pts[:, 1])]

                    anchors.append(KeypointMatch(
                        name="synthetic_pbl3", raw_pt=tuple(infield_top), snapped_pt=tuple(infield_top),
                        pitch_pt=(PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN), conf=0.90, is_synthetic=True
                    ))
                    anchors.append(KeypointMatch(
                        name="synthetic_pbl4", raw_pt=tuple(infield_bot), snapped_pt=tuple(infield_bot),
                        pitch_pt=(PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX), conf=0.90, is_synthetic=True
                    ))
                else:
                    infield_pts = sorted_by_x[:2]
                    goal_line_pts = sorted_by_x[2:]
                    infield_top = infield_pts[np.argmin(infield_pts[:, 1])]
                    infield_bot = infield_pts[np.argmax(infield_pts[:, 1])]

                    anchors.append(KeypointMatch(
                        name="synthetic_pbr3", raw_pt=tuple(infield_top), snapped_pt=tuple(infield_top),
                        pitch_pt=(HALF_LINE_X * 2 - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN),
                        conf=0.90, is_synthetic=True
                    ))
                    anchors.append(KeypointMatch(
                        name="synthetic_pbr4", raw_pt=tuple(infield_bot), snapped_pt=tuple(infield_bot),
                        pitch_pt=(HALF_LINE_X * 2 - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX),
                        conf=0.90, is_synthetic=True
                    ))

        return anchors

    def _validate_and_snap_keypoints(self, res_kp: Any, zones: List[DetectedZone],
                                     side: str) -> List[KeypointMatch]:
        """
        Validates raw candidate keypoints against zone contours.
        Snaps slight drift to the nearest boundary and rejects wild outliers.
        """
        validated: List[KeypointMatch] = []
        if res_kp.boxes is None or res_kp.keypoints is None:
            return validated

        zone_dict: Dict[str, DetectedZone] = {z.name: z for z in zones}

        for box, kpt in zip(res_kp.boxes, res_kp.keypoints):
            cls_id = int(box.cls[0])
            name = self.keypoints_model.names[cls_id]
            conf = float(box.conf[0])
            if conf < CONF_KEYPOINT:
                continue

            xy = kpt.xy[0].cpu().numpy()[0]
            raw_pt = (float(xy[0]), float(xy[1]))

            # Correct Left/Right keypoint label if model inverted it
            canonical_name = name
            if side == "left":
                if name.startswith("pbr"):
                    canonical_name = "pbl" + name[3:]
                elif name.startswith("gr"):
                    canonical_name = "gl" + name[2:]
            elif side == "right":
                if name.startswith("pbl"):
                    canonical_name = "pbr" + name[3:]
                elif name.startswith("gl"):
                    canonical_name = "gr" + name[2:]

            if canonical_name not in KEYPOINT_PITCH_COORDS:
                continue

            target_pitch_coord = KEYPOINT_PITCH_COORDS[canonical_name]

            # Find matching zone mask for spatial verification
            relevant_zone = None
            if "pbl" in canonical_name or "pbr" in canonical_name:
                relevant_zone = zone_dict.get("18Yard") or zone_dict.get("18Yard Circle")
            elif "gl" in canonical_name or "gr" in canonical_name:
                relevant_zone = zone_dict.get("5Yard")
            elif canonical_name in ["cu", "cd", "cl", "cr", "cul", "cur", "cll", "clr"]:
                relevant_zone = (zone_dict.get("First Half Central Circle") or
                                 zone_dict.get("Second Half Central Circle"))

            snapped_pt = raw_pt

            if relevant_zone is not None:
                poly = relevant_zone.poly
                # Signed distance to contour (+ inside, - outside)
                dist = cv2.pointPolygonTest(poly, raw_pt, True)

                # Outlier rejection: if keypoint is way outside its zone, discard
                if dist < -KEYPOINT_MAX_ZONE_DIST_PX:
                    continue

                # Snapping: snap to nearest vertex or edge
                dists_to_vertices = np.linalg.norm(poly - np.array(raw_pt), axis=1)
                min_idx = int(np.argmin(dists_to_vertices))
                if dists_to_vertices[min_idx] <= KEYPOINT_MAX_ZONE_DIST_PX:
                    closest_v = poly[min_idx]
                    # Snap 70% towards vertex to maintain stability
                    snapped_pt = (
                        float(0.3 * raw_pt[0] + 0.7 * closest_v[0]),
                        float(0.3 * raw_pt[1] + 0.7 * closest_v[1])
                    )

            validated.append(KeypointMatch(
                name=canonical_name,
                raw_pt=raw_pt,
                snapped_pt=snapped_pt,
                pitch_pt=target_pitch_coord,
                conf=conf,
                is_synthetic=False
            ))

        return validated

    def _merge_landmarks(self, synthetic: List[KeypointMatch],
                         validated: List[KeypointMatch]) -> List[KeypointMatch]:
        """
        Combines synthetic anchors and validated keypoints, de-duplicating points that map
        to the exact same 2D metric target.
        """
        merged: List[KeypointMatch] = []
        seen_pitch_pts = set()

        # Prioritize high-precision synthetic anchors first
        for anchor in synthetic:
            pt_key = (round(anchor.pitch_pt[0], 1), round(anchor.pitch_pt[1], 1))
            seen_pitch_pts.add(pt_key)
            merged.append(anchor)

        # Append non-duplicate validated keypoints
        for kp in validated:
            pt_key = (round(kp.pitch_pt[0], 1), round(kp.pitch_pt[1], 1))
            if pt_key not in seen_pitch_pts:
                seen_pitch_pts.add(pt_key)
                merged.append(kp)

        return merged

    def get_player_zone_associations(self, foot_points: np.ndarray,
                                     zones: List[DetectedZone]) -> List[List[str]]:
        """
        Tests each player foot coordinate against all detected zone masks.
        Returns a list of zone names containing each player's feet.
        """
        associations: List[List[str]] = []
        for pt in foot_points:
            pt_tuple = (float(pt[0]), float(pt[1]))
            inside_zones = []
            for z in zones:
                if cv2.pointPolygonTest(z.poly, pt_tuple, False) >= 0:
                    inside_zones.append(z.name)
            associations.append(inside_zones)
        return associations
