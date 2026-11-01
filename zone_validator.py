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
    PITCH_WIDTH,
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

    def _find_halfway_line_from_fields(self, zones: List[DetectedZone],
                                       h: int, w: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Finds the halfway line in image coordinates by intersecting First Half Field
        and Second Half Field mask boundaries. Returns (top_pt, bottom_pt) or None.
        """
        first_half = None
        second_half = None
        for z in zones:
            if z.name == "First Half Field" and first_half is None:
                first_half = z
            elif z.name == "Second Half Field" and second_half is None:
                second_half = z

        if first_half is None or second_half is None:
            return None

        kernel = np.ones((7, 7), np.uint8)
        d1 = cv2.dilate(first_half.mask, kernel, iterations=2)
        d2 = cv2.dilate(second_half.mask, kernel, iterations=2)
        inter = cv2.bitwise_and(d1, d2)

        pts = np.column_stack(np.where(inter > 0))  # y, x
        if len(pts) < 10:
            return None

        pts_xy = pts[:, [1, 0]].astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        y_min = float(pts_xy[:, 1].min())
        y_max = float(pts_xy[:, 1].max())

        if abs(vy) < 1e-6:
            return None

        x_at_ymin = float(x0 + (y_min - y0) * (vx / vy))
        x_at_ymax = float(x0 + (y_max - y0) * (vx / vy))

        return ((x_at_ymin, y_min), (x_at_ymax, y_max))

    def _extract_synthetic_anchors(self, zones: List[DetectedZone], side: str,
                                   h: int, w: int) -> List[KeypointMatch]:
        """
        Extracts robust geometric landmarks directly from high-confidence zone masks.
        Uses curvature analysis on semicircle masks to distinguish the straight halfway-line
        edge from the curved arc edge.
        """
        anchors: List[KeypointMatch] = []

        # Collect all zones by name (keep first highest-confidence instance of each)
        zone_dict: Dict[str, DetectedZone] = {}
        for z in zones:
            if z.name not in zone_dict or z.conf > zone_dict[z.name].conf:
                zone_dict[z.name] = z

        # ------------------------------------------------------------------
        # A. Central Circle Anchors (Curvature-Aware)
        # ------------------------------------------------------------------
        # Process each visible half-circle independently using curvature analysis.
        # The straight edge of a semicircle mask IS the halfway line (X=52.5).
        # The curved arc edge extends to cl (43.35, 34.0) or cr (61.65, 34.0).
        for cc_name in ["First Half Central Circle", "Second Half Central Circle"]:
            cc_zone = zone_dict.get(cc_name)
            if cc_zone is None or len(cc_zone.poly) < 20:
                continue

            poly = cc_zone.poly.astype(np.float32)

            # Fit a minimum-area bounding rectangle to distinguish straight vs curved edge
            rect = cv2.minAreaRect(poly)
            box_pts = cv2.boxPoints(rect).astype(np.float32)

            # Find the centroid of the mask
            M = cv2.moments(poly)
            if M["m00"] < 1:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Determine which side (screen-left or screen-right) the straight edge lies on.
            # The straight (halfway-line) edge is the side facing toward the other half.
            # For "First Half Central Circle", the straight edge is the rightmost vertical edge.
            # For "Second Half Central Circle", the straight edge is the leftmost vertical edge.
            if cc_name == "First Half Central Circle":
                # Straight edge is the rightmost vertical part of the contour
                # Curved arc extends to the LEFT (minimum X)
                straight_edge_pts = poly[poly[:, 0] > cx]
                arc_extreme_idx = np.argmin(poly[:, 0])
            else:
                # Straight edge is the leftmost vertical part of the contour
                # Curved arc extends to the RIGHT (maximum X)
                straight_edge_pts = poly[poly[:, 0] < cx]
                arc_extreme_idx = np.argmax(poly[:, 0])

            # --- Extract straight edge top and bottom (halfway line intersects center circle) ---
            if len(straight_edge_pts) > 5:
                # Fit a line to the straight edge points
                vx_l, vy_l, x0_l, y0_l = cv2.fitLine(
                    straight_edge_pts, cv2.DIST_L2, 0, 0.01, 0.01
                ).flatten()

                # Top and bottom of the straight edge
                se_y_min = float(straight_edge_pts[:, 1].min())
                se_y_max = float(straight_edge_pts[:, 1].max())

                if abs(vy_l) > 1e-6:
                    se_x_at_ymin = float(x0_l + (se_y_min - y0_l) * (vx_l / vy_l))
                    se_x_at_ymax = float(x0_l + (se_y_max - y0_l) * (vx_l / vy_l))
                else:
                    se_x_at_ymin = float(x0_l)
                    se_x_at_ymax = float(x0_l)

                top_pt = (se_x_at_ymin, se_y_min)
                bot_pt = (se_x_at_ymax, se_y_max)
                center_pt = ((top_pt[0] + bot_pt[0]) / 2.0, (top_pt[1] + bot_pt[1]) / 2.0)

                # Top of straight edge = center circle top on halfway line: (52.5, 24.85)
                anchors.append(KeypointMatch(
                    name=f"synthetic_cc_top_{cc_name[:5]}",
                    raw_pt=top_pt, snapped_pt=top_pt,
                    pitch_pt=(HALF_LINE_X, PITCH_CENTER_Y - CENTER_CIRCLE_RADIUS),
                    conf=0.95, is_synthetic=True
                ))

                # Bottom of straight edge = center circle bottom on halfway line: (52.5, 43.15)
                anchors.append(KeypointMatch(
                    name=f"synthetic_cc_bot_{cc_name[:5]}",
                    raw_pt=bot_pt, snapped_pt=bot_pt,
                    pitch_pt=(HALF_LINE_X, PITCH_CENTER_Y + CENTER_CIRCLE_RADIUS),
                    conf=0.95, is_synthetic=True
                ))

                # Center spot: (52.5, 34.0)
                anchors.append(KeypointMatch(
                    name=f"synthetic_center_spot_{cc_name[:5]}",
                    raw_pt=center_pt, snapped_pt=center_pt,
                    pitch_pt=(HALF_LINE_X, PITCH_CENTER_Y),
                    conf=0.95, is_synthetic=True
                ))

            # --- Curved arc extreme (cl or cr) ---
            arc_pt = (float(poly[arc_extreme_idx, 0]), float(poly[arc_extreme_idx, 1]))

            if cc_name == "First Half Central Circle":
                # Arc extends LEFT -> cl: (43.35, 34.0)
                anchors.append(KeypointMatch(
                    name="synthetic_cl",
                    raw_pt=arc_pt, snapped_pt=arc_pt,
                    pitch_pt=(HALF_LINE_X - CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y),
                    conf=0.93, is_synthetic=True
                ))
            else:
                # Arc extends RIGHT -> cr: (61.65, 34.0)
                anchors.append(KeypointMatch(
                    name="synthetic_cr",
                    raw_pt=arc_pt, snapped_pt=arc_pt,
                    pitch_pt=(HALF_LINE_X + CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y),
                    conf=0.93, is_synthetic=True
                ))

        # ------------------------------------------------------------------
        # A2. Touchline / Far-Touchline Anchor from First Half/Second Half Field boundary
        # ------------------------------------------------------------------
        hw_line = self._find_halfway_line_from_fields(zones, h, w)
        if hw_line is not None:
            top_hw, bot_hw = hw_line
            # cu: (52.5, 0.0) — top touchline intersection
            anchors.append(KeypointMatch(
                name="synthetic_cu", raw_pt=top_hw, snapped_pt=top_hw,
                pitch_pt=(HALF_LINE_X, 0.0), conf=0.92, is_synthetic=True
            ))
            # cd: (52.5, 68.0) — bottom touchline intersection
            anchors.append(KeypointMatch(
                name="synthetic_cd", raw_pt=bot_hw, snapped_pt=bot_hw,
                pitch_pt=(HALF_LINE_X, PITCH_WIDTH), conf=0.92, is_synthetic=True
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
                    raw_pt=apex_pt, snapped_pt=apex_pt,
                    pitch_pt=PENALTY_ARC_LEFT_APEX,
                    conf=0.92, is_synthetic=True
                ))
            else:
                idx = np.argmin(box_circle.poly[:, 0])
                apex_pt = (float(box_circle.poly[idx, 0]), float(box_circle.poly[idx, 1]))
                anchors.append(KeypointMatch(
                    name="synthetic_arc_right_apex",
                    raw_pt=apex_pt, snapped_pt=apex_pt,
                    pitch_pt=PENALTY_ARC_RIGHT_APEX,
                    conf=0.92, is_synthetic=True
                ))

        # ------------------------------------------------------------------
        # B2. 18Yard ∩ 18Yard Circle intersection points (pbl2/pbl5 or pbr2/pbr5)
        # ------------------------------------------------------------------
        box_18 = zone_dict.get("18Yard")
        if box_18 is not None and box_circle is not None:
            kernel3 = np.ones((5, 5), np.uint8)
            d_18 = cv2.dilate(box_18.mask, kernel3, iterations=2)
            d_circ = cv2.dilate(box_circle.mask, kernel3, iterations=2)
            inter_mask = cv2.bitwise_and(d_18, d_circ)

            pts_inter = np.column_stack(np.where(inter_mask > 0))  # y, x
            if len(pts_inter) > 5:
                pts_inter_xy = pts_inter[:, [1, 0]]
                # Split into top cluster and bottom cluster by Y
                median_y = np.median(pts_inter_xy[:, 1])
                top_cluster = pts_inter_xy[pts_inter_xy[:, 1] < median_y]
                bot_cluster = pts_inter_xy[pts_inter_xy[:, 1] >= median_y]

                if len(top_cluster) > 2:
                    top_center = top_cluster.mean(axis=0)
                    if side == "left":
                        anchors.append(KeypointMatch(
                            name="synthetic_pbl2",
                            raw_pt=(float(top_center[0]), float(top_center[1])),
                            snapped_pt=(float(top_center[0]), float(top_center[1])),
                            pitch_pt=(PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN + 12.85),
                            conf=0.90, is_synthetic=True
                        ))
                    else:
                        anchors.append(KeypointMatch(
                            name="synthetic_pbr2",
                            raw_pt=(float(top_center[0]), float(top_center[1])),
                            snapped_pt=(float(top_center[0]), float(top_center[1])),
                            pitch_pt=(105.0 - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN + 12.85),
                            conf=0.90, is_synthetic=True
                        ))

                if len(bot_cluster) > 2:
                    bot_center = bot_cluster.mean(axis=0)
                    if side == "left":
                        anchors.append(KeypointMatch(
                            name="synthetic_pbl5",
                            raw_pt=(float(bot_center[0]), float(bot_center[1])),
                            snapped_pt=(float(bot_center[0]), float(bot_center[1])),
                            pitch_pt=(PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX - 12.85),
                            conf=0.90, is_synthetic=True
                        ))
                    else:
                        anchors.append(KeypointMatch(
                            name="synthetic_pbr5",
                            raw_pt=(float(bot_center[0]), float(bot_center[1])),
                            snapped_pt=(float(bot_center[0]), float(bot_center[1])),
                            pitch_pt=(105.0 - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX - 12.85),
                            conf=0.90, is_synthetic=True
                        ))

        # ------------------------------------------------------------------
        # C. 18-Yard Box Polygon Corners (robust even when clipped)
        # ------------------------------------------------------------------
        if box_18 is not None:
            poly_f = box_18.poly.astype(np.float32)

            if side == "left":
                # Infield top corner pbl3 (16.5, 13.84): rightmost point with minimum Y
                # Find the rightmost X points (infield edge)
                x_threshold = np.percentile(poly_f[:, 0], 80)
                infield_edge = poly_f[poly_f[:, 0] >= x_threshold]

                if len(infield_edge) > 2:
                    # Top infield corner: highest Y among infield edge points
                    top_idx = np.argmin(infield_edge[:, 1])
                    bot_idx = np.argmax(infield_edge[:, 1])
                    infield_top = infield_edge[top_idx]
                    infield_bot = infield_edge[bot_idx]

                    anchors.append(KeypointMatch(
                        name="synthetic_pbl3",
                        raw_pt=(float(infield_top[0]), float(infield_top[1])),
                        snapped_pt=(float(infield_top[0]), float(infield_top[1])),
                        pitch_pt=(PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN),
                        conf=0.90, is_synthetic=True
                    ))
                    anchors.append(KeypointMatch(
                        name="synthetic_pbl4",
                        raw_pt=(float(infield_bot[0]), float(infield_bot[1])),
                        snapped_pt=(float(infield_bot[0]), float(infield_bot[1])),
                        pitch_pt=(PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX),
                        conf=0.90, is_synthetic=True
                    ))
            else:
                # Right side: infield edge has minimum X
                x_threshold = np.percentile(poly_f[:, 0], 20)
                infield_edge = poly_f[poly_f[:, 0] <= x_threshold]

                if len(infield_edge) > 2:
                    top_idx = np.argmin(infield_edge[:, 1])
                    bot_idx = np.argmax(infield_edge[:, 1])
                    infield_top = infield_edge[top_idx]
                    infield_bot = infield_edge[bot_idx]

                    anchors.append(KeypointMatch(
                        name="synthetic_pbr3",
                        raw_pt=(float(infield_top[0]), float(infield_top[1])),
                        snapped_pt=(float(infield_top[0]), float(infield_top[1])),
                        pitch_pt=(105.0 - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN),
                        conf=0.90, is_synthetic=True
                    ))
                    anchors.append(KeypointMatch(
                        name="synthetic_pbr4",
                        raw_pt=(float(infield_bot[0]), float(infield_bot[1])),
                        snapped_pt=(float(infield_bot[0]), float(infield_bot[1])),
                        pitch_pt=(105.0 - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX),
                        conf=0.90, is_synthetic=True
                    ))

        return anchors

    def _validate_and_snap_keypoints(self, res_kp: Any, zones: List[DetectedZone],
                                     side: str) -> List[KeypointMatch]:
        """
        Validates raw candidate keypoints against zone contours.
        Snaps slight drift to the nearest boundary and rejects wild outliers.
        NEVER snaps cu/cd/hlu/hll (touchline points) to the Central Circle mask.
        """
        validated: List[KeypointMatch] = []
        if res_kp.boxes is None or res_kp.keypoints is None:
            return validated

        zone_dict: Dict[str, DetectedZone] = {z.name: z for z in zones}

        # Touchline / halfway-line keypoints that should NOT be snapped to central circle
        TOUCHLINE_KPS = {"cu", "cd", "hlu", "hll"}

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

            # ---- BUG 2 FIX: Assign correct zone for validation ----
            # Touchline points (cu, cd, hlu, hll) should validate against the
            # First/Second Half Field boundary, NOT the Central Circle.
            relevant_zone = None
            if canonical_name in TOUCHLINE_KPS:
                # Validate against First/Second Half Field (not central circle)
                relevant_zone = (zone_dict.get("First Half Field") or
                                 zone_dict.get("Second Half Field"))
            elif "pbl" in canonical_name or "pbr" in canonical_name:
                relevant_zone = zone_dict.get("18Yard") or zone_dict.get("18Yard Circle")
            elif "gl" in canonical_name or "gr" in canonical_name:
                relevant_zone = zone_dict.get("5Yard")
            elif canonical_name in ["cl", "cr"]:
                relevant_zone = (zone_dict.get("First Half Central Circle") or
                                 zone_dict.get("Second Half Central Circle"))
            elif canonical_name in ["cul", "cur", "cll", "clr"]:
                relevant_zone = (zone_dict.get("First Half Central Circle") or
                                 zone_dict.get("Second Half Central Circle"))

            snapped_pt = raw_pt

            if relevant_zone is not None and canonical_name not in TOUCHLINE_KPS:
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
            if pt_key not in seen_pitch_pts:
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
