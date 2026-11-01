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
    PITCH_LENGTH,
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
        self.last_known_geometry: Optional[Dict[str, Any]] = None

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

    def get_player_zone_associations(self, foot_points: Any,
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

    def get_zone_player_counts(self, foot_points: List[Tuple[float, float]],
                               zones: List[DetectedZone]) -> Dict[str, int]:
        """
        Counts the number of players inside each detected pitch zone.
        """
        counts: Dict[str, int] = {}
        for z in zones:
            counts[z.name] = 0

        for pt in foot_points:
            pt_tuple = (float(pt[0]), float(pt[1]))
            for z in zones:
                if cv2.pointPolygonTest(z.poly, pt_tuple, False) >= 0:
                    counts[z.name] += 1
        return counts

    def extract_reference_geometry(self, zones: List[DetectedZone],
                                   h: int, w: int,
                                   pitch_side: str = "left") -> Dict[str, Any]:
        """
        Extracts true, unclipped pitch reference lines and features from high-confidence
        zone segmentation masks (ignoring all image borders within 15px of 0, 0, W, H).
        """
        halfway_line = None
        cc_top = None
        cc_bot = None
        cc_spot = None
        cl = None
        cr = None

        # 1. Halfway line & central circle features
        # First Half Central Circle (curves left; straight diameter is on the right)
        for z in zones:
            if z.name == "First Half Central Circle":
                poly = z.poly
                unc = poly[(poly[:, 0] > 15) & (poly[:, 0] < w - 15) & (poly[:, 1] > 15) & (poly[:, 1] < h - 15)]
                if len(unc) >= 10:
                    p85 = np.percentile(unc[:, 0], 85)
                    straight = unc[unc[:, 0] >= p85]
                    if len(straight) >= 5:
                        vx, vy, x0, y0 = cv2.fitLine(straight.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                        halfway_line = (float(vx), float(vy), float(x0), float(y0))
                        y_top = float(straight[:, 1].min())
                        y_bot = float(straight[:, 1].max())
                        y_sp = (y_top + y_bot) / 2.0
                        slope = vx / (vy if abs(vy) > 1e-4 else 1e-4)
                        cc_top = (float(x0 + slope * (y_top - y0)), y_top)
                        cc_bot = (float(x0 + slope * (y_bot - y0)), y_bot)
                        cc_spot = (float(x0 + slope * (y_sp - y0)), y_sp)

                    cl_idx = np.argmin(unc[:, 0])
                    cl = (float(unc[cl_idx, 0]), float(unc[cl_idx, 1]))
                break

        # Second Half Central Circle (curves right; straight diameter is on the left)
        for z in zones:
            if z.name == "Second Half Central Circle":
                poly = z.poly
                unc = poly[(poly[:, 0] > 15) & (poly[:, 0] < w - 15) & (poly[:, 1] > 15) & (poly[:, 1] < h - 15)]
                if len(unc) >= 10:
                    cr_idx = np.argmax(unc[:, 0])
                    cr = (float(unc[cr_idx, 0]), float(unc[cr_idx, 1]))
                    if halfway_line is None:
                        p15 = np.percentile(unc[:, 0], 15)
                        straight = unc[unc[:, 0] <= p15]
                        if len(straight) >= 5:
                            vx, vy, x0, y0 = cv2.fitLine(straight.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                            halfway_line = (float(vx), float(vy), float(x0), float(y0))
                            y_top = float(straight[:, 1].min())
                            y_bot = float(straight[:, 1].max())
                            y_sp = (y_top + y_bot) / 2.0
                            slope = vx / (vy if abs(vy) > 1e-4 else 1e-4)
                            cc_top = (float(x0 + slope * (y_top - y0)), y_top)
                            cc_bot = (float(x0 + slope * (y_bot - y0)), y_bot)
                            cc_spot = (float(x0 + slope * (y_sp - y0)), y_sp)
                break

        # Fallback halfway line from boundary between First Half Field & Second Half Field
        if halfway_line is None:
            hw_pts = self._find_halfway_line_from_fields(zones, h, w)
            if hw_pts is not None:
                p_top, p_bot = hw_pts
                dx = p_bot[0] - p_top[0]
                dy = p_bot[1] - p_top[1]
                mag = np.hypot(dx, dy)
                if mag > 10.0:
                    halfway_line = (dx / mag, dy / mag, p_top[0], p_top[1])
                    cc_top = p_top
                    cc_bot = p_bot
                    cc_spot = ((p_top[0] + p_bot[0]) / 2.0, (p_top[1] + p_bot[1]) / 2.0)

        # 2. 18-Yard features (Infield line, apex of D-arc, top corner)
        line_18 = None
        arc_apex = None
        top_18_y = None

        for z in zones:
            if z.name == "18Yard Circle":
                poly = z.poly
                unc = poly[(poly[:, 0] > 15) & (poly[:, 0] < w - 15) & (poly[:, 1] > 15) & (poly[:, 1] < h - 15)]
                if len(unc) >= 10:
                    idx = np.argmax(unc[:, 0]) if pitch_side == "left" else np.argmin(unc[:, 0])
                    arc_apex = (float(unc[idx, 0]), float(unc[idx, 1]))
                break

        for z in zones:
            if z.name == "18Yard":
                poly = z.poly
                unc = poly[(poly[:, 0] > 15) & (poly[:, 0] < w - 15) & (poly[:, 1] > 15) & (poly[:, 1] < h - 15)]
                if len(unc) >= 10:
                    if pitch_side == "left":
                        p80 = np.percentile(unc[:, 0], 80)
                        infield = unc[unc[:, 0] >= p80]
                    else:
                        p20 = np.percentile(unc[:, 0], 20)
                        infield = unc[unc[:, 0] <= p20]
                    if len(infield) >= 4:
                        vx, vy, x0, y0 = cv2.fitLine(infield.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                        line_18 = (float(vx), float(vy), float(x0), float(y0))

                    # Top horizontal edge if not touching image top border
                    min_y_unc = unc[:, 1].min()
                    if min_y_unc > 20.0:
                        top_18_y = float(min_y_unc)
                break

        # 3. Far touchline (where green pitch meets advertising boards, Y = 0.0m)
        far_touchline = None
        field_pts = []
        for z in zones:
            if "Field" in z.name:
                poly = z.poly
                unc = poly[(poly[:, 0] > 15) & (poly[:, 0] < w - 15) & (poly[:, 1] > 15) & (poly[:, 1] < h - 15)]
                if len(unc) > 0:
                    field_pts.append(unc)

        if field_pts:
            all_f = np.vstack(field_pts)
            bins = np.arange(0, w, 50)
            top_pts = []
            for i in range(len(bins) - 1):
                pts_bin = all_f[(all_f[:, 0] >= bins[i]) & (all_f[:, 0] < bins[i + 1])]
                if len(pts_bin) > 0:
                    min_y_pt = pts_bin[np.argmin(pts_bin[:, 1])]
                    if min_y_pt[1] > 15:  # unclipped
                        top_pts.append(min_y_pt)
            if len(top_pts) >= 5:
                top_pts = np.array(top_pts)
                vx_f, vy_f, x0_f, y0_f = cv2.fitLine(top_pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                far_touchline = (float(vx_f), float(vy_f), float(x0_f), float(y0_f))

        # 4. Projective vertical mapping along Halfway Line
        # Fit strictly monotonic rational curve: Y(u) = (a * u) / (1 + c * u)
        # with u = y - y_far_h >= 0.
        halfway_proj = None
        if halfway_line is not None and far_touchline is not None and cc_top is not None and cc_bot is not None:
            vx_h, vy_h, x0_h, y0_h = halfway_line
            vx_f, vy_f, x0_f, y0_f = far_touchline
            slope_h = vx_h / (vy_h if abs(vy_h) > 1e-4 else 1e-4)
            slope_f = vy_f / (vx_f if abs(vx_f) > 1e-4 else 1e-4)

            denom = 1.0 - slope_f * slope_h
            if abs(denom) > 1e-4:
                y_far_h = (y0_f + slope_f * (x0_h - slope_h * y0_h - x0_f)) / denom
                x_far_h = x0_h + slope_h * (y_far_h - y0_h)

                u_top = max(cc_top[1] - y_far_h, 1.0)
                u_spot = max(cc_spot[1] - y_far_h, 2.0)
                u_bot = max(cc_bot[1] - y_far_h, 3.0)

                u_vals = np.array([u_top, u_spot, u_bot], dtype=np.float32)
                Y_vals = np.array([24.85, 34.00, 43.15], dtype=np.float32)

                # Linear regression on 1/Y = (1/a)*(1/u) + (c/a)
                x_lin = 1.0 / u_vals
                y_lin = 1.0 / Y_vals
                poly_res = np.polyfit(x_lin, y_lin, 1)
                m, b = poly_res[0], poly_res[1]

                a = 1.0 / m if m > 0 else 0.30
                c = b * a if b > 0 else 0.005
                halfway_proj = (float(a), float(c), float(x_far_h), float(y_far_h))

        # Merge with history if any feature was temporarily missing
        current_geom = {
            "pitch_side": pitch_side,
            "halfway_line": halfway_line,
            "cc_top": cc_top,
            "cc_bot": cc_bot,
            "cc_spot": cc_spot,
            "cl": cl,
            "cr": cr,
            "line_18": line_18,
            "arc_apex": arc_apex,
            "top_18_y": top_18_y,
            "far_touchline": far_touchline,
            "halfway_proj": halfway_proj
        }

        if self.last_known_geometry is not None:
            for k in ["halfway_line", "cc_top", "cc_bot", "cc_spot", "cl", "cr", "line_18", "arc_apex", "far_touchline", "halfway_proj"]:
                if current_geom[k] is None and self.last_known_geometry.get(k) is not None:
                    current_geom[k] = self.last_known_geometry[k]

        self.last_known_geometry = current_geom
        return current_geom

    def compute_zone_relative_positions(self, foot_points: Any,
                                        zones: List[DetectedZone],
                                        frame_shape: Tuple[int, int]) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        """
        Pipeline A: High-Reliability Zone-Relative Coordinate Interpolation.
        Uses true unclipped reference pitch lines to compute direct (X_zone, Y_zone) metric positions
        for each player foot contact point.
        """
        h, w = frame_shape[:2]
        pitch_side = self.last_known_side
        geom = self.extract_reference_geometry(zones, h, w, pitch_side=pitch_side)

        halfway_line = geom["halfway_line"]
        cl = geom["cl"]
        cr = geom["cr"]
        arc_apex = geom["arc_apex"]
        line_18 = geom["line_18"]
        far_touchline = geom["far_touchline"]
        halfway_proj = geom["halfway_proj"]

        # Default fallbacks if video starts without detected features
        vx_h, vy_h, x0_h, y0_h = halfway_line if halfway_line else (0.12, -0.99, w * 0.88, h * 0.45)
        slope_h = vx_h / (vy_h if abs(vy_h) > 1e-4 else 1e-4)

        if line_18:
            vx_18, vy_18, x0_18, y0_18 = line_18
            slope_18 = vx_18 / (vy_18 if abs(vy_18) > 1e-4 else 1e-4)
        else:
            slope_18 = slope_h
            x0_18, y0_18 = (w * 0.15, h * 0.3)

        vx_f, vy_f, x0_f, y0_f = far_touchline if far_touchline else (0.99, 0.08, w * 0.5, h * 0.15)
        slope_f = vy_f / (vx_f if abs(vx_f) > 1e-4 else 1e-4)

        if halfway_proj:
            a_proj, c_proj, x_far_h, y_far_h = halfway_proj
        else:
            a_proj, c_proj, x_far_h, y_far_h = (0.30, 0.005, w * 0.9, h * 0.2)

        p_zones: List[Tuple[float, float]] = []

        for pt in foot_points:
            xc, yb = float(pt[0]), float(pt[1])

            # ------------------------------------------------------------------
            # 1. Y_zone: Local vertical cross-section & monotonic projective mapping
            # ------------------------------------------------------------------
            y_far_at_x = y0_f + slope_f * (xc - x0_f)
            dy = yb - y_far_at_x
            y_eff = y_far_h + dy
            u = y_eff - y_far_h

            if u >= 0:
                Y_m = (a_proj * u) / (1.0 + c_proj * u)
            else:
                Y_m = a_proj * u  # slightly negative for sideline/bench

            Y_m = float(np.clip(Y_m, -2.0, PITCH_WIDTH + 2.0))

            # ------------------------------------------------------------------
            # 2. X_zone: Perspective piecewise line-ratio interpolation
            # ------------------------------------------------------------------
            x_half = x0_h + slope_h * (yb - y0_h)
            refs = [(x_half, 52.5)]

            if cl is not None:
                x_cl = cl[0] + slope_h * (yb - cl[1])
                refs.append((x_cl, 43.35))
            if cr is not None:
                x_cr = cr[0] + slope_h * (yb - cr[1])
                refs.append((x_cr, 61.65))
            if arc_apex is not None:
                x_apex = arc_apex[0] + slope_18 * (yb - arc_apex[1])
                X_apex_m = 20.15 if pitch_side == "left" else 84.85
                refs.append((x_apex, X_apex_m))
            if line_18 is not None:
                x_18 = x0_18 + slope_18 * (yb - y0_18)
                X_18_m = 16.5 if pitch_side == "left" else 88.5
                refs.append((x_18, X_18_m))

            refs = sorted(refs, key=lambda r: r[0])

            if len(refs) == 1:
                scale = 0.05
                X_m = 52.5 + (xc - refs[0][0]) * scale
            elif xc <= refs[0][0]:
                dx_px = refs[1][0] - refs[0][0]
                dX_m = refs[1][1] - refs[0][1]
                scale = dX_m / max(dx_px, 1e-4)
                X_m = refs[0][1] + (xc - refs[0][0]) * scale
            elif xc >= refs[-1][0]:
                dx_px = refs[-1][0] - refs[-2][0]
                dX_m = refs[-1][1] - refs[-2][1]
                scale = dX_m / max(dx_px, 1e-4)
                X_m = refs[-1][1] + (xc - refs[-1][0]) * scale
            else:
                X_m = 52.5
                for j in range(len(refs) - 1):
                    if refs[j][0] <= xc <= refs[j + 1][0]:
                        t = (xc - refs[j][0]) / max(refs[j + 1][0] - refs[j][0], 1e-4)
                        X_m = refs[j][1] + t * (refs[j + 1][1] - refs[j][1])
                        break

            X_m = float(np.clip(X_m, -2.0, PITCH_LENGTH + 2.0))
            p_zones.append((round(X_m, 2), round(Y_m, 2)))

        return p_zones, geom
