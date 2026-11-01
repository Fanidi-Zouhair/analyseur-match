"""
Zone Validator Module — Segmentation-Only Edition (Phase 1 MVP).

Uses football_pitch.pt (99.3% precision) ONLY.
No football_keypoints.pt, no cv2.findHomography, no synthetic anchors.

Responsibilities:
  1. Detect the 7 pitch zones from football_pitch.pt masks.
  2. Determine pitch side (left / right) from zone centroids.
  3. Associate player foot-points to zones (point-in-polygon test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import MODEL_PITCH_PATH, CONF_ZONE, ZONE_CLASSES


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectedZone:
    cls_id: int
    name: str
    conf: float
    poly: np.ndarray      # (N, 2) int32 contour
    mask: np.ndarray      # 2-D binary uint8 mask  (H×W, values 0/255)
    side: str             # 'left', 'right', 'center', or 'unknown'


# ---------------------------------------------------------------------------
# ZoneValidator
# ---------------------------------------------------------------------------

class ZoneValidator:
    """
    Segmentation-only pitch-zone processor.
    Loads football_pitch.pt and performs zone detection + player association.
    Does NOT load or use football_keypoints.pt.
    """

    def __init__(self, pitch_model_path: str = MODEL_PITCH_PATH,
                 keypoints_model_path: Optional[str] = None):  # param kept for API compat, ignored
        self.pitch_model = YOLO(pitch_model_path)
        # keypoints_model_path intentionally not loaded
        self.last_known_side: str = "left"

    # ------------------------------------------------------------------
    # Public: process_frame
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Tuple[List[DetectedZone], list]:
        """
        Run zone segmentation on one frame.

        Returns:
            zones   — list of DetectedZone objects.
            matches — always an empty list (keypoint pipeline disabled).
        """
        h, w = frame.shape[:2]
        detected_zones: List[DetectedZone] = []

        res = self.pitch_model(frame, conf=CONF_ZONE, verbose=False)[0]

        if res.boxes is not None and res.masks is not None:
            for box, mask_data in zip(res.boxes, res.masks):
                cls_id = int(box.cls[0])
                name = self.pitch_model.names[cls_id]
                conf = float(box.conf[0])
                poly = mask_data.xy[0].astype(np.int32)
                if len(poly) < 3:
                    continue

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)

                detected_zones.append(DetectedZone(
                    cls_id=cls_id, name=name, conf=conf,
                    poly=poly, mask=mask, side="unknown"
                ))

        # Build zone_dict (highest-conf instance per name)
        zone_dict = self._build_zone_dict(detected_zones)

        # Determine pitch side and annotate zones
        side = self._detect_side(zone_dict, w)
        self.last_known_side = side

        for z in detected_zones:
            if z.name in ("18Yard", "18Yard Circle", "5Yard"):
                z.side = side
            elif "First Half" in z.name:
                z.side = "left"
            elif "Second Half" in z.name:
                z.side = "right"
            else:
                z.side = "center"

        return detected_zones, []   # empty matches — no keypoints

    # ------------------------------------------------------------------
    # Public: player zone association helpers
    # ------------------------------------------------------------------

    def get_player_zone_associations(
        self,
        foot_points: List[Tuple[float, float]],
        zones: List[DetectedZone]
    ) -> List[List[str]]:
        """
        For each foot-point, return the list of zone names whose mask contains it.
        Uses point-in-polygon test on the mask image.
        Priority order from most to least specific is preserved.
        """
        priority = [
            "5Yard", "18Yard", "18Yard Circle",
            "First Half Central Circle", "Second Half Central Circle",
            "First Half Field", "Second Half Field",
        ]
        zone_dict = self._build_zone_dict(zones)

        associations: List[List[str]] = []
        for (px, py) in foot_points:
            ipx, ipy = int(round(px)), int(round(py))
            found: List[str] = []
            for name in priority:
                z = zone_dict.get(name)
                if z is None:
                    continue
                if 0 <= ipy < z.mask.shape[0] and 0 <= ipx < z.mask.shape[1]:
                    if z.mask[ipy, ipx] > 0:
                        found.append(name)
            associations.append(found)
        return associations

    def get_zone_player_counts(
        self,
        foot_points: List[Tuple[float, float]],
        zones: List[DetectedZone]
    ) -> Dict[str, int]:
        """Returns a dict of {zone_name: player_count}."""
        associations = self.get_player_zone_associations(foot_points, zones)
        counts: Dict[str, int] = {}
        for zone_list in associations:
            seen = set()
            for name in zone_list:
                if name not in seen:
                    counts[name] = counts.get(name, 0) + 1
                    seen.add(name)
        return counts

    def build_zone_dict(self, zones: List[DetectedZone]) -> Dict[str, DetectedZone]:
        """Public wrapper around _build_zone_dict."""
        return self._build_zone_dict(zones)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_zone_dict(self, zones: List[DetectedZone]) -> Dict[str, DetectedZone]:
        """Keep highest-confidence instance of each zone name."""
        d: Dict[str, DetectedZone] = {}
        for z in zones:
            if z.name not in d or z.conf > d[z.name].conf:
                d[z.name] = z
        return d

    def _detect_side(self, zone_dict: Dict[str, DetectedZone], frame_width: int) -> str:
        """
        Determine left / right penalty side using ONLY mask X-centroids.

        Priority:
          1. 5Yard + 18Yard both visible: side based on relative centroid X.
          2. 18Yard + 18Yard Circle both visible: side based on relative centroid X.
          3. Single penalty zone visible: compare centroid vs frame mid (>50 px gap required).
          4. Border clipping: zone touches frame edge.
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

        # Rule 1: 5Yard & 18Yard
        if "5Yard" in cx and "18Yard" in cx:
            side = "left" if cx["5Yard"] < cx["18Yard"] else "right"
            self.last_known_side = side
            return side

        # Rule 2: 18Yard & 18Yard Circle
        if "18Yard" in cx and "18Yard Circle" in cx:
            side = "left" if cx["18Yard"] < cx["18Yard Circle"] else "right"
            self.last_known_side = side
            return side

        # Rule 3: single zone centroid with confidence margin
        for name in pz_names:
            if name in cx:
                dist = cx[name] - mid
                if abs(dist) > 50:
                    side = "left" if dist < 0 else "right"
                    self.last_known_side = side
                    return side

        # Rule 4: border clipping
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

        return self.last_known_side
