"""
Player, Goalkeeper, and Referee Detection & Tracking Module.
Uses best.pt combined with ByteTrack for persistent multi-object tracking.
Extracts bottom-center bounding box points (x_center, y_bottom) representing foot contact.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from config import MODEL_DET_PATH, TARGET_CLASSES, CONF_DETECTION


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    conf: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    foot_pt: Tuple[float, float]             # (x_center, y_bottom)
    pitch_pt: Optional[Tuple[float, float]] = None # Final fused & smoothed (X_m, Y_m)
    p_zone: Optional[Tuple[float, float]] = None   # Pipeline A (Zone-Relative)
    p_kp: Optional[Tuple[float, float]] = None     # Pipeline B (Keypoint / H)
    discrepancy: Optional[float] = None            # ||p_kp - p_zone|| in meters
    is_fused: bool = False
    zones: List[str] = field(default_factory=list)


class PlayerTracker:
    """
    Manages detection and ByteTrack tracking for match participants.
    """

    def __init__(self, model_path: str = MODEL_DET_PATH,
                 target_classes: Dict[int, str] = TARGET_CLASSES,
                 conf_thresh: float = CONF_DETECTION):
        self.model = YOLO(model_path)
        self.target_classes = target_classes
        self.conf_thresh = conf_thresh
        self.class_ids = list(target_classes.keys())

    def update(self, frame: np.ndarray) -> List[TrackedObject]:
        """
        Runs detection and ByteTrack on the current frame.
        Returns a list of TrackedObject instances.
        """
        tracked_objects: List[TrackedObject] = []

        res = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=self.class_ids,
            conf=self.conf_thresh,
            verbose=False
        )[0]

        if res.boxes is None:
            return tracked_objects

        for box in res.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.target_classes:
                continue

            class_name = self.target_classes[cls_id]
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

            # Calculate bottom-center foot contact point: (x_center, y_bottom)
            xc = (x1 + x2) / 2.0
            yb = y2

            tracked_objects.append(TrackedObject(
                track_id=track_id,
                class_id=cls_id,
                class_name=class_name,
                conf=conf,
                bbox=(x1, y1, x2, y2),
                foot_pt=(xc, yb),
                pitch_pt=None,
                zones=[]
            ))

        return tracked_objects

