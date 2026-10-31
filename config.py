"""
Configuration module for Football Match Analyzer (Phase 1 MVP).
Defines standard pitch dimensions, 2D coordinates for landmarks and zones,
model paths, and pipeline hyperparameters.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

# ==============================================================================
# Model File Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Object detection model (player, goalkeeper, referee)
MODEL_DET_PATH = str(BASE_DIR / "best.pt")

# Pitch Keypoint / Pose model (36 landmark classes)
MODEL_KEYPOINTS_PATH = str(BASE_DIR / "football_keypoints.pt")

# Pitch Zone Segmentation model (supports both spellings)
_PITCH_MODEL = BASE_DIR / "football_pitch.pt"
if not _PITCH_MODEL.exists():
    _PITCH_MODEL = BASE_DIR / "football_pith.pt"
MODEL_PITCH_PATH = str(_PITCH_MODEL)

# ==============================================================================
# Standard FIFA Football Pitch Template (Meters)
# ==============================================================================
# Pitch Length (X-axis): 0 to 105.0 m
# Pitch Width  (Y-axis): 0 to 68.0 m
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Pitch key reference measurements
HALF_LINE_X = PITCH_LENGTH / 2.0         # 52.5 m
PITCH_CENTER_Y = PITCH_WIDTH / 2.0       # 34.0 m
CENTER_CIRCLE_RADIUS = 9.15              # 9.15 m

# 18-Yard Box (Penalty Area): extends 16.5m from goal line, width 40.32m
PENALTY_BOX_DEPTH = 16.5                 # 16.5 m
PENALTY_BOX_WIDTH = 40.32                # 40.32 m
PENALTY_BOX_Y_MIN = PITCH_CENTER_Y - (PENALTY_BOX_WIDTH / 2.0)  # 13.84 m
PENALTY_BOX_Y_MAX = PITCH_CENTER_Y + (PENALTY_BOX_WIDTH / 2.0)  # 54.16 m

# 5-Yard / 6-Yard Box (Goal Area): extends 5.5m from goal line, width 18.32m
GOAL_BOX_DEPTH = 5.5                     # 5.5 m
GOAL_BOX_WIDTH = 18.32                   # 18.32 m
GOAL_BOX_Y_MIN = PITCH_CENTER_Y - (GOAL_BOX_WIDTH / 2.0)        # 24.84 m
GOAL_BOX_Y_MAX = PITCH_CENTER_Y + (GOAL_BOX_WIDTH / 2.0)        # 43.16 m

# Penalty spots
PENALTY_SPOT_DIST = 11.0                 # 11.0 m from goal line
PENALTY_SPOT_LEFT = (PENALTY_SPOT_DIST, PITCH_CENTER_Y)               # (11.0, 34.0)
PENALTY_SPOT_RIGHT = (PITCH_LENGTH - PENALTY_SPOT_DIST, PITCH_CENTER_Y) # (94.0, 34.0)

# Penalty Arc (D-arc): radius 9.15m centered at penalty spot
# Apex of left D-arc: 11.0 + 9.15 = 20.15 m
PENALTY_ARC_LEFT_APEX = (PENALTY_SPOT_DIST + CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y)
# Apex of right D-arc: 94.0 - 9.15 = 84.85 m
PENALTY_ARC_RIGHT_APEX = (PITCH_LENGTH - PENALTY_SPOT_DIST - CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y)

# ==============================================================================
# 2D Metric Pitch Coordinates for All 36 Keypoint Classes
# ==============================================================================
KEYPOINT_PITCH_COORDS: Dict[str, Tuple[float, float]] = {
    # Center Pitch & Halfway Line
    "cu": (HALF_LINE_X, 0.0),                               # Center Up (top touchline)
    "cd": (HALF_LINE_X, PITCH_WIDTH),                        # Center Down (bottom touchline)
    "cl": (HALF_LINE_X - CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y), # Center Left (43.35, 34.0)
    "cr": (HALF_LINE_X + CENTER_CIRCLE_RADIUS, PITCH_CENTER_Y), # Center Right (61.65, 34.0)
    "cul": (HALF_LINE_X - 6.47, PITCH_CENTER_Y - 6.47),     # Center Upper-Left quadrant
    "cur": (HALF_LINE_X + 6.47, PITCH_CENTER_Y - 6.47),     # Center Upper-Right quadrant
    "cll": (HALF_LINE_X - 6.47, PITCH_CENTER_Y + 6.47),     # Center Lower-Left quadrant
    "clr": (HALF_LINE_X + 6.47, PITCH_CENTER_Y + 6.47),     # Center Lower-Right quadrant
    "hlu": (HALF_LINE_X, 0.0),                               # Halfway line upper
    "hll": (HALF_LINE_X, PITCH_WIDTH),                       # Halfway line lower

    # Pitch Boundary Corners (Quadrant flags)
    "qlul": (0.0, 0.0),                                     # Quadrant Upper-Left
    "qlll": (0.0, PITCH_WIDTH),                             # Quadrant Lower-Left
    "qlur": (PITCH_LENGTH, 0.0),                            # Quadrant Upper-Right
    "qllr": (PITCH_LENGTH, PITCH_WIDTH),                    # Quadrant Lower-Right

    # Left Penalty Box (pbl1 .. pbl6)
    "pbl1": (0.0, PENALTY_BOX_Y_MIN),                       # Goal line top corner (0.0, 13.84)
    "pbl2": (PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN + 12.85), # Penalty arc top intersection (16.5, 26.69)
    "pbl3": (PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN),         # Infield top corner (16.5, 13.84)
    "pbl4": (PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX),         # Infield bottom corner (16.5, 54.16)
    "pbl5": (PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX - 12.85), # Penalty arc bottom intersection (16.5, 41.31)
    "pbl6": (0.0, PENALTY_BOX_Y_MAX),                       # Goal line bottom corner (0.0, 54.16)
    "pl": PENALTY_SPOT_LEFT,                                # Left penalty spot (11.0, 34.0)

    # Right Penalty Box (pbr1 .. pbr6)
    "pbr1": (PITCH_LENGTH, PENALTY_BOX_Y_MIN),              # Goal line top corner (105.0, 13.84)
    "pbr2": (PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN + 12.85), # Penalty arc top intersection (88.5, 26.69)
    "pbr3": (PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN),         # Infield top corner (88.5, 13.84)
    "pbr4": (PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX),         # Infield bottom corner (88.5, 54.16)
    "pbr5": (PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX - 12.85), # Penalty arc bottom intersection (88.5, 41.31)
    "pbr6": (PITCH_LENGTH, PENALTY_BOX_Y_MAX),              # Goal line bottom corner (105.0, 54.16)
    "pr": PENALTY_SPOT_RIGHT,                               # Right penalty spot (94.0, 34.0)

    # Goal Areas (6-yard boxes: gl1..gl4 for left, gr1..gr4 for right)
    "gl1": (0.0, GOAL_BOX_Y_MIN),                           # Left goal line top (0.0, 24.84)
    "gl2": (GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN),                # Left infield top (5.5, 24.84)
    "gl3": (GOAL_BOX_DEPTH, GOAL_BOX_Y_MAX),                # Left infield bottom (5.5, 43.16)
    "gl4": (0.0, GOAL_BOX_Y_MAX),                           # Left goal line bottom (0.0, 43.16)

    "gr1": (PITCH_LENGTH, GOAL_BOX_Y_MIN),                  # Right goal line top (105.0, 24.84)
    "gr2": (PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN), # Right infield top (99.5, 24.84)
    "gr3": (PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_MAX), # Right infield bottom (99.5, 43.16)
    "gr4": (PITCH_LENGTH, GOAL_BOX_Y_MAX),                  # Right goal line bottom (105.0, 43.16)
}

# ==============================================================================
# Pitch Zone Semantic Labels (football_pitch.pt)
# ==============================================================================
ZONE_CLASSES = {
    0: "18Yard",
    1: "18Yard Circle",
    2: "5Yard",
    3: "First Half Central Circle",
    4: "First Half Field",
    5: "Second Half Central Circle",
    6: "Second Half Field",
}

# Mapping between pitch zones and expected keypoint families
ZONE_KEYPOINT_MAPPING = {
    "18Yard_left": ["pbl1", "pbl2", "pbl3", "pbl4", "pbl5", "pbl6", "pl"],
    "18Yard_right": ["pbr1", "pbr2", "pbr3", "pbr4", "pbr5", "pbr6", "pr"],
    "5Yard_left": ["gl1", "gl2", "gl3", "gl4"],
    "5Yard_right": ["gr1", "gr2", "gr3", "gr4"],
    "CentralCircle": ["cu", "cd", "cl", "cr", "cul", "cur", "cll", "clr", "hlu", "hll"],
}

# 2D Template Bounding Box Polygons for Zones (for player zone clamping)
ZONE_TEMPLATE_POLYGONS = {
    "18Yard_left": [
        (0.0, PENALTY_BOX_Y_MIN),
        (PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN),
        (PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX),
        (0.0, PENALTY_BOX_Y_MAX)
    ],
    "18Yard_right": [
        (PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MIN),
        (PITCH_LENGTH, PENALTY_BOX_Y_MIN),
        (PITCH_LENGTH, PENALTY_BOX_Y_MAX),
        (PITCH_LENGTH - PENALTY_BOX_DEPTH, PENALTY_BOX_Y_MAX)
    ],
    "5Yard_left": [
        (0.0, GOAL_BOX_Y_MIN),
        (GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN),
        (GOAL_BOX_DEPTH, GOAL_BOX_Y_MAX),
        (0.0, GOAL_BOX_Y_MAX)
    ],
    "5Yard_right": [
        (PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_MIN),
        (PITCH_LENGTH, GOAL_BOX_Y_MIN),
        (PITCH_LENGTH, GOAL_BOX_Y_MAX),
        (PITCH_LENGTH - GOAL_BOX_DEPTH, GOAL_BOX_Y_MAX)
    ],
    "First Half Field": [
        (0.0, 0.0),
        (HALF_LINE_X, 0.0),
        (HALF_LINE_X, PITCH_WIDTH),
        (0.0, PITCH_WIDTH)
    ],
    "Second Half Field": [
        (HALF_LINE_X, 0.0),
        (PITCH_LENGTH, 0.0),
        (PITCH_LENGTH, PITCH_WIDTH),
        (HALF_LINE_X, PITCH_WIDTH)
    ]
}

# ==============================================================================
# Detection & Tracking Config (best.pt)
# ==============================================================================
TARGET_CLASSES = {
    2: "goalkeeper",
    3: "player",
    6: "referee"
}

# Confidence thresholds
CONF_ZONE = 0.25
CONF_KEYPOINT = 0.20
CONF_DETECTION = 0.35

# Keypoint validation distance threshold (in pixels)
KEYPOINT_MAX_ZONE_DIST_PX = 80.0

# ==============================================================================
# Homography & Smoothing Parameters
# ==============================================================================
RANSAC_REPROJ_THRESHOLD = 5.0
HOMOGRAPHY_EMA_ALPHA = 0.25            # Temporal smoothing factor for H matrix
MAX_HOMOGRAPHY_CACHE_AGE = 30          # Fallback duration (frames)

# ==============================================================================
# Visualization Styling
# ==============================================================================
COLOR_PLAYER = (255, 60, 60)          # BGR: Blueish / Red
COLOR_GOALKEEPER = (0, 200, 255)      # BGR: Yellow / Orange
COLOR_REFEREE = (0, 255, 255)         # BGR: Bright Yellow
COLOR_ZONE_FILL = (0, 180, 0)         # Translucent pitch green
COLOR_KEYPOINT_RAW = (0, 0, 255)      # Red for raw keypoint
COLOR_KEYPOINT_SNAPPED = (0, 255, 0)  # Green for snapped/validated
COLOR_SYNTHETIC_ANCHOR = (255, 0, 255)# Magenta for synthetic anchors

# Radar canvas size (pixels)
RADAR_CANVAS_WIDTH = 640
RADAR_CANVAS_HEIGHT = int(RADAR_CANVAS_WIDTH * (PITCH_WIDTH / PITCH_LENGTH)) # ~414 px

