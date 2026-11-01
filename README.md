# Football Match Analyzer (Phase 1 MVP)

An end-to-end Computer Vision pipeline for tactical football analysis. The system detects and tracks players, goalkeepers, and referees from broadcast camera footage, projecting their real-time positions onto a 2D tactical pitch (radar view).

## Architecture: Zone-First Guided Homography
- **Anchor Zone Segmentation (`football_pitch.pt`)**: High-precision segmentation (18-yard box, 6-yard box, halfway line, center circle).
- **Synthetic Landmark Extraction**: Extracts deterministic geometric corners and intersections directly from high-confidence zone masks, avoiding dependency on noisy keypoints.
- **Zone-Guided Keypoint Validation (`football_keypoints.pt`)**: Snaps drifting keypoints to physical boundaries and rejects spatial outliers.
- **Stabilized Homography**: Estimates $H$ via RANSAC with Exponential Moving Average (EMA) temporal smoothing.
- **Multi-Object Tracking (`best.pt` + ByteTrack)**: Persistent tracking of players, goalkeepers, and referees with bottom-center foot contact points.
- **Zone-Consistency Clamping**: Ensures player radar projections strictly respect their segmented zone boundaries.

## Project Structure
```
├── config.py           # Pitch template dimensions, landmark coordinates, hyperparameters
├── inspect_models.py   # Model inspection and diagnostic utility
├── zone_validator.py   # Pitch zone segmentation, anchor extraction & snapping
├── homography.py       # Homography estimation, temporal smoothing & projection
├── tracker.py          # Player/GK/Referee detection & ByteTrack tracking
├── visualizer.py       # Broadcast overlay & 2D tactical pitch radar rendering
├── main.py             # Pipeline runner and video export
└── requirements.txt    # Project dependencies
```

## Setup & Usage

### 1. Activate the Virtual Environment
On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Or call `.\.venv\Scripts\python.exe` directly)*

### 2. Run Analysis on a Match Video
```bash
# Full video processing
python main.py --video match.mp4 --output annotated_match.mp4

# Fast validation test (e.g. 50 frames)
python main.py --video match.mp4 --max-frames 50
```

