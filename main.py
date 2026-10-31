"""
Main Pipeline Orchestrator for Football Match Analyzer (Phase 1 MVP).
Executes Zone-First Homography, Player/GK/Referee Tracking, 2D Radar Projection,
and Video Export.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from config import (
    MODEL_PITCH_PATH,
    MODEL_KEYPOINTS_PATH,
    MODEL_DET_PATH
)
from zone_validator import ZoneValidator
from homography import HomographyEstimator
from tracker import PlayerTracker
from visualizer import TacticalVisualizer


def process_video(video_path: str,
                  output_path: Optional[str] = None,
                  max_frames: Optional[int] = None,
                  save_snapshots: bool = True) -> str:
    """
    Runs the full Phase 1 MVP computer vision pipeline on a target football video.
    """
    video_p = Path(video_path)
    if not video_p.exists():
        raise FileNotFoundError(f"Video not found at: {video_path}")

    if output_path is None:
        output_path = str(video_p.parent / f"annotated_{video_p.stem}.mp4")

    print("=" * 80)
    print(f"FOOTBALL MATCH ANALYZER - PHASE 1 MVP")
    print(f"Input Video : {video_path}")
    print(f"Output Video: {output_path}")
    if max_frames:
        print(f"Frame Limit : {max_frames} frames (fast validation mode)")
    print("=" * 80)

    # Open video capture
    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = min(total_frames, max_frames) if max_frames else total_frames

    print(f"Video Properties: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
          f"{fps:.2f} FPS, Total Frames: {total_frames} (Processing {target_frames} frames)")

    # Initialize Modules
    print("\nInitializing Computer Vision Modules...")
    zone_validator = ZoneValidator(pitch_model_path=MODEL_PITCH_PATH,
                                   keypoints_model_path=MODEL_KEYPOINTS_PATH)
    homography_estimator = HomographyEstimator()
    player_tracker = PlayerTracker(model_path=MODEL_DET_PATH)
    visualizer = TacticalVisualizer()
    print("All models and modules loaded successfully.\n")

    # Read first frame to determine composite canvas dimensions
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame from video.")

    # Process first frame to obtain dimensions
    zones, matches = zone_validator.process_frame(first_frame)
    H = homography_estimator.update(matches)
    tracked_objs = player_tracker.update(first_frame)

    radar_img = visualizer.draw_tactical_pitch(tracked_objs)
    bcast_img = visualizer.draw_broadcast_frame(first_frame, zones, matches, tracked_objs)
    comp_first = visualizer.create_composite_dashboard(
        bcast_img, radar_img, frame_idx=0, total_frames=target_frames,
        fps=fps, inlier_count=homography_estimator.inlier_count, tracked_objects=tracked_objs
    )

    comp_h, comp_w = comp_first.shape[:2]
    print(f"Composite Dashboard Resolution: {comp_w}x{comp_h}")

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (comp_w, comp_h))

    # Reset video to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    start_time = time.time()
    processed_count = 0

    snapshot_dir = Path("snapshots")
    if save_snapshots:
        snapshot_dir.mkdir(exist_ok=True)

    snapshot_indices = {
        int(target_frames * 0.1),
        int(target_frames * 0.3),
        int(target_frames * 0.6),
        int(target_frames * 0.9)
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames is not None and processed_count >= max_frames):
                break

            # ------------------------------------------------------------------
            # Step 1: Zone Detection & Synthetic/Snapped Landmark Extraction
            # ------------------------------------------------------------------
            zones, matches = zone_validator.process_frame(frame)

            # ------------------------------------------------------------------
            # Step 2: Homography Estimation & Temporal EMA Stabilization
            # ------------------------------------------------------------------
            H = homography_estimator.update(matches)

            # ------------------------------------------------------------------
            # Step 3: Multi-Object Detection & Tracking (best.pt + ByteTrack)
            # ------------------------------------------------------------------
            tracked_objects = player_tracker.update(frame)

            # ------------------------------------------------------------------
            # Step 4: 2D Radar Projection & Zone Consistency Enforcement
            # ------------------------------------------------------------------
            if len(tracked_objects) > 0 and H is not None:
                foot_points = np.array([o.foot_pt for o in tracked_objects], dtype=np.float32)

                # Get zones containing each player's feet
                zone_assocs = zone_validator.get_player_zone_associations(foot_points, zones)

                # Raw projection
                proj_pts = homography_estimator.project_points(foot_points)

                # Enforce zone consistency
                clamped_pts = homography_estimator.enforce_zone_consistency(
                    proj_pts, zone_assocs, pitch_side=zone_validator.last_known_side
                )

                for obj, pt_2d, z_list in zip(tracked_objects, clamped_pts, zone_assocs):
                    if not (np.isnan(pt_2d[0]) or np.isnan(pt_2d[1])):
                        obj.pitch_pt = (float(pt_2d[0]), float(pt_2d[1]))
                    obj.zones = z_list

            # ------------------------------------------------------------------
            # Step 5: Render Visual Displays
            # ------------------------------------------------------------------
            radar_canvas = visualizer.draw_tactical_pitch(tracked_objects)
            broadcast_canvas = visualizer.draw_broadcast_frame(frame, zones, matches, tracked_objects)

            dashboard = visualizer.create_composite_dashboard(
                broadcast_img=broadcast_canvas,
                radar_img=radar_canvas,
                frame_idx=processed_count + 1,
                total_frames=target_frames,
                fps=fps,
                inlier_count=homography_estimator.inlier_count,
                tracked_objects=tracked_objects
            )

            writer.write(dashboard)

            # Save snapshot frames for quick inspection
            if save_snapshots and processed_count in snapshot_indices:
                snap_path = snapshot_dir / f"{video_p.stem}_frame_{processed_count:04d}.jpg"
                cv2.imwrite(str(snap_path), dashboard)

            processed_count += 1
            if processed_count % 25 == 0 or processed_count == target_frames:
                elapsed = time.time() - start_time
                curr_fps = processed_count / elapsed
                rem_frames = target_frames - processed_count
                eta_sec = rem_frames / curr_fps if curr_fps > 0 else 0
                print(f"[{video_p.name}] Frame {processed_count:04d}/{target_frames:04d} | "
                      f"Speed: {curr_fps:.1f} FPS | Inliers: {homography_estimator.inlier_count:02d} | "
                      f"Tracked: {len(tracked_objects):02d} | ETA: {eta_sec:.0f}s")

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - start_time
    avg_fps = processed_count / total_time if total_time > 0 else 0.0

    print("=" * 80)
    print(f"Processing Complete for {video_p.name}!")
    print(f"Total Frames Processed : {processed_count}")
    print(f"Total Time             : {total_time:.1f}s (Average {avg_fps:.1f} FPS)")
    print(f"Saved Video Output     : {output_path}")
    print("=" * 80 + "\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Football Match Analyzer (Phase 1 MVP)")
    parser.add_argument("--video", type=str, default="match.mp4",
                        help="Path to input video file (e.g., match.mp4, match2.mp4, match3.mp4)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to export annotated video")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional maximum number of frames to process (for fast testing)")
    parser.add_argument("--no-snapshots", action="store_true",
                        help="Disable saving diagnostic snapshot images")

    args = parser.parse_args()
    process_video(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
        save_snapshots=not args.no_snapshots
    )


if __name__ == "__main__":
    main()

