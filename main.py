"""
Main Pipeline Orchestrator — Segmentation-Only Edition (Phase 1 MVP).

Pipeline:
  1. football_pitch.pt → zone masks + side detection.
  2. best.pt + ByteTrack → player / GK / referee tracking.
  3. seg_mapper.SegMapper → 100% mask-based 2D position (no homography).
  4. TacticalVisualizer → broadcast + radar composite.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import MODEL_PITCH_PATH, MODEL_DET_PATH
from zone_validator import ZoneValidator
from seg_mapper import SegMapper
from tracker import PlayerTracker
from visualizer import TacticalVisualizer


# Frames at which we print full diagnostic tables
DIAGNOSTIC_FRAMES = {1, 70, 138}

# Snapshot frame indices (absolute) — overridden in process_video by % of total
FIXED_SNAPSHOT_FRAMES = {1, 70, 138}


def print_diagnostic_table(
    frame_idx: int,
    tracked_objects,
    zone_counts: dict,
    pitch_side: str,
) -> None:
    sep = "─" * 72
    print(f"\n{'═' * 72}")
    print(f"  DIAGNOSTIC TABLE — Frame {frame_idx:04d}  |  Pitch side: {pitch_side.upper()}")
    print(sep)
    print(f"  {'Zone':<30} {'Player Count':>12}")
    print(sep)
    for zn, cnt in sorted(zone_counts.items(), key=lambda x: -x[1]):
        print(f"  {zn:<30} {cnt:>12}")
    print(sep)

    # XY ranges
    xs = [o.pitch_pt[0] for o in tracked_objects if o.pitch_pt is not None and not np.isnan(o.pitch_pt[0])]
    ys = [o.pitch_pt[1] for o in tracked_objects if o.pitch_pt is not None and not np.isnan(o.pitch_pt[1])]
    if xs:
        print(f"  X range: [{min(xs):.1f}, {max(xs):.1f}] m  (std={float(np.std(xs)):.2f})")
        print(f"  Y range: [{min(ys):.1f}, {max(ys):.1f}] m  (std={float(np.std(ys)):.2f})")
    print(f"{'═' * 72}\n")


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    save_snapshots: bool = True,
) -> str:
    video_p = Path(video_path)
    if not video_p.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = str(video_p.parent / f"annotated_{video_p.stem}.mp4")

    print("=" * 72)
    print("FOOTBALL MATCH ANALYZER — PHASE 1 MVP (Segmentation-Only)")
    print(f"Input : {video_path}")
    print(f"Output: {output_path}")
    if max_frames:
        print(f"Limit : {max_frames} frames")
    print("=" * 72)

    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = min(total_frames, max_frames) if max_frames else total_frames

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {vid_w}×{vid_h}, {fps:.2f} FPS, {total_frames} frames "
          f"(processing {target_frames})")

    # ── Initialise modules ─────────────────────────────────────────────
    print("\nLoading models…")
    zone_validator  = ZoneValidator(pitch_model_path=MODEL_PITCH_PATH)
    seg_mapper      = SegMapper()
    player_tracker  = PlayerTracker(model_path=MODEL_DET_PATH)
    visualizer      = TacticalVisualizer()
    print("Models loaded.\n")

    # ── Determine composite resolution from first frame ────────────────
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    zones_f, _ = zone_validator.process_frame(first_frame)
    tracked_f   = player_tracker.update(first_frame)
    radar_f     = visualizer.draw_tactical_pitch(tracked_f)
    bcast_f     = visualizer.draw_broadcast_frame(first_frame, zones_f, [], tracked_f)
    comp_f      = visualizer.create_composite_dashboard(
        bcast_f, radar_f,
        frame_idx=0, total_frames=target_frames, fps=fps,
        pitch_side=zone_validator.last_known_side,
        tracked_objects=tracked_f,
    )
    comp_h, comp_w = comp_f.shape[:2]
    print(f"Dashboard resolution: {comp_w}×{comp_h}")

    # ── VideoWriter ────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (comp_w, comp_h))

    # Reset and re-initialise for clean run
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    zone_validator  = ZoneValidator(pitch_model_path=MODEL_PITCH_PATH)
    seg_mapper      = SegMapper()
    player_tracker  = PlayerTracker(model_path=MODEL_DET_PATH)

    snapshot_dir = Path("snapshots")
    if save_snapshots:
        snapshot_dir.mkdir(exist_ok=True)

    # Always snapshot frames 1, 70, 138 + proportional checkpoints
    snapshot_indices = set(FIXED_SNAPSHOT_FRAMES)
    for frac in (0.1, 0.3, 0.6, 0.9):
        snapshot_indices.add(int(target_frames * frac))

    start_time    = time.time()
    processed_cnt = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames is not None and processed_cnt >= max_frames):
                break

            frame_idx = processed_cnt + 1  # 1-based

            # ── Step 1: Zone detection ─────────────────────────────────
            zones, _ = zone_validator.process_frame(frame)
            zone_dict = zone_validator.build_zone_dict(zones)
            pitch_side = zone_validator.last_known_side

            # ── Step 2: Player detection + tracking ────────────────────
            tracked_objects = player_tracker.update(frame)

            # ── Step 3: Zone association ───────────────────────────────
            foot_points = [o.foot_pt for o in tracked_objects]
            zone_assocs  = zone_validator.get_player_zone_associations(foot_points, zones)
            for obj, z_list in zip(tracked_objects, zone_assocs):
                obj.zones = z_list

            # ── Step 4: 2D position mapping (pure segmentation) ────────
            h_frame, w_frame = frame.shape[:2]
            for obj, z_list in zip(tracked_objects, zone_assocs):
                raw_xy = seg_mapper.map_player(
                    foot_px=obj.foot_pt,
                    zone_dict=zone_dict,
                    player_zones=z_list,
                    side=pitch_side,
                    frame_shape=(h_frame, w_frame),
                )
                smoothed_xy = seg_mapper.apply_ema(obj.track_id, raw_xy)
                obj.pitch_pt = smoothed_xy
                obj.p_zone   = raw_xy

            # ── Step 5: Zone player counts ─────────────────────────────
            zone_counts = zone_validator.get_zone_player_counts(foot_points, zones)

            # ── Diagnostic tables for specific frames ─────────────────
            if frame_idx in DIAGNOSTIC_FRAMES:
                print_diagnostic_table(frame_idx, tracked_objects, zone_counts, pitch_side)

            # ── Step 6: Render ─────────────────────────────────────────
            radar_canvas  = visualizer.draw_tactical_pitch(tracked_objects)
            bcast_canvas  = visualizer.draw_broadcast_frame(frame, zones, [], tracked_objects)
            dashboard = visualizer.create_composite_dashboard(
                broadcast_img=bcast_canvas,
                radar_img=radar_canvas,
                frame_idx=frame_idx,
                total_frames=target_frames,
                fps=fps,
                pitch_side=pitch_side,
                tracked_objects=tracked_objects,
                zone_counts=zone_counts,
            )

            writer.write(dashboard)

            # ── Save snapshots ─────────────────────────────────────────
            if save_snapshots and frame_idx in snapshot_indices:
                snap_path = snapshot_dir / f"{video_p.stem}_frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(snap_path), dashboard)
                print(f"  [snapshot] saved → {snap_path}")

            processed_cnt += 1

            if processed_cnt % 25 == 0 or processed_cnt == target_frames:
                elapsed = time.time() - start_time
                spd     = processed_cnt / elapsed if elapsed > 0 else 0
                eta     = (target_frames - processed_cnt) / spd if spd > 0 else 0
                print(f"[{video_p.name}] Frame {frame_idx:04d}/{target_frames:04d} | "
                      f"{spd:.1f} FPS | Side: {pitch_side} | "
                      f"Tracked: {len(tracked_objects):02d} | ETA: {eta:.0f}s")

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - start_time
    avg_fps    = processed_cnt / total_time if total_time > 0 else 0

    print("=" * 72)
    print(f"Done! {processed_cnt} frames in {total_time:.1f}s ({avg_fps:.1f} FPS avg)")
    print(f"Output: {output_path}")
    print("=" * 72)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Football Match Analyzer — Phase 1 MVP")
    parser.add_argument("--video",      type=str, default="match.mp4")
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-snapshots", action="store_true")
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
        save_snapshots=not args.no_snapshots,
    )


if __name__ == "__main__":
    main()
