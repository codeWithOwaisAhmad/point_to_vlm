"""
STEP 1: Synthetic Point Cloud Generator

Purpose: Generate fake but realistically-shaped point cloud data so we can
test the entire pipeline (PointNeXt -> Projection -> LLaMA) before:
  (a) ScanNet/ScanQA access is approved, and
  (b) RealSense data collection begins.

Output format matches what we expect from real ScanQA point clouds:
  - N x 6 array: x, y, z, r, g, b
  - N = 4096 points after sampling
  - Coordinates in meters, roughly room-scale (a few meters per axis)
  - Colors normalized to [0, 1]

Run this first. Everything else in the pipeline consumes this output format.
"""

import numpy as np
import os

NUM_POINTS = 4096
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def generate_dummy_room_point_cloud(num_points=NUM_POINTS, seed=None):
    """
    Generates a synthetic point cloud that loosely resembles a room scan:
    - Points clustered around a few "objects" (so it's not pure random noise)
    - Realistic room-scale coordinate range (~3m x 3m x 2.5m)
    - RGB colors in [0, 1]

    Returns:
        point_cloud: np.ndarray of shape (num_points, 6) -> [x, y, z, r, g, b]
    """
    rng = np.random.default_rng(seed)

    # Simulate 4-6 "objects" in the room as Gaussian blobs of points.
    num_objects = rng.integers(4, 7)
    points_per_object = num_points // num_objects

    all_points = []
    all_colors = []

    for _ in range(num_objects):
        # Random object center within room bounds
        center = rng.uniform(low=[-1.5, -1.5, 0.0], high=[1.5, 1.5, 2.0])
        # Random object spread (size)
        spread = rng.uniform(low=0.05, high=0.4, size=3)

        obj_points = rng.normal(loc=center, scale=spread, size=(points_per_object, 3))
        obj_color = rng.uniform(low=0.0, high=1.0, size=3)  # one base color per object
        obj_colors = np.tile(obj_color, (points_per_object, 1))
        # Add slight color noise so it's not flat
        obj_colors = np.clip(obj_colors + rng.normal(0, 0.03, obj_colors.shape), 0, 1)

        all_points.append(obj_points)
        all_colors.append(obj_colors)

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    # Pad or trim to exactly num_points
    if points.shape[0] < num_points:
        pad = num_points - points.shape[0]
        idx = rng.integers(0, points.shape[0], size=pad)
        points = np.concatenate([points, points[idx]], axis=0)
        colors = np.concatenate([colors, colors[idx]], axis=0)
    elif points.shape[0] > num_points:
        points = points[:num_points]
        colors = colors[:num_points]

    point_cloud = np.concatenate([points, colors], axis=1).astype(np.float32)
    return point_cloud


def validate_shape(point_cloud):
    """Sanity check that matches what PointNeXt and ScanQA expect."""
    assert point_cloud.shape == (NUM_POINTS, 6), f"Expected ({NUM_POINTS}, 6), got {point_cloud.shape}"
    assert point_cloud.dtype == np.float32
    colors = point_cloud[:, 3:6]
    assert colors.min() >= 0.0 and colors.max() <= 1.0, "Colors must be normalized to [0,1]"
    print(f"Shape check passed: {point_cloud.shape}, dtype={point_cloud.dtype}")
    print(f"  XYZ range: x=[{point_cloud[:,0].min():.2f}, {point_cloud[:,0].max():.2f}] "
          f"y=[{point_cloud[:,1].min():.2f}, {point_cloud[:,1].max():.2f}] "
          f"z=[{point_cloud[:,2].min():.2f}, {point_cloud[:,2].max():.2f}]")
    print(f"  RGB range: [{colors.min():.2f}, {colors.max():.2f}]")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate a handful of dummy "rooms" for testing batching later
    for i in range(5):
        pc = generate_dummy_room_point_cloud(seed=i)
        validate_shape(pc)
        out_path = os.path.join(OUTPUT_DIR, f"dummy_room_{i:02d}.npy")
        np.save(out_path, pc)
        print(f"Saved: {out_path}\n")

    print("Step 1 complete. Dummy point clouds are in:", OUTPUT_DIR)
