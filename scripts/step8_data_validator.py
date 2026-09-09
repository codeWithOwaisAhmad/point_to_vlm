"""
STEP 8: Data Quality Validator

Run this on EVERY room you capture with the RealSense camera, immediately
after recording, before moving to the next room. If a room fails, re-record
it on the spot rather than discovering the problem during training.

This script CAN be fully tested in this sandbox using synthetic data with
deliberately injected flaws (holes, out-of-range depth, narrow spread) to
confirm the validator actually catches problems, not just passes everything.

CHECKS IMPLEMENTED (from our earlier discussion):
  1. Depth coverage: % of pixels with valid (non-zero, non-NaN) depth
  2. Depth range: % of valid depth values within plausible indoor range (0.3-3.5m)
  3. Spatial spread: point cloud should span a reasonable room-scale area,
     not be clustered in one tight corner
  4. Color-depth alignment: NOT automatable reliably - this remains a
     VISUAL check you must do yourself (see reminder at the bottom)
"""

import numpy as np

MIN_DEPTH_COVERAGE = 0.85    # 85% of pixels must have valid depth
MIN_DEPTH_RANGE_VALID = 0.90  # 90% of valid depths must be in plausible range
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 3.5
MIN_SPATIAL_SPREAD_M = 0.5    # minimum spread per axis after sampling


def check_depth_coverage(depth_map):
    """
    Args:
        depth_map: 2D np.ndarray of raw depth values in meters, where
                   0 or NaN indicates an invalid/missing depth pixel.
    Returns:
        coverage_ratio: float in [0, 1]
        passed: bool
    """
    valid_mask = (depth_map > 0) & (~np.isnan(depth_map))
    coverage_ratio = valid_mask.sum() / depth_map.size
    return coverage_ratio, coverage_ratio >= MIN_DEPTH_COVERAGE


def check_depth_range(depth_map):
    """Checks what fraction of VALID depth pixels fall in plausible indoor range."""
    valid_mask = (depth_map > 0) & (~np.isnan(depth_map))
    valid_depths = depth_map[valid_mask]

    if len(valid_depths) == 0:
        return 0.0, False

    in_range_mask = (valid_depths >= DEPTH_MIN_M) & (valid_depths <= DEPTH_MAX_M)
    in_range_ratio = in_range_mask.sum() / len(valid_depths)
    return in_range_ratio, in_range_ratio >= MIN_DEPTH_RANGE_VALID


def check_spatial_spread(point_cloud):
    """
    Args:
        point_cloud: (N, 6) array [x, y, z, r, g, b]
    Returns:
        spread: dict with x/y/z spread in meters
        passed: bool
    """
    xyz = point_cloud[:, :3]
    spread = {
        "x": float(xyz[:, 0].max() - xyz[:, 0].min()),
        "y": float(xyz[:, 1].max() - xyz[:, 1].min()),
        "z": float(xyz[:, 2].max() - xyz[:, 2].min()),
    }
    passed = all(v >= MIN_SPATIAL_SPREAD_M for v in spread.values())
    return spread, passed


def validate_room_capture(depth_map, point_cloud, room_name="unknown"):
    """
    Runs all automated checks and prints a report. Returns overall pass/fail.

    NOTE: depth_map and point_cloud are checked SEPARATELY because depth_map
    is the raw 2D sensor output (before point cloud conversion) and
    point_cloud is the post-processed (4096, 6) array used by the model.
    You need both - depth_map catches sensor-level problems, point_cloud
    catches problems introduced during your Open3D processing step.
    """
    print(f"\n{'='*60}")
    print(f"Validating: {room_name}")
    print(f"{'='*60}")

    coverage, coverage_ok = check_depth_coverage(depth_map)
    status = "PASS" if coverage_ok else "FAIL"
    print(f"Depth coverage:  {coverage*100:.1f}%  (need >={MIN_DEPTH_COVERAGE*100:.0f}%)  [{status}]")

    range_ratio, range_ok = check_depth_range(depth_map)
    status = "PASS" if range_ok else "FAIL"
    print(f"Depth in range:  {range_ratio*100:.1f}%  (need >={MIN_DEPTH_RANGE_VALID*100:.0f}%)  [{status}]")

    spread, spread_ok = check_spatial_spread(point_cloud)
    status = "PASS" if spread_ok else "FAIL"
    print(f"Spatial spread:  x={spread['x']:.2f}m y={spread['y']:.2f}m z={spread['z']:.2f}m "
          f"(need >={MIN_SPATIAL_SPREAD_M}m each)  [{status}]")

    overall_pass = coverage_ok and range_ok and spread_ok
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL - RE-RECORD THIS ROOM'}")
    print(f"{'='*60}")

    print("\nREMINDER: Also do the VISUAL check manually - open the point")
    print("cloud in Open3D and confirm with your own eyes that colors align")
    print("with geometry and the room looks like a room, not a blob of noise.")
    print("This cannot be automated reliably and must not be skipped.")

    return overall_pass


def _make_good_depth_map(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.5, 3.0, size=(h, w))
    # simulate small realistic holes (5% invalid)
    holes = rng.random((h, w)) < 0.05
    depth[holes] = 0
    return depth


def _make_bad_depth_map_low_coverage(h=480, w=640, seed=1):
    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.5, 3.0, size=(h, w))
    # simulate a BAD capture: 40% holes (e.g. lots of glass/reflective surfaces)
    holes = rng.random((h, w)) < 0.40
    depth[holes] = 0
    return depth


def _make_bad_depth_map_out_of_range(h=480, w=640, seed=2):
    rng = np.random.default_rng(seed)
    # simulate sensor noise causing many out-of-range readings
    depth = rng.uniform(0.1, 8.0, size=(h, w))  # way outside 0.3-3.5m range
    return depth


if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pc_path = os.path.join(data_dir, "dummy_room_00.npy")
    point_cloud = np.load(pc_path)

    print("TEST 1: Good synthetic depth map (should PASS)")
    good_depth = _make_good_depth_map()
    result1 = validate_room_capture(good_depth, point_cloud, room_name="test_good_capture")
    assert result1 == True, "Good capture should pass but didn't!"

    print("\n\nTEST 2: Bad depth map - low coverage (should FAIL on coverage)")
    bad_depth_coverage = _make_bad_depth_map_low_coverage()
    result2 = validate_room_capture(bad_depth_coverage, point_cloud, room_name="test_bad_coverage")
    assert result2 == False, "Bad coverage capture should fail but passed!"

    print("\n\nTEST 3: Bad depth map - out of range (should FAIL on range)")
    bad_depth_range = _make_bad_depth_map_out_of_range()
    result3 = validate_room_capture(bad_depth_range, point_cloud, room_name="test_bad_range")
    assert result3 == False, "Bad range capture should fail but passed!"

    print("\n\nAll validator self-tests passed - the validator correctly")
    print("distinguishes good captures from bad ones on synthetic data.")
    print("\nStep 8 complete. Use this script on every real RealSense")
    print("capture during data collection.")
