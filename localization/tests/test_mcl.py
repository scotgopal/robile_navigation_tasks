"""
Unit tests for MCL node: distance map, motion model, resampling, effective sample size,
and measurement update (likelihood field). Uses synthetic data; optional fixture map_sample.npz
from scripts/save_map_for_tests.py for integration-style tests.
"""

import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow importing mcl_node from parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcl_node import (
    MCLNode,
    _find_nearest_obstacle_cell,
    compute_distance_map,
)

# -----------------------------------------------------------------------------
# Distance map and obstacle search (no ROS)
# -----------------------------------------------------------------------------


def test_find_nearest_obstacle_out_of_bounds():
    grid = np.zeros((5, 5), dtype=np.int32)
    d, nx, ny = _find_nearest_obstacle_cell(grid, -1, 0, 10)
    assert d == float("inf") and nx is None and ny is None
    d, nx, ny = _find_nearest_obstacle_cell(grid, 10, 10, 10)
    assert d == float("inf") and nx is None and ny is None


def test_find_nearest_obstacle_occupied_cell():
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[2, 2] = 100
    d, nx, ny = _find_nearest_obstacle_cell(grid, 2, 2, 5)
    assert d == 0 and nx == 2 and ny == 2


def test_find_nearest_obstacle_neighbor():
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[3, 3] = 100
    d, nx, ny = _find_nearest_obstacle_cell(grid, 2, 3, 5)
    assert d == 1.0 and nx == 3 and ny == 3


def test_compute_distance_map_edt():
    # Small grid: free (0), one occupied cell at (1,1)
    grid = np.zeros((3, 3), dtype=np.int32)
    grid[1, 1] = 100
    res = 0.1
    dist = compute_distance_map(grid, res, max_dist=1.0, method="edt")
    assert dist.shape == grid.shape
    assert dist[1, 1] == 0.0
    # Neighbors at 1 cell = 0.1 m
    assert dist[0, 1] == pytest.approx(0.1, abs=1e-5)
    assert dist[1, 0] == pytest.approx(0.1, abs=1e-5)
    # Diagonal sqrt(2)*0.1
    assert dist[0, 0] == pytest.approx(0.1 * math.sqrt(2), abs=1e-5)


def test_compute_distance_map_expand():
    grid = np.zeros((3, 3), dtype=np.int32)
    grid[1, 1] = 100
    res = 0.1
    dist = compute_distance_map(grid, res, max_dist=1.0, method="expand")
    assert dist.shape == grid.shape
    assert dist[1, 1] == 0.0
    assert dist[0, 1] == pytest.approx(0.1, abs=1e-5)


def test_compute_distance_map_unknown_treated_as_free():
    # -1 = unknown, should be free for EDT (distance computed)
    grid = np.full((3, 3), -1, dtype=np.int32)
    grid[1, 1] = 100
    res = 0.1
    dist = compute_distance_map(grid, res, max_dist=1.0, method="edt")
    assert dist[1, 1] == 0.0
    assert dist[0, 0] == pytest.approx(0.1 * math.sqrt(2), abs=1e-5)


# -----------------------------------------------------------------------------
# Motion model (requires rclpy and MCLNode)
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rclpy_context():
    import rclpy
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def mcl_node(rclpy_context):
    import rclpy
    from rclpy.parameter import Parameter
    node = MCLNode()
    # Deterministic motion for tests: no noise
    node.set_parameters([
        Parameter("motion_noise_scale", rclpy.Parameter.Type.DOUBLE, 0.0),
        Parameter("alpha1", rclpy.Parameter.Type.DOUBLE, 0.1),
        Parameter("alpha2", rclpy.Parameter.Type.DOUBLE, 0.1),
        Parameter("alpha3", rclpy.Parameter.Type.DOUBLE, 0.1),
        Parameter("alpha4", rclpy.Parameter.Type.DOUBLE, 0.1),
    ])
    yield node
    node.destroy_node()


def test_motion_update_deterministic_forward(mcl_node):
    """With noise_scale=0, all particles get same motion: pure forward 1 m."""
    n = 10
    mcl_node._particles = np.zeros((n, 3))
    mcl_node._particles[:, 0] = 0.0
    mcl_node._particles[:, 1] = 0.0
    mcl_node._particles[:, 2] = 0.0
    dx, dy, dth = 1.0, 0.0, 0.0
    mcl_node._motion_update(dx, dy, dth)
    # All particles should move to (1, 0), theta=0
    np.testing.assert_allclose(mcl_node._particles[:, 0], 1.0)
    np.testing.assert_allclose(mcl_node._particles[:, 1], 0.0)
    np.testing.assert_allclose(mcl_node._particles[:, 2], 0.0)


def test_motion_update_deterministic_rotate_only(mcl_node):
    """Pure rotation: dth = 0.5 rad, no translation."""
    n = 5
    mcl_node._particles = np.zeros((n, 3))
    mcl_node._particles[:, 2] = 0.0
    dx, dy, dth = 0.0, 0.0, 0.5
    mcl_node._motion_update(dx, dy, dth)
    np.testing.assert_allclose(mcl_node._particles[:, 0], 0.0)
    np.testing.assert_allclose(mcl_node._particles[:, 1], 0.0)
    np.testing.assert_allclose(mcl_node._particles[:, 2], 0.5)


def test_motion_update_deterministic_forward_then_rotate(mcl_node):
    """Particle at (0,0,0): move (1,0,0) then (0,0,pi/2). Final pose (1,0,pi/2)."""
    mcl_node._particles = np.array([[0.0, 0.0, 0.0]])
    mcl_node._motion_update(1.0, 0.0, 0.0)
    mcl_node._motion_update(0.0, 0.0, math.pi / 2)
    np.testing.assert_allclose(mcl_node._particles[0, 0], 1.0, atol=1e-5)
    np.testing.assert_allclose(mcl_node._particles[0, 1], 0.0, atol=1e-5)
    np.testing.assert_allclose(mcl_node._particles[0, 2], math.pi / 2, atol=1e-5)


# -----------------------------------------------------------------------------
# Resampling and N_eff (algorithm only; can test with node or standalone)
# -----------------------------------------------------------------------------


def test_effective_sample_size_uniform():
    w = np.ones(10) / 10
    n_eff = 1.0 / np.sum(w * w)
    assert n_eff == pytest.approx(10.0)


def test_effective_sample_size_one_dominant():
    w = np.zeros(10)
    w[0] = 1.0
    n_eff = 1.0 / np.sum(w * w)
    assert n_eff == pytest.approx(1.0)


def test_resample_systematic_reproduces_count():
    random.seed(42)
    N = 100
    particles = np.random.randn(N, 3)
    weights = np.random.rand(N)
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    u0 = random.random()
    indices = np.zeros(N, dtype=np.int64)
    for i in range(N):
        u = (u0 + i) / N
        indices[i] = np.searchsorted(cdf, u, side="left")
    indices = np.clip(indices, 0, N - 1)
    new_particles = particles[indices].copy()
    assert new_particles.shape == (N, 3)


# -----------------------------------------------------------------------------
# Measurement update: correct pose gets higher weight (needs map + LaserScan)
# -----------------------------------------------------------------------------


def _make_synthetic_map_and_metadata():
    """Small map: 20x20, resolution 0.05, origin (0,0). Wall along left column."""
    h, w = 20, 20
    grid = np.zeros((h, w), dtype=np.int32)
    grid[:, 0] = 100
    resolution = 0.05
    origin_x, origin_y = 0.0, 0.0
    return grid, {
        "origin_x": origin_x,
        "origin_y": origin_y,
        "resolution": resolution,
        "width": w,
        "height": h,
    }


def test_measurement_update_correct_pose_higher_weight(rclpy_context):
    """Particle at true pose should get higher weight than particle far away."""
    import rclpy
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import LaserScan

    node = MCLNode()
    node.set_parameters([
        Parameter("max_beams", rclpy.Parameter.Type.INTEGER, 10),
        Parameter("sigma_hit", rclpy.Parameter.Type.DOUBLE, 0.25),
        Parameter("likelihood_max_dist", rclpy.Parameter.Type.DOUBLE, 2.0),
        Parameter("map_uncertainty_scale", rclpy.Parameter.Type.DOUBLE, 0.0),
        Parameter("z_max_log_likelihood", rclpy.Parameter.Type.DOUBLE, -4.6),
        Parameter("max_range_ratio_for_z_max", rclpy.Parameter.Type.DOUBLE, 0.98),
        Parameter("laser_origin_x", rclpy.Parameter.Type.DOUBLE, 0.0),
        Parameter("laser_origin_y", rclpy.Parameter.Type.DOUBLE, 0.0),
        Parameter("base_frame_id", rclpy.Parameter.Type.STRING, "base_footprint"),
    ])
    grid, metadata = _make_synthetic_map_and_metadata()
    node._map = grid
    node._map_metadata = metadata
    node._dist_map = compute_distance_map(
        grid, metadata["resolution"], 2.0, method="edt"
    )
    node._laser_origin_tf = None
    node._laser_origin_tf_frame = None
    node._scan_frame_warned = True

    # True pose: robot at world (0.5, 0.5) facing +x (0 rad). Wall at x=0 is 0.5 m away along -x.
    true_x, true_y, true_th = 0.5, 0.5, 0.0
    # Scan: beam 0 at angle_min = -pi (points -x in map when theta=0) with range 0.5 m (hits wall).
    num_beams = 10
    angle_min = -math.pi
    angle_increment = 2.0 * math.pi / num_beams
    ranges = [3.0] * num_beams
    ranges[0] = 0.5  # beam at -pi (backward / -x) sees wall at 0.5 m

    scan = LaserScan()
    scan.header.frame_id = "base_footprint"
    scan.angle_min = angle_min
    scan.angle_increment = angle_increment
    scan.angle_max = angle_min + (num_beams - 1) * angle_increment
    scan.range_min = 0.1
    scan.range_max = 10.0
    scan.ranges = ranges

    # Two particles: one at true pose, one at wrong pose (far away)
    node._particles = np.array([
        [true_x, true_y, true_th],
        [2.0, 2.0, 0.0],
    ])
    node._weights = np.ones(2) / 2
    node._measurement_update(scan)
    # Particle 0 (correct pose) should have higher weight than particle 1
    assert node._weights[0] > node._weights[1], (
        "Correct pose particle should have higher weight than wrong pose; "
        f"got w0={node._weights[0]:.4f}, w1={node._weights[1]:.4f}"
    )
    node.destroy_node()


# -----------------------------------------------------------------------------
# Optional: load fixture map and run distance-map consistency
# -----------------------------------------------------------------------------


def test_recovery_inject_particle_count_unchanged(rclpy_context):
    """Recovery inject replaces a fraction of particles; count unchanged, injected ones in map bounds."""
    import rclpy
    from rclpy.parameter import Parameter

    node = MCLNode()
    N = 100
    node._particles = np.random.randn(N, 3)
    node._weights = np.ones(N) / N
    node._map_metadata = {
        "origin_x": 0.0,
        "origin_y": 0.0,
        "resolution": 0.05,
        "width": 20,
        "height": 20,
    }
    node._recovery_inject(0.1)
    assert node._particles.shape == (N, 3)
    # Injected particles (10%) are in [0, 1] x [0, 1]; at least 10 should be in bounds
    in_bounds = (
        (node._particles[:, 0] >= 0.0) & (node._particles[:, 0] <= 1.0) &
        (node._particles[:, 1] >= 0.0) & (node._particles[:, 1] <= 1.0)
    )
    assert np.sum(in_bounds) >= 10
    node.destroy_node()


def test_world_to_map_cell_convention():
    """Validate world (x,y) -> map cell (mx, my) matches MCL node convention: origin at (0,0), row=my, col=mx."""
    ox, oy = 0.0, 0.0
    res = 0.05
    w, h = 20, 10
    # World (ox, oy) -> cell (0, 0); world (ox + res, oy) -> col 1; world (ox, oy + res) -> row 1
    mx0 = int((ox - ox) / res)
    my0 = int((oy - oy) / res)
    assert mx0 == 0 and my0 == 0
    mx1 = int((ox + res - ox) / res)
    my1 = int((oy + res - oy) / res)
    assert mx1 == 1 and my1 == 1
    # Indexing: dist_map[my, mx] with shape (height, width) = (h, w)
    assert 0 <= mx1 < w and 0 <= my1 < h


def test_fixture_map_distance_map_consistent():
    """If map_sample.npz exists, distance map from it should be consistent (occupied -> 0)."""
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "map_sample.npz"
    if not fixture_path.exists():
        pytest.skip("No fixtures/map_sample.npz (run scripts/save_map_for_tests.py)")
    data = np.load(fixture_path)
    grid = data["grid"]
    resolution = float(data["resolution"])
    dist = compute_distance_map(grid, resolution, max_dist=2.0, method="edt")
    assert dist.shape == grid.shape
    # All occupied cells (grid > 0) should have distance 0
    occupied = grid > 0
    np.testing.assert_allclose(dist[occupied], 0.0, atol=1e-5)
    # Free cells should have positive distance (or at least non-negative)
    free = grid <= 0
    assert np.all(dist[free] >= 0)
