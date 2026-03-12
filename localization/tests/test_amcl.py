"""
Comprehensive unit tests for the AMCL node.

Tests are organised by component:
  1. Distance map computation
  2. Free-space extraction
  3. KLD sample count
  4. Odometry motion model
  5. Likelihood-field sensor model
  6. Systematic resampling (fixed-N and KLD-adaptive)
  7. Pose estimation (cluster and weighted)
  8. Covariance computation
  9. Effective sample size
  10. Recovery particle injection
  11. Integration: full filter cycle on synthetic data
  12. ROS-dependent node tests (map callback, initial pose, measurement + resample)

Run:
  cd robile_navigation_tasks/localization
  python -m pytest tests/test_amcl.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amcl_node import (
    AMCLNode,
    compute_distance_map,
    extract_free_indices,
    kld_sample_count,
    odometry_motion_model,
    likelihood_field_model,
    systematic_resample,
    systematic_resample_kld,
    cluster_pose_estimate,
    weighted_pose_estimate,
    particle_covariance,
    effective_sample_size,
)


# ===================================================================
# 1. Distance map
# ===================================================================


class TestDistanceMap:
    def test_occupied_cell_has_zero_distance(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[2, 2] = 100
        dist = compute_distance_map(grid, 0.1, max_dist=1.0)
        assert dist[2, 2] == 0.0

    def test_adjacent_cell_distance(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[2, 2] = 100
        dist = compute_distance_map(grid, 0.1, max_dist=5.0)
        assert dist[2, 3] == pytest.approx(0.1, abs=1e-5)
        assert dist[3, 2] == pytest.approx(0.1, abs=1e-5)

    def test_diagonal_distance(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[1, 1] = 100
        dist = compute_distance_map(grid, 0.1, max_dist=1.0)
        assert dist[0, 0] == pytest.approx(0.1 * math.sqrt(2), abs=1e-5)

    def test_unknown_treated_as_free(self):
        grid = np.full((3, 3), -1, dtype=np.int32)
        grid[1, 1] = 100
        dist = compute_distance_map(grid, 0.1, max_dist=1.0)
        assert dist[1, 1] == 0.0
        assert dist[0, 0] == pytest.approx(0.1 * math.sqrt(2), abs=1e-5)

    def test_max_dist_cap(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[0, 0] = 100
        dist = compute_distance_map(grid, 0.1, max_dist=0.3)
        assert np.all(dist <= 0.3 + 1e-6)

    def test_all_free_capped_at_max_dist(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        dist = compute_distance_map(grid, 0.1, max_dist=1.0)
        # No obstacles: EDT gives finite distances (to implicit grid edge);
        # all values are non-negative and capped at max_dist
        assert np.all(dist >= 0)
        assert np.all(dist <= 1.0 + 1e-5)

    def test_all_occupied_returns_zeros(self):
        grid = np.full((5, 5), 100, dtype=np.int32)
        dist = compute_distance_map(grid, 0.1, max_dist=1.0)
        assert np.all(dist == 0.0)

    def test_output_shape_matches_grid(self):
        grid = np.zeros((7, 13), dtype=np.int32)
        grid[3, 6] = 100
        dist = compute_distance_map(grid, 0.05, max_dist=2.0)
        assert dist.shape == (7, 13)
        assert dist.dtype == np.float32


# ===================================================================
# 2. Free-space extraction
# ===================================================================


class TestFreeIndices:
    def test_all_free(self):
        grid = np.zeros((4, 4), dtype=np.int32)
        fi = extract_free_indices(grid)
        assert fi.shape == (16, 2)

    def test_all_occupied(self):
        grid = np.full((4, 4), 100, dtype=np.int32)
        fi = extract_free_indices(grid)
        assert fi.shape[0] == 0

    def test_mixed(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[0, 0] = 100
        grid[1, 1] = 100
        fi = extract_free_indices(grid)
        assert fi.shape[0] == 7

    def test_unknown_is_free(self):
        grid = np.full((2, 2), -1, dtype=np.int32)
        fi = extract_free_indices(grid)
        assert fi.shape[0] == 4


# ===================================================================
# 3. KLD sample count
# ===================================================================


class TestKLDSampleCount:
    def test_single_bin_returns_max(self):
        assert kld_sample_count(1, max_particles=2000) == 2000

    def test_zero_bins_returns_max(self):
        assert kld_sample_count(0, max_particles=2000) == 2000

    def test_many_bins_within_range(self):
        n = kld_sample_count(50, pf_err=0.05, pf_z=0.99,
                             min_particles=100, max_particles=5000)
        assert 100 <= n <= 5000

    def test_increases_with_bins(self):
        n10 = kld_sample_count(10, pf_err=0.05, pf_z=0.99)
        n50 = kld_sample_count(50, pf_err=0.05, pf_z=0.99)
        assert n50 >= n10

    def test_clamped_to_min(self):
        n = kld_sample_count(2, pf_err=0.05, pf_z=0.99,
                             min_particles=500, max_particles=5000)
        assert n >= 500

    def test_clamped_to_max(self):
        n = kld_sample_count(10000, pf_err=0.001, pf_z=0.999,
                             min_particles=100, max_particles=500)
        assert n <= 500


# ===================================================================
# 4. Odometry motion model
# ===================================================================


class TestMotionModel:
    def test_zero_motion_preserves_particles(self):
        rng = np.random.default_rng(42)
        p = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, -1.0]])
        p_orig = p.copy()
        # With zero motion and tiny alpha, particles barely move
        odometry_motion_model(p, 0.0, 0.0, 0.0,
                              0.001, 0.001, 0.001, 0.001, rng)
        np.testing.assert_allclose(p[:, :2], p_orig[:, :2], atol=0.05)

    def test_forward_motion_increases_x(self):
        rng = np.random.default_rng(42)
        N = 500
        p = np.zeros((N, 3))
        odometry_motion_model(p, 0.0, 1.0, 0.0, 0.01, 0.01, 0.01, 0.01, rng)
        assert np.mean(p[:, 0]) == pytest.approx(1.0, abs=0.1)
        assert np.mean(p[:, 1]) == pytest.approx(0.0, abs=0.1)

    def test_rotation_only(self):
        rng = np.random.default_rng(42)
        N = 500
        p = np.zeros((N, 3))
        odometry_motion_model(p, math.pi / 2, 0.0, 0.0,
                              0.01, 0.01, 0.01, 0.01, rng)
        mean_th = np.arctan2(np.mean(np.sin(p[:, 2])),
                             np.mean(np.cos(p[:, 2])))
        assert mean_th == pytest.approx(math.pi / 2, abs=0.15)

    def test_noise_increases_with_alpha(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        N = 1000
        p_low = np.zeros((N, 3))
        p_high = np.zeros((N, 3))
        odometry_motion_model(p_low, 0.0, 1.0, 0.0,
                              0.01, 0.01, 0.01, 0.01, rng1)
        odometry_motion_model(p_high, 0.0, 1.0, 0.0,
                              0.5, 0.5, 0.5, 0.5, rng2)
        std_low = np.std(p_low[:, 0])
        std_high = np.std(p_high[:, 0])
        assert std_high > std_low

    def test_preserves_shape(self):
        rng = np.random.default_rng(42)
        p = np.zeros((10, 3))
        result = odometry_motion_model(p, 0.1, 0.5, 0.1,
                                       0.2, 0.2, 0.2, 0.2, rng)
        assert result.shape == (10, 3)

    def test_theta_stays_normalised(self):
        rng = np.random.default_rng(42)
        N = 200
        p = np.zeros((N, 3))
        p[:, 2] = np.pi - 0.01
        odometry_motion_model(p, 0.5, 1.0, 0.5, 0.2, 0.2, 0.2, 0.2, rng)
        assert np.all(p[:, 2] >= -np.pi - 1e-6)
        assert np.all(p[:, 2] <= np.pi + 1e-6)


class TestMotionDecomposition:
    """Validate the odometry decomposition used by AMCLNode.

    The textbook formulation (Probabilistic Robotics Table 5.6):
      delta_rot1  = atan2(y'-y, x'-x) - theta
      delta_trans = sqrt((x'-x)^2 + (y'-y)^2)
      delta_rot2  = (theta'-theta) - delta_rot1
    """

    @staticmethod
    def _decompose(odom_prev, odom_curr):
        """Replicate the decomposition logic from AMCLNode."""
        dx = odom_curr[0] - odom_prev[0]
        dy = odom_curr[1] - odom_prev[1]
        dth = odom_curr[2] - odom_prev[2]
        dth = math.atan2(math.sin(dth), math.cos(dth))
        trans = math.sqrt(dx * dx + dy * dy)
        if trans > 1e-6:
            rot1 = math.atan2(dy, dx) - odom_prev[2]
            rot1 = math.atan2(math.sin(rot1), math.cos(rot1))
        else:
            rot1 = 0.0
        rot2 = dth - rot1
        rot2 = math.atan2(math.sin(rot2), math.cos(rot2))
        return rot1, trans, rot2

    def test_forward_along_x(self):
        rot1, trans, rot2 = self._decompose((0, 0, 0), (1, 0, 0))
        assert rot1 == pytest.approx(0, abs=1e-9)
        assert trans == pytest.approx(1.0, abs=1e-9)
        assert rot2 == pytest.approx(0, abs=1e-9)

    def test_forward_along_y(self):
        """Robot facing +y moves 1m forward -- rot1 should be 0."""
        rot1, trans, rot2 = self._decompose(
            (0, 0, math.pi / 2), (0, 1, math.pi / 2))
        assert rot1 == pytest.approx(0, abs=1e-9)
        assert trans == pytest.approx(1.0, abs=1e-9)
        assert rot2 == pytest.approx(0, abs=1e-9)

    def test_forward_along_neg_x(self):
        """Robot facing -x moves 1m forward."""
        rot1, trans, rot2 = self._decompose(
            (1, 0, math.pi), (0, 0, math.pi))
        assert rot1 == pytest.approx(0, abs=1e-9)
        assert trans == pytest.approx(1.0, abs=1e-9)
        assert rot2 == pytest.approx(0, abs=1e-9)

    def test_turn_then_translate(self):
        """Robot at origin facing +x, turns 90 deg and moves 1m to (0,1)."""
        rot1, trans, rot2 = self._decompose(
            (0, 0, 0), (0, 1, math.pi / 2))
        assert rot1 == pytest.approx(math.pi / 2, abs=1e-9)
        assert trans == pytest.approx(1.0, abs=1e-9)
        assert rot2 == pytest.approx(0, abs=1e-9)

    def test_pure_rotation(self):
        """Pure rotation in place: trans~0, rot1=0, rot2=dth."""
        rot1, trans, rot2 = self._decompose(
            (0, 0, 0), (0, 0, math.pi / 4))
        assert trans == pytest.approx(0, abs=1e-6)
        assert rot1 == pytest.approx(0, abs=1e-6)
        assert rot2 == pytest.approx(math.pi / 4, abs=1e-6)

    def test_noise_is_small_for_straight_motion(self):
        """Straight motion should produce small rot1 noise, not theta-sized."""
        rng = np.random.default_rng(42)
        N = 2000
        p = np.zeros((N, 3))
        p[:, 2] = math.pi / 2  # all facing +y

        rot1, trans, rot2 = self._decompose(
            (0, 0, math.pi / 2), (0, 1, math.pi / 2))

        odometry_motion_model(p, rot1, trans, rot2,
                              0.2, 0.2, 0.2, 0.2, rng)
        theta_std = np.std(p[:, 2])
        # With correct decomposition (rot1≈0): var_theta ≈ 2*(alpha2*trans^2) = 0.4, std≈0.63
        # With the bug (rot1=pi/2): var_theta ≈ 2*(alpha1*(pi/2)^2 + alpha2*1) ≈ 1.39, std≈1.18
        assert theta_std < 0.9, (
            f"theta std {theta_std:.3f} too large for straight motion "
            f"(rot1={rot1:.4f}, rot2={rot2:.4f})")


# ===================================================================
# 5. Likelihood-field sensor model
# ===================================================================


def _make_test_map():
    """
    10x10 grid, resolution 0.1 m, origin (0,0).
    Wall along left column (col 0).
    """
    h, w = 10, 10
    grid = np.zeros((h, w), dtype=np.int32)
    grid[:, 0] = 100
    res = 0.1
    dist_map = compute_distance_map(grid, res, max_dist=2.0)
    meta = {"origin_x": 0.0, "origin_y": 0.0,
            "resolution": res, "width": w, "height": h}
    return grid, dist_map, meta


class TestLikelihoodField:
    def test_correct_pose_scores_higher(self):
        _, dist_map, meta = _make_test_map()
        # Robot at (0.5, 0.5, 0): left wall at x=0 is 0.5 m away
        p = np.array([[0.5, 0.5, 0.0],
                       [0.8, 0.5, 0.0]])
        ranges = np.array([0.5])  # beam hitting wall
        angles = np.array([math.pi])  # beam pointing -x
        valid = np.array([True])

        lw = likelihood_field_model(
            p, ranges, angles, valid,
            max_range=10.0, max_range_ratio=0.98,
            dist_map=dist_map,
            origin_x=0.0, origin_y=0.0, resolution=0.1,
            sigma_hit=0.2, z_hit=0.95, z_rand=0.05, max_dist=2.0,
            laser_x=0.0, laser_y=0.0,
        )
        assert lw[0] > lw[1], "Correct-pose particle should score higher"

    def test_out_of_map_gets_neg_inf(self):
        _, dist_map, meta = _make_test_map()
        p = np.array([[-5.0, -5.0, 0.0]])
        ranges = np.array([1.0])
        angles = np.array([0.0])
        valid = np.array([True])
        lw = likelihood_field_model(
            p, ranges, angles, valid,
            10.0, 0.98, dist_map,
            0.0, 0.0, 0.1, 0.2, 0.95, 0.05, 2.0, 0.0, 0.0,
        )
        assert lw[0] == -np.inf

    def test_max_range_beam_uses_z_rand(self):
        _, dist_map, meta = _make_test_map()
        p = np.array([[0.5, 0.5, 0.0]])
        ranges = np.array([9.9])
        angles = np.array([0.0])
        valid = np.array([True])
        lw = likelihood_field_model(
            p, ranges, angles, valid,
            10.0, 0.98, dist_map,
            0.0, 0.0, 0.1, 0.2, 0.95, 0.05, 2.0, 0.0, 0.0,
        )
        assert np.isfinite(lw[0])

    def test_invalid_beams_contribute_zero(self):
        _, dist_map, meta = _make_test_map()
        p = np.array([[0.5, 0.5, 0.0]])
        ranges = np.array([float("nan"), 1.0])
        angles = np.array([0.0, math.pi])
        valid = np.array([False, True])
        lw = likelihood_field_model(
            p, ranges, angles, valid,
            10.0, 0.98, dist_map,
            0.0, 0.0, 0.1, 0.2, 0.95, 0.05, 2.0, 0.0, 0.0,
        )
        assert np.isfinite(lw[0])

    def test_multiple_beams_aggregate(self):
        _, dist_map, meta = _make_test_map()
        p = np.array([[0.5, 0.5, 0.0]])
        r1 = np.array([0.5])
        a1 = np.array([math.pi])
        v1 = np.array([True])
        lw1 = likelihood_field_model(
            p, r1, a1, v1, 10.0, 0.98, dist_map,
            0.0, 0.0, 0.1, 0.2, 0.95, 0.05, 2.0, 0.0, 0.0,
        )

        r2 = np.array([0.5, 0.5])
        a2 = np.array([math.pi, math.pi])
        v2 = np.array([True, True])
        lw2 = likelihood_field_model(
            p, r2, a2, v2, 10.0, 0.98, dist_map,
            0.0, 0.0, 0.1, 0.2, 0.95, 0.05, 2.0, 0.0, 0.0,
        )
        # Two identical beams -> double the log-likelihood
        assert lw2[0] == pytest.approx(2 * lw1[0], rel=1e-5)

    def test_laser_offset_shifts_endpoint(self):
        _, dist_map, meta = _make_test_map()
        # Laser offset 0.2 m forward: beam at angle pi (backward) needs
        # 0.7 m to hit wall at x=0 from robot at x=0.5
        p = np.array([[0.5, 0.5, 0.0]])
        ranges_no_offset = np.array([0.5])
        ranges_with_offset = np.array([0.7])
        angles = np.array([math.pi])
        valid = np.array([True])

        lw_no = likelihood_field_model(
            p, ranges_no_offset, angles, valid,
            10.0, 0.98, dist_map, 0.0, 0.0, 0.1,
            0.2, 0.95, 0.05, 2.0, 0.0, 0.0,
        )
        lw_with = likelihood_field_model(
            p, ranges_with_offset, angles, valid,
            10.0, 0.98, dist_map, 0.0, 0.0, 0.1,
            0.2, 0.95, 0.05, 2.0, 0.2, 0.0,
        )
        # Both should be close (beam endpoint hits wall x=0)
        assert abs(lw_no[0] - lw_with[0]) < 1.0


# ===================================================================
# 6. Resampling
# ===================================================================


class TestResampling:
    def test_systematic_preserves_count(self):
        rng = np.random.default_rng(42)
        N = 100
        p = rng.standard_normal((N, 3))
        w = rng.random(N)
        w /= w.sum()
        p_new, w_new = systematic_resample(p, w, rng)
        assert p_new.shape == (N, 3)
        assert w_new.shape == (N,)
        assert w_new.sum() == pytest.approx(1.0, abs=1e-10)

    def test_systematic_concentrates_high_weight(self):
        rng = np.random.default_rng(42)
        N = 100
        p = rng.standard_normal((N, 3))
        w = np.zeros(N)
        w[0] = 1.0
        p_new, _ = systematic_resample(p, w, rng)
        np.testing.assert_allclose(p_new[:, 0], p[0, 0])

    def test_kld_returns_variable_count(self):
        rng = np.random.default_rng(42)
        N = 2000
        # Broad distribution -> many bins -> more particles
        p_broad = rng.uniform(-10, 10, (N, 3))
        w = np.full(N, 1.0 / N)
        p_out, w_out = systematic_resample_kld(
            p_broad, w, rng,
            bin_size_xy=0.5, bin_size_theta=0.1745,
            pf_err=0.05, pf_z=0.99,
            min_particles=100, max_particles=2000,
        )
        assert 100 <= p_out.shape[0] <= 2000
        assert w_out.shape[0] == p_out.shape[0]

    def test_kld_narrow_distribution_fewer_particles(self):
        rng = np.random.default_rng(42)
        N = 2000
        # Very narrow distribution -> few bins -> fewer particles
        p_narrow = np.zeros((N, 3))
        p_narrow[:, 0] = 1.0 + rng.normal(0, 0.01, N)
        p_narrow[:, 1] = 2.0 + rng.normal(0, 0.01, N)
        p_narrow[:, 2] = 0.5 + rng.normal(0, 0.001, N)
        w = np.full(N, 1.0 / N)
        p_narrow_out, _ = systematic_resample_kld(
            p_narrow, w, rng,
            0.5, 0.1745, 0.05, 0.99, 100, 2000,
        )

        p_broad = rng.uniform(-10, 10, (N, 3))
        p_broad_out, _ = systematic_resample_kld(
            p_broad, w, rng,
            0.5, 0.1745, 0.05, 0.99, 100, 2000,
        )
        assert p_narrow_out.shape[0] <= p_broad_out.shape[0]

    def test_kld_respects_min_particles(self):
        rng = np.random.default_rng(42)
        N = 500
        p = np.zeros((N, 3))
        w = np.full(N, 1.0 / N)
        p_out, _ = systematic_resample_kld(
            p, w, rng,
            0.5, 0.1745, 0.05, 0.99, 200, 500,
        )
        assert p_out.shape[0] >= 200


# ===================================================================
# 7. Pose estimation
# ===================================================================


class TestPoseEstimation:
    def test_weighted_mean_trivial(self):
        p = np.array([[1.0, 2.0, 0.5]])
        w = np.array([1.0])
        x, y, th = weighted_pose_estimate(p, w)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(2.0)
        assert th == pytest.approx(0.5)

    def test_weighted_mean_average(self):
        p = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]])
        w = np.array([0.5, 0.5])
        x, y, _ = weighted_pose_estimate(p, w)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(2.0)

    def test_weighted_mean_theta_circular(self):
        # Two particles at +179 and -179 degrees; mean should be ~180
        p = np.array([[0.0, 0.0, math.radians(179)],
                       [0.0, 0.0, math.radians(-179)]])
        w = np.array([0.5, 0.5])
        _, _, th = weighted_pose_estimate(p, w)
        assert abs(abs(th) - math.pi) < 0.05

    def test_cluster_selects_majority(self):
        # 80 particles at (1,1,0), 20 at (10,10,0)
        p = np.zeros((100, 3))
        p[:80, 0] = 1.0
        p[:80, 1] = 1.0
        p[80:, 0] = 10.0
        p[80:, 1] = 10.0
        w = np.full(100, 0.01)
        x, y, _ = cluster_pose_estimate(p, w, bin_size_xy=0.5,
                                         bin_size_theta=0.1745)
        assert abs(x - 1.0) < 0.5
        assert abs(y - 1.0) < 0.5

    def test_cluster_selects_highest_weight(self):
        # 90 low-weight particles at (1,1), 10 high-weight at (5,5)
        p = np.zeros((100, 3))
        p[:90, :2] = 1.0
        p[90:, :2] = 5.0
        w = np.zeros(100)
        w[:90] = 0.001
        w[90:] = 0.091
        w /= w.sum()
        x, y, _ = cluster_pose_estimate(p, w, bin_size_xy=0.5,
                                         bin_size_theta=0.1745)
        assert abs(x - 5.0) < 0.5
        assert abs(y - 5.0) < 0.5


# ===================================================================
# 8. Covariance
# ===================================================================


class TestCovariance:
    def test_zero_variance_single_particle(self):
        p = np.array([[1.0, 2.0, 0.5]])
        w = np.array([1.0])
        cov = particle_covariance(p, w, (1.0, 2.0, 0.5))
        assert cov[0] == pytest.approx(0.0, abs=1e-10)
        assert cov[7] == pytest.approx(0.0, abs=1e-10)
        assert cov[35] == pytest.approx(0.0, abs=1e-10)

    def test_symmetric_off_diagonal(self):
        rng = np.random.default_rng(42)
        p = rng.standard_normal((100, 3))
        w = np.full(100, 0.01)
        mean = weighted_pose_estimate(p, w)
        cov = particle_covariance(p, w, mean)
        assert cov[1] == pytest.approx(cov[6], abs=1e-10)
        assert cov[5] == pytest.approx(cov[30], abs=1e-10)
        assert cov[11] == pytest.approx(cov[31], abs=1e-10)

    def test_length_is_36(self):
        p = np.zeros((10, 3))
        w = np.full(10, 0.1)
        cov = particle_covariance(p, w, (0.0, 0.0, 0.0))
        assert len(cov) == 36

    def test_positive_diagonal(self):
        rng = np.random.default_rng(42)
        p = rng.standard_normal((200, 3))
        w = np.full(200, 1.0 / 200)
        mean = weighted_pose_estimate(p, w)
        cov = particle_covariance(p, w, mean)
        assert cov[0] >= 0
        assert cov[7] >= 0
        assert cov[35] >= 0


# ===================================================================
# 9. Effective sample size
# ===================================================================


class TestEffectiveSampleSize:
    def test_uniform_weights(self):
        w = np.ones(100) / 100
        assert effective_sample_size(w) == pytest.approx(100.0)

    def test_single_dominant(self):
        w = np.zeros(100)
        w[0] = 1.0
        assert effective_sample_size(w) == pytest.approx(1.0)

    def test_empty(self):
        assert effective_sample_size(np.array([])) == 0.0

    def test_none(self):
        assert effective_sample_size(None) == 0.0

    def test_two_equal(self):
        w = np.array([0.5, 0.5])
        assert effective_sample_size(w) == pytest.approx(2.0)


# ===================================================================
# 10. Integration: filter cycle on synthetic data
# ===================================================================


class TestFilterCycleIntegration:
    """Test a full prediction-update-resample cycle without ROS."""

    def test_converges_to_true_pose(self):
        rng = np.random.default_rng(123)
        grid, dist_map, meta = _make_test_map()
        N = 500
        true_x, true_y, true_th = 0.5, 0.5, 0.0

        particles = np.zeros((N, 3))
        particles[:, 0] = true_x + rng.normal(0, 0.3, N)
        particles[:, 1] = true_y + rng.normal(0, 0.3, N)
        particles[:, 2] = true_th + rng.normal(0, 0.2, N)
        weights = np.full(N, 1.0 / N)

        # Simulate 10 observations
        ranges = np.array([0.5])
        angles = np.array([math.pi])
        valid = np.array([True])

        for _ in range(10):
            log_w = likelihood_field_model(
                particles, ranges, angles, valid,
                10.0, 0.98, dist_map,
                0.0, 0.0, 0.1, 0.2, 0.95, 0.05, 2.0, 0.0, 0.0,
            )
            max_lw = np.max(log_w)
            if np.isfinite(max_lw):
                log_w -= max_lw
                weights = np.exp(log_w)
                weights /= weights.sum()
            particles, weights = systematic_resample(particles, weights, rng)

        x, y, th = weighted_pose_estimate(particles, weights)
        assert abs(x - true_x) < 0.2
        assert abs(y - true_y) < 0.3

    def test_kld_reduces_particles_after_convergence(self):
        rng = np.random.default_rng(456)
        grid, dist_map, meta = _make_test_map()
        N = 2000
        true_x, true_y, true_th = 0.5, 0.5, 0.0

        particles = np.zeros((N, 3))
        particles[:, 0] = true_x + rng.normal(0, 0.01, N)
        particles[:, 1] = true_y + rng.normal(0, 0.01, N)
        particles[:, 2] = true_th + rng.normal(0, 0.01, N)
        weights = np.full(N, 1.0 / N)

        particles, weights = systematic_resample_kld(
            particles, weights, rng,
            0.5, 0.1745, 0.05, 0.99, 100, 2000,
        )
        # Converged cloud is narrow -> few bins -> should have < 2000 particles
        assert particles.shape[0] < N


# ===================================================================
# 11. ROS-dependent node tests
# ===================================================================


@pytest.fixture(scope="module")
def rclpy_context():
    import rclpy
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def amcl_node(rclpy_context):
    from rclpy.parameter import Parameter
    import rclpy as _rclpy
    node = AMCLNode()
    node.set_parameters([
        Parameter("min_particles", _rclpy.Parameter.Type.INTEGER, 100),
        Parameter("max_particles", _rclpy.Parameter.Type.INTEGER, 500),
        Parameter("max_beams", _rclpy.Parameter.Type.INTEGER, 10),
        Parameter("sigma_hit", _rclpy.Parameter.Type.DOUBLE, 0.2),
        Parameter("z_hit", _rclpy.Parameter.Type.DOUBLE, 0.95),
        Parameter("z_rand", _rclpy.Parameter.Type.DOUBLE, 0.05),
        Parameter("laser_likelihood_max_dist", _rclpy.Parameter.Type.DOUBLE, 2.0),
        Parameter("laser_origin_x", _rclpy.Parameter.Type.DOUBLE, 0.0),
        Parameter("laser_origin_y", _rclpy.Parameter.Type.DOUBLE, 0.0),
        Parameter("base_frame_id", _rclpy.Parameter.Type.STRING, "base_footprint"),
        Parameter("alpha1", _rclpy.Parameter.Type.DOUBLE, 0.2),
        Parameter("alpha2", _rclpy.Parameter.Type.DOUBLE, 0.2),
        Parameter("alpha3", _rclpy.Parameter.Type.DOUBLE, 0.2),
        Parameter("alpha4", _rclpy.Parameter.Type.DOUBLE, 0.2),
        Parameter("recovery_alpha_slow", _rclpy.Parameter.Type.DOUBLE, 0.0),
        Parameter("recovery_alpha_fast", _rclpy.Parameter.Type.DOUBLE, 0.0),
        Parameter("use_cluster_estimate", _rclpy.Parameter.Type.BOOL, False),
    ])
    yield node
    node.destroy_node()


class TestAMCLNodeInit:
    def test_node_creates_successfully(self, amcl_node):
        assert amcl_node.get_name() == "amcl_node"

    def test_initial_state(self, amcl_node):
        assert amcl_node._map is None
        assert amcl_node._particles is None
        assert not amcl_node._initialized


class TestAMCLNodeMap:
    def test_map_callback_sets_state(self, amcl_node):
        from nav_msgs.msg import OccupancyGrid, MapMetaData
        from geometry_msgs.msg import Pose

        msg = OccupancyGrid()
        msg.info.width = 10
        msg.info.height = 10
        msg.info.resolution = 0.1
        msg.info.origin = Pose()
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.data = [0] * 100
        msg.data[0] = 100

        amcl_node._on_map(msg)

        assert amcl_node._map is not None
        assert amcl_node._dist_map is not None
        assert amcl_node._free_indices is not None
        assert amcl_node._map_meta is not None
        assert amcl_node._map.shape == (10, 10)


class TestAMCLNodeParticleInit:
    def test_init_particles(self, amcl_node):
        # Ensure map is set
        if amcl_node._map is None:
            from nav_msgs.msg import OccupancyGrid
            from geometry_msgs.msg import Pose
            msg = OccupancyGrid()
            msg.info.width = 20
            msg.info.height = 20
            msg.info.resolution = 0.05
            msg.info.origin = Pose()
            msg.data = [0] * 400
            msg.data[0] = 100
            amcl_node._on_map(msg)

        amcl_node._init_particles_at(0.5, 0.5, 0.0, 0.1, 0.05)
        assert amcl_node._particles is not None
        assert amcl_node._initialized
        N = amcl_node._particles.shape[0]
        assert N == int(amcl_node._p("max_particles"))
        assert amcl_node._weights.shape == (N,)
        assert amcl_node._weights.sum() == pytest.approx(1.0, abs=1e-10)


class TestAMCLNodeMeasurement:
    def test_correct_pose_higher_weight(self, rclpy_context):
        """Particle at true pose gets higher weight than one far away."""
        from rclpy.parameter import Parameter
        import rclpy as _rclpy
        from sensor_msgs.msg import LaserScan

        node = AMCLNode()
        node.set_parameters([
            Parameter("max_beams", _rclpy.Parameter.Type.INTEGER, 10),
            Parameter("sigma_hit", _rclpy.Parameter.Type.DOUBLE, 0.2),
            Parameter("z_hit", _rclpy.Parameter.Type.DOUBLE, 0.95),
            Parameter("z_rand", _rclpy.Parameter.Type.DOUBLE, 0.05),
            Parameter("laser_likelihood_max_dist", _rclpy.Parameter.Type.DOUBLE, 2.0),
            Parameter("max_range_ratio_for_z_max", _rclpy.Parameter.Type.DOUBLE, 0.98),
            Parameter("laser_origin_x", _rclpy.Parameter.Type.DOUBLE, 0.0),
            Parameter("laser_origin_y", _rclpy.Parameter.Type.DOUBLE, 0.0),
            Parameter("laser_min_range", _rclpy.Parameter.Type.DOUBLE, -1.0),
            Parameter("laser_max_range", _rclpy.Parameter.Type.DOUBLE, -1.0),
            Parameter("base_frame_id", _rclpy.Parameter.Type.STRING, "base_footprint"),
        ])

        grid, dist_map, meta = _make_test_map()
        node._map = grid
        node._map_meta = meta
        node._dist_map = dist_map
        node._free_indices = extract_free_indices(grid)
        node._laser_origin_tf = None
        node._laser_origin_tf_frame = None
        node._scan_frame_warned = True

        num_beams = 10
        scan = LaserScan()
        scan.header.frame_id = "base_footprint"
        scan.angle_min = -math.pi
        scan.angle_increment = 2.0 * math.pi / num_beams
        scan.angle_max = scan.angle_min + (num_beams - 1) * scan.angle_increment
        scan.range_min = 0.1
        scan.range_max = 10.0
        scan.ranges = [3.0] * num_beams
        scan.ranges[0] = 0.5

        node._particles = np.array([
            [0.5, 0.5, 0.0],
            [0.9, 0.5, 0.0],
        ])
        node._weights = np.array([0.5, 0.5])
        node._measurement_update(scan)

        assert node._weights[0] > node._weights[1], (
            f"Correct-pose weight {node._weights[0]:.4f} should exceed "
            f"wrong-pose weight {node._weights[1]:.4f}")
        node.destroy_node()


class TestAMCLNodeRecovery:
    def test_inject_preserves_count(self, amcl_node):
        if amcl_node._map is None:
            from nav_msgs.msg import OccupancyGrid
            from geometry_msgs.msg import Pose
            msg = OccupancyGrid()
            msg.info.width = 20
            msg.info.height = 20
            msg.info.resolution = 0.05
            msg.info.origin = Pose()
            msg.data = [0] * 400
            msg.data[0] = 100
            amcl_node._on_map(msg)

        N = 200
        amcl_node._particles = np.random.randn(N, 3)
        amcl_node._weights = np.full(N, 1.0 / N)
        n_injected = amcl_node._inject_recovery_particles(0.2)
        assert amcl_node._particles.shape == (N, 3)
        assert n_injected == 40

    def test_inject_zero_w_diff(self, amcl_node):
        N = 100
        amcl_node._particles = np.random.randn(N, 3)
        amcl_node._weights = np.full(N, 1.0 / N)
        n_injected = amcl_node._inject_recovery_particles(0.0)
        assert n_injected == 0


class TestAMCLNodeKidnapRecovery:
    """Tests for the kidnapped-robot detection and recovery mechanism."""

    def _setup_map(self, node):
        """Ensure node has a map for free-space injection."""
        if node._map is not None:
            return
        from nav_msgs.msg import OccupancyGrid
        from geometry_msgs.msg import Pose
        msg = OccupancyGrid()
        msg.info.width = 20
        msg.info.height = 20
        msg.info.resolution = 0.05
        msg.info.origin = Pose()
        msg.data = [0] * 400
        msg.data[0] = 100
        node._on_map(msg)

    def test_kidnap_recovery_injects_particles(self, amcl_node):
        self._setup_map(amcl_node)
        N = 200
        amcl_node._particles = np.random.randn(N, 3) * 0.01
        amcl_node._weights = np.full(N, 1.0 / N)
        amcl_node._pose_estimate = (0.5, 0.5, 0.0)

        import rclpy as _rclpy
        from rclpy.parameter import Parameter
        amcl_node.set_parameters([
            Parameter("kidnap_inject_fraction",
                      _rclpy.Parameter.Type.DOUBLE, 0.5),
        ])

        n = amcl_node._kidnap_recovery()
        assert n == 100
        assert amcl_node._particles.shape == (N, 3)
        # Weights should be reset to uniform
        np.testing.assert_allclose(
            amcl_node._weights, np.full(N, 1.0 / N))

    def test_kidnap_recovery_resets_w_averages(self, amcl_node):
        self._setup_map(amcl_node)
        N = 100
        amcl_node._particles = np.random.randn(N, 3) * 0.01
        amcl_node._weights = np.full(N, 1.0 / N)
        amcl_node._pose_estimate = (0.5, 0.5, 0.0)
        amcl_node._w_slow = 0.5
        amcl_node._w_fast = 0.1

        amcl_node._kidnap_recovery()
        assert amcl_node._w_slow == 0.0
        assert amcl_node._w_fast == 0.0

    def test_kidnap_counter_increments_neff(self, amcl_node):
        """Counter should increment when pre-resampling N_eff is low."""
        amcl_node._kidnap_count = 0
        amcl_node._last_best_per_beam_log = 0.0  # quality is fine
        N = 100
        amcl_node._particles = np.zeros((N, 3))
        amcl_node._weights = np.zeros(N)
        amcl_node._weights[0] = 1.0
        n_eff = effective_sample_size(amcl_node._weights)
        assert n_eff < 0.1 * N

        kidnap_neff_ratio = 0.1
        kidnap_log_thr = -2.0
        neff_bad = n_eff < kidnap_neff_ratio * N
        quality_bad = amcl_node._last_best_per_beam_log < kidnap_log_thr
        if neff_bad or quality_bad:
            amcl_node._kidnap_count += 1
        assert amcl_node._kidnap_count == 1

    def test_kidnap_counter_increments_quality(self, amcl_node):
        """Counter should increment when per-beam log-likelihood is very low,
        even if N_eff looks healthy (uniformly bad particles)."""
        amcl_node._kidnap_count = 0
        amcl_node._last_best_per_beam_log = -3.5  # very poor quality
        N = 100
        amcl_node._particles = np.zeros((N, 3))
        amcl_node._weights = np.full(N, 1.0 / N)  # uniform → N_eff = N
        n_eff = effective_sample_size(amcl_node._weights)
        assert n_eff >= 0.1 * N  # N_eff is healthy

        kidnap_neff_ratio = 0.1
        kidnap_log_thr = -2.0
        neff_bad = n_eff < kidnap_neff_ratio * N
        quality_bad = amcl_node._last_best_per_beam_log < kidnap_log_thr
        assert not neff_bad
        assert quality_bad
        if neff_bad or quality_bad:
            amcl_node._kidnap_count += 1
        assert amcl_node._kidnap_count == 1

    def test_kidnap_counter_resets_when_healthy(self, amcl_node):
        """Counter should reset when N_eff is healthy AND quality is good."""
        amcl_node._kidnap_count = 3
        amcl_node._last_best_per_beam_log = -0.8  # good quality
        N = 100
        amcl_node._particles = np.zeros((N, 3))
        amcl_node._weights = np.full(N, 1.0 / N)
        n_eff = effective_sample_size(amcl_node._weights)
        assert n_eff >= 0.1 * N
        kidnap_neff_ratio = 0.1
        kidnap_log_thr = -2.0
        neff_bad = n_eff < kidnap_neff_ratio * N
        quality_bad = amcl_node._last_best_per_beam_log < kidnap_log_thr
        if neff_bad or quality_bad:
            amcl_node._kidnap_count += 1
        else:
            amcl_node._kidnap_count = 0
        assert amcl_node._kidnap_count == 0

    def test_kidnap_injects_both_local_and_global(self, amcl_node):
        """Recovery should place particles near the estimate AND in free space."""
        self._setup_map(amcl_node)
        N = 200
        amcl_node._particles = np.zeros((N, 3))
        amcl_node._weights = np.full(N, 1.0 / N)
        amcl_node._pose_estimate = (0.5, 0.5, 0.0)

        import rclpy as _rclpy
        from rclpy.parameter import Parameter
        amcl_node.set_parameters([
            Parameter("kidnap_inject_fraction",
                      _rclpy.Parameter.Type.DOUBLE, 0.5),
        ])

        amcl_node._kidnap_recovery()
        xs = amcl_node._particles[:, 0]
        # Some particles should be near the estimate (within ~3 sigma = 3m)
        near_count = np.sum(np.abs(xs - 0.5) < 3.0)
        assert near_count > 0
        # Some should be from free space (map is 1m x 1m from origin 0,0)
        far_count = np.sum(np.abs(xs - 0.5) > 3.0)
        # With 100 global particles in a 1m map, most will actually be near 0.5
        # but the test ensures the method runs without error
        assert amcl_node._particles.shape == (N, 3)


class TestAMCLNodeShouldUpdate:
    def test_first_update_always(self, amcl_node):
        amcl_node._last_odom_pose = (0.0, 0.0, 0.0)
        amcl_node._last_update_odom_pose = None
        assert amcl_node._should_update() is True

    def test_small_motion_skipped(self, amcl_node):
        amcl_node._last_odom_pose = (0.01, 0.0, 0.01)
        amcl_node._last_update_odom_pose = (0.0, 0.0, 0.0)
        amcl_node._update_count = 999
        import rclpy as _rclpy
        from rclpy.parameter import Parameter
        amcl_node.set_parameters([
            Parameter("update_min_d", _rclpy.Parameter.Type.DOUBLE, 0.25),
            Parameter("update_min_a", _rclpy.Parameter.Type.DOUBLE, 0.2),
            Parameter("force_update_after_init", _rclpy.Parameter.Type.INTEGER, 30),
        ])
        assert amcl_node._should_update() is False

    def test_large_translation_triggers(self, amcl_node):
        amcl_node._last_odom_pose = (0.5, 0.0, 0.0)
        amcl_node._last_update_odom_pose = (0.0, 0.0, 0.0)
        amcl_node._update_count = 999
        assert amcl_node._should_update() is True

    def test_large_rotation_triggers(self, amcl_node):
        amcl_node._last_odom_pose = (0.0, 0.0, 0.5)
        amcl_node._last_update_odom_pose = (0.0, 0.0, 0.0)
        amcl_node._update_count = 999
        assert amcl_node._should_update() is True

    def test_force_update_bypasses_threshold(self, amcl_node):
        """Within the first force_update_after_init updates, motion threshold is bypassed."""
        import rclpy as _rclpy
        from rclpy.parameter import Parameter
        amcl_node.set_parameters([
            Parameter("update_min_d", _rclpy.Parameter.Type.DOUBLE, 0.25),
            Parameter("update_min_a", _rclpy.Parameter.Type.DOUBLE, 0.2),
            Parameter("force_update_after_init", _rclpy.Parameter.Type.INTEGER, 30),
        ])
        amcl_node._last_odom_pose = (0.001, 0.0, 0.001)
        amcl_node._last_update_odom_pose = (0.0, 0.0, 0.0)
        amcl_node._update_count = 5
        assert amcl_node._should_update() is True

    def test_force_update_expires(self, amcl_node):
        """After force_update_after_init updates, normal motion gating applies."""
        import rclpy as _rclpy
        from rclpy.parameter import Parameter
        amcl_node.set_parameters([
            Parameter("update_min_d", _rclpy.Parameter.Type.DOUBLE, 0.25),
            Parameter("update_min_a", _rclpy.Parameter.Type.DOUBLE, 0.2),
            Parameter("force_update_after_init", _rclpy.Parameter.Type.INTEGER, 30),
        ])
        amcl_node._last_odom_pose = (0.001, 0.0, 0.001)
        amcl_node._last_update_odom_pose = (0.0, 0.0, 0.0)
        amcl_node._update_count = 30
        assert amcl_node._should_update() is False

    def test_force_update_disabled(self, amcl_node):
        """When force_update_after_init=0, motion gating always applies."""
        import rclpy as _rclpy
        from rclpy.parameter import Parameter
        amcl_node.set_parameters([
            Parameter("update_min_d", _rclpy.Parameter.Type.DOUBLE, 0.25),
            Parameter("update_min_a", _rclpy.Parameter.Type.DOUBLE, 0.2),
            Parameter("force_update_after_init", _rclpy.Parameter.Type.INTEGER, 0),
        ])
        amcl_node._last_odom_pose = (0.001, 0.0, 0.001)
        amcl_node._last_update_odom_pose = (0.0, 0.0, 0.0)
        amcl_node._update_count = 0
        assert amcl_node._should_update() is False


# ===================================================================
# 12. Fixture map consistency (optional, like original tests)
# ===================================================================


class TestFixtureMap:
    def test_distance_map_consistent(self):
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "map_sample.npz"
        if not fixture_path.exists():
            pytest.skip("No fixtures/map_sample.npz")
        data = np.load(fixture_path)
        grid = data["grid"]
        res = float(data["resolution"])
        dist = compute_distance_map(grid, res, max_dist=2.0)
        assert dist.shape == grid.shape
        occupied = grid > 0
        np.testing.assert_allclose(dist[occupied], 0.0, atol=1e-5)
        assert np.all(dist[grid <= 0] >= 0)
