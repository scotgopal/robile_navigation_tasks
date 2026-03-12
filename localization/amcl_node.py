#!/usr/bin/env python3
"""
Adaptive Monte Carlo Localization (AMCL) ROS2 node.

Implements a production-quality particle filter with:
  - KLD-sampling for adaptive particle count (Fox 2003)
  - Differential-drive odometry motion model (Thrun, Probabilistic Robotics Ch.5)
  - Likelihood-field observation model with z_hit/z_rand mixture
  - Exponential weight averaging for automatic recovery detection (w_slow/w_fast)
  - Low-variance (systematic) resampling
  - Cluster-based pose estimation via spatial binning
  - Free-space-only particle injection for recovery
  - Precomputed EDT-based distance map

Compatible with Nav2 (Humble); frame and topic names match AMCL-style defaults.

Run:
  python3 amcl_node.py --ros-args --params-file config/amcl_params.yaml
  python3 amcl_localization.py --config config/amcl_params.yaml

Requires: map (nav_msgs/OccupancyGrid), scan (sensor_msgs/LaserScan),
          odom (nav_msgs/Odometry).
Python deps: numpy, scipy, rclpy, tf_transformations.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.special import erfinv

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.logging import LoggingSeverity, set_logger_level
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from geometry_msgs.msg import (
    PoseStamped,
    PoseArray,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf_transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_matrix,
    translation_matrix,
    concatenate_matrices,
    translation_from_matrix,
    quaternion_from_matrix,
)


# ---------------------------------------------------------------------------
# Pure-function helpers (no ROS dependency, easily testable)
# ---------------------------------------------------------------------------


def compute_distance_map(
    grid: np.ndarray,
    resolution: float,
    max_dist: float,
) -> np.ndarray:
    """
    Precompute distance (metres) from each cell to the nearest occupied cell.

    Uses SciPy EDT. ROS convention: occupied = (grid > 0); free/unknown treated
    as traversable for the distance transform.

    Parameters
    ----------
    grid : np.ndarray
        2-D occupancy grid (0 free, 100 occupied, -1 unknown).
    resolution : float
        Map resolution in m/cell.
    max_dist : float
        Distances are capped at this value (metres).

    Returns
    -------
    np.ndarray
        float32 array, same shape as *grid*: distance in metres to nearest
        occupied cell, capped at *max_dist*.  Occupied cells get 0.
    """
    free = (grid <= 0).astype(np.float64)
    dist_cells = distance_transform_edt(free)
    dist_m = (dist_cells * resolution).astype(np.float32)
    return np.minimum(dist_m, np.float32(max_dist))


def extract_free_indices(grid: np.ndarray) -> np.ndarray:
    """
    Return (N, 2) int32 array of (row, col) indices where grid <= 0 (free space).

    Parameters
    ----------
    grid : np.ndarray
        2-D occupancy grid.

    Returns
    -------
    np.ndarray
        Shape (N, 2) with columns [row, col].
    """
    rows, cols = np.where(grid <= 0)
    return np.column_stack((rows, cols)).astype(np.int32)


def kld_sample_count(
    k: int,
    pf_err: float = 0.05,
    pf_z: float = 0.99,
    min_particles: int = 100,
    max_particles: int = 5000,
) -> int:
    """
    Compute required number of particles for KLD-sampling (Fox 2003).

    Uses the Wilson-Hilferty approximation to the chi-squared quantile,
    matching Nav2 AMCL's ``pf_resample_limit``.

    Parameters
    ----------
    k : int
        Number of occupied histogram bins.
    pf_err : float
        Maximum allowed KL-divergence (epsilon).
    pf_z : float
        Upper quantile of the standard normal (e.g. 0.99 -> z ~ 2.33).
    min_particles : int
        Minimum particle count.
    max_particles : int
        Maximum particle count.

    Returns
    -------
    int
        Required particle count, clamped to [min_particles, max_particles].
    """
    if k <= 1:
        return max_particles

    z = math.sqrt(2.0) * erfinv(2.0 * pf_z - 1.0)
    a = 1.0
    b = 2.0 / (9.0 * (k - 1))
    c = math.sqrt(b) * z
    x = a - b + c
    if x <= 0:
        return max_particles
    n = int(math.ceil((k - 1) / (2.0 * pf_err) * x * x * x))
    return max(min_particles, min(n, max_particles))


def odometry_motion_model(
    particles: np.ndarray,
    delta_rot1: float,
    delta_trans: float,
    delta_rot2: float,
    alpha1: float,
    alpha2: float,
    alpha3: float,
    alpha4: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply the odometry-based motion model (Probabilistic Robotics, Table 5.6).

    Noise variances are proportional to *squared* motion magnitudes, matching
    Nav2 AMCL and the textbook formulation.

    Parameters
    ----------
    particles : np.ndarray
        (N, 3) array [x, y, theta].
    delta_rot1, delta_trans, delta_rot2 : float
        Odometry decomposition.
    alpha1 .. alpha4 : float
        Motion noise parameters.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Updated (N, 3) particle array.
    """
    N = particles.shape[0]
    dr1_sq = delta_rot1 * delta_rot1
    dt_sq = delta_trans * delta_trans
    dr2_sq = delta_rot2 * delta_rot2

    var_rot1 = alpha1 * dr1_sq + alpha2 * dt_sq
    var_trans = alpha3 * dt_sq + alpha4 * (dr1_sq + dr2_sq)
    var_rot2 = alpha1 * dr2_sq + alpha2 * dt_sq

    EPS = 1e-12
    noisy_rot1 = delta_rot1 + rng.normal(0.0, math.sqrt(max(var_rot1, EPS)), N)
    noisy_trans = delta_trans + rng.normal(0.0, math.sqrt(max(var_trans, EPS)), N)
    noisy_rot2 = delta_rot2 + rng.normal(0.0, math.sqrt(max(var_rot2, EPS)), N)

    th = particles[:, 2]
    particles[:, 0] += noisy_trans * np.cos(th + noisy_rot1)
    particles[:, 1] += noisy_trans * np.sin(th + noisy_rot1)
    particles[:, 2] += noisy_rot1 + noisy_rot2
    particles[:, 2] = np.arctan2(np.sin(particles[:, 2]),
                                  np.cos(particles[:, 2]))
    return particles


def likelihood_field_model(
    particles: np.ndarray,
    ranges: np.ndarray,
    angles: np.ndarray,
    valid_mask: np.ndarray,
    max_range: float,
    max_range_ratio: float,
    dist_map: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution: float,
    sigma_hit: float,
    z_hit: float,
    z_rand: float,
    max_dist: float,
    laser_x: float,
    laser_y: float,
) -> np.ndarray:
    """
    Compute log-likelihood of laser scan for each particle using the
    likelihood-field model with z_hit / z_rand mixture.

    Parameters
    ----------
    particles : np.ndarray
        (N, 3) particle poses [x, y, theta].
    ranges : np.ndarray
        (B,) laser ranges for selected beams.
    angles : np.ndarray
        (B,) laser beam angles in sensor frame.
    valid_mask : np.ndarray
        (B,) bool mask for finite, in-range beams.
    max_range : float
        Laser maximum range (metres).
    max_range_ratio : float
        Beam is treated as max-range when r >= max_range * max_range_ratio.
    dist_map : np.ndarray
        (H, W) precomputed distance-to-obstacle map (metres).
    origin_x, origin_y : float
        Map origin in world frame.
    resolution : float
        Map resolution (m/cell).
    sigma_hit : float
        Standard deviation for the hit component.
    z_hit : float
        Mixture weight for the Gaussian hit component.
    z_rand : float
        Mixture weight for the uniform random component.
    max_dist : float
        Distance map cap (metres).
    laser_x, laser_y : float
        Laser origin in the base frame.

    Returns
    -------
    np.ndarray
        (N,) log-likelihood per particle.
    """
    h, w = dist_map.shape
    N = particles.shape[0]
    B = ranges.shape[0]

    px = particles[:, 0:1]
    py = particles[:, 1:2]
    pth = particles[:, 2:3]

    r_b = ranges.reshape(1, B)
    a_b = angles.reshape(1, B)
    v_b = valid_mask.reshape(1, B)

    cos_th = np.cos(pth)
    sin_th = np.sin(pth)

    lx_map = px + laser_x * cos_th - laser_y * sin_th
    ly_map = py + laser_x * sin_th + laser_y * cos_th

    wx = lx_map + r_b * np.cos(a_b + pth)
    wy = ly_map + r_b * np.sin(a_b + pth)

    mx = np.clip(((wx - origin_x) / resolution).astype(np.int32), 0, w - 1)
    my = np.clip(((wy - origin_y) / resolution).astype(np.int32), 0, h - 1)

    d = dist_map[my, mx]

    inv_2sigma_sq = 1.0 / (2.0 * sigma_hit * sigma_hit)
    p_hit = z_hit * np.exp(-d * d * inv_2sigma_sq)
    p_rand = z_rand / max_range if max_range > 0.0 else 0.0
    p_total = p_hit + p_rand
    p_total = np.maximum(p_total, 1e-300)
    log_beam = np.log(p_total)

    is_max_range = r_b >= max_range * max_range_ratio
    log_max = np.log(max(1e-300, z_rand / max_range if max_range > 0 else 1e-300))
    log_per_beam = np.where(
        is_max_range & v_b, log_max,
        np.where(v_b, log_beam, 0.0)
    )
    log_weights = np.sum(log_per_beam, axis=1)

    x_lo, x_hi = origin_x, origin_x + w * resolution
    y_lo, y_hi = origin_y, origin_y + h * resolution
    outside = (
        (px.ravel() < x_lo) | (px.ravel() >= x_hi) |
        (py.ravel() < y_lo) | (py.ravel() >= y_hi)
    )
    log_weights[outside] = -np.inf

    return log_weights


def systematic_resample(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Low-variance (systematic) resampling.

    Parameters
    ----------
    particles : np.ndarray
        (N, 3) particle poses.
    weights : np.ndarray
        (N,) normalised weights.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Resampled particles (N, 3) and uniform weights (N,).
    """
    N = particles.shape[0]
    cdf = np.cumsum(weights)
    u0 = rng.uniform(0.0, 1.0 / N)
    u = u0 + np.arange(N, dtype=np.float64) / N
    indices = np.searchsorted(cdf, u, side="left")
    indices = np.clip(indices, 0, N - 1)
    return particles[indices].copy(), np.full(N, 1.0 / N)


def systematic_resample_kld(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    bin_size_xy: float,
    bin_size_theta: float,
    pf_err: float,
    pf_z: float,
    min_particles: int,
    max_particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multinomial resampling with KLD-based adaptive particle count.

    Each new particle is drawn independently from the full weighted
    distribution (matching Nav2 AMCL). Occupied histogram bins are tracked
    to compute the KLD stopping criterion.

    Parameters
    ----------
    particles : np.ndarray
        (N_old, 3) source particle poses.
    weights : np.ndarray
        (N_old,) normalised weights.
    rng : np.random.Generator
        Random generator.
    bin_size_xy : float
        Spatial bin size (metres) for KLD histogram.
    bin_size_theta : float
        Angular bin size (radians) for KLD histogram.
    pf_err, pf_z : float
        KLD-sampling parameters.
    min_particles, max_particles : int
        Particle count bounds.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Resampled particles (N_new, 3) and uniform weights (N_new,).
    """
    N_old = particles.shape[0]
    cdf = np.cumsum(weights)

    new_particles = []
    occupied_bins: set[tuple[int, int, int]] = set()
    kld_limit = max_particles

    for i in range(max_particles):
        u = rng.random()
        idx = int(np.searchsorted(cdf, u, side="left"))
        idx = min(idx, N_old - 1)
        p = particles[idx].copy()
        new_particles.append(p)

        bx = int(math.floor(p[0] / bin_size_xy))
        by = int(math.floor(p[1] / bin_size_xy))
        bth = int(math.floor(p[2] / bin_size_theta))
        b = (bx, by, bth)
        if b not in occupied_bins:
            occupied_bins.add(b)
            k = len(occupied_bins)
            kld_limit = kld_sample_count(
                k, pf_err, pf_z, min_particles, max_particles,
            )

        if i + 1 >= kld_limit and i + 1 >= min_particles:
            break

    out = np.array(new_particles)
    N_new = out.shape[0]
    return out, np.full(N_new, 1.0 / N_new)


def cluster_pose_estimate(
    particles: np.ndarray,
    weights: np.ndarray,
    bin_size_xy: float = 0.5,
    bin_size_theta: float = math.radians(10.0),
) -> tuple[float, float, float]:
    """
    Compute the pose estimate as the weighted mean of the highest-weight cluster.

    Particles are binned into spatial-angular cells.  The cluster with the
    highest total weight is selected and its weighted mean is returned.

    Parameters
    ----------
    particles : np.ndarray
        (N, 3) particle poses.
    weights : np.ndarray
        (N,) normalised weights.
    bin_size_xy : float
        Spatial bin size (metres).
    bin_size_theta : float
        Angular bin size (radians).

    Returns
    -------
    tuple of (float, float, float)
        (x, y, theta) pose estimate.
    """
    N = particles.shape[0]
    bx = np.floor(particles[:, 0] / bin_size_xy).astype(np.int64)
    by = np.floor(particles[:, 1] / bin_size_xy).astype(np.int64)
    bth = np.floor(particles[:, 2] / bin_size_theta).astype(np.int64)

    bins: dict[tuple[int, int, int], list[int]] = {}
    for i in range(N):
        key = (int(bx[i]), int(by[i]), int(bth[i]))
        bins.setdefault(key, []).append(i)

    best_key = max(bins, key=lambda k: sum(weights[i] for i in bins[k]))
    idxs = np.array(bins[best_key])
    w_sub = weights[idxs]
    w_sum = w_sub.sum()
    if w_sum <= 0:
        w_sub = np.ones_like(w_sub) / len(w_sub)
        w_sum = 1.0

    x = float(np.average(particles[idxs, 0], weights=w_sub))
    y = float(np.average(particles[idxs, 1], weights=w_sub))
    theta = float(np.arctan2(
        np.average(np.sin(particles[idxs, 2]), weights=w_sub),
        np.average(np.cos(particles[idxs, 2]), weights=w_sub),
    ))
    return x, y, theta


def weighted_pose_estimate(
    particles: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    """
    Standard weighted-mean pose estimate (circular mean for theta).

    Parameters
    ----------
    particles : np.ndarray
        (N, 3) particle poses.
    weights : np.ndarray
        (N,) normalised weights.

    Returns
    -------
    tuple of (float, float, float)
        (x, y, theta).
    """
    x = float(np.average(particles[:, 0], weights=weights))
    y = float(np.average(particles[:, 1], weights=weights))
    theta = float(np.arctan2(
        np.average(np.sin(particles[:, 2]), weights=weights),
        np.average(np.cos(particles[:, 2]), weights=weights),
    ))
    return x, y, theta


def particle_covariance(
    particles: np.ndarray,
    weights: np.ndarray,
    mean_pose: tuple[float, float, float],
) -> list[float]:
    """
    Compute 6x6 covariance (row-major, 36 elements) from weighted particles.

    Only x, y, yaw entries are populated; others are zero.

    Parameters
    ----------
    particles : np.ndarray
        (N, 3) particle poses.
    weights : np.ndarray
        (N,) normalised weights.
    mean_pose : tuple of (float, float, float)
        Mean (x, y, theta) around which covariance is computed.

    Returns
    -------
    list of float
        36-element row-major covariance.
    """
    xm, ym, thm = mean_pose
    x = particles[:, 0]
    y = particles[:, 1]
    dth = np.arctan2(np.sin(particles[:, 2] - thm),
                     np.cos(particles[:, 2] - thm))
    cov_xx = float(np.average((x - xm) ** 2, weights=weights))
    cov_yy = float(np.average((y - ym) ** 2, weights=weights))
    cov_thth = float(np.average(dth ** 2, weights=weights))
    cov_xy = float(np.average((x - xm) * (y - ym), weights=weights))
    cov_xth = float(np.average((x - xm) * dth, weights=weights))
    cov_yth = float(np.average((y - ym) * dth, weights=weights))

    cov = [0.0] * 36
    cov[0] = cov_xx
    cov[1] = cov[6] = cov_xy
    cov[5] = cov[30] = cov_xth
    cov[7] = cov_yy
    cov[11] = cov[31] = cov_yth
    cov[35] = cov_thth
    return cov


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Effective sample size: N_eff = 1 / sum(w_i^2).

    Parameters
    ----------
    weights : np.ndarray
        (N,) normalised weights.

    Returns
    -------
    float
        Effective particle count; N when uniform, 1 when degenerate.
    """
    if weights is None or weights.size == 0:
        return 0.0
    s = np.sum(weights * weights)
    return float(1.0 / s) if s > 0 else 0.0


# ---------------------------------------------------------------------------
# AMCL ROS2 Node
# ---------------------------------------------------------------------------


class AMCLNode(Node):
    """
    Adaptive Monte Carlo Localization node.

    Subscribes to map, scan, odom, and initialpose.
    Publishes pose estimate, covariance, particle cloud, path, and map->odom TF.
    """

    def __init__(self) -> None:
        super().__init__("amcl_node")
        self.log = self.get_logger()
        self._declare_parameters()

        self._rng = np.random.default_rng(seed=None)

        # State
        self._map: Optional[np.ndarray] = None
        self._map_meta: Optional[dict] = None
        self._dist_map: Optional[np.ndarray] = None
        self._free_indices: Optional[np.ndarray] = None

        self._particles: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._last_odom_pose: Optional[tuple[float, float, float]] = None
        self._last_update_odom_pose: Optional[tuple[float, float, float]] = None
        self._update_count: int = 0
        self._pose_estimate: Optional[tuple[float, float, float]] = None
        self._initialized: bool = False

        # Recovery: exponential moving averages of average weight
        self._w_slow: float = 0.0
        self._w_fast: float = 0.0
        self._kidnap_count: int = 0
        self._last_best_per_beam_log: float = 0.0

        # Laser origin caching
        self._laser_origin_tf: Optional[tuple[float, float]] = None
        self._laser_origin_tf_frame: Optional[str] = None
        self._scan_frame_warned: bool = False

        # Throttled logging timestamps
        self._scan_skip_log_time: float = 0.0
        self._tf_warn_time: float = 0.0

        # Path display
        self._path_poses: list[PoseStamped] = []

        # QoS profiles
        qos_map = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
        )

        self._state_cb_group = MutuallyExclusiveCallbackGroup()
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()

        # Subscriptions -- filter-critical callbacks share one group
        self._sub_map = self.create_subscription(
            OccupancyGrid, self._p("map_topic"), self._on_map, qos_map)
        self._sub_scan = self.create_subscription(
            LaserScan, self._p("scan_topic"), self._on_scan, qos_sensor,
            callback_group=self._state_cb_group)
        self._sub_odom = self.create_subscription(
            Odometry, self._p("odom_topic"), self._on_odom, 10,
            callback_group=self._state_cb_group)
        self._sub_initial_pose = self.create_subscription(
            PoseWithCovarianceStamped, self._p("initial_pose_topic"),
            self._on_initial_pose, 10, callback_group=self._state_cb_group)

        # Publishers
        self._pub_pose = self.create_publisher(
            PoseStamped, f"~/{self._p('publish_pose_topic')}", 10)
        self._pub_pose_cov = self.create_publisher(
            PoseWithCovarianceStamped,
            f"~/{self._p('publish_pose_with_covariance_topic')}", 10)
        self._pub_particles = self.create_publisher(
            PoseArray, f"~/{self._p('publish_particle_cloud_topic')}", 10)
        self._pub_path = self.create_publisher(
            Path, f"~/{self._p('publish_path_topic')}", 10)

        self._tf_broadcaster = TransformBroadcaster(self)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Timers in a SEPARATE callback group so TF republishing and
        # particle cloud display never starve the filter callbacks.
        cloud_hz = float(self._p("particle_cloud_publish_rate"))
        self._timer_cloud = self.create_timer(
            1.0 / max(cloud_hz, 0.1), self._publish_particle_cloud,
            callback_group=self._timer_cb_group)

        tf_hz = float(self._p("tf_publish_rate"))
        if tf_hz > 0:
            self._timer_tf = self.create_timer(
                1.0 / tf_hz, self._publish_map_odom_tf_idle,
                callback_group=self._timer_cb_group)

        self.log.info(
            "AMCL node started. Waiting for map and initial pose. "
            f"Particles: {self._p('min_particles')}..{self._p('max_particles')} (KLD). "
            f"Recovery: alpha_slow={self._p('recovery_alpha_slow')}, "
            f"alpha_fast={self._p('recovery_alpha_fast')}."
        )

    # -- Parameter helpers ---------------------------------------------------

    def _p(self, name: str):
        """Shorthand for reading a declared parameter value."""
        return self.get_parameter(name).value

    def _declare_parameters(self) -> None:
        d = self.declare_parameter
        # Frames
        d("map_frame_id", "map")
        d("odom_frame_id", "odom")
        d("base_frame_id", "base_footprint")
        # Topics
        d("scan_topic", "scan")
        d("odom_topic", "odom")
        d("map_topic", "map")
        d("initial_pose_topic", "initialpose")

        # Particle filter -- KLD adaptive
        d("min_particles", 500)
        d("max_particles", 2000)
        d("pf_err", 0.05)
        d("pf_z", 0.99)
        d("kld_bin_size_xy", 0.5)
        d("kld_bin_size_theta", 0.1745)  # ~10 degrees

        # Resampling
        d("resample_interval", 1)
        d("resample_neff_ratio", 0.5)

        # Recovery (exponential-average weight tracking)
        d("recovery_alpha_slow", 0.001)
        d("recovery_alpha_fast", 0.1)

        # Kidnapped-robot detection
        d("kidnap_neff_ratio", 0.1)
        d("kidnap_consecutive_threshold", 5)
        d("kidnap_inject_fraction", 0.5)
        d("kidnap_per_beam_log_threshold", -2.0)

        # Update gating
        d("update_min_d", 0.25)
        d("update_min_a", 0.2)
        d("force_update_after_init", 30)

        # Initial pose spread
        d("initial_pose_std_xy", 0.5)
        d("initial_pose_std_theta", 0.26)  # ~15 degrees
        d("initial_pose_params_override", True)

        # Motion model (differential drive, squared-magnitude noise)
        d("alpha1", 0.2)
        d("alpha2", 0.2)
        d("alpha3", 0.2)
        d("alpha4", 0.2)

        # Sensor model
        d("laser_origin_x", 0.0)
        d("laser_origin_y", 0.0)
        d("max_beams", 60)
        d("sigma_hit", 0.2)
        d("z_hit", 0.5)
        d("z_rand", 0.5)
        d("laser_likelihood_max_dist", 2.0)
        d("max_range_ratio_for_z_max", 0.98)
        d("laser_max_range", -1.0)
        d("laser_min_range", -1.0)

        # Pose estimation
        d("use_cluster_estimate", True)
        d("cluster_bin_xy", 0.5)
        d("cluster_bin_theta", 0.1745)

        # Publishing
        d("publish_pose_topic", "pose")
        d("publish_pose_with_covariance_topic", "amcl_pose")
        d("publish_particle_cloud_topic", "particle_cloud")
        d("particle_cloud_publish_rate", 5.0)
        d("publish_path_topic", "path")
        d("path_max_poses", 200)
        d("path_decay_time", 0.0)
        d("tf_publish_rate", 1.0)
        d("transform_tolerance", 1.0)

    # -- Map callback --------------------------------------------------------

    def _on_map(self, msg: OccupancyGrid) -> None:
        if self._map is not None:
            self.log.info("Ignoring duplicate map message (already received).")
            return
        t0 = time.perf_counter()
        self._map_meta = {
            "origin_x": msg.info.origin.position.x,
            "origin_y": msg.info.origin.position.y,
            "resolution": msg.info.resolution,
            "width": msg.info.width,
            "height": msg.info.height,
        }
        self._map = np.array(msg.data, dtype=np.int32).reshape(
            (msg.info.height, msg.info.width))
        max_dist = float(self._p("laser_likelihood_max_dist"))
        self._dist_map = compute_distance_map(
            self._map, self._map_meta["resolution"], max_dist)
        self._free_indices = extract_free_indices(self._map)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.log.info(
            f"Map received: {self._map_meta['width']}x{self._map_meta['height']}, "
            f"res={self._map_meta['resolution']:.3f} m/cell, "
            f"free_cells={self._free_indices.shape[0]}, "
            f"dist_map computed in {elapsed:.1f} ms"
        )

    # -- Initial pose --------------------------------------------------------

    def _on_initial_pose(self, msg: PoseWithCovarianceStamped) -> None:
        if self._map is None:
            self.log.warning(
                "Initial pose received but map not yet available; ignoring.")
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        std_xy = float(self._p("initial_pose_std_xy"))
        std_theta = float(self._p("initial_pose_std_theta"))
        if not self._p("initial_pose_params_override"):
            cov = getattr(msg.pose, "covariance", None)
            if cov is not None and len(cov) >= 36:
                vx, vy, vyaw = float(cov[0]), float(cov[7]), float(cov[35])
                if vx > 0 and np.isfinite(vx):
                    std_xy = math.sqrt(vx)
                if vy > 0 and np.isfinite(vy):
                    std_xy = (std_xy + math.sqrt(vy)) / 2.0
                if vyaw > 0 and np.isfinite(vyaw):
                    std_theta = math.sqrt(vyaw)

        self._init_particles_at(p.x, p.y, yaw, std_xy, std_theta)

    def _init_particles_at(
        self, x: float, y: float, theta: float,
        std_xy: float, std_theta: float,
    ) -> None:
        n = int(self._p("max_particles"))
        std_xy = max(1e-6, std_xy)
        std_theta = max(1e-6, std_theta)
        self._particles = np.zeros((n, 3))
        self._particles[:, 0] = x + self._rng.normal(0, std_xy, n)
        self._particles[:, 1] = y + self._rng.normal(0, std_xy, n)
        self._particles[:, 2] = theta + self._rng.normal(0, std_theta, n)
        self._particles[:, 2] = np.arctan2(
            np.sin(self._particles[:, 2]),
            np.cos(self._particles[:, 2]))
        self._weights = np.full(n, 1.0 / n)
        self._pose_estimate = (x, y, theta)
        self._initialized = True
        self._update_count = 0
        self._w_slow = 0.0
        self._w_fast = 0.0
        self._kidnap_count = 0
        self._last_best_per_beam_log = 0.0
        self._last_odom_pose = None
        self._last_update_odom_pose = None
        self._scan_frame_warned = False
        self._laser_origin_tf = None
        self._laser_origin_tf_frame = None
        self.log.info(
            f"Particles initialized: N={n} at ({x:.3f}, {y:.3f}, "
            f"{math.degrees(theta):.1f} deg), "
            f"std_xy={std_xy:.3f} m, std_theta={math.degrees(std_theta):.1f} deg"
        )

    # -- Odometry callback ---------------------------------------------------

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._last_odom_pose = (p.x, p.y, theta)

    def _apply_motion_since_last_update(self) -> None:
        """Apply motion model once with the full odom delta since the last
        filter update, matching Nav2 AMCL behaviour.

        Uses the textbook decomposition (Probabilistic Robotics, Table 5.6):
          delta_rot1  = atan2(y'-y, x'-x) - theta
          delta_trans = sqrt((x'-x)^2 + (y'-y)^2)
          delta_rot2  = (theta'-theta) - delta_rot1
        """
        if self._particles is None or self._last_odom_pose is None:
            return
        if self._last_update_odom_pose is None:
            return
        dx = self._last_odom_pose[0] - self._last_update_odom_pose[0]
        dy = self._last_odom_pose[1] - self._last_update_odom_pose[1]
        dth = self._last_odom_pose[2] - self._last_update_odom_pose[2]
        dth = math.atan2(math.sin(dth), math.cos(dth))

        trans = math.sqrt(dx * dx + dy * dy)
        if trans > 1e-6:
            rot1 = math.atan2(dy, dx) - self._last_update_odom_pose[2]
            rot1 = math.atan2(math.sin(rot1), math.cos(rot1))
        else:
            rot1 = 0.0
        rot2 = dth - rot1
        rot2 = math.atan2(math.sin(rot2), math.cos(rot2))

        odometry_motion_model(
            self._particles,
            rot1, trans, rot2,
            float(self._p("alpha1")),
            float(self._p("alpha2")),
            float(self._p("alpha3")),
            float(self._p("alpha4")),
            self._rng,
        )

    # -- Scan callback (main filter cycle) -----------------------------------

    def _on_scan(self, msg: LaserScan) -> None:
        if self._map is None or not self._initialized or self._particles is None:
            return
        if self._last_odom_pose is None:
            return

        if not self._should_update():
            return

        t_total_start = time.perf_counter()

        # --- Motion update (predict) using full delta since last update ---
        self._apply_motion_since_last_update()

        # --- Measurement update (correct) ---
        t0 = time.perf_counter()
        avg_w = self._measurement_update(msg)
        t_meas = (time.perf_counter() - t0) * 1000.0

        self._update_count += 1
        self._last_update_odom_pose = self._last_odom_pose

        # --- Recovery tracking via exponential weight averages ---
        w_diff = 0.0
        if avg_w is not None and avg_w > 0:
            alpha_slow = float(self._p("recovery_alpha_slow"))
            alpha_fast = float(self._p("recovery_alpha_fast"))
            if alpha_slow > 0 and alpha_fast > 0:
                if self._w_slow == 0.0:
                    self._w_slow = avg_w
                else:
                    self._w_slow += alpha_slow * (avg_w - self._w_slow)
                if self._w_fast == 0.0:
                    self._w_fast = avg_w
                else:
                    self._w_fast += alpha_fast * (avg_w - self._w_fast)
                if self._w_slow > 0.0:
                    w_diff = max(0.0, 1.0 - self._w_fast / self._w_slow)

        # --- Resampling ---
        t0 = time.perf_counter()
        resample_interval = int(self._p("resample_interval"))
        do_resample = (resample_interval > 0 and
                       self._update_count % resample_interval == 0)
        n_eff = effective_sample_size(self._weights)
        N = self._particles.shape[0]
        neff_ratio = float(self._p("resample_neff_ratio"))
        if neff_ratio > 0 and n_eff < neff_ratio * N:
            do_resample = True

        n_recovery = 0
        if do_resample:
            self._particles, self._weights = systematic_resample_kld(
                self._particles, self._weights, self._rng,
                float(self._p("kld_bin_size_xy")),
                float(self._p("kld_bin_size_theta")),
                float(self._p("pf_err")),
                float(self._p("pf_z")),
                int(self._p("min_particles")),
                int(self._p("max_particles")),
            )
            if w_diff > 0.0:
                n_recovery = self._inject_recovery_particles(w_diff)
                self._w_slow = 0.0
                self._w_fast = 0.0
        t_resample = (time.perf_counter() - t0) * 1000.0

        # --- Kidnapped-robot detection ---
        # Two complementary signals:
        #   1. Pre-resampling N_eff collapse (particles disagree on weights)
        #   2. Poor absolute scan quality (best particle's per-beam log
        #      likelihood below threshold — all particles uniformly bad)
        n_kidnap_injected = 0
        kidnap_neff_ratio = float(self._p("kidnap_neff_ratio"))
        kidnap_threshold = int(self._p("kidnap_consecutive_threshold"))
        kidnap_log_thr = float(self._p("kidnap_per_beam_log_threshold"))
        N = self._particles.shape[0]

        neff_bad = (kidnap_neff_ratio > 0 and
                    n_eff < kidnap_neff_ratio * N)
        quality_bad = (kidnap_log_thr < 0 and
                       self._last_best_per_beam_log < kidnap_log_thr)
        if kidnap_threshold > 0 and (neff_bad or quality_bad):
            self._kidnap_count += 1
        else:
            self._kidnap_count = 0

        if self._kidnap_count >= kidnap_threshold > 0:
            n_kidnap_injected = self._kidnap_recovery()
            self._kidnap_count = 0

        # --- Pose estimation ---
        t0 = time.perf_counter()
        if self._p("use_cluster_estimate"):
            self._pose_estimate = cluster_pose_estimate(
                self._particles, self._weights,
                float(self._p("cluster_bin_xy")),
                float(self._p("cluster_bin_theta")),
            )
        else:
            self._pose_estimate = weighted_pose_estimate(
                self._particles, self._weights)
        t_pose = (time.perf_counter() - t0) * 1000.0

        # --- Publish ---
        self._publish_pose_and_tf(msg.header.stamp)

        t_total = (time.perf_counter() - t_total_start) * 1000.0
        n_eff_post = effective_sample_size(self._weights)
        x, y, th = self._pose_estimate
        N = self._particles.shape[0]

        is_periodic = (self._update_count <= 5 or
                       self._update_count % 10 == 0)
        is_event = (n_recovery > 0 or n_kidnap_injected > 0 or
                    self._kidnap_count >= max(1, kidnap_threshold - 2))

        if is_periodic or is_event:
            extra = ""
            pbl = self._last_best_per_beam_log
            extra += f" pbl={pbl:.2f}"
            if self._w_slow > 0:
                extra += f" w_slow={self._w_slow:.4f} w_fast={self._w_fast:.4f}"
            if w_diff > 0:
                extra += f" w_diff={w_diff:.3f}"
            if n_recovery > 0:
                extra += f" recovery={n_recovery}"
            if n_kidnap_injected > 0:
                extra += f" KIDNAP_RECOVERY={n_kidnap_injected}"
            elif self._kidnap_count > 0:
                extra += f" kidnap_streak={self._kidnap_count}/{kidnap_threshold}"
            self.log.info(
                f"AMCL #{self._update_count}: "
                f"pose=({x:.3f}, {y:.3f}, {math.degrees(th):.1f} deg) "
                f"N={N} N_eff={n_eff:.0f}/{n_eff_post:.0f} "
                f"total={t_total:.1f} ms{extra}"
            )
        else:
            self.log.debug(
                f"AMCL #{self._update_count}: "
                f"pose=({x:.3f}, {y:.3f}, {math.degrees(th):.1f} deg) "
                f"N={N} N_eff={n_eff_post:.0f}"
            )

    def _should_update(self) -> bool:
        """Check whether the robot has moved enough to trigger a filter update."""
        if self._last_update_odom_pose is None:
            return True
        force_n = int(self._p("force_update_after_init"))
        if force_n > 0 and self._update_count < force_n:
            return True
        min_d = float(self._p("update_min_d"))
        min_a = float(self._p("update_min_a"))
        dx = self._last_odom_pose[0] - self._last_update_odom_pose[0]
        dy = self._last_odom_pose[1] - self._last_update_odom_pose[1]
        dist = math.sqrt(dx * dx + dy * dy)
        da = abs(math.atan2(
            math.sin(self._last_odom_pose[2] - self._last_update_odom_pose[2]),
            math.cos(self._last_odom_pose[2] - self._last_update_odom_pose[2]),
        ))
        if dist >= min_d or da >= min_a:
            return True
        now = time.monotonic()
        if now - self._scan_skip_log_time >= 10.0:
            self.log.debug(
                f"Scan skipped: motion below threshold "
                f"(d={dist:.3f}<{min_d}, da={da:.3f}<{min_a})")
            self._scan_skip_log_time = now
        return False

    def _measurement_update(self, msg: LaserScan) -> Optional[float]:
        """
        Run the likelihood-field sensor model. Returns the average weight
        (before normalisation) for recovery tracking, or None on failure.

        Also stores ``_last_best_per_beam_log`` for kidnap quality detection.
        """
        if self._particles is None or self._weights is None:
            return None
        if self._dist_map is None or self._map_meta is None:
            return None

        max_beams = int(self._p("max_beams"))
        sigma_hit = float(self._p("sigma_hit"))
        z_hit_w = float(self._p("z_hit"))
        z_rand_w = float(self._p("z_rand"))
        max_dist = float(self._p("laser_likelihood_max_dist"))
        max_range_ratio = float(self._p("max_range_ratio_for_z_max"))

        msg_rmin = getattr(msg, "range_min", 0.0)
        msg_rmax = getattr(msg, "range_max", float("inf"))
        param_rmin = float(self._p("laser_min_range"))
        param_rmax = float(self._p("laser_max_range"))
        min_range = param_rmin if param_rmin >= 0 else float(msg_rmin)
        max_range = param_rmax if param_rmax > 0 else float(msg_rmax)
        if not np.isfinite(max_range) or max_range <= min_range:
            max_range = 100.0

        num_readings = len(msg.ranges)
        if num_readings == 0:
            return None
        if num_readings <= max_beams:
            indices = np.arange(num_readings)
        else:
            step = num_readings / max_beams
            indices = (np.arange(max_beams) * step).astype(np.int32)
            indices = np.clip(indices, 0, num_readings - 1)

        r = np.array(msg.ranges, dtype=np.float64)[indices]
        angles = msg.angle_min + indices * msg.angle_increment
        valid = np.isfinite(r) & (r >= min_range) & (r <= max_range)
        if np.sum(valid) == 0:
            self.log.debug("Measurement update skipped: no valid beams.")
            return None

        lox, loy = self._resolve_laser_origin(msg)

        log_weights = likelihood_field_model(
            self._particles, r, angles, valid,
            max_range, max_range_ratio,
            self._dist_map,
            self._map_meta["origin_x"], self._map_meta["origin_y"],
            self._map_meta["resolution"],
            sigma_hit, z_hit_w, z_rand_w, max_dist,
            lox, loy,
        )

        max_log = np.max(log_weights)
        n_valid = int(np.sum(valid))
        self._last_best_per_beam_log = (
            float(max_log / n_valid) if n_valid > 0 else -999.0)

        if not np.isfinite(max_log):
            self.log.warning(
                "All particles have -inf log weight (all outside map?); "
                "using uniform weights.")
            N = self._particles.shape[0]
            self._weights = np.full(N, 1.0 / N)
            return None

        log_weights -= max_log
        self._weights = np.exp(log_weights)
        total = np.sum(self._weights)
        if total <= 0:
            self.log.warning("Weights sum to zero; using uniform weights.")
            N = self._particles.shape[0]
            self._weights = np.full(N, 1.0 / N)
            return None

        avg_unnorm = float(total / self._particles.shape[0])
        self._weights /= total
        return avg_unnorm

    def _resolve_laser_origin(
        self, msg: LaserScan,
    ) -> tuple[float, float]:
        """Determine the laser sensor origin in the base frame."""
        base_frame = self._p("base_frame_id")
        scan_frame = getattr(msg.header, "frame_id", "") or ""
        lox = float(self._p("laser_origin_x"))
        loy = float(self._p("laser_origin_y"))

        if not scan_frame or scan_frame == base_frame:
            return lox, loy

        if self._laser_origin_tf_frame == scan_frame:
            if self._laser_origin_tf is not None:
                return self._laser_origin_tf
            return lox, loy

        t = None
        for _ in range(3):
            try:
                t = self._tf_buffer.lookup_transform(
                    base_frame, scan_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2))
                break
            except Exception:
                time.sleep(0.1)
                rclpy.spin_once(self, timeout_sec=0.05)

        if t is not None:
            lox = t.transform.translation.x
            loy = t.transform.translation.y
            self._laser_origin_tf = (lox, loy)
            self._laser_origin_tf_frame = scan_frame
            if not self._scan_frame_warned:
                self.log.info(
                    f"Laser origin from TF ({base_frame}->{scan_frame}): "
                    f"x={lox:.3f}, y={loy:.3f}")
                self._scan_frame_warned = True
        else:
            if not self._scan_frame_warned:
                self.log.warning(
                    f"Scan frame '{scan_frame}' differs from base '{base_frame}' "
                    f"but TF lookup failed. Using laser_origin params "
                    f"({lox:.3f}, {loy:.3f}). Incorrect origin degrades "
                    f"localization quality.")
                self._scan_frame_warned = True

        return lox, loy

    def _inject_recovery_particles(self, w_diff: float) -> int:
        """
        Replace a fraction (w_diff) of particles with uniform samples from
        free space. Returns the number of particles injected.
        """
        if (self._particles is None or self._free_indices is None or
                self._map_meta is None or w_diff <= 0):
            return 0
        N = self._particles.shape[0]
        n_inject = max(1, int(N * w_diff))
        n_inject = min(n_inject, N)

        res = self._map_meta["resolution"]
        ox = self._map_meta["origin_x"]
        oy = self._map_meta["origin_y"]
        n_free = self._free_indices.shape[0]
        if n_free == 0:
            return 0

        chosen = self._rng.choice(n_free, size=n_inject, replace=True)
        rows = self._free_indices[chosen, 0]
        cols = self._free_indices[chosen, 1]

        inject_x = ox + (cols + self._rng.uniform(0, 1, n_inject)) * res
        inject_y = oy + (rows + self._rng.uniform(0, 1, n_inject)) * res
        inject_th = self._rng.uniform(-math.pi, math.pi, n_inject)

        replace_idx = self._rng.choice(N, size=n_inject, replace=False)
        self._particles[replace_idx, 0] = inject_x
        self._particles[replace_idx, 1] = inject_y
        self._particles[replace_idx, 2] = inject_th
        self._weights[replace_idx] = 1.0 / N

        w_sum = self._weights.sum()
        if w_sum > 0:
            self._weights /= w_sum

        self.log.info(
            f"Recovery: injected {n_inject}/{N} particles from free space "
            f"(w_diff={w_diff:.3f})")
        return n_inject

    def _kidnap_recovery(self) -> int:
        """Inject a large batch of particles when catastrophic divergence is
        detected (N_eff has been critically low for several consecutive
        updates). Half the injected particles are spread widely around the
        current best estimate; the other half are drawn from free space."""
        if (self._particles is None or self._free_indices is None or
                self._map_meta is None):
            return 0
        frac = float(self._p("kidnap_inject_fraction"))
        N = self._particles.shape[0]
        n_inject = max(1, int(N * frac))
        n_inject = min(n_inject, N)

        res = self._map_meta["resolution"]
        ox = self._map_meta["origin_x"]
        oy = self._map_meta["origin_y"]
        n_free = self._free_indices.shape[0]
        if n_free == 0:
            return 0

        replace_idx = self._rng.choice(N, size=n_inject, replace=False)

        n_around = n_inject // 2
        n_global = n_inject - n_around

        if n_around > 0 and self._pose_estimate is not None:
            ex, ey, eth = self._pose_estimate
            self._particles[replace_idx[:n_around], 0] = (
                ex + self._rng.normal(0, 1.0, n_around))
            self._particles[replace_idx[:n_around], 1] = (
                ey + self._rng.normal(0, 1.0, n_around))
            self._particles[replace_idx[:n_around], 2] = (
                eth + self._rng.normal(0, math.pi / 2, n_around))
        else:
            n_global = n_inject

        if n_global > 0:
            global_idx = replace_idx[n_around:] if n_around > 0 else replace_idx
            chosen = self._rng.choice(n_free, size=n_global, replace=True)
            rows = self._free_indices[chosen, 0]
            cols = self._free_indices[chosen, 1]
            self._particles[global_idx, 0] = (
                ox + (cols + self._rng.uniform(0, 1, n_global)) * res)
            self._particles[global_idx, 1] = (
                oy + (rows + self._rng.uniform(0, 1, n_global)) * res)
            self._particles[global_idx, 2] = self._rng.uniform(
                -math.pi, math.pi, n_global)

        self._weights = np.full(N, 1.0 / N)
        self._w_slow = 0.0
        self._w_fast = 0.0

        self.log.warning(
            f"KIDNAP RECOVERY: injected {n_inject}/{N} particles "
            f"({n_around} near estimate, {n_global} from free space)")
        return n_inject

    # -- Publishing ----------------------------------------------------------

    def _publish_pose_and_tf(self, stamp) -> None:
        if self._pose_estimate is None:
            return
        gf = self._p("map_frame_id")
        x, y, theta = self._pose_estimate
        q = quaternion_from_euler(0, 0, theta)

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = gf
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        self._pub_pose.publish(ps)

        cov = particle_covariance(
            self._particles, self._weights, self._pose_estimate)
        pwc = PoseWithCovarianceStamped()
        pwc.header.stamp = stamp
        pwc.header.frame_id = gf
        pwc.pose.pose = ps.pose
        pwc.pose.covariance = cov
        self._pub_pose_cov.publish(pwc)

        self._update_path(ps, stamp, gf)
        self._publish_map_odom_tf(stamp)

    def _update_path(self, ps: PoseStamped, stamp, frame_id: str) -> None:
        copy = PoseStamped()
        copy.header.stamp = stamp
        copy.header.frame_id = frame_id
        copy.pose = ps.pose
        self._path_poses.append(copy)

        decay = float(self._p("path_decay_time"))
        if decay > 0:
            cutoff = stamp.sec + stamp.nanosec * 1e-9 - decay
            self._path_poses = [
                p for p in self._path_poses
                if (p.header.stamp.sec + p.header.stamp.nanosec * 1e-9) >= cutoff
            ]
        max_poses = int(self._p("path_max_poses"))
        if len(self._path_poses) > max_poses:
            self._path_poses = self._path_poses[-max_poses:]

        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.poses = self._path_poses
        self._pub_path.publish(msg)

    def _publish_map_odom_tf(self, stamp, short_timeout: bool = False) -> None:
        if self._pose_estimate is None:
            return
        gf = self._p("map_frame_id")
        odom_frame = self._p("odom_frame_id")
        base_frame = self._p("base_frame_id")
        q = quaternion_from_euler(0, 0, self._pose_estimate[2])

        timeout_s = 0.01 if short_timeout else 0.1
        timeout_dur = rclpy.duration.Duration(seconds=timeout_s)
        lookup_time = rclpy.time.Time.from_msg(stamp)
        used_stamp = True
        try:
            odom_to_base = self._tf_buffer.lookup_transform(
                base_frame, odom_frame, lookup_time,
                timeout=timeout_dur)
        except Exception:
            used_stamp = False
            try:
                odom_to_base = self._tf_buffer.lookup_transform(
                    base_frame, odom_frame, rclpy.time.Time(),
                    timeout=timeout_dur)
            except Exception as e:
                now = time.monotonic()
                if now - self._tf_warn_time >= 5.0:
                    self.log.warning(
                        f"Cannot publish map->odom: lookup "
                        f"{odom_frame}->{base_frame} failed: {e}")
                    self._tf_warn_time = now
                return

        T_base_odom = concatenate_matrices(
            translation_matrix([
                odom_to_base.transform.translation.x,
                odom_to_base.transform.translation.y,
                odom_to_base.transform.translation.z,
            ]),
            quaternion_matrix([
                odom_to_base.transform.rotation.x,
                odom_to_base.transform.rotation.y,
                odom_to_base.transform.rotation.z,
                odom_to_base.transform.rotation.w,
            ]),
        )
        T_map_base = concatenate_matrices(
            translation_matrix([self._pose_estimate[0],
                                self._pose_estimate[1], 0.0]),
            quaternion_matrix(list(q)),
        )
        T_map_odom = T_map_base @ T_base_odom
        trans = translation_from_matrix(T_map_odom)
        quat = quaternion_from_matrix(T_map_odom)

        t = TransformStamped()
        base_stamp = stamp if used_stamp else odom_to_base.header.stamp
        tf_tolerance = self._p("transform_tolerance")
        tf_stamp = rclpy.time.Time.from_msg(base_stamp) + \
            rclpy.duration.Duration(seconds=tf_tolerance)
        t.header.stamp = tf_stamp.to_msg()
        t.header.frame_id = gf
        t.child_frame_id = odom_frame
        t.transform.translation.x = float(trans[0])
        t.transform.translation.y = float(trans[1])
        t.transform.translation.z = float(trans[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self._tf_broadcaster.sendTransform(t)

    def _publish_map_odom_tf_idle(self) -> None:
        if not self._initialized or self._pose_estimate is None:
            return
        self._publish_map_odom_tf(self.get_clock().now().to_msg(),
                                  short_timeout=True)

    def _publish_particle_cloud(self) -> None:
        if self._particles is None:
            return
        gf = self._p("map_frame_id")
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = gf

        N = self._particles.shape[0]
        for i in range(N):
            p = PoseStamped().pose
            p.position.x = float(self._particles[i, 0])
            p.position.y = float(self._particles[i, 1])
            p.position.z = 0.0
            q = quaternion_from_euler(0, 0, float(self._particles[i, 2]))
            p.orientation.x = q[0]
            p.orientation.y = q[1]
            p.orientation.z = q[2]
            p.orientation.w = q[3]
            pa.poses.append(p)
        self._pub_particles.publish(pa)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args=None) -> None:
    from localization import parse_config_args
    ros_args = parse_config_args(args, method="amcl")
    debug = "--debug" in ros_args
    if debug:
        ros_args.remove("--debug")
    rclpy.init(args=ros_args)
    node = AMCLNode()
    if debug:
        set_logger_level(node.get_name(), LoggingSeverity.DEBUG)
        node.log.debug("Debug logging enabled.")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.log.info("Shutting down (keyboard interrupt).")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(args=sys.argv[1:])
