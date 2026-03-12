#!/usr/bin/env python3
"""
Monte Carlo Localization (Particle Filter) ROS2 node.

Vanilla MCL: motion model from odometry, likelihood-field observation model,
systematic resampling. Publishes pose estimate, map->odom TF, and particle cloud.
Compatible with Nav2 (Humble); frame and topic names match AMCL-style defaults.

Run (from workspace or with PYTHONPATH including this directory):
  python3 mcl_node.py
  python3 localization.py
With parameters:
  python3 mcl_node.py --ros-args --params-file config/mcl_params.yaml

Requires: map (nav_msgs/OccupancyGrid), scan (sensor_msgs/LaserScan), odom
(nav_msgs/Odometry). Python deps: numpy, scipy, rclpy, tf_transformations.
Send initial pose via topic initialpose (geometry_msgs/PoseWithCovarianceStamped);
the node does not auto-initialize. The robot must publish odom -> base_footprint
TF (e.g. from diff_drive or wheel odometry).

Stability: update threshold odom-vs-odom; out-of-map particles penalized; N_eff-based resampling.
Convergence (good initial pose): resample_interval=0 (resample only when N_eff low); tighter alphas (0.1);
softer sigma_hit (0.25); more beams (90); smaller update_min_*; min_updates_before_resample.

For localization quality close to Nav2 AMCL: ensure scan frame matches base or set
laser_origin_x/y; use periodic resampling (resample_interval > 0) so the cloud can
concentrate; tune sigma_hit and map_uncertainty_scale for your map.
Suggested later: recovery (particle injection when N_eff low), KLD-sampling, beam model (z_short, z_rand).
"""

from __future__ import annotations

import math
import random
import sys
import time
from typing import Optional

import numpy as np
import rclpy
from scipy.ndimage import distance_transform_edt
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.logging import LoggingSeverity, set_logger_level
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from tf_transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_matrix,
    translation_matrix,
    concatenate_matrices,
    translation_from_matrix,
    quaternion_from_matrix,
)


# -----------------------------------------------------------------------------
# Map and likelihood field helpers
# -----------------------------------------------------------------------------


def _find_nearest_obstacle_cell(
    grid: np.ndarray,
    mx: int,
    my: int,
    max_search: int,
) -> tuple[float, Optional[int], Optional[int]]:
    """
    Expanding-square search: distance (cells) and (nx, ny) of nearest occupied cell.

    ROS convention: occupied = (grid > 0). Unknown (-1) and 0 are free.

    Parameters
    ----------
    grid : np.ndarray
        2D occupancy grid (0 free, >0 occupied, -1 unknown).
    mx, my : int
        Column and row indices of the query cell.
    max_search : int
        Maximum search radius in cells.

    Returns
    -------
    tuple of (float, int | None, int | None)
        (distance_cells, nx, ny); (inf, None, None) if none found or out of bounds.
    """
    h, w = grid.shape
    if mx < 0 or mx >= w or my < 0 or my >= h:
        return float("inf"), None, None
    for r in range(0, max_search + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx == 0 and dy == 0 and r > 0:
                    continue
                nx, ny = mx + dx, my + dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] > 0:
                    d_cells = math.sqrt(float(dx * dx + dy * dy))
                    return d_cells, nx, ny
    return float("inf"), None, None


def compute_distance_map(
    grid: np.ndarray,
    resolution: float,
    max_dist: float,
    method: str = "edt",
) -> np.ndarray:
    """
    Precompute distance (meters) from each cell to the nearest occupied cell.

    Uses either SciPy EDT (O(n), fast) or expanding-square search per cell.
    ROS convention: occupied = (grid > 0); free = (grid <= 0), so unknown (-1)
    is treated as free for the distance transform.

    Parameters
    ----------
    grid : np.ndarray
        2D occupancy grid (0 free, 100 occupied, -1 unknown).
    resolution : float
        Map resolution in m/cell.
    max_dist : float
        Maximum distance in meters; values are capped and used as search limit for expand.
    method : {'edt', 'expand'}, optional
        'edt': SciPy Euclidean distance transform (default, fast).
        'expand': Expanding-square search per cell (slow on large maps).

    Returns
    -------
    np.ndarray
        2D float32 array, same shape as grid: distance in meters to nearest
        occupied cell, capped at max_dist. Occupied cells get 0.
    """
    h, w = grid.shape
    max_dist = float(max_dist)
    if method == "edt":
        # EDT: 1 = free (distance computed), 0 = occupied (source).
        # ROS: free = (grid <= 0), occupied = (grid > 0)
        free = (grid <= 0).astype(np.float64)
        dist_cells = distance_transform_edt(free)
        dist_m = (dist_cells * resolution).astype(np.float32)
        return np.minimum(dist_m, max_dist)

    elif method == "expand":
        max_search = min(int(max_dist / resolution) + 1, max(h, w) + 1)
        dist_m = np.full((h, w), max_dist, dtype=np.float32)
        for my in range(h):
            for mx in range(w):
                d_cells, _, _ = _find_nearest_obstacle_cell(grid, mx, my, max_search)
                d_m = min(d_cells * resolution, max_dist)
                dist_m[my, mx] = d_m
    else:
        raise ValueError(f"Invalid distance map method: {method}")
    return dist_m


# -----------------------------------------------------------------------------
# MCL Node
# -----------------------------------------------------------------------------


class MCLNode(Node):
    """
    Monte Carlo Localization node using odometry motion model and
    likelihood-field observation model.
    """

    def __init__(self) -> None:
        super().__init__("mcl_node")
        self.log = self.get_logger()
        self._init_parameters()

        # State
        self._map: Optional[np.ndarray] = None
        self._map_metadata: Optional[dict] = (
            None  # origin_x, origin_y, resolution, width, height
        )
        self._dist_map: Optional[np.ndarray] = (
            None  # precomputed distance to obstacle (m)
        )

        self._particles: Optional[np.ndarray] = None  # (N, 3) x, y, theta
        self._weights: Optional[np.ndarray] = None  # (N,)
        self._last_odom_pose: Optional[tuple[float, float, float]] = None
        self._last_update_odom_pose: Optional[tuple[float, float, float]] = (
            None  # odom at last measurement update (for update_min_*)
        )
        self._last_odom_time: Optional[rclpy.time.Time] = None
        self._update_counter: int = 0
        self._pose_estimate: Optional[tuple[float, float, float]] = None
        self._path_poses: list[PoseStamped] = []
        self._tf_lookup_warn_time: float = 0.0
        self._scan_skip_log_time: float = 0.0
        self._scan_frame_warned: bool = False
        self._laser_origin_tf: Optional[tuple[float, float]] = (
            None  # (lox, loy) from TF when scan frame != base
        )
        self._laser_origin_tf_frame: Optional[str] = None

        # QoS: map is latched; scan/odom best-effort
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

        # Interdependent callbacks (scan, odom, initial_pose, particle_cloud) share state;
        # club them in one mutually exclusive group. Map stays in default (called once).
        self._state_cb_group = MutuallyExclusiveCallbackGroup()

        self._map_sub = self.create_subscription(
            OccupancyGrid,
            self.get_parameter("map_topic").value,
            self._cb_map,
            qos_map,
        )
        self._scan_sub = self.create_subscription(
            LaserScan,
            self.get_parameter("scan_topic").value,
            self._cb_scan,
            qos_sensor,
            callback_group=self._state_cb_group,
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self._cb_odom,
            10,
            callback_group=self._state_cb_group,
        )
        self._initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("initial_pose_topic").value,
            self._cb_initial_pose,
            10,
            callback_group=self._state_cb_group,
        )

        self._pose_pub = self.create_publisher(
            PoseStamped,
            f"~/{self._get_param('publish_pose_topic')}",
            10,
        )
        self._pose_cov_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            f"~/{self._get_param('publish_pose_with_covariance_topic')}",
            10,
        )
        self._particle_cloud_pub = self.create_publisher(
            PoseArray,
            f"~/{self._get_param('publish_particle_cloud_topic')}",
            10,
        )
        self._path_pub = self.create_publisher(
            Path,
            f"~/{self._get_param('publish_path_topic')}",
            10,
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Throttled particle cloud (reads shared state; same group as scan/odom/initial_pose)
        self._particle_cloud_timer = self.create_timer(
            1.0 / self.get_parameter("particle_cloud_publish_rate").value,
            self._publish_particle_cloud,
            callback_group=self._state_cb_group,
        )
        # Republish map->odom periodically so transform stays valid during inactivity
        tf_rate = float(self.get_parameter("tf_publish_rate").value)
        if tf_rate > 0:
            self._tf_timer = self.create_timer(
                1.0 / tf_rate,
                self._publish_map_odom_tf_idle,
                callback_group=self._state_cb_group,
            )
        else:
            self._tf_timer = None

        self.log.info(
            "MCL node started; waiting for map and initial pose. "
            "map->odom is published on scan updates and at tf_publish_rate when idle."
        )
        self._initialized = False

    def _get_param(self, name: str):
        return self.get_parameter(name).value

    def _init_parameters(self) -> None:
        # Declare parameters (Nav2/AMCL-aligned defaults)
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_footprint")
        self.declare_parameter("scan_topic", "scan")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("map_topic", "map")
        self.declare_parameter("initial_pose_topic", "initialpose")

        self.declare_parameter("num_particles", 1000)
        # Resampling: 0 = only when N_eff low; >0 = also every N updates (helps cloud concentrate)
        self.declare_parameter("resample_interval", 5)
        self.declare_parameter(
            "resample_alpha", 0.5
        )  # resample when N_eff < resample_alpha * N
        self.declare_parameter("min_updates_before_resample", 2)
        self.declare_parameter(
            "resample_first_n_updates", 5
        )  # resample every update for first N (fast convergence)
        self.declare_parameter(
            "recovery_n_eff_ratio", 0.0
        )  # inject when N_eff < this * N; 0 = disabled (often dilutes cloud)
        self.declare_parameter(
            "recovery_fraction", 0.1
        )  # fraction of particles to replace when recovering
        self.declare_parameter(
            "recovery_min_updates", 25
        )  # only run recovery after this many updates (avoid diluting initial convergence)
        self.declare_parameter(
            "recovery_around_estimate_fraction", 0.5
        )  # fraction of injected particles spawned near current pose (rest uniform in map)
        self.declare_parameter(
            "recovery_around_std_xy", 0.5
        )  # std (m) for spawning around pose estimate
        self.declare_parameter("update_min_d", 0.02)
        self.declare_parameter("update_min_a", 0.05)
        # For first N updates, run measurement on every scan (ignore motion threshold); good for ~10 Hz scan to converge fast
        self.declare_parameter("update_ignore_motion_first_n", 60)
        # Initial cloud spread (used when initial_pose_params_override is true, or when message has no valid covariance)
        self.declare_parameter("initial_pose_std_xy", 0.02)
        self.declare_parameter("initial_pose_std_theta", 0.01)
        self.declare_parameter(
            "initial_pose_params_override", True
        )  # true = always use params; false = use message covariance when present

        # Motion model noise (tighter = less spread, measurement can focus distribution faster)
        self.declare_parameter("motion_noise_scale", 1.0)
        self.declare_parameter("alpha1", 0.1)
        self.declare_parameter("alpha2", 0.1)
        self.declare_parameter("alpha3", 0.1)
        self.declare_parameter("alpha4", 0.1)

        # Sensor: laser origin in base frame (m); set if scan is in base_laser or laser is offset from base
        self.declare_parameter("laser_origin_x", 0.0)
        self.declare_parameter("laser_origin_y", 0.0)
        # Likelihood field: use all beams when scan has <= max_beams; else subsample (loses resolution).
        self.declare_parameter("max_beams", 360)
        self.declare_parameter(
            "sigma_hit", 0.15
        )  # smaller = peakier likelihood, faster convergence (0.25 = flatter)
        self.declare_parameter("likelihood_max_dist", 2.0)
        # Map / environment discrepancy: soften likelihood so map errors or new obstacles don't kill good particles
        self.declare_parameter(
            "map_uncertainty_scale", 0.1
        )  # sigma_eff = sigma_hit * (1 + this); 0 = trust map fully
        self.declare_parameter(
            "z_max_log_likelihood", -4.6
        )  # log prob for max-range beams (~0.01); avoids penalizing "obstacle not in map"
        self.declare_parameter(
            "max_range_ratio_for_z_max", 0.98
        )  # beam treated as max-range when r >= max_range * this
        # Distance map: 'edt' (SciPy, fast) or 'expand' (expanding-square, fallback).
        self.declare_parameter("distance_map_method", "edt")

        # Pose / particle cloud publishing
        self.declare_parameter("publish_pose_topic", "pose")
        self.declare_parameter("publish_pose_with_covariance_topic", "amcl_pose")
        self.declare_parameter("publish_particle_cloud_topic", "particle_cloud")
        self.declare_parameter("particle_cloud_publish_rate", 5.0)
        self.declare_parameter("publish_path_topic", "path")
        self.declare_parameter("path_max_poses", 100)
        # Path decay: only show poses from the last path_decay_time seconds (0 = disabled).
        self.declare_parameter("path_decay_time", 0.0)
        # Republish map->odom at this rate when idle so the transform does not expire (default 1 Hz).
        self.declare_parameter("tf_publish_rate", 1.0)

    def _cb_map(self, msg: OccupancyGrid) -> None:
        if self._map is not None:
            return
        self._map_metadata = {
            "origin_x": msg.info.origin.position.x,
            "origin_y": msg.info.origin.position.y,
            "resolution": msg.info.resolution,
            "width": msg.info.width,
            "height": msg.info.height,
        }
        self._map = np.array(msg.data, dtype=np.int32).reshape(
            (msg.info.height, msg.info.width)
        )
        method = self._get_param("distance_map_method")
        self._dist_map = compute_distance_map(
            self._map,
            self._map_metadata["resolution"],
            self._get_param("likelihood_max_dist"),
            method=method,
        )
        self.log.info(
            f"Map received: wxh = {self._map_metadata['width']}x{self._map_metadata['height']}, "
            f"resolution = {self._map_metadata['resolution']:.3f} m/cell, distance_map = {method}"
        )

    def _initialize_particles_at(
        self,
        x: float,
        y: float,
        theta: float,
        std_xy: Optional[float] = None,
        std_theta: Optional[float] = None,
    ) -> None:
        n = self._get_param("num_particles")
        if std_xy is None:
            std_xy = float(self._get_param("initial_pose_std_xy"))
        if std_theta is None:
            std_theta = float(self._get_param("initial_pose_std_theta"))
        std_xy = max(1e-6, std_xy)
        std_theta = max(1e-6, std_theta)
        self._particles = np.zeros((n, 3))
        self._particles[:, 0] = x + np.random.normal(0, std_xy, n)
        self._particles[:, 1] = y + np.random.normal(0, std_xy, n)
        self._particles[:, 2] = theta + np.random.normal(0, std_theta, n)
        self._normalize_theta_column(2)
        self._weights = np.ones(n) / n
        self._pose_estimate = (x, y, theta)
        self._initialized = True
        if float(self._get_param("motion_noise_scale")) <= 0:
            self.log.warn(
                "motion_noise_scale is 0: all particles get identical motion, so the likelihood "
                "field cannot discriminate poses and localization will not converge. Set to 1 (or tune alphas) for map-based correction."
            )
        self.log.info(
            f"{n} particles initialized at (x:{x:.2f}, y:{y:.2f}, theta:{theta:.2f} rad) "
            f"with std_xy={std_xy:.3f} m, std_theta={std_theta:.3f} rad"
        )

    def _normalize_theta_column(self, col: int) -> None:
        if self._particles is None:
            return
        self._particles[:, col] = np.arctan2(
            np.sin(self._particles[:, col]),
            np.cos(self._particles[:, col]),
        )

    def _cb_initial_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """Initialize or re-initialize particles at the given pose (e.g. from RViz 2D Pose Estimate).
        When initial_pose_params_override is true (default), always use initial_pose_std_xy and
        initial_pose_std_theta from params. When false, use message covariance when valid, else params.
        """
        if self._map is None:
            self.log.warn("Initial pose received but map not yet available.")
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        std_xy, std_theta = None, None
        if not self._get_param("initial_pose_params_override"):
            cov = getattr(msg.pose, "covariance", None)
            if cov is not None and len(cov) >= 36:
                var_x = float(cov[0])
                var_y = float(cov[7])
                var_yaw = float(cov[35])
                if var_x > 0 and np.isfinite(var_x):
                    std_xy = np.sqrt(var_x)
                if var_y > 0 and np.isfinite(var_y):
                    std_xy = (
                        np.sqrt(var_y)
                        if std_xy is None
                        else (std_xy + np.sqrt(var_y)) / 2.0
                    )
                if var_yaw > 0 and np.isfinite(var_yaw):
                    std_theta = np.sqrt(var_yaw)
        self._initialize_particles_at(p.x, p.y, yaw, std_xy=std_xy, std_theta=std_theta)
        self._scan_frame_warned = False
        self._laser_origin_tf = None
        self._laser_origin_tf_frame = None

        self._last_odom_pose = None
        self._last_update_odom_pose = None
        self._last_odom_time = None

    def _cb_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
        now = rclpy.time.Time.from_msg(msg.header.stamp)

        if self._last_odom_pose is not None and self._particles is not None:
            dx = p.x - self._last_odom_pose[0]
            dy = p.y - self._last_odom_pose[1]
            dth = theta - self._last_odom_pose[2]
            dth = math.atan2(math.sin(dth), math.cos(dth))
            self._motion_update(dx, dy, dth)

        self._last_odom_pose = (p.x, p.y, theta)
        self._last_odom_time = now

    def _motion_update(self, dx: float, dy: float, dth: float) -> None:
        """Apply odometry delta with Gaussian noise to all particles."""
        if self._particles is None:
            return
        noise_scale = float(self._get_param("motion_noise_scale"))
        a1, a2, a3, a4 = (
            self._get_param("alpha1"),
            self._get_param("alpha2"),
            self._get_param("alpha3"),
            self._get_param("alpha4"),
        )
        trans = math.sqrt(dx * dx + dy * dy)
        if trans > 1e-6:
            rot1 = math.atan2(dy, dx) - self._last_odom_pose[2]
            rot1 = math.atan2(math.sin(rot1), math.cos(rot1))
        else:
            rot1 = 0.0
        rot2 = dth - rot1
        rot2 = math.atan2(math.sin(rot2), math.cos(rot2))

        rot1_std = noise_scale * (a1 * abs(rot1) + a2 * trans)
        trans_std = noise_scale * (a3 * trans + a4 * (abs(rot1) + abs(rot2)))
        rot2_std = noise_scale * (a1 * abs(rot2) + a2 * trans)

        n = self._particles.shape[0]
        if noise_scale <= 0:
            rot1_noisy = np.full(n, rot1)
            trans_noisy = np.full(n, trans)
            rot2_noisy = np.full(n, rot2)
        else:
            rot1_noisy = rot1 + np.random.normal(0, max(1e-6, rot1_std), n)
            trans_noisy = trans + np.random.normal(0, max(1e-6, trans_std), n)
            rot2_noisy = rot2 + np.random.normal(0, max(1e-6, rot2_std), n)

        # Apply in particle frame: rotate by (particle_theta + rot1), move trans_noisy, rotate rot2
        th = self._particles[:, 2]
        self._particles[:, 0] += trans_noisy * np.cos(th + rot1_noisy)
        self._particles[:, 1] += trans_noisy * np.sin(th + rot1_noisy)
        self._particles[:, 2] += rot1_noisy + rot2_noisy
        self._normalize_theta_column(2)

        # Do not clamp (x,y) to map here. Standard MCL lets particles go anywhere; the
        # observation model gives zero weight to out-of-map particles, and resampling
        # removes them. Clamping would distort the posterior at boundaries. If particles
        # spread too much on rotation, reduce motion noise (e.g. alpha4).

    def _cb_scan(self, msg: LaserScan) -> None:
        if self._map is None:
            return
        if not self._initialized:
            return
        if self._particles is None:
            return

        # Check update threshold: min translation/rotation in *odom* frame since last measurement update
        if self._last_odom_pose is None:
            return
        # When we've never run an update, ref would equal current odom (dx=dy=da=0) and we'd never pass.
        # Always run at least one update after init. For first update_ignore_motion_first_n updates, run on every scan
        # (e.g. at 10 Hz scan that is ~6 s) so the filter can converge; then apply motion threshold.
        if self._last_update_odom_pose is not None:
            ignore_motion_first_n = self._get_param("update_ignore_motion_first_n")
            if self._update_counter >= ignore_motion_first_n:
                min_d = self._get_param("update_min_d")
                min_a = self._get_param("update_min_a")
                dx = self._last_odom_pose[0] - self._last_update_odom_pose[0]
                dy = self._last_odom_pose[1] - self._last_update_odom_pose[1]
                if math.sqrt(dx * dx + dy * dy) < min_d:
                    da = abs(
                        math.atan2(
                            math.sin(
                                self._last_odom_pose[2] - self._last_update_odom_pose[2]
                            ),
                            math.cos(
                                self._last_odom_pose[2] - self._last_update_odom_pose[2]
                            ),
                        )
                    )
                    if da < min_a:
                        now = time.monotonic()
                        if now - self._scan_skip_log_time >= 10.0:
                            self.log.debug(
                                f"Scan skipped: motion below threshold (d={math.sqrt(dx * dx + dy * dy):.3f}<{min_d} m, "
                                f"da={da:.3f}<{min_a} rad). Move robot to trigger updates."
                            )
                            self._scan_skip_log_time = now
                        return

        t0 = time.perf_counter()
        self._measurement_update(msg)
        t_meas = (time.perf_counter() - t0) * 1000.0

        self._update_counter += 1
        resample_interval = self._get_param("resample_interval")
        resample_alpha = float(self._get_param("resample_alpha"))
        min_before_resample = self._get_param("min_updates_before_resample")
        resample_first_n = self._get_param("resample_first_n_updates")
        recovery_n_eff_ratio = float(self._get_param("recovery_n_eff_ratio"))
        recovery_fraction = float(self._get_param("recovery_fraction"))
        recovery_min_updates = self._get_param("recovery_min_updates")
        n_eff = (
            self._effective_sample_size() if self._weights is not None else float("inf")
        )
        N = self._particles.shape[0] if self._particles is not None else 0
        do_resample = False
        if self._update_counter <= resample_first_n:
            do_resample = True
        elif self._update_counter >= min_before_resample:
            if resample_interval > 0 and self._update_counter % resample_interval == 0:
                do_resample = True
            if resample_alpha > 0 and N > 0 and n_eff < resample_alpha * N:
                do_resample = True
        t_resample = 0.0
        if do_resample:
            t0r = time.perf_counter()
            self._resample_systematic()
            # Only recover after enough updates so we don't dilute the cloud during initial convergence
            if (
                self._update_counter >= recovery_min_updates
                and recovery_n_eff_ratio > 0
                and recovery_fraction > 0
                and n_eff < recovery_n_eff_ratio * N
            ):
                self._recovery_inject(recovery_fraction)
            t_resample = (time.perf_counter() - t0r) * 1000.0

        t0u = time.perf_counter()
        self._update_pose_estimate()
        self._last_update_odom_pose = (
            self._last_odom_pose
        )  # for next update_min_* check (odom vs odom)
        # Publish pose, covariance, path and map->odom TF all with scan timestamp.
        self._publish_pose_and_tf(msg.header.stamp)
        t_publish = (time.perf_counter() - t0u) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        n_eff = self._effective_sample_size() if self._weights is not None else 0.0
        N = self._particles.shape[0] if self._particles is not None else 0
        self.log.debug(
            f"MCL update: meas={t_meas:.1f} resample={t_resample:.1f} publish={t_publish:.1f} total={total_ms:.1f} ms | "
            f"N={N} N_eff={n_eff:.0f} beams={self._get_param('max_beams')}"
        )

    def _measurement_update(self, msg: LaserScan) -> None:
        """Likelihood-field model: weight particles by scan likelihood (vectorized).
        Beam angles in sensor frame + particle theta for map frame. Assumes scan in base frame.
        Map discrepancy: sigma inflated by map_uncertainty_scale; max-range beams use z_max_log_likelihood.
        """
        if self._particles is None or self._weights is None or self._dist_map is None:
            return
        # max_beams: max beams to use (compute cap). We use all beams when scan has <= max_beams.
        # Subsampling when scan has more than max_beams reduces angular resolution and can hurt accuracy;
        # set max_beams >= your laser's beam count when accuracy is priority, or accept subsampling for CPU.
        max_beams_param = self._get_param("max_beams")
        sigma_hit = float(self._get_param("sigma_hit"))
        msg_range_min = getattr(msg, "range_min", 0.0)
        msg_range_max = getattr(msg, "range_max", float("inf"))
        if np.isfinite(msg_range_max) and msg_range_max > msg_range_min:
            min_range = float(msg_range_min)
            max_range = float(msg_range_max)
        else:
            min_range = 0.0
            max_range = 100.0

        num_readings = len(msg.ranges)
        if num_readings == 0:
            return
        if num_readings <= max_beams_param:
            indices = np.arange(num_readings)
        else:
            step = num_readings / max_beams_param
            indices = (np.arange(max_beams_param) * step).astype(np.int32)
            indices = np.clip(indices, 0, num_readings - 1)
        r = np.array(msg.ranges, dtype=np.float64)[indices]
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        angles = angle_min + indices * angle_increment
        valid = np.isfinite(r) & (r <= max_range)
        if min_range >= 0:
            valid &= r >= min_range
        if np.sum(valid) == 0:
            self.log.debug("Measurement update skipped: no valid beams in range.")
            return

        base_frame = self._get_param("base_frame_id")
        scan_frame = getattr(msg.header, "frame_id", "") or ""
        lox = float(self._get_param("laser_origin_x"))
        loy = float(self._get_param("laser_origin_y"))
        if scan_frame and scan_frame != base_frame:
            if self._laser_origin_tf_frame != scan_frame:
                # First few TF lookups often fail while the buffer fills; retry before giving up.
                t = None
                for _ in range(4):
                    try:
                        t = self._tf_buffer.lookup_transform(
                            base_frame,
                            scan_frame,
                            rclpy.time.Time(),
                            timeout=rclpy.duration.Duration(seconds=0.3),
                        )
                        break
                    except Exception:
                        time.sleep(0.15)
                        rclpy.spin_once(self, timeout_sec=0.05)
                if t is not None:
                    lox = t.transform.translation.x
                    loy = t.transform.translation.y
                    self._laser_origin_tf = (lox, loy)
                    self._laser_origin_tf_frame = scan_frame
                    if not self._scan_frame_warned:
                        self.log.info(
                            f"Using laser origin from TF {base_frame} -> {scan_frame}: "
                            f"x={lox:.3f} m, y={loy:.3f} m"
                        )
                        self._scan_frame_warned = True
                else:
                    self._laser_origin_tf = None
                    # Do not set _laser_origin_tf_frame so we retry on next scan (early lookups often fail)
                    if not self._scan_frame_warned:
                        self.log.warn(
                            f"Scan frame_id is '{scan_frame}' but base_frame_id is '{base_frame}'. "
                            "TF lookup failed after retries. Using laser_origin_x/y (currently 0, 0). "
                            "Wrong laser origin often causes particles not to converge: set laser_origin_x and laser_origin_y in your config to the laser position in base frame."
                        )
                        self._scan_frame_warned = True
            elif self._laser_origin_tf is not None:
                lox, loy = self._laser_origin_tf

        N = self._particles.shape[0]
        ox = self._map_metadata["origin_x"]
        oy = self._map_metadata["origin_y"]
        res = self._map_metadata["resolution"]
        h, w = self._dist_map.shape

        # Beam endpoints in map: base pose + laser origin in base (from TF or params) + range in beam direction
        n_beams = len(indices)
        px = self._particles[:, 0:1]
        py = self._particles[:, 1:2]
        pth = self._particles[:, 2:3]
        r_b = r.reshape(1, n_beams)
        angles_b = angles.reshape(1, n_beams)
        cos_pth = np.cos(pth)
        sin_pth = np.sin(pth)
        laser_x_map = px + lox * cos_pth - loy * sin_pth
        laser_y_map = py + lox * sin_pth + loy * cos_pth
        wx = laser_x_map + r_b * np.cos(angles_b + pth)
        wy = laser_y_map + r_b * np.sin(angles_b + pth)
        mx = np.int32((wx - ox) / res)
        my = np.int32((wy - oy) / res)
        mx = np.clip(mx, 0, w - 1)
        my = np.clip(my, 0, h - 1)

        d = self._dist_map[my, mx]
        # Inflate sigma when map may have errors or environment has changed (manual/auto map, new obstacles)
        map_uncertainty = float(self._get_param("map_uncertainty_scale"))
        sigma_eff = sigma_hit * (1.0 + map_uncertainty)
        inv_sigma_sq = 1.0 / (sigma_eff * sigma_eff)
        likelihood_max_dist = float(self._get_param("likelihood_max_dist"))
        log_floor = -0.5 * inv_sigma_sq * (likelihood_max_dist**2)
        hit_log = np.maximum(-0.5 * inv_sigma_sq * (d * d), log_floor)
        # Max-range beams: use fixed log-likelihood so "no hit" / "obstacle not in map" don't kill the particle
        z_max_log = float(self._get_param("z_max_log_likelihood"))
        max_range_ratio = float(self._get_param("max_range_ratio_for_z_max"))
        is_max_range_beam = (r_b >= max_range * max_range_ratio) & valid.reshape(
            1, n_beams
        )
        log_like_per_beam = np.where(
            is_max_range_beam,
            z_max_log,
            np.where(valid.reshape(1, n_beams), hit_log, 0.0),
        )
        log_weights = np.sum(log_like_per_beam, axis=1)

        # Penalize particles outside map bounds (beam endpoints were clipped; pose may be invalid)
        ox, oy = self._map_metadata["origin_x"], self._map_metadata["origin_y"]
        x_lo, x_hi = ox, ox + w * res
        y_lo, y_hi = oy, oy + h * res
        outside = (
            (px.ravel() < x_lo)
            | (px.ravel() >= x_hi)
            | (py.ravel() < y_lo)
            | (py.ravel() >= y_hi)
        )
        log_weights[outside] = -np.inf

        # Normalize to weights (numerically stable). If all log_weights are -inf, max is -inf and subtract gives NaN.
        max_log = np.max(log_weights)
        if np.isfinite(max_log):
            log_weights -= max_log
            self._weights = np.exp(log_weights)
            s = np.sum(self._weights)
            if s > 0:
                self._weights /= s
            else:
                self.log.debug(
                    "Weights sum to zero (e.g. all particles outside map); using uniform weights."
                )
                self._weights = np.ones(N) / N
        else:
            self.log.debug(
                "All particles have -inf log weight (e.g. all outside map); using uniform weights."
            )
            self._weights = np.ones(N) / N

    def _effective_sample_size(self) -> float:
        """Effective sample size N_eff = 1 / sum(w_i^2). Low N_eff indicates particle depletion."""
        if self._weights is None or self._weights.size == 0:
            return 0.0
        w = self._weights.ravel()
        s = np.sum(w * w)
        return float(1.0 / s) if s > 0 else 0.0

    def _resample_systematic(self) -> None:
        """Systematic resampling."""
        if self._particles is None or self._weights is None:
            return
        N = self._particles.shape[0]
        cdf = np.cumsum(self._weights)
        u0 = random.random()
        indices = np.zeros(N, dtype=np.int64)
        for i in range(N):
            u = (u0 + i) / N
            indices[i] = np.searchsorted(cdf, u, side="left")
        indices = np.clip(indices, 0, N - 1)
        self._particles = self._particles[indices].copy()
        self._weights = np.ones(N) / N

    def _recovery_inject(self, fraction: float) -> None:
        """Replace a fraction of particles: some near current pose estimate, rest uniform in map. Helps when N_eff is very low."""
        if self._particles is None or self._map_metadata is None or fraction <= 0:
            return
        N = self._particles.shape[0]
        n_inject = max(1, int(N * fraction))
        ox = self._map_metadata["origin_x"]
        oy = self._map_metadata["origin_y"]
        res = self._map_metadata["resolution"]
        w = self._map_metadata["width"]
        h = self._map_metadata["height"]
        x_lo, x_hi = ox, ox + w * res
        y_lo, y_hi = oy, oy + h * res
        idx = np.random.choice(N, size=n_inject, replace=False)
        around_frac = float(self._get_param("recovery_around_estimate_fraction"))
        around_std = float(self._get_param("recovery_around_std_xy"))
        n_around = (
            int(n_inject * around_frac)
            if self._pose_estimate is not None and around_frac > 0 and around_std > 0
            else 0
        )
        n_uniform = n_inject - n_around
        if n_around > 0 and self._pose_estimate is not None:
            pe = self._pose_estimate
            self._particles[idx[:n_around], 0] = pe[0] + np.random.normal(
                0, around_std, n_around
            )
            self._particles[idx[:n_around], 1] = pe[1] + np.random.normal(
                0, around_std, n_around
            )
            self._particles[idx[:n_around], 2] = pe[2] + np.random.normal(
                0, math.pi / 2, n_around
            )
        if n_uniform > 0:
            u_idx = idx[n_around:] if n_around > 0 else idx
            self._particles[u_idx, 0] = np.random.uniform(x_lo, x_hi, n_uniform)
            self._particles[u_idx, 1] = np.random.uniform(y_lo, y_hi, n_uniform)
            self._particles[u_idx, 2] = np.random.uniform(-math.pi, math.pi, n_uniform)
        self._normalize_theta_column(2)
        self.log.info(
            f"Recovery: injected {n_inject} particles ({n_around} near estimate, {n_uniform} uniform in map)."
        )

    def _update_pose_estimate(self) -> None:
        """Set pose estimate as weighted mean of particles (circular mean for theta)."""
        if self._particles is None or self._weights is None:
            return
        self._pose_estimate = (
            float(np.average(self._particles[:, 0], weights=self._weights)),
            float(np.average(self._particles[:, 1], weights=self._weights)),
            float(
                np.arctan2(
                    np.average(np.sin(self._particles[:, 2]), weights=self._weights),
                    np.average(np.cos(self._particles[:, 2]), weights=self._weights),
                )
            ),
        )

    def _particle_covariance_6x6(self, stamp, frame_id: str) -> Optional[list[float]]:
        """
        Compute 6x6 pose covariance (row-major) from weighted particles.
        Only x, y, yaw are filled; other entries zero. Used for RViz uncertainty ellipse.
        """
        if (
            self._particles is None
            or self._weights is None
            or self._particles.shape[0] < 2
        ):
            return None
        xm, ym, thm = (
            self._pose_estimate[0],
            self._pose_estimate[1],
            self._pose_estimate[2],
        )
        x = self._particles[:, 0]
        y = self._particles[:, 1]
        th = self._particles[:, 2]
        w = self._weights
        # Circular difference for theta
        dth = np.arctan2(np.sin(th - thm), np.cos(th - thm))
        cov_xx = float(np.average((x - xm) ** 2, weights=w))
        cov_yy = float(np.average((y - ym) ** 2, weights=w))
        cov_thth = float(np.average(dth**2, weights=w))
        cov_xy = float(np.average((x - xm) * (y - ym), weights=w))
        cov_xth = float(np.average((x - xm) * dth, weights=w))
        cov_yth = float(np.average((y - ym) * dth, weights=w))
        # 6x6 row-major: [0]=xx, [7]=yy, [35]=yawyaw; [1]=[6]=xy, [5]=[30]=x,yaw, [11]=[31]=y,yaw
        cov = [0.0] * 36
        cov[0] = cov_xx
        cov[1] = cov[6] = cov_xy
        cov[5] = cov[30] = cov_xth
        cov[7] = cov_yy
        cov[11] = cov[31] = cov_yth
        cov[35] = cov_thth
        return cov

    def _publish_pose_and_tf(self, stamp) -> None:
        if self._pose_estimate is None:
            return
        global_frame = self._get_param("map_frame_id")

        # Pose in map frame (map -> base_footprint)
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = global_frame
        ps.pose.position.x = self._pose_estimate[0]
        ps.pose.position.y = self._pose_estimate[1]
        ps.pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, self._pose_estimate[2])
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        self._pose_pub.publish(ps)

        # Pose with covariance (RViz: PoseWithCovariance display shows uncertainty ellipse)
        cov_flat = self._particle_covariance_6x6(stamp, global_frame)
        if cov_flat is not None:
            pwc = PoseWithCovarianceStamped()
            pwc.header.stamp = stamp
            pwc.header.frame_id = global_frame
            pwc.pose.pose = ps.pose
            pwc.pose.covariance = cov_flat
            self._pose_cov_pub.publish(pwc)

        # Path (trajectory of pose estimates for RViz Path display)
        path_max = self._get_param("path_max_poses")
        path_decay_time = self._get_param("path_decay_time")
        ps_copy = PoseStamped()
        ps_copy.header.stamp = stamp
        ps_copy.header.frame_id = global_frame
        ps_copy.pose.position.x = self._pose_estimate[0]
        ps_copy.pose.position.y = self._pose_estimate[1]
        ps_copy.pose.position.z = 0.0
        ps_copy.pose.orientation.x = q[0]
        ps_copy.pose.orientation.y = q[1]
        ps_copy.pose.orientation.z = q[2]
        ps_copy.pose.orientation.w = q[3]
        self._path_poses.append(ps_copy)
        if path_decay_time > 0:
            # Time-based decay: keep only poses from the last path_decay_time seconds
            now_sec = stamp.sec + stamp.nanosec * 1e-9
            cutoff = now_sec - path_decay_time
            self._path_poses = [
                p
                for p in self._path_poses
                if (p.header.stamp.sec + p.header.stamp.nanosec * 1e-9) >= cutoff
            ]
        if len(self._path_poses) > path_max:
            self._path_poses = self._path_poses[-path_max:]
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = global_frame
        path_msg.poses = self._path_poses
        self._path_pub.publish(path_msg)

        self._publish_map_odom_tf(stamp)

    def _publish_map_odom_tf(self, stamp) -> None:
        """Publish map->odom transform using current pose estimate and odom->base lookup."""
        if self._pose_estimate is None:
            return
        global_frame = self._get_param("map_frame_id")
        odom_frame = self._get_param("odom_frame_id")
        base_frame = self._get_param("base_frame_id")
        q = quaternion_from_euler(0, 0, self._pose_estimate[2])

        # Lookup odom->base: try stamp first (for scan sync), then latest
        lookup_time = rclpy.time.Time.from_msg(stamp)
        used_stamp = True
        try:
            odom_to_base = self._tf_buffer.lookup_transform(
                base_frame,
                odom_frame,
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except Exception:
            used_stamp = False
            try:
                odom_to_base = self._tf_buffer.lookup_transform(
                    base_frame,
                    odom_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
            except Exception as e2:
                now = time.monotonic()
                if now - self._tf_lookup_warn_time >= 5.0:
                    self.log.warn(
                        f"Cannot publish map->odom: lookup {odom_frame} -> {base_frame} failed ({e2}). "
                        "Ensure odom -> base_footprint is published (e.g. by diff_drive or wheel odometry)."
                    )
                    self._tf_lookup_warn_time = now
                return
        T_base_odom = concatenate_matrices(
            translation_matrix(
                [
                    odom_to_base.transform.translation.x,
                    odom_to_base.transform.translation.y,
                    odom_to_base.transform.translation.z,
                ]
            ),
            quaternion_matrix(
                [
                    odom_to_base.transform.rotation.x,
                    odom_to_base.transform.rotation.y,
                    odom_to_base.transform.rotation.z,
                    odom_to_base.transform.rotation.w,
                ]
            ),
        )
        T_map_base = concatenate_matrices(
            translation_matrix([self._pose_estimate[0], self._pose_estimate[1], 0.0]),
            quaternion_matrix(list(q)),
        )
        T_map_odom = T_map_base @ T_base_odom
        trans = translation_from_matrix(T_map_odom)
        quat = quaternion_from_matrix(T_map_odom)

        t = TransformStamped()
        t.header.stamp = stamp if used_stamp else odom_to_base.header.stamp
        t.header.frame_id = global_frame
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
        """Republish map->odom with current time so transform does not expire during inactivity."""
        if not self._initialized or self._pose_estimate is None:
            return
        stamp = self.get_clock().now().to_msg()
        self._publish_map_odom_tf(stamp)

    def _publish_particle_cloud(self) -> None:
        if self._particles is None:
            return
        global_frame = self._get_param("map_frame_id")
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = global_frame
        # Clip to map bounds for display only (filter state is not modified)
        if self._map_metadata is not None:
            ox = self._map_metadata["origin_x"]
            oy = self._map_metadata["origin_y"]
            res = self._map_metadata["resolution"]
            w = self._map_metadata["width"]
            h = self._map_metadata["height"]
            x_lo, x_hi = ox, ox + w * res
            y_lo, y_hi = oy, oy + h * res
        else:
            x_lo = y_lo = -np.inf
            x_hi = y_hi = np.inf
        for i in range(self._particles.shape[0]):
            p = PoseStamped().pose
            p.position.x = float(np.clip(self._particles[i, 0], x_lo, x_hi))
            p.position.y = float(np.clip(self._particles[i, 1], y_lo, y_hi))
            p.position.z = 0.0
            q = quaternion_from_euler(0, 0, self._particles[i, 2])
            p.orientation.x = q[0]
            p.orientation.y = q[1]
            p.orientation.z = q[2]
            p.orientation.w = q[3]
            pa.poses.append(p)
        self._particle_cloud_pub.publish(pa)


def main(args=None) -> None:
    from localization import parse_config_args

    ros_args = parse_config_args(args, method="mcl")
    if "--debug" in ros_args:
        ros_args.remove("--debug")
        debug = True
    else:
        debug = False
    rclpy.init(args=ros_args)
    node = MCLNode()
    if debug:
        set_logger_level(node.get_name(), LoggingSeverity.DEBUG)
        node.log.debug("Debug mode enabled")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.log.error("Keyboard interrupt received")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(args=sys.argv[1:])
