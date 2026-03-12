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
Send initial pose via topic initialpose
(geometry_msgs/PoseWithCovarianceStamped) or the node initializes at (0,0,0)
on first scan. The robot must publish odom -> base_footprint TF (e.g. from
diff_drive or wheel odometry).

Suggested improvements (for later):
- Effective sample size (N_eff) and adaptive resampling / recovery (random
  particle injection when N_eff drops below threshold).
- KLD-sampling to adapt particle count.
- Beam skip / likelihood field outlier terms (z_short, z_max, z_rand) for
  robustness to dynamic obstacles and specular reflections.
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
        self._last_odom_time: Optional[rclpy.time.Time] = None
        self._update_counter: int = 0
        self._pose_estimate: Optional[tuple[float, float, float]] = None
        self._path_poses: list[PoseStamped] = []

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

        self.log.info("MCL node started; waiting for map and initial pose or params.")
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
        self.declare_parameter("resample_interval", 1)
        self.declare_parameter("update_min_d", 0.05)
        self.declare_parameter("update_min_a", 0.1)

        # Motion model noise (differential drive style)
        self.declare_parameter("alpha1", 0.2)  # rotation from rotation
        self.declare_parameter("alpha2", 0.2)  # rotation from translation
        self.declare_parameter("alpha3", 0.2)  # translation from translation
        self.declare_parameter("alpha4", 0.2)  # translation from rotation

        # Likelihood field
        self.declare_parameter("max_beams", 60)
        self.declare_parameter("sigma_hit", 0.2)
        self.declare_parameter("laser_max_range", 100.0)
        self.declare_parameter("laser_min_range", -1.0)
        self.declare_parameter("likelihood_max_dist", 2.0)
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

    def _cb_map(self, msg: OccupancyGrid) -> None:
        self.log.debug("_cb_map: fired")
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

    def _initialize_particles_at(self, x: float, y: float, theta: float) -> None:
        n = self._get_param("num_particles")
        # Small noise around (x, y, theta)
        self._particles = np.zeros((n, 3))
        self._particles[:, 0] = x + np.random.normal(0, 0.05, n)
        self._particles[:, 1] = y + np.random.normal(0, 0.05, n)
        self._particles[:, 2] = theta + np.random.normal(0, 0.02, n)
        self._normalize_theta_column(2)
        self._weights = np.ones(n) / n
        self._pose_estimate = (x, y, theta)
        self._initialized = True
        self.log.info(
            f"{n} particles initialized at around (x:{x:.2f}, y:{y:.2f}, theta:{theta:.2f} rad)"
        )

    def _normalize_theta_column(self, col: int) -> None:
        if self._particles is None:
            return
        self._particles[:, col] = np.arctan2(
            np.sin(self._particles[:, col]),
            np.cos(self._particles[:, col]),
        )

    def _cb_initial_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """Initialize or re-initialize particles at the given pose (e.g. from RViz 2D Pose Estimate)."""
        self.log.debug("_cb_initial_pose: fired")
        if self._map is None:
            self.log.warn("Initial pose received but map not yet available.")
            return
        self.log.debug(f"Initial pose received: {msg}")
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._initialize_particles_at(p.x, p.y, yaw)

        self._last_odom_pose = None
        self._last_odom_time = None

    def _cb_odom(self, msg: Odometry) -> None:
        self.log.debug("_cb_odom: fired")
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
        a1, a2, a3, a4 = (
            self._get_param("alpha1"),
            self._get_param("alpha2"),
            self._get_param("alpha3"),
            self._get_param("alpha4"),
        )
        trans = math.sqrt(dx * dx + dy * dy)
        rot1 = math.atan2(dy, dx) if trans > 1e-6 else 0.0
        rot2 = dth - rot1

        rot1_std = a1 * abs(rot1) + a2 * trans
        trans_std = a3 * trans + a4 * (abs(rot1) + abs(rot2))
        rot2_std = a1 * abs(rot2) + a2 * trans

        n = self._particles.shape[0]
        rot1_noisy = rot1 + np.random.normal(0, max(1e-6, rot1_std), n)
        trans_noisy = trans + np.random.normal(0, max(1e-6, trans_std), n)
        rot2_noisy = rot2 + np.random.normal(0, max(1e-6, rot2_std), n)

        # Apply in particle frame: rotate by (particle_theta + rot1), move trans_noisy, rotate rot2
        th = self._particles[:, 2]
        self._particles[:, 0] += trans_noisy * np.cos(th + rot1_noisy)
        self._particles[:, 1] += trans_noisy * np.sin(th + rot1_noisy)
        self._particles[:, 2] += rot1_noisy + rot2_noisy
        self._normalize_theta_column(2)

    def _cb_scan(self, msg: LaserScan) -> None:
        if self._map is None:
            return
        if not self._initialized:
            return
        if self._particles is None:
            return
        self.log.debug("_cb_scan: fired")

        # Check update threshold (min translation / rotation since last update)
        if self._last_odom_pose is None:
            return
        min_d = self._get_param("update_min_d")
        min_a = self._get_param("update_min_a")
        if self._pose_estimate is not None:
            dx = self._last_odom_pose[0] - self._pose_estimate[0]
            dy = self._last_odom_pose[1] - self._pose_estimate[1]
            if math.sqrt(dx * dx + dy * dy) < min_d:
                da = abs(self._last_odom_pose[2] - self._pose_estimate[2])
                if da < min_a:
                    return

        t0 = time.perf_counter()
        self._measurement_update(msg)
        t_meas = (time.perf_counter() - t0) * 1000.0

        self._update_counter += 1
        resample_interval = self._get_param("resample_interval")
        t_resample = 0.0
        if resample_interval > 0 and self._update_counter % resample_interval == 0:
            t0r = time.perf_counter()
            self._resample_systematic()
            t_resample = (time.perf_counter() - t0r) * 1000.0

        t0u = time.perf_counter()
        self._update_pose_estimate()
        # Publish pose, covariance, path and map->odom TF all with scan timestamp.
        self._publish_pose_and_tf(msg.header.stamp)
        t_publish = (time.perf_counter() - t0u) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        self.log.debug(
            f"MCL timing (ms): measurement={t_meas:.1f} resample={t_resample:.1f} publish={t_publish:.1f} total={total_ms:.1f} "
            f"(particles={self._particles.shape[0] if self._particles is not None else 0} max_beams={self._get_param('max_beams')})"
        )

    def _measurement_update(self, msg: LaserScan) -> None:
        """Likelihood-field model: weight particles by scan likelihood (vectorized)."""
        if self._particles is None or self._weights is None or self._dist_map is None:
            return
        max_beams = self._get_param("max_beams")
        sigma_hit = float(self._get_param("sigma_hit"))
        max_range = self._get_param("laser_max_range")
        min_range = self._get_param("laser_min_range")

        num_readings = len(msg.ranges)
        if num_readings == 0:
            return
        step = max(1, num_readings // max_beams)
        indices = np.arange(0, num_readings, step)[:max_beams]
        r = np.array(msg.ranges, dtype=np.float64)[indices]
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        angles = angle_min + indices * angle_increment
        valid = np.isfinite(r) & (r <= max_range)
        if min_range >= 0:
            valid &= r >= min_range
        if np.sum(valid) == 0:
            return

        N = self._particles.shape[0]
        ox = self._map_metadata["origin_x"]
        oy = self._map_metadata["origin_y"]
        res = self._map_metadata["resolution"]
        h, w = self._dist_map.shape

        # All beam endpoints for all particles: (N, n_beams)
        n_beams = len(indices)
        px = self._particles[:, 0:1]
        py = self._particles[:, 1:2]
        pth = self._particles[:, 2:3]
        r_b = r.reshape(1, n_beams)
        angles_b = angles.reshape(1, n_beams)
        wx = px + r_b * np.cos(angles_b + pth)
        wy = py + r_b * np.sin(angles_b + pth)
        mx = np.int32((wx - ox) / res)
        my = np.int32((wy - oy) / res)
        mx = np.clip(mx, 0, w - 1)
        my = np.clip(my, 0, h - 1)

        d = self._dist_map[my, mx]
        inv_sigma_sq = 1.0 / (sigma_hit * sigma_hit)
        log_weights = np.sum(
            np.where(valid, -0.5 * inv_sigma_sq * (d * d), 0.0), axis=1
        )

        # Normalize to weights (numerically stable)
        log_weights -= np.max(log_weights)
        self._weights = np.exp(log_weights)
        s = np.sum(self._weights)
        if s > 0:
            self._weights /= s
        else:
            self._weights = np.ones(N) / N

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
        odom_frame = self._get_param("odom_frame_id")
        base_frame = self._get_param("base_frame_id")

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

        # TF: lookup(base, odom) returns transform from odom to base (T_base_odom).
        # We want map->odom so that map->base = map->odom * odom->base, i.e.
        # T_map_base = T_map_odom * T_odom_base, so T_map_odom = T_map_base * T_base_odom.
        # (Composition: point_in_map = T_map_base * point_in_base, point_in_base = T_base_odom * point_in_odom.)
        scan_time = rclpy.time.Time.from_msg(stamp)
        used_scan_time = True
        try:
            odom_to_base = self._tf_buffer.lookup_transform(
                base_frame,
                odom_frame,
                scan_time,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except Exception:
            used_scan_time = False
            try:
                odom_to_base = self._tf_buffer.lookup_transform(
                    base_frame,
                    odom_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
            except Exception as e2:
                self.log.debug(f"TF lookup odom->base at latest failed: {e2}")
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
            translation_matrix(
                [
                    self._pose_estimate[0],
                    self._pose_estimate[1],
                    0.0,
                ]
            ),
            quaternion_matrix(list(q)),
        )
        T_map_odom = T_map_base @ T_base_odom
        trans = translation_from_matrix(T_map_odom)
        quat = quaternion_from_matrix(T_map_odom)

        t = TransformStamped()
        t.header.stamp = stamp if used_scan_time else odom_to_base.header.stamp
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

    def _publish_particle_cloud(self) -> None:
        if self._particles is None:
            return
        self.log.debug("_publish_particle_cloud: fired")
        global_frame = self._get_param("map_frame_id")
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = global_frame
        for i in range(self._particles.shape[0]):
            p = PoseStamped().pose
            p.position.x = float(self._particles[i, 0])
            p.position.y = float(self._particles[i, 1])
            p.position.z = 0.0
            q = quaternion_from_euler(0, 0, self._particles[i, 2])
            p.orientation.x = q[0]
            p.orientation.y = q[1]
            p.orientation.z = q[2]
            p.orientation.w = q[3]
            pa.poses.append(p)
        self._particle_cloud_pub.publish(pa)


def main(args=None) -> None:
    # When run directly, use same launcher logic as localization.py (--config, -h/--help)
    from localization import parse_config_args

    ros_args = parse_config_args(args)
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
