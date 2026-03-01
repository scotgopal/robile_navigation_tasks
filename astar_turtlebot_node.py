#!/usr/bin/env python3
"""
A* Path Planning for TurtleBot3
================================
A ROS 2 node that uses the simple A* planner to navigate a TurtleBot.

Workflow:
  1. Loads the occupancy map from a PGM/YAML file (same as simple.py)
  2. Subscribes to /amcl_pose (or /odom) for the robot's current position
  3. Subscribes to /goal_pose (from RViz "2D Goal Pose") for the target
  4. Runs A* on the grid to compute a path
  5. Publishes the path on /planned_path for RViz visualisation
  6. Follows the path with a pure-pursuit controller publishing /cmd_vel

Usage:
  ros2 run <your_pkg> astar_turtlebot_node.py          # defaults
  ros2 run <your_pkg> astar_turtlebot_node.py --ros-args \
      -p map_yaml:=maps/map_task1.yaml \
      -p use_odom:=true \
      -p lookahead:=0.4 \
      -p linear_speed:=0.18 \
      -p goal_tolerance:=0.15

Or simply:
  python3 astar_turtlebot_node.py
"""

import os
import sys
import math
import heapq
from typing import List, Tuple, Optional, Set
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from std_msgs.msg import Header


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation about Z) from a quaternion — no external deps."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# ============================================================================
# PART 1 — Simple Grid & A* (extracted from simple.py, self-contained)
# ============================================================================

class SimpleGrid:
    """A 2-D occupancy grid: 0 = free, 1 = obstacle."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[0] * width for _ in range(height)]

    def add_obstacle(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = 1

    def is_free(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.grid[y][x] == 0


def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance — admissible for 8-connected grids."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar(grid: SimpleGrid,
          start: Tuple[int, int],
          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """Run A* on *grid* from *start* to *goal* (pixel coords).

    Returns a list of (x, y) pixel waypoints, or ``None``.
    """
    if not grid.is_free(*start) or not grid.is_free(*goal):
        return None

    MOVES = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
    ]

    open_set: list = []
    closed: Set[Tuple[int, int]] = set()
    nodes: dict = {start: {"g": 0.0, "parent": None}}
    heapq.heappush(open_set, (_heuristic(start, goal), 0.0, start[0], start[1]))

    while open_set:
        _f, _g, cx, cy = heapq.heappop(open_set)
        cur = (cx, cy)
        if cur in closed:
            continue
        if cur == goal:
            path = []
            pos = cur
            while pos is not None:
                path.append(pos)
                pos = nodes[pos]["parent"]
            path.reverse()
            return path
        closed.add(cur)
        g_cur = nodes[cur]["g"]
        for dx, dy, cost in MOVES:
            nx, ny = cx + dx, cy + dy
            nb = (nx, ny)
            if not grid.is_free(nx, ny) or nb in closed:
                continue
            g_new = g_cur + cost
            if g_new < nodes.get(nb, {}).get("g", float("inf")):
                nodes[nb] = {"g": g_new, "parent": cur}
                heapq.heappush(open_set, (g_new + _heuristic(nb, goal), g_new, nx, ny))
    return None


# ── helpers for loading the PGM map ──────────────────────────────────────────

def load_grid_from_pgm(pgm_path: str, yaml_path: str = None):
    """Return (SimpleGrid, resolution, origin_x, origin_y)."""
    from PIL import Image
    import numpy as np
    import yaml as _yaml

    img = Image.open(pgm_path)
    arr = np.asarray(img, dtype=float)
    height, width = arr.shape

    # defaults
    resolution = 0.05
    origin = [-11.3, -5.21, 0.0]
    free_thresh = 0.196
    occupied_thresh = 0.65
    negate = False

    if yaml_path is None:
        candidate = os.path.splitext(pgm_path)[0] + ".yaml"
        if os.path.isfile(candidate):
            yaml_path = candidate
    if yaml_path:
        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f)
        resolution = cfg.get("resolution", resolution)
        origin = cfg.get("origin", origin)
        free_thresh = cfg.get("free_thresh", free_thresh)
        occupied_thresh = cfg.get("occupied_thresh", occupied_thresh)
        negate = bool(cfg.get("negate", 0))

    if negate:
        occ = arr / 255.0
    else:
        occ = (255.0 - arr) / 255.0

    grid = SimpleGrid(width, height)
    for y in range(height):
        for x in range(width):
            if occ[y, x] > free_thresh:          # not clearly free → obstacle
                grid.add_obstacle(x, y)

    return grid, resolution, origin[0], origin[1]


def _inflate_grid(grid: SimpleGrid, radius_cells: int) -> SimpleGrid:
    """Return a copy of *grid* with obstacles inflated by *radius_cells*."""
    inflated = SimpleGrid(grid.width, grid.height)
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.grid[y][x] == 1:
                for dy in range(-radius_cells, radius_cells + 1):
                    for dx in range(-radius_cells, radius_cells + 1):
                        inflated.add_obstacle(x + dx, y + dy)
    return inflated


# ============================================================================
# PART 2 — ROS 2 Node
# ============================================================================

class AStarTurtleBotNode(Node):
    """ROS 2 node: A* planning + pure-pursuit path following for TurtleBot."""

    def __init__(self):
        super().__init__("astar_turtlebot")

        # ── parameters ───────────────────────────────────────────────────
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("map_pgm", "")
        self.declare_parameter("use_odom", True)        # True → /odom, False → /amcl_pose
        self.declare_parameter("inflation_radius", 3)   # cells to inflate obstacles
        self.declare_parameter("lookahead", 0.35)       # pure-pursuit lookahead (m)
        self.declare_parameter("linear_speed", 0.18)    # m/s
        self.declare_parameter("angular_speed_max", 1.0)  # rad/s clamp
        self.declare_parameter("goal_tolerance", 0.15)  # m
        self.declare_parameter("path_downsample", 5)    # publish every N-th waypoint
        self.declare_parameter("control_rate", 10.0)    # Hz

        self._map_yaml = self.get_parameter("map_yaml").value
        self._map_pgm = self.get_parameter("map_pgm").value
        self._use_odom = self.get_parameter("use_odom").value
        self._inflation = self.get_parameter("inflation_radius").value
        self._lookahead = self.get_parameter("lookahead").value
        self._linear_speed = self.get_parameter("linear_speed").value
        self._ang_max = self.get_parameter("angular_speed_max").value
        self._goal_tol = self.get_parameter("goal_tolerance").value
        self._downsample = max(1, self.get_parameter("path_downsample").value)
        self._rate = self.get_parameter("control_rate").value

        # ── state ────────────────────────────────────────────────────────
        self._grid: Optional[SimpleGrid] = None
        self._grid_inflated: Optional[SimpleGrid] = None
        self._resolution: float = 0.05
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0

        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_yaw: float = 0.0
        self._have_pose = False

        self._path_world: List[Tuple[float, float]] = []  # world coords
        self._path_index: int = 0
        self._following = False

        # ── load map ─────────────────────────────────────────────────────
        self._try_load_map_from_params()

        # ── publishers ───────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self._path_pub = self.create_publisher(Path, "planned_path", 10)

        # ── subscribers ──────────────────────────────────────────────────
        # Robot pose
        if self._use_odom:
            self.create_subscription(Odometry, "odom", self._odom_cb, 10)
            self.get_logger().info("Listening on /odom for robot pose")
        else:
            self.create_subscription(
                PoseWithCovarianceStamped, "amcl_pose", self._amcl_cb, 10)
            self.get_logger().info("Listening on /amcl_pose for robot pose")

        # Goal from RViz "2D Goal Pose"
        self.create_subscription(PoseStamped, "goal_pose", self._goal_cb, 10)

        # Optionally subscribe to /map topic (if no file was loaded)
        if self._grid is None:
            latching_qos = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self.create_subscription(
                OccupancyGrid, "map", self._map_topic_cb, latching_qos)
            self.get_logger().info(
                "No map file loaded — waiting for /map topic …")

        # ── control timer ────────────────────────────────────────────────
        self._timer = self.create_timer(1.0 / self._rate, self._control_loop)

        self.get_logger().info("A* TurtleBot node ready — "
                               "send a goal with RViz '2D Goal Pose'")

    # ─── map loading ─────────────────────────────────────────────────────

    def _try_load_map_from_params(self):
        """Try to load the map from the file parameters."""
        pgm = self._map_pgm
        yml = self._map_yaml

        # Resolve relative paths w.r.t. the workspace
        ws = os.path.dirname(os.path.abspath(__file__))
        if pgm and not os.path.isabs(pgm):
            pgm = os.path.join(ws, pgm)
        if yml and not os.path.isabs(yml):
            yml = os.path.join(ws, yml)

        # Auto-detect companion files
        if not pgm and yml:
            import yaml as _y
            with open(yml) as f:
                cfg = _y.safe_load(f)
            pgm = os.path.join(os.path.dirname(yml), cfg["image"])
        if pgm and not yml:
            candidate = os.path.splitext(pgm)[0] + ".yaml"
            if os.path.isfile(candidate):
                yml = candidate

        # Default: look for the map shipped with this repo
        if not pgm:
            default = os.path.join(ws, "maps", "map_task1.pgm")
            if os.path.isfile(default):
                pgm = default
                yml = os.path.splitext(pgm)[0] + ".yaml"
                self.get_logger().info(f"Auto-detected map: {pgm}")

        if pgm and os.path.isfile(pgm):
            self._load_pgm(pgm, yml)

    def _load_pgm(self, pgm_path: str, yaml_path: str = None):
        grid, res, ox, oy = load_grid_from_pgm(pgm_path, yaml_path)
        self._resolution = res
        self._origin_x = ox
        self._origin_y = oy
        if self._inflation > 0:
            self._grid = grid  # keep raw grid for reference
            self._grid_inflated = _inflate_grid(grid, self._inflation)
            self.get_logger().info(
                f"Inflated obstacles by {self._inflation} cells "
                f"(~{self._inflation * res:.2f} m)")
        else:
            self._grid = grid
            self._grid_inflated = grid
        self.get_logger().info(
            f"Map loaded: {grid.width}x{grid.height} px, "
            f"res={res} m/px, origin=({ox:.2f}, {oy:.2f})")

    def _map_topic_cb(self, msg: OccupancyGrid):
        """Build a SimpleGrid from a /map topic message."""
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        grid = SimpleGrid(w, h)
        for idx, val in enumerate(msg.data):
            if val < 0 or val > 50:          # unknown (-1) or occupied (>50)
                x = idx % w
                y = idx // w
                grid.add_obstacle(x, y)

        self._resolution = res
        self._origin_x = ox
        self._origin_y = oy
        if self._inflation > 0:
            self._grid = grid
            self._grid_inflated = _inflate_grid(grid, self._inflation)
        else:
            self._grid = grid
            self._grid_inflated = grid
        self.get_logger().info(
            f"Map from topic: {w}x{h}, res={res}, origin=({ox:.2f},{oy:.2f})")

    # ─── coordinate conversion ───────────────────────────────────────────

    def _world_to_pixel(self, wx: float, wy: float) -> Tuple[int, int]:
        px = int(round((wx - self._origin_x) / self._resolution))
        py = int(round((wy - self._origin_y) / self._resolution))
        return px, py

    def _pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        wx = px * self._resolution + self._origin_x
        wy = py * self._resolution + self._origin_y
        return wx, wy

    # ─── pose callbacks ──────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._robot_x = p.x
        self._robot_y = p.y
        self._robot_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self._have_pose = True

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._robot_x = p.x
        self._robot_y = p.y
        self._robot_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self._have_pose = True

    # ─── goal callback (from RViz) ───────────────────────────────────────

    def _goal_cb(self, msg: PoseStamped):
        if self._grid_inflated is None:
            self.get_logger().warn("No map loaded yet — ignoring goal")
            return
        if not self._have_pose:
            self.get_logger().warn("No robot pose yet — ignoring goal")
            return

        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.get_logger().info(f"New goal received: ({gx:.2f}, {gy:.2f})")

        # Convert both poses to pixel coordinates
        sx, sy = self._world_to_pixel(self._robot_x, self._robot_y)
        ex, ey = self._world_to_pixel(gx, gy)

        self.get_logger().info(
            f"Planning A*: pixel ({sx},{sy}) → ({ex},{ey})")

        # Clamp to map bounds
        sx = max(0, min(sx, self._grid_inflated.width - 1))
        sy = max(0, min(sy, self._grid_inflated.height - 1))
        ex = max(0, min(ex, self._grid_inflated.width - 1))
        ey = max(0, min(ey, self._grid_inflated.height - 1))

        # If start/goal is inside an obstacle, find nearest free cell
        sx, sy = self._nearest_free(sx, sy)
        ex, ey = self._nearest_free(ex, ey)

        if (sx, sy) == (-1, -1) or (ex, ey) == (-1, -1):
            self.get_logger().error("Could not find free cells near start/goal")
            return

        # ── Run A* ──
        path_px = astar(self._grid_inflated, (sx, sy), (ex, ey))

        if path_px is None:
            self.get_logger().warn("A* found no path!")
            self._following = False
            return

        self.get_logger().info(f"Path found: {len(path_px)} waypoints (pixels)")

        # Convert to world coordinates
        self._path_world = [self._pixel_to_world(px, py) for px, py in path_px]
        self._path_index = 0
        self._following = True

        # Publish for RViz
        self._publish_path()

    def _nearest_free(self, x: int, y: int, max_r: int = 15) -> Tuple[int, int]:
        """BFS spiral to find the nearest free cell in the inflated grid."""
        g = self._grid_inflated
        if g.is_free(x, y):
            return x, y
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if g.is_free(x + dx, y + dy):
                        return x + dx, y + dy
        return -1, -1

    # ─── path publisher ──────────────────────────────────────────────────

    def _publish_path(self):
        msg = Path()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for i, (wx, wy) in enumerate(self._path_world):
            if i % self._downsample != 0 and i != len(self._path_world) - 1:
                continue
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self._path_pub.publish(msg)
        self.get_logger().info(
            f"Published path with {len(msg.poses)} poses on /planned_path")

    # ─── control loop (pure pursuit) ─────────────────────────────────────

    def _control_loop(self):
        if not self._following or not self._path_world:
            return

        # Current robot position
        rx, ry, ryaw = self._robot_x, self._robot_y, self._robot_yaw

        # Advance the path index to the first waypoint beyond the lookahead
        # (skip waypoints we have already passed)
        while self._path_index < len(self._path_world) - 1:
            wx, wy = self._path_world[self._path_index]
            d = math.hypot(wx - rx, wy - ry)
            if d > self._lookahead:
                break
            self._path_index += 1

        # Target waypoint
        tx, ty = self._path_world[self._path_index]
        dist = math.hypot(tx - rx, ty - ry)

        # Check if we are at the final goal
        gx, gy = self._path_world[-1]
        goal_dist = math.hypot(gx - rx, gy - ry)
        if goal_dist < self._goal_tol:
            self.get_logger().info("Goal reached!")
            self._stop()
            self._following = False
            return

        # Pure pursuit steering
        angle_to_target = math.atan2(ty - ry, tx - rx)
        heading_error = self._normalize_angle(angle_to_target - ryaw)

        cmd = Twist()

        # If heading error is large, rotate in place first
        if abs(heading_error) > math.radians(30):
            cmd.linear.x = 0.0
            cmd.angular.z = max(-self._ang_max,
                                min(self._ang_max, 1.5 * heading_error))
        else:
            cmd.linear.x = self._linear_speed
            cmd.angular.z = max(-self._ang_max,
                                min(self._ang_max, 2.0 * heading_error))

        self._cmd_pub.publish(cmd)

    def _stop(self):
        cmd = Twist()
        self._cmd_pub.publish(cmd)

    @staticmethod
    def _normalize_angle(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a


# ============================================================================
# MAIN
# ============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = AStarTurtleBotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down …")
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
