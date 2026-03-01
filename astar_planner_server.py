#!/usr/bin/env python3
"""
A* Global Planner Server for Nav2  (based on maps/simple.py)
=============================================================
A standalone ROS 2 node that acts as a drop-in replacement for
Nav2's ``planner_server``.  It re-uses the A* implementation from
``maps/simple.py`` (``SimpleGrid`` + ``astar_simple``).

Architecture
------------
::

    ┌──────────────┐         ┌──────────────────────┐
    │  map_server   │──/map──▶│  astar_planner_server │
    └──────────────┘         │  (this node)          │
                              │                      │
    ┌──────────────┐  action  │  compute_path_to_pose │
    │ bt_navigator  │────────▶│  ───────────────────▶ │──▶ /plan
    └──────────────┘         └──────────────────────┘

How to run
----------
**Terminal 1** – start this planner server::

    python3 astar_planner_server.py
    # or with parameters:
    python3 astar_planner_server.py --ros-args \
        -p use_sim_time:=true \
        -p inflation_radius_cells:=3 \
        -p path_downsample:=3

**Terminal 2** – start Nav2 *without* its built-in planner_server.
Use the provided ``astar_tb3_launch.py`` which already excludes
``planner_server`` from the lifecycle manager, or manually launch
Nav2 components (controller_server, bt_navigator, …).

Once both are up, clicking **2D Goal Pose** in RViz will route
through our A* planner automatically.
"""

import os
import sys
import math
import time
from typing import List, Tuple, Optional, Set

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Quaternion
from builtin_interfaces.msg import Duration

from tf2_ros import Buffer, TransformListener, TransformException

# ────────────────────────────────────────────────────────────────────
# Import A* algorithm from maps/simple.py
# ────────────────────────────────────────────────────────────────────
# Add the project root to sys.path so ``from maps.simple import ...``
# works regardless of the working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from maps.simple import SimpleGrid, astar_simple, heuristic_xy   # noqa: E402


# ────────────────────────────────────────────────────────────────────
# Obstacle inflation helper
# ────────────────────────────────────────────────────────────────────

def inflate_grid(grid: SimpleGrid, radius: int) -> SimpleGrid:
    """Return a *new* SimpleGrid with obstacles inflated by *radius* cells.

    Every cell within Euclidean distance ``radius`` of an obstacle is
    also marked as an obstacle.  This gives the planner a safety margin
    so the robot (which has a non-zero footprint) does not clip walls.
    """
    out = SimpleGrid(grid.width, grid.height)
    r2 = radius * radius
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.grid[y][x] == 1:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx * dx + dy * dy <= r2:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                                out.grid[ny][nx] = 1
    return out


# ────────────────────────────────────────────────────────────────────
# ROS 2 node
# ────────────────────────────────────────────────────────────────────

class AStarPlannerServer(Node):
    """
    Global planner node that wraps ``maps/simple.py``'s A* algorithm.

    Subscriptions:
        /map  (nav_msgs/msg/OccupancyGrid)

    Action server:
        compute_path_to_pose  (nav2_msgs/action/ComputePathToPose)

    Publishers:
        /plan  (nav_msgs/msg/Path)   – for RViz visualisation
    """

    def __init__(self):
        super().__init__("planner_server")

        # ── Parameters ───────────────────────────────────────────
        self.declare_parameter("inflation_radius_cells", 3)
        self.declare_parameter("path_downsample", 1)
        self.declare_parameter("snap_radius", 15)

        self._inflation = (
            self.get_parameter("inflation_radius_cells")
            .get_parameter_value().integer_value
        )
        self._downsample = (
            self.get_parameter("path_downsample")
            .get_parameter_value().integer_value
        )
        self._snap_radius = (
            self.get_parameter("snap_radius")
            .get_parameter_value().integer_value
        )

        # ── Map state (populated by /map callback) ───────────────
        self._grid: Optional[SimpleGrid] = None        # inflated grid
        self._raw_grid: Optional[SimpleGrid] = None     # original grid
        self._resolution: float = 0.05
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0

        # ── TF (robot pose lookup when start not in request) ─────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── /map subscriber (latching QoS to get the last map) ───
        latching_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._map_sub = self.create_subscription(
            OccupancyGrid, "/map", self._map_callback, latching_qos
        )

        # ── /plan publisher (for RViz) ───────────────────────────
        self._path_pub = self.create_publisher(Path, "/plan", 10)

        # ── Action server: compute_path_to_pose ──────────────────
        # This is what Nav2's bt_navigator calls to get a global plan.
        self._action_server = ActionServer(
            self,
            ComputePathToPose,
            "compute_path_to_pose",
            execute_callback=self._execute_callback,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        self.get_logger().info(
            "  A* Planner Server READY  (using maps/simple.py A*)"
        )
        self.get_logger().info(
            f"  inflation={self._inflation} cells, "
            f"downsample={self._downsample}, "
            f"snap_radius={self._snap_radius}"
        )
        self.get_logger().info(
            "  Waiting for /map topic and compute_path_to_pose goals …"
        )
        self.get_logger().info(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    # ── /map callback ────────────────────────────────────────────

    def _map_callback(self, msg: OccupancyGrid):
        """Convert a nav_msgs/OccupancyGrid into a SimpleGrid."""
        w = msg.info.width
        h = msg.info.height
        self._resolution = msg.info.resolution
        self._origin_x = msg.info.origin.position.x
        self._origin_y = msg.info.origin.position.y

        # Build a SimpleGrid from the OccupancyGrid data.
        # OccupancyGrid values: -1 = unknown, 0 = free, 100 = occupied
        # We treat unknown and occupied (> 50) as obstacles.
        grid = SimpleGrid(w, h)
        for i, val in enumerate(msg.data):
            if val < 0 or val > 50:
                x = i % w
                y = i // w
                grid.add_obstacle(x, y)

        self._raw_grid = grid

        # Inflate obstacles for robot footprint safety margin
        if self._inflation > 0:
            self._grid = inflate_grid(grid, self._inflation)
        else:
            self._grid = grid

        free_cells = sum(
            1 for y in range(h) for x in range(w)
            if self._grid.grid[y][x] == 0
        )
        self.get_logger().info(
            f"Map received: {w}×{h}, resolution={self._resolution:.3f} m/cell, "
            f"free={free_cells}/{w * h}"
        )

    # ── coordinate helpers ───────────────────────────────────────

    def _world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates (metres) → grid cell (x, y)."""
        gx = int(round((wx - self._origin_x) / self._resolution))
        gy = int(round((wy - self._origin_y) / self._resolution))
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid cell (x, y) → world coordinates (metres)."""
        wx = gx * self._resolution + self._origin_x
        wy = gy * self._resolution + self._origin_y
        return wx, wy

    def _snap_to_free(self, x: int, y: int) -> Tuple[int, int]:
        """Find nearest free cell within ``snap_radius`` via ring search."""
        grid = self._grid
        if grid.is_free(x, y):
            return x, y
        for r in range(1, self._snap_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = x + dx, y + dy
                        if grid.is_free(nx, ny):
                            return nx, ny
        return -1, -1

    # ── downsample path ──────────────────────────────────────────

    def _downsample_path(
        self, path: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Keep every Nth waypoint (always keep first and last)."""
        if self._downsample <= 1 or len(path) <= 2:
            return path
        out = [path[0]]
        for i in range(1, len(path) - 1):
            if i % self._downsample == 0:
                out.append(path[i])
        out.append(path[-1])
        return out

    # ── action callback ──────────────────────────────────────────

    def _execute_callback(self, goal_handle):
        """Handle a ComputePathToPose goal from bt_navigator."""
        request = goal_handle.request
        result = ComputePathToPose.Result()

        # ── guard: need a map first ──────────────────────────────
        if self._grid is None:
            self.get_logger().error("No map received yet — cannot plan")
            goal_handle.abort()
            return result

        # ── goal position (always from the request) ──────────────
        goal_wx = request.goal.pose.position.x
        goal_wy = request.goal.pose.position.y

        # ── start position (from request or TF lookup) ───────────
        if request.use_start:
            start_wx = request.start.pose.position.x
            start_wy = request.start.pose.position.y
        else:
            try:
                t = self._tf_buffer.lookup_transform(
                    "map", "base_link", rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
                start_wx = t.transform.translation.x
                start_wy = t.transform.translation.y
            except TransformException as exc:
                self.get_logger().error(f"TF lookup failed: {exc}")
                goal_handle.abort()
                return result

        self.get_logger().info(
            f"Planning: ({start_wx:.2f}, {start_wy:.2f}) → "
            f"({goal_wx:.2f}, {goal_wy:.2f})"
        )

        # ── world → grid coords, clamp to map bounds ────────────
        sx, sy = self._world_to_grid(start_wx, start_wy)
        gx, gy = self._world_to_grid(goal_wx, goal_wy)

        w, h = self._grid.width, self._grid.height
        sx = max(0, min(sx, w - 1))
        sy = max(0, min(sy, h - 1))
        gx = max(0, min(gx, w - 1))
        gy = max(0, min(gy, h - 1))

        # Snap to nearest free cell if start/goal are in an obstacle
        sx, sy = self._snap_to_free(sx, sy)
        gx, gy = self._snap_to_free(gx, gy)
        if sx < 0 or gx < 0:
            self.get_logger().error(
                "No free cell found near start or goal — aborting"
            )
            goal_handle.abort()
            return result

        # ── run A* from maps/simple.py ───────────────────────────
        t0 = time.monotonic()
        path_cells = astar_simple(self._grid, (sx, sy), (gx, gy))
        dt = time.monotonic() - t0

        if path_cells is None:
            self.get_logger().warn(
                f"A* found NO path  ({dt:.3f}s)"
            )
            goal_handle.abort()
            return result

        # Apply down-sampling to reduce waypoint count
        path_cells = self._downsample_path(path_cells)

        self.get_logger().info(
            f"A* path: {len(path_cells)} waypoints in {dt:.3f}s"
        )

        # ── build nav_msgs/msg/Path ──────────────────────────────
        now = self.get_clock().now().to_msg()
        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = "map"

        for px, py in path_cells:
            wx, wy = self._grid_to_world(px, py)
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = "map"
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        # Orient each pose toward the next waypoint
        for i in range(len(path_msg.poses) - 1):
            p0 = path_msg.poses[i].pose.position
            p1 = path_msg.poses[i + 1].pose.position
            yaw = math.atan2(p1.y - p0.y, p1.x - p0.x)
            path_msg.poses[i].pose.orientation = _yaw_to_quat(yaw)
        if len(path_msg.poses) >= 2:
            path_msg.poses[-1].pose.orientation = (
                path_msg.poses[-2].pose.orientation
            )

        # ── publish /plan for RViz ───────────────────────────────
        self._path_pub.publish(path_msg)

        # ── return result to bt_navigator ────────────────────────
        result.path = path_msg
        result.planning_time = Duration(
            sec=int(dt), nanosec=int((dt - int(dt)) * 1e9)
        )
        goal_handle.succeed()
        return result


def _yaw_to_quat(yaw: float) -> Quaternion:
    """Convert a yaw angle (radians) to a geometry_msgs/Quaternion."""
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


# ────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerServer()
    # Use multi-threaded executor so TF + action can run concurrently
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
