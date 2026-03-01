#!/usr/bin/env python3
"""
ROS2 A* Global Planner Server for Nav2

This implements a global planner for Nav2 using the A* algorithm.
It subscribes to /map for the costmap data and /goal_pose for goal
requests, using TF2 to determine the robot's current position.

Key features:
- A* search with 8-connected grid movement
- Cost-aware planning using costmap values
- Path smoothing via gradient descent
- Subscribes to /goal_pose for planning requests
- TF2 transform support for robot pose lookup
"""

import math
import heapq
import time as pytime
from collections import deque
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

# ROS2 messages
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

# TF2 transformations
import tf2_ros
from tf2_ros import TransformException


# ============================================================================
# A* Algorithm Implementation
# ============================================================================

class AStarNode:
    """A node in the A* search graph (renamed to avoid clash with rclpy.Node)."""

    __slots__ = ('x', 'y', 'g', 'h', 'f', 'parent')

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.g = float('inf')
        self.h = 0.0
        self.f = float('inf')
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"({self.x},{self.y})"


class GridCostmap:
    """Wrapper around OccupancyGrid data for the A* planner."""

    def __init__(self):
        self.width: int = 0
        self.height: int = 0
        self.resolution: float = 0.0
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        self.data: list = []
        self.frame_id: str = "map"

        # Cost thresholds (OccupancyGrid values 0-100, -1 = unknown)
        self.lethal_threshold: int = 90
        self.inscribed_threshold: int = 50

    # ------------------------------------------------------------------
    def update_from_occupancy_grid(self, msg: OccupancyGrid):
        """Populate from a nav_msgs/OccupancyGrid message."""
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.data = list(msg.data)
        self.frame_id = msg.header.frame_id

    # ------------------------------------------------------------------
    def world_to_map(self, wx: float, wy: float) -> Tuple[int, int]:
        """World → map-cell coordinates."""
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """Map-cell → world coordinates (centre of cell)."""
        wx = (mx + 0.5) * self.resolution + self.origin_x
        wy = (my + 0.5) * self.resolution + self.origin_y
        return wx, wy

    # ------------------------------------------------------------------
    def in_bounds(self, mx: int, my: int) -> bool:
        return 0 <= mx < self.width and 0 <= my < self.height

    def is_free(self, mx: int, my: int) -> bool:
        """Return True when the cell is traversable."""
        if not self.in_bounds(mx, my):
            return False
        idx = my * self.width + mx
        if idx >= len(self.data):
            return False
        val = self.data[idx]
        return 0 <= val < self.lethal_threshold

    def get_cost(self, mx: int, my: int) -> int:
        if not self.in_bounds(mx, my):
            return 100
        idx = my * self.width + mx
        if idx >= len(self.data):
            return 100
        return self.data[idx]

    def inflate_obstacles(self, inflation_radius_cells: int):
        """
        Inflate obstacles by *inflation_radius_cells* using BFS.

        Cells within the inflation radius of a lethal cell are assigned
        a cost that decays linearly from ``lethal_threshold`` at distance 0
        to ``inscribed_threshold`` at the edge. The original data is never
        lowered — only raised.
        """
        if inflation_radius_cells <= 0:
            return

        w, h = self.width, self.height
        inflated = list(self.data)  # copy

        # BFS from every lethal cell
        queue: deque = deque()
        visited = [False] * (w * h)

        for y in range(h):
            for x in range(w):
                idx = y * w + x
                val = self.data[idx]
                if val >= self.lethal_threshold or val < 0:  # lethal or unknown
                    queue.append((x, y, 0))
                    visited[idx] = True
                    inflated[idx] = max(inflated[idx], self.lethal_threshold)

        while queue:
            cx, cy, dist = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    nidx = ny * w + nx
                    if visited[nidx]:
                        continue
                    # Actual Euclidean distance from the nearest obstacle
                    step = 1.414 if (dx != 0 and dy != 0) else 1.0
                    new_dist = dist + step
                    if new_dist > inflation_radius_cells:
                        continue
                    visited[nidx] = True
                    # Linear decay: lethal at dist 0 → inscribed at edge
                    ratio = 1.0 - (new_dist / inflation_radius_cells)
                    cost = int(
                        self.inscribed_threshold
                        + ratio * (self.lethal_threshold - self.inscribed_threshold)
                    )
                    if cost > inflated[nidx]:
                        inflated[nidx] = cost
                    queue.append((nx, ny, new_dist))

        self.data = inflated


# ------------------------------------------------------------------
# Heuristic
# ------------------------------------------------------------------
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance – admissible for 8-connected grids."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


# ------------------------------------------------------------------
# A* search
# ------------------------------------------------------------------
# 8-connected movements: (dx, dy, base_cost)
_MOVEMENTS = [
    (1, 0, 1.0), (-1, 0, 1.0),
    (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, 1.414), (1, -1, 1.414),
    (-1, 1, 1.414), (-1, -1, 1.414),
]


def astar_search(
    grid: GridCostmap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    use_costmap_weights: bool = True,
    max_iterations: int = 500_000,
) -> Optional[List[Tuple[int, int]]]:
    """
    A* search on an 8-connected grid.

    Returns a list of (mx, my) map cells from *start* to *goal*,
    or ``None`` when no path exists.
    """

    if not grid.is_free(*start):
        return None
    if not grid.is_free(*goal):
        return None

    open_set: list = []
    closed_set: set = set()
    all_nodes: dict = {}  # (x,y) -> {'g': float, 'parent': (px,py)|None}

    all_nodes[start] = {'g': 0.0, 'parent': None}
    heapq.heappush(open_set, (heuristic(start, goal), 0.0, start[0], start[1]))

    iterations = 0
    while open_set:
        iterations += 1
        if iterations > max_iterations:
            return None  # safety bail-out

        _f, _g, cx, cy = heapq.heappop(open_set)
        current = (cx, cy)

        if current in closed_set:
            continue

        # Goal reached – reconstruct path
        if current == goal:
            path: list = []
            pos = current
            while pos is not None:
                path.append(pos)
                pos = all_nodes[pos]['parent']
            path.reverse()
            return path

        closed_set.add(current)
        g_cur = all_nodes[current]['g']

        for dx, dy, move_cost in _MOVEMENTS:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)

            if not grid.is_free(nx, ny) or neighbor in closed_set:
                continue

            # Costmap-aware weighting
            weight = 1.0
            if use_costmap_weights:
                cost_val = grid.get_cost(nx, ny)
                if cost_val < grid.inscribed_threshold:
                    weight = 1.0 + (cost_val / grid.inscribed_threshold) * 0.5
                else:
                    weight = 2.0

            tentative_g = g_cur + move_cost * weight
            old_g = all_nodes.get(neighbor, {}).get('g', float('inf'))

            if tentative_g < old_g:
                all_nodes[neighbor] = {'g': tentative_g, 'parent': current}
                f_new = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_new, tentative_g, nx, ny))

    return None  # no path


# ------------------------------------------------------------------
# Path smoothing
# ------------------------------------------------------------------
def smooth_path(
    path: List[Tuple[float, float]],
    weight_smooth: float = 0.5,
    max_iterations: int = 100,
    tolerance: float = 0.001,
) -> List[Tuple[float, float]]:
    """Gradient-descent smoothing (keeps start & goal fixed)."""

    if len(path) <= 2:
        return list(path)

    smoothed = [list(p) for p in path]
    n = len(smoothed)

    for _ in range(max_iterations):
        max_change = 0.0
        for i in range(1, n - 1):
            for dim in (0, 1):
                prev = smoothed[i - 1][dim]
                curr = smoothed[i][dim]
                nxt = smoothed[i + 1][dim]
                new_val = curr + weight_smooth * ((prev + nxt) / 2.0 - curr)
                change = abs(new_val - curr)
                if change > max_change:
                    max_change = change
                smoothed[i][dim] = new_val
        if max_change < tolerance:
            break

    return [(p[0], p[1]) for p in smoothed]


# ============================================================================
# ROS 2 Global Planner Node  (standalone + Nav2 action server)
# ============================================================================

class AStarGlobalPlanner(Node):
    """
    A standalone ROS 2 node that:

    * subscribes to ``/map`` (OccupancyGrid)
    * subscribes to ``/goal_pose`` (PoseStamped) for goal requests
    * looks up the robot pose via TF2 (map → base_link)
    * publishes the resulting ``/plan`` (Path)
    """

    def __init__(self):
        super().__init__('astar_global_planner')

        # ---- Parameters ------------------------------------------------
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('plan_topic', '/plan')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('use_costmap_weights', True)
        self.declare_parameter('do_smooth_path', True)
        self.declare_parameter('lethal_threshold', 90)
        self.declare_parameter('inscribed_threshold', 50)
        self.declare_parameter('robot_radius', 0.3)  # metres of padding around obstacles

        self._map_topic = self.get_parameter('map_topic').value
        self._plan_topic = self.get_parameter('plan_topic').value
        self._goal_topic = self.get_parameter('goal_topic').value
        self._use_costmap_weights = self.get_parameter('use_costmap_weights').value
        self._do_smooth = self.get_parameter('do_smooth_path').value
        self._robot_radius = self.get_parameter('robot_radius').value

        # ---- Grid / map state ------------------------------------------
        self.grid = GridCostmap()
        self.grid.lethal_threshold = self.get_parameter('lethal_threshold').value
        self.grid.inscribed_threshold = self.get_parameter('inscribed_threshold').value
        self._map_received = False

        # ---- TF2 -------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Subscribers ------------------------------------------------
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, self._map_topic, self._map_cb, map_qos
        )

        # ---- Goal subscriber --------------------------------------------
        self.goal_sub = self.create_subscription(
            PoseStamped, self._goal_topic, self._goal_pose_cb, 10
        )

        # ---- Publishers -------------------------------------------------
        self.path_pub = self.create_publisher(Path, self._plan_topic, 10)

        self.get_logger().info('A* Global Planner initialised')
        self.get_logger().info(f'  map topic : {self._map_topic}')
        self.get_logger().info(f'  goal topic: {self._goal_topic}')
        self.get_logger().info(f'  plan topic: {self._plan_topic}')

    # ------------------------------------------------------------------
    # Map callback
    # ------------------------------------------------------------------
    def _map_cb(self, msg: OccupancyGrid):
        self.grid.update_from_occupancy_grid(msg)

        # Inflate obstacles so the path keeps robot_radius clearance
        if self._robot_radius > 0.0 and self.grid.resolution > 0.0:
            inflation_cells = int(
                math.ceil(self._robot_radius / self.grid.resolution)
            )
            self.grid.inflate_obstacles(inflation_cells)
            self.get_logger().info(
                f'Obstacles inflated by {inflation_cells} cells '
                f'({self._robot_radius:.2f} m)'
            )

        self._map_received = True
        self.get_logger().info(
            f'Map received: {self.grid.width}x{self.grid.height} '
            f'@ {self.grid.resolution:.3f} m/cell'
        )

    # ------------------------------------------------------------------
    # Goal pose callback
    # ------------------------------------------------------------------
    def _goal_pose_cb(self, msg: PoseStamped):
        """Handle a new goal received on /goal_pose."""
        self.get_logger().info(
            f'Received goal pose: ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f})'
        )

        start_pose = self._get_robot_pose()
        if start_pose is None:
            self.get_logger().error('Could not determine start pose from TF')
            return

        start_world = (start_pose.pose.position.x, start_pose.pose.position.y)
        goal_world = (msg.pose.position.x, msg.pose.position.y)

        path_msg = self.compute_path(start_world, goal_world)

        if path_msg is not None and len(path_msg.poses) > 0:
            self.path_pub.publish(path_msg)
            self.get_logger().info(
                f'Path published ({len(path_msg.poses)} poses)'
            )
        else:
            self.get_logger().warn('Planning failed – no valid path')

    # ------------------------------------------------------------------
    # Core planning method
    # ------------------------------------------------------------------
    def compute_path(
        self,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float],
    ) -> Optional[Path]:
        """Compute a path (world coords) and return a Path msg."""

        if not self._map_received:
            self.get_logger().warn('No map received yet')
            return None

        start_map = self.grid.world_to_map(*start_world)
        goal_map = self.grid.world_to_map(*goal_world)

        self.get_logger().info(
            f'Planning  world {start_world} → {goal_world}  |  '
            f'map {start_map} → {goal_map}'
        )

        t0 = pytime.monotonic()
        path_cells = astar_search(
            self.grid,
            start_map,
            goal_map,
            use_costmap_weights=self._use_costmap_weights,
        )
        dt = pytime.monotonic() - t0

        if path_cells is None:
            self.get_logger().warn(f'A* found no path ({dt:.3f}s)')
            return None

        self.get_logger().info(
            f'A* found path: {len(path_cells)} cells in {dt:.3f}s'
        )

        # Convert to world coords
        path_world = [self.grid.map_to_world(mx, my) for mx, my in path_cells]

        # Optional smoothing
        if self._do_smooth and len(path_world) > 2:
            path_world = smooth_path(path_world)

        # Build Path msg
        path_msg = self._build_path_msg(path_world)
        return path_msg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_path_msg(
        self, path_world: List[Tuple[float, float]]
    ) -> Path:
        msg = Path()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.grid.frame_id

        prev_x, prev_y = path_world[0]
        for x, y in path_world:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0

            # Compute orientation along path direction
            dx = x - prev_x
            dy = y - prev_y
            yaw = math.atan2(dy, dx)
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)

            msg.poses.append(pose)
            prev_x, prev_y = x, y

        return msg

    def _get_robot_pose(self) -> Optional[PoseStamped]:
        """Look up the robot base_link pose in the map frame via TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp = t.header.stamp
            ps.pose.position.x = t.transform.translation.x
            ps.pose.position.y = t.transform.translation.y
            ps.pose.position.z = t.transform.translation.z
            ps.pose.orientation = t.transform.rotation
            return ps
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None


# ============================================================================
# Entry point
# ============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = AStarGlobalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down A* Global Planner')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
