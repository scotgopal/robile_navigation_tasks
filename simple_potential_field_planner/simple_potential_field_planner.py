#!/usr/bin/env python3
"""
Simplified Potential Field-Based Path Planning Node for Robile Robot

This node implements a basic potential field algorithm with:
- Parabolic attractive potential towards the goal
- Inverse square repulsive potential from obstacles
- Simple stuck detection (logs warning, no escape maneuver)
- Trajectory visualization with color changes per goal
- Force visualization (attractive, repulsive, total)
"""

import math
import signal
import random
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float64MultiArray
import tf2_ros
import tf_transformations
from tf2_geometry_msgs import do_transform_pose


class SimplePotentialFieldPlanner(Node):
    """
    Simplified Potential Field Planner for autonomous navigation.

    Uses:
    - Parabolic attractive potential
    - Inverse square repulsive potential
    - Fixed (non-adaptive) gains
    """

    # ==========================================================================
    # Parameter Type Hints (set dynamically via load_parameters)
    # ==========================================================================

    # Potential field parameters
    k_attr: float
    k_rep: float
    rho_0: float

    # Goal tolerances
    goal_dist_tolerance: float
    goal_angle_tolerance: float

    # Velocity limits
    max_linear_vel: float
    max_angular_vel: float
    min_angular_vel: float
    min_linear_vel: float

    # Control gains
    angular_gain: float

    # Stuck detection
    stuck_threshold: float
    stuck_timeout: float

    # Visualization
    force_visualization_scale: float
    force_arrow_max_length: float

    # Goal pose marker (for clear/delete)
    GOAL_MARKER_NS = "goal_pose"
    GOAL_MARKER_ID = 0

    # ==========================================================================
    # Parameter Definitions
    # ==========================================================================

    PARAMETER_DEFINITIONS = {
        # Potential field parameters
        "k_attr": (0.5, "Attractive force gain"),
        "k_rep": (0.8, "Repulsive force gain"),
        "rho_0": (1.5, "Influence distance for repulsive force (meters)"),
        # Goal tolerances
        "goal_dist_tolerance": (0.3, "Distance tolerance for goal reached (meters)"),
        "goal_angle_tolerance": (
            0.1,
            "Angle tolerance for final orientation (radians)",
        ),
        # Velocity limits
        "max_linear_vel": (0.5, "Maximum linear velocity (m/s)"),
        "max_angular_vel": (1.0, "Maximum angular velocity (rad/s)"),
        "min_angular_vel": (0.1, "Minimum angular velocity (rad/s)"),
        "min_linear_vel": (0.1, "Minimum linear velocity (m/s)"),
        # Control gains
        "angular_gain": (1.0, "Angular velocity gain"),
        # Stuck detection
        "stuck_threshold": (0.05, "Movement threshold for stuck detection (meters)"),
        "stuck_timeout": (5.0, "Time before logging stuck warning (seconds)"),
        # Visualization
        "force_visualization_scale": (
            2.0,
            "Scale factor for force visualization arrows",
        ),
        "force_arrow_max_length": (
            2.0,
            "Max arrow length in RViz (meters); caps haywire arrows",
        ),
    }

    # ==========================================================================
    # Initialization
    # ==========================================================================

    def __init__(self):
        """Initialize the simplified potential field planner node."""
        super().__init__("simple_potential_field_planner")
        self.logger = self.get_logger()

        # Declare and load parameters
        self.declare_all_parameters()
        self.load_parameters()
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Navigation state
        self.goal_odom = None  # [x, y, theta]
        self.latest_scan = None
        self.goal_reached_logged = False
        self.goal_achieved = False  # True once goal reached; stop control, keep marker
        self.orientation_phase_logged = False

        # Stuck detection state
        self.last_robot_pose = None
        self.stuck_time = 0.0
        self.stuck_logged = False

        # Trajectory state
        self.trajectory = []
        self.last_trajectory_pose = None
        self.trajectory_color = self._generate_random_color()
        self.max_trajectory_length = 1000
        self.trajectory_update_threshold = 0.05
        self.teleop_active = False
        self.last_cmd_vel_time = None

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.trajectory_publisher = self.create_publisher(Path, "/trajectory", 10)
        self.force_visualization_publisher = self.create_publisher(
            MarkerArray, "/potential_field_forces", 10
        )
        self.force_magnitudes_publisher = self.create_publisher(
            Float64MultiArray, "/potential_field_force_magnitudes", 10
        )
        self.goal_pose_marker_publisher = self.create_publisher(
            MarkerArray, "/goal_pose_marker", 10
        )

        # Subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.goal_pose_subscriber = self.create_subscription(
            PoseStamped, "/goal_pose", self.goal_pose_callback, 10
        )
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.trajectory_timer = self.create_timer(0.5, self.publish_trajectory)  # 2 Hz

        self.logger.info("Simple Potential Field Planner initialized")
        self.logger.info("Waiting for goal pose from /goal_pose topic")

    # ==========================================================================
    # Parameter Management
    # ==========================================================================

    def declare_all_parameters(self):
        """Declare all parameters from PARAMETER_DEFINITIONS."""
        for name, (default, description) in self.PARAMETER_DEFINITIONS.items():
            self.declare_parameter(
                name, default, ParameterDescriptor(description=description)
            )

    def load_parameters(self):
        """Load all parameters from ROS2 parameter server."""
        for name in self.PARAMETER_DEFINITIONS:
            setattr(self, name, self.get_parameter(name).value)

    def parameter_callback(self, params: list[Parameter]) -> SetParametersResult:
        """Callback for parameter updates."""
        for param in params:
            if param.name in self.PARAMETER_DEFINITIONS:
                setattr(self, param.name, param.value)
        self.logger.info("Parameters updated")
        return SetParametersResult(successful=True)

    # ==========================================================================
    # Callbacks
    # ==========================================================================

    def scan_callback(self, msg: LaserScan):
        """Store latest laser scan data."""
        self.latest_scan = msg

    def cmd_vel_callback(self, msg: Twist):
        """Monitor cmd_vel to detect teleop activity."""
        # If we receive cmd_vel that we didn't publish (teleop), mark as teleop active
        if msg.linear.x != 0.0 or msg.angular.z != 0.0:
            self.last_cmd_vel_time = self.get_clock().now()

    def goal_pose_callback(self, msg: PoseStamped):
        """Process new goal pose."""
        try:
            goal_frame = msg.header.frame_id
            self.logger.info(
                f"Received goal in frame '{goal_frame}' at "
                f"({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
            )

            # Transform to odom frame
            goal_pose_odom = self._transform_pose_to_odom(msg, goal_frame)
            if goal_pose_odom is None:
                return

            # Extract goal [x, y, theta]
            self.goal_odom = self._extract_pose_array(goal_pose_odom)

            # Clear previous goal visualization and show new goal
            self._clear_goal_marker()

            # Reset state for new goal
            self.stuck_time = 0.0
            self.stuck_logged = False
            self.goal_reached_logged = False
            self.goal_achieved = False
            self.orientation_phase_logged = False

            # Start new trajectory with new color
            self.trajectory = []
            self.last_trajectory_pose = None
            self.trajectory_color = self._generate_random_color()

            self._publish_goal_marker()

            self.logger.info(
                f"Goal set: x={self.goal_odom[0]:.2f}, y={self.goal_odom[1]:.2f}, "
                f"theta={self.goal_odom[2]:.2f} rad (odom frame)"
            )

        except Exception as e:
            self.logger.error(f"Error processing goal pose: {e}")

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _transform_pose_to_odom(
        self, msg: PoseStamped, source_frame: str
    ) -> PoseStamped | None:
        """Transform PoseStamped to odom frame."""
        if source_frame == "odom":
            return msg

        try:
            if self.tf_buffer.can_transform("odom", source_frame, rclpy.time.Time()):
                transform = self.tf_buffer.lookup_transform(
                    "odom", source_frame, rclpy.time.Time()
                )
                return do_transform_pose(msg, transform)
        except Exception as e:
            self.logger.warn(f"Cannot transform from {source_frame} to odom: {e}")
        return None

    def _extract_pose_array(self, pose_stamped: PoseStamped) -> np.ndarray:
        """Extract [x, y, theta] from PoseStamped."""
        pos = pose_stamped.pose.position
        orient = pose_stamped.pose.orientation
        quaternion = (orient.x, orient.y, orient.z, orient.w)
        euler = tf_transformations.euler_from_quaternion(quaternion)
        return np.array([pos.x, pos.y, euler[2]])

    def _generate_random_color(self) -> ColorRGBA:
        """Generate a random vibrant color for trajectory."""
        hue = random.random()
        # Convert HSV to RGB (S=1.0, V=1.0 for vibrant colors)
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return ColorRGBA(r=float(r), g=float(g), b=float(b), a=0.8)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    # ==========================================================================
    # Robot Pose
    # ==========================================================================

    def get_robot_pose_in_odom(self) -> tuple | None:
        """Get current robot pose (x, y, theta) in odom frame."""
        try:
            if not self.tf_buffer.can_transform("odom", "base_link", rclpy.time.Time()):
                return None

            transform = self.tf_buffer.lookup_transform(
                "odom", "base_link", rclpy.time.Time()
            )
            t = transform.transform.translation
            r = transform.transform.rotation
            euler = tf_transformations.euler_from_quaternion((r.x, r.y, r.z, r.w))
            return (t.x, t.y, euler[2])

        except Exception as e:
            self.logger.error(f"Error getting robot pose: {e}")
            return None

    def transform_goal_to_base_link(self, robot_pose_odom: tuple) -> np.ndarray | None:
        """Transform goal from odom to base_link frame."""
        if robot_pose_odom is None or self.goal_odom is None:
            return None

        robot_x, robot_y, robot_theta = robot_pose_odom
        dx = self.goal_odom[0] - robot_x
        dy = self.goal_odom[1] - robot_y

        # Rotate to base_link frame
        cos_theta = math.cos(-robot_theta)
        sin_theta = math.sin(-robot_theta)
        return np.array(
            [dx * cos_theta - dy * sin_theta, dx * sin_theta + dy * cos_theta]
        )

    # ==========================================================================
    # Potential Field Calculations
    # ==========================================================================

    def calculate_attractive_force(self, goal_base_link: np.ndarray) -> np.ndarray:
        """
        Calculate parabolic attractive force towards goal.

        F_attr = k_attr * goal_vector
        """
        if goal_base_link is None:
            return np.array([0.0, 0.0])

        distance = np.linalg.norm(goal_base_link)
        if distance < 1e-6:
            return np.array([0.0, 0.0])

        # Parabolic potential: U = (1/2) * k_attr * distance^2
        # Force: F = -grad(U) = k_attr * direction
        direction = goal_base_link / distance
        return self.k_attr * direction

    def calculate_repulsive_force(self) -> np.ndarray:
        """
        Calculate inverse square repulsive force from obstacles.

        F_rep = k_rep * (1/distance - 1/rho_0) * (1/distance^2) * direction
        """
        if self.latest_scan is None:
            return np.array([0.0, 0.0])

        ranges = np.array(self.latest_scan.ranges)
        valid_mask = (
            np.isfinite(ranges)
            & (ranges >= self.latest_scan.range_min)
            & (ranges <= self.latest_scan.range_max)
        )
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return np.array([0.0, 0.0])

        v_rep_total = np.array([0.0, 0.0])
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment

        for idx in valid_indices:
            dist = ranges[idx]
            if dist >= self.rho_0 or dist <= 0:
                continue

            # Obstacle position in base_link frame
            angle = angle_min + idx * angle_increment
            obs_x = dist * math.cos(angle)
            obs_y = dist * math.sin(angle)

            # Repulsive force direction (away from obstacle)
            direction = np.array([-obs_x, -obs_y]) / dist

            # Inverse square law
            force_magnitude = (
                self.k_rep * (1.0 / dist - 1.0 / self.rho_0) * (1.0 / dist**2)
            )
            v_rep_total += force_magnitude * direction

        return v_rep_total

    # ==========================================================================
    # Velocity Calculation
    # ==========================================================================

    def calculate_velocity(
        self, v_total: np.ndarray, distance_to_goal: float, angle_to_goal: float
    ) -> tuple[float, float]:
        """
        Calculate velocity commands from total force.

        Returns (linear_x, angular_z)
        """
        # Calculate desired direction from force
        force_magnitude = np.linalg.norm(v_total)

        if force_magnitude < 1e-6:
            # No force - stop
            return 0.0, 0.0

        # Desired angle from force direction
        desired_angle = math.atan2(v_total[1], v_total[0])
        angle_error = self.normalize_angle(desired_angle)

        # Angular velocity (proportional to angle error)
        angular_z = np.clip(
            self.angular_gain * angle_error, -self.max_angular_vel, self.max_angular_vel
        )

        # Apply minimum angular velocity
        if 0 < abs(angular_z) < self.min_angular_vel:
            angular_z = math.copysign(self.min_angular_vel, angular_z)

        # Linear velocity (proportional to force magnitude and alignment)
        alignment = max(0.0, math.cos(angle_error))
        linear_x = np.clip(force_magnitude * alignment, 0.0, self.max_linear_vel)

        # Apply minimum linear velocity when reasonably aligned
        if (
            abs(angle_error) < math.pi / 3
            and distance_to_goal > self.goal_dist_tolerance
        ):
            linear_x = max(linear_x, self.min_linear_vel)

        return linear_x, angular_z

    def calculate_orientation_velocity(
        self, robot_pose_odom: tuple
    ) -> tuple[float, float]:
        """Calculate velocity for final orientation alignment."""
        angle_error = self.normalize_angle(self.goal_odom[2] - robot_pose_odom[2])

        if abs(angle_error) < self.goal_angle_tolerance:
            return 0.0, 0.0

        angular_z = np.clip(
            self.angular_gain * angle_error, -self.max_angular_vel, self.max_angular_vel
        )

        if 0 < abs(angular_z) < self.min_angular_vel:
            angular_z = math.copysign(self.min_angular_vel, angular_z)

        return 0.0, angular_z

    # ==========================================================================
    # Stuck Detection
    # ==========================================================================

    def check_stuck_status(self, robot_pose: tuple) -> bool:
        """
        Check if robot is stuck (not making progress).

        Returns True if stuck warning should be logged.
        """
        if self.last_robot_pose is None:
            self.last_robot_pose = robot_pose
            return False

        # Calculate movement
        dx = robot_pose[0] - self.last_robot_pose[0]
        dy = robot_pose[1] - self.last_robot_pose[1]
        movement = math.sqrt(dx * dx + dy * dy)

        # Update stuck timer
        if movement < self.stuck_threshold:
            self.stuck_time += 0.1  # Timer period
        else:
            self.stuck_time = 0.0
            self.stuck_logged = False

        # Update last pose
        self.last_robot_pose = robot_pose

        # Check if stuck
        if self.stuck_time > self.stuck_timeout and not self.stuck_logged:
            self.logger.warn(
                f"Robot appears to be stuck! Movement: {movement:.4f}m in 0.1s"
            )
            self.stuck_logged = True
            return True

        return False

    # ==========================================================================
    # Trajectory Management
    # ==========================================================================

    def update_trajectory(self, robot_pose: tuple):
        """Update trajectory if robot moved significantly and not in teleop."""
        if robot_pose is None:
            return

        # Check if teleop is active (received cmd_vel in last 0.5 seconds)
        if self.last_cmd_vel_time is not None:
            time_since_cmd = (
                self.get_clock().now() - self.last_cmd_vel_time
            ).nanoseconds / 1e9
            if time_since_cmd < 0.5:
                return  # Don't log trajectory during teleop

        # Check if moved enough to update
        if self.last_trajectory_pose is not None:
            dx = robot_pose[0] - self.last_trajectory_pose[0]
            dy = robot_pose[1] - self.last_trajectory_pose[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < self.trajectory_update_threshold:
                return

        # Add pose to trajectory
        pose_stamped = self._create_pose_stamped(robot_pose)
        self.trajectory.append(pose_stamped)

        # Limit trajectory length
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)

        self.last_trajectory_pose = robot_pose

    def publish_trajectory(self):
        """Publish trajectory for visualization."""
        if not self.trajectory:
            return

        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.poses = self.trajectory.copy()

        self.trajectory_publisher.publish(path_msg)

        # Publish trajectory line marker with color
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width
        marker.color = self.trajectory_color

        for pose_stamped in self.trajectory:
            point = Point()
            point.x = pose_stamped.pose.position.x
            point.y = pose_stamped.pose.position.y
            point.z = 0.05
            marker.points.append(point)

        marker_array = MarkerArray()
        marker_array.markers.append(marker)

        # Note: You may want a separate publisher for trajectory markers if needed
        # For now, trajectory is just published as Path

    def _create_pose_stamped(self, pose: tuple) -> PoseStamped:
        """Create PoseStamped from (x, y, theta) tuple."""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "odom"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = float(pose[0])
        pose_stamped.pose.position.y = float(pose[1])
        pose_stamped.pose.position.z = 0.0

        quaternion = tf_transformations.quaternion_from_euler(0, 0, pose[2])
        pose_stamped.pose.orientation.x = quaternion[0]
        pose_stamped.pose.orientation.y = quaternion[1]
        pose_stamped.pose.orientation.z = quaternion[2]
        pose_stamped.pose.orientation.w = quaternion[3]

        return pose_stamped

    # ==========================================================================
    # Goal Pose Visualization
    # ==========================================================================

    # Bright orange for goal pose marker
    GOAL_MARKER_COLOR = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)

    def _clear_goal_marker(self):
        """Publish a DELETE marker to clear the goal pose from RViz."""
        delete_marker = Marker()
        delete_marker.header.frame_id = "odom"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = self.GOAL_MARKER_NS
        delete_marker.id = self.GOAL_MARKER_ID
        delete_marker.action = Marker.DELETE
        arr = MarkerArray()
        arr.markers.append(delete_marker)
        self.goal_pose_marker_publisher.publish(arr)

    def _publish_goal_marker(self):
        """Publish the current goal pose as a bright orange arrow in odom frame."""
        if self.goal_odom is None:
            return
        gx, gy, theta = self.goal_odom[0], self.goal_odom[1], self.goal_odom[2]
        arrow_length = 0.5
        tip_x = gx + arrow_length * math.cos(theta)
        tip_y = gy + arrow_length * math.sin(theta)
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = self.GOAL_MARKER_NS
        marker.id = self.GOAL_MARKER_ID
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [
            Point(x=gx, y=gy, z=0.0),
            Point(x=tip_x, y=tip_y, z=0.0),
        ]
        marker.scale.x = 0.08
        marker.scale.y = 0.12
        marker.scale.z = 0.15
        marker.color = self.GOAL_MARKER_COLOR
        arr = MarkerArray()
        arr.markers.append(marker)
        self.goal_pose_marker_publisher.publish(arr)

    # ==========================================================================
    # Force Visualization
    # ==========================================================================

    def publish_force_visualization(
        self,
        robot_pose: tuple,
        v_attr: np.ndarray,
        v_rep: np.ndarray,
        v_total: np.ndarray,
    ):
        """Publish force visualization markers."""
        if robot_pose is None:
            return

        marker_array = MarkerArray()

        # Transform forces to odom frame
        cos_t = math.cos(robot_pose[2])
        sin_t = math.sin(robot_pose[2])

        def transform_to_odom(force: np.ndarray) -> tuple:
            return (
                force[0] * cos_t - force[1] * sin_t,
                force[0] * sin_t + force[1] * cos_t,
            )

        # Attractive force (green)
        marker = self._create_arrow_marker(
            robot_pose,
            v_attr,
            transform_to_odom,
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),
            z_offset=0.1,
            marker_id=0,
        )
        if marker:
            marker_array.markers.append(marker)

        # Repulsive force (red)
        marker = self._create_arrow_marker(
            robot_pose,
            v_rep,
            transform_to_odom,
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),
            z_offset=0.15,
            marker_id=1,
        )
        if marker:
            marker_array.markers.append(marker)

        # Total force (blue, thicker)
        marker = self._create_arrow_marker(
            robot_pose,
            v_total,
            transform_to_odom,
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
            z_offset=0.2,
            marker_id=2,
            scale_mult=1.5,
        )
        if marker:
            marker_array.markers.append(marker)

        # Force magnitude text markers (for debugging real vs simulation)
        mag_attr = float(np.linalg.norm(v_attr))
        mag_rep = float(np.linalg.norm(v_rep))
        mag_tot = float(np.linalg.norm(v_total))
        for text_marker in self._create_force_magnitude_text_markers(
            robot_pose, mag_attr, mag_rep, mag_tot
        ):
            marker_array.markers.append(text_marker)

        self.force_visualization_publisher.publish(marker_array)

        # Publish magnitudes for rqt_plot / recording (attr, rep, total)
        msg = Float64MultiArray()
        msg.data = [mag_attr, mag_rep, mag_tot]
        self.force_magnitudes_publisher.publish(msg)

    def _create_arrow_marker(
        self,
        robot_pose: tuple,
        force: np.ndarray,
        transform_fn,
        color: ColorRGBA,
        z_offset: float,
        marker_id: int,
        scale_mult: float = 1.0,
    ) -> Marker | None:
        """Create arrow marker for force visualization."""
        magnitude = np.linalg.norm(force)
        if magnitude < 0.01:
            return None

        fx, fy = transform_fn(force)
        arrow_length = min(
            magnitude * self.force_visualization_scale, self.force_arrow_max_length
        )

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = Point(x=robot_pose[0], y=robot_pose[1], z=z_offset)
        end = Point(
            x=robot_pose[0] + fx * arrow_length / magnitude,
            y=robot_pose[1] + fy * arrow_length / magnitude,
            z=z_offset,
        )
        marker.points = [start, end]

        marker.scale.x = 0.08 * scale_mult  # Shaft diameter
        marker.scale.y = 0.12 * scale_mult  # Head diameter
        marker.scale.z = 0.15 * scale_mult  # Head length
        marker.color = color

        return marker

    def _create_force_magnitude_text_markers(
        self, robot_pose: tuple, mag_attr: float, mag_rep: float, mag_tot: float
    ) -> list:
        """
        Create TEXT_VIEW_FACING markers showing force magnitudes for debugging.

        Returns
        -------
        list of Marker
            One marker with F_attr, F_rep, F_tot (ids 10, 11, 12 for namespace).
        """
        markers = []
        ns = "force_magnitudes"
        rx, ry, _ = robot_pose
        z_offset = 0.5
        scale_z = 0.12

        text = f"F_attr={mag_attr:.3f}\nF_rep={mag_rep:.3f}\nF_tot={mag_tot:.3f}"
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = 10
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = rx
        marker.pose.position.y = ry
        marker.pose.position.z = z_offset
        marker.pose.orientation.w = 1.0
        marker.scale.z = scale_z
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
        marker.text = text
        markers.append(marker)
        return markers

    # ==========================================================================
    # Main Control Loop
    # ==========================================================================

    def control_loop(self):
        """Main control loop - calculate and publish velocity commands."""
        robot_pose = self.get_robot_pose_in_odom()
        if robot_pose is None:
            return

        # Update trajectory
        self.update_trajectory(robot_pose)

        # No goal - stop
        if self.goal_odom is None:
            self._publish_stop()
            return

        # Goal already achieved - stop applying control (marker stays visible)
        if self.goal_achieved:
            self._publish_stop()
            return

        # Transform goal to base_link
        goal_base_link = self.transform_goal_to_base_link(robot_pose)
        if goal_base_link is None:
            return

        distance_to_goal = np.linalg.norm(goal_base_link)
        angle_to_goal = math.atan2(goal_base_link[1], goal_base_link[0])
        angle_error_final = abs(self.normalize_angle(self.goal_odom[2] - robot_pose[2]))

        # Check if goal reached
        if distance_to_goal < self.goal_dist_tolerance:
            # Distance reached - now align orientation
            if angle_error_final >= self.goal_angle_tolerance:
                if not self.orientation_phase_logged:
                    self.logger.info(
                        f"Position reached! Now aligning orientation. "
                        f"Current error: {math.degrees(angle_error_final):.1f}°"
                    )
                    self.orientation_phase_logged = True

                linear_x, angular_z = self.calculate_orientation_velocity(robot_pose)
                self._publish_velocity(linear_x, angular_z)
                return
            else:
                # Goal fully reached - forget for control, keep visualization
                self.goal_achieved = True
                self._publish_stop()
                if not self.goal_reached_logged:
                    self.logger.info("Goal reached!")
                    self.goal_reached_logged = True
                return

        # Calculate potential field forces
        v_attr = self.calculate_attractive_force(goal_base_link)
        v_rep = self.calculate_repulsive_force()
        v_total = v_attr + v_rep

        # Visualize forces
        self.publish_force_visualization(robot_pose, v_attr, v_rep, v_total)

        # Check stuck status
        self.check_stuck_status(robot_pose)

        # Calculate velocity
        linear_x, angular_z = self.calculate_velocity(
            v_total, distance_to_goal, angle_to_goal
        )

        # Publish velocity
        self._publish_velocity(linear_x, angular_z)

    def _publish_stop(self):
        """Publish zero velocity."""
        self.cmd_vel_publisher.publish(Twist())

    def _publish_velocity(self, linear_x: float, angular_z: float):
        """Publish velocity command."""
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.cmd_vel_publisher.publish(cmd)

    def stop_robot(self):
        """Stop robot safely on shutdown."""
        try:
            self._clear_goal_marker()
            self._publish_stop()
            self.logger.info("Robot stopped")
        except Exception:
            pass


# ==============================================================================
# Main Entry Point
# ==============================================================================


def _parse_config_args(args=None):
    """
    Parse command line for --config <path> and convert to ROS2 --params-file.

    Allows both standalone usage (--config path/to/config.yaml) and standard
    ROS2 usage (--ros-args --params-file path). When the node is launched via
    a launch file with parameters, the launch file passes parameters directly
    and no --config is needed.

    Parameters
    ----------
    args : list, optional
        Command line arguments (defaults to sys.argv).

    Returns
    -------
    list
        Arguments to pass to rclpy.init(); --config and its value are
        replaced by --ros-args --params-file <path>.
    """
    if args is None:
        args = sys.argv
    argv = list(args)
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config":
            if i + 1 < len(argv):
                path = argv[i + 1]
                out.append("--ros-args")
                out.append("--params-file")
                out.append(path)
                i += 2
                continue
        out.append(argv[i])
        i += 1
    return out


def main(args=None):
    """
    Run the simple potential field planner node.

    Parameters
    ----------
    args : list, optional
        Command line arguments. Use --config <path> to load a YAML config file,
        or --ros-args --params-file <path> (ROS2 standard). When run from a
        launch file, parameters are typically passed by the launch file.
    """
    ros_args = _parse_config_args(args)
    rclpy.init(args=ros_args)
    planner = SimplePotentialFieldPlanner()

    def signal_handler(sig, frame):
        planner.logger.info("Shutdown signal received")
        planner.stop_robot()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.logger.info("Shutting down...")
        planner.stop_robot()
    except Exception as e:
        planner.logger.error(f"Error: {e}")
        planner.stop_robot()
    finally:
        planner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(args=sys.argv)
