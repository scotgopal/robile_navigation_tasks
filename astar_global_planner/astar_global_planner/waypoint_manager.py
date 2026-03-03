#!/usr/bin/env python3
"""
Waypoint Manager Node

Subscribes to /waypoints (PoseStamped) and forwards them one-by-one
to /local_goal_pose.  The first waypoint is published immediately.  Every
subsequent waypoint is queued and only published after the previous
one is confirmed as reached via /goal_status (GoalStatusArray).

Goal identification:
  Each goal in the GoalStatusArray carries a unique UUID (goal_id).
  When a new waypoint is published, the manager records the set of
  known goal IDs.  Any *new* goal ID that appears is attributed to
  the current waypoint, ensuring correct identification even when
  multiple goals exist in the status list.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatusArray, GoalStatus
from unique_identifier_msgs.msg import UUID as UuidMsg


class WaypointManager(Node):
    def __init__(self):
        super().__init__("waypoint_manager")

        # ---- Parameters ------------------------------------------------
        self.declare_parameter("cooldown_sec", 2.0)

        self._cooldown_sec = (
            self.get_parameter("cooldown_sec").get_parameter_value().double_value
        )

        # ---- State -----------------------------------------------------
        self._queue: list = []  # [(PoseStamped, int), ...] waiting
        self._active_wp = None  # waypoint currently navigating to
        self._wp_count: int = 0  # total waypoints received (for logging)
        self._active_wp_number: int = 0  # 1-based index of the active waypoint

        # Set of goal UUIDs that were already present *before* the
        # current waypoint was published.  Used to detect new goals.
        self._known_goal_ids: set = set()
        # The UUID assigned to the current in-flight waypoint (filled in
        # once a new goal ID appears in the status list).
        self._current_goal_id = None

        # Cooldown: ignore status messages briefly after publishing a new
        # goal so stale SUCCEEDED from the previous goal is not misread.
        self._cooldown_active: bool = False
        self._cooldown_timer = None

        # ---- Publisher --------------------------------------------------
        self.goal_pub = self.create_publisher(PoseStamped, "/local_goal_pose", 10)

        # ---- Subscribers ------------------------------------------------
        self.waypoint_sub = self.create_subscription(
            PoseStamped, "/waypoints", self._waypoint_cb, 10
        )
        self.status_sub = self.create_subscription(
            GoalStatusArray, "/goal_status", self._status_cb, 10
        )

        self.get_logger().info("Waypoint Manager initialised")
        self.get_logger().info("  Listening for waypoints on  /waypoints")
        self.get_logger().info("  Publishing goals to         /local_goal_pose")
        self.get_logger().info("  Monitoring status on        /goal_status")

    # ==================================================================
    # /waypoints callback
    # ==================================================================
    def _waypoint_cb(self, msg: PoseStamped):
        """Handle an incoming waypoint.

        If no waypoint is currently being navigated, publish it
        immediately.  Otherwise queue it and wait for the active
        waypoint to be confirmed before publishing the next one.
        """
        self._wp_count += 1
        self.get_logger().info(
            f"Received waypoint #{self._wp_count}: "
            f"({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

        if self._active_wp is None:
            # Nothing in-flight — publish immediately
            self._publish_waypoint(msg, self._wp_count)
        else:
            # Currently navigating — queue for later
            self._queue.append((msg, self._wp_count))
            self.get_logger().info(
                f"  Queued (waiting for current goal to be reached). "
                f"Queue size: {len(self._queue)}"
            )

    # ==================================================================
    # Goal publishing
    # ==================================================================
    def _publish_waypoint(self, wp: PoseStamped, wp_number: int):
        """Publish a single waypoint to /local_goal_pose and mark it active."""
        self._active_wp = wp
        self._active_wp_number = wp_number
        self._current_goal_id = None  # will be detected in status callback

        wp.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(wp)

        self.get_logger().info(
            f"[Waypoint {wp_number}] Published goal: "
            f"x={wp.pose.position.x:.2f}, y={wp.pose.position.y:.2f}"
        )

        # Start cooldown to avoid processing stale status
        self._start_cooldown()

    def _advance_to_next(self):
        """Publish the next queued waypoint, or finish if queue is empty."""
        if self._queue:
            next_wp, next_number = self._queue.pop(0)
            self._publish_waypoint(next_wp, next_number)
        else:
            self._active_wp = None
            self._cancel_cooldown()
            self.get_logger().info(
                "=== All waypoints reached! Navigation complete. ==="
            )

    # ==================================================================
    # /goal_status callback
    # ==================================================================
    def _status_cb(self, msg: GoalStatusArray):
        """Check whether the current goal has been reached or failed."""
        if self._active_wp is None:
            return

        if not msg.status_list:
            return

        # During cooldown, only process if we see SUCCEEDED for a new goal ID.
        # (Planner may publish SUCCEEDED only once when reached; we must not drop it.)
        if self._cooldown_active:
            has_new_succeeded = any(
                s.status == GoalStatus.STATUS_SUCCEEDED
                and bytes(s.goal_info.goal_id.uuid) not in self._known_goal_ids
                for s in msg.status_list
            )
            if not has_new_succeeded:
                return

        # ---- Identify the current goal by its UUID -------------------
        # After the cooldown expires the first new goal ID we see is the
        # one we just published. During cooldown we only get here for new SUCCEEDED.
        current_ids = {
            bytes(status.goal_info.goal_id.uuid) for status in msg.status_list
        }

        if self._current_goal_id is None:
            new_ids = current_ids - self._known_goal_ids
            if new_ids:
                self._current_goal_id = new_ids.pop()
                self._known_goal_ids.update(current_ids)
                self.get_logger().info(
                    f"[Waypoint {self._active_wp_number}] Tracking goal ID: "
                    f"{self._uuid_to_hex(self._current_goal_id)}"
                )

        if self._current_goal_id is None:
            return

        # ---- Find the status entry for our tracked goal ---------------
        for status in msg.status_list:
            goal_id = bytes(status.goal_info.goal_id.uuid)
            if goal_id != self._current_goal_id:
                continue

            if status.status == GoalStatus.STATUS_SUCCEEDED:
                wp = self._active_wp
                self.get_logger().info(
                    f"[Waypoint {self._active_wp_number}] "
                    f"REACHED — "
                    f"x={wp.pose.position.x:.2f}, y={wp.pose.position.y:.2f}  "
                    f"(id: {self._uuid_to_hex(goal_id)})"
                )
                self._advance_to_next()
                return

            elif status.status == GoalStatus.STATUS_ABORTED:
                wp = self._active_wp
                self.get_logger().warn(
                    f"[Waypoint {self._active_wp_number}] "
                    f"ABORTED — "
                    f"x={wp.pose.position.x:.2f}, y={wp.pose.position.y:.2f}  "
                    f"(id: {self._uuid_to_hex(goal_id)})  "
                    f"Skipping to next waypoint."
                )
                self._advance_to_next()
                return

            elif status.status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn(
                    f"[Waypoint {self._active_wp_number}] "
                    f"CANCELED (id: {self._uuid_to_hex(goal_id)}) — "
                    f"stopping navigation."
                )
                self._reset_navigation()
                return

            # STATUS_EXECUTING / STATUS_UNKNOWN / STATUS_ACCEPTED →
            # still in progress, keep waiting.
            break

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _uuid_to_hex(uuid_bytes: bytes) -> str:
        """Return a short hex representation of a 16-byte UUID."""
        return uuid_bytes.hex()[:12] + "..."

    def _reset_navigation(self):
        self._active_wp = None
        self._queue.clear()
        self._known_goal_ids.clear()
        self._current_goal_id = None
        self._cancel_cooldown()

    # ---- Cooldown ----------------------------------------------------
    def _start_cooldown(self):
        self._cancel_cooldown()
        self._cooldown_active = True
        self._cooldown_timer = self.create_timer(self._cooldown_sec, self._end_cooldown)

    def _end_cooldown(self):
        if self._cooldown_timer is not None:
            self._cooldown_timer.cancel()
            self.destroy_timer(self._cooldown_timer)
            self._cooldown_timer = None
        self._cooldown_active = False
        self.get_logger().debug("Cooldown ended — now monitoring goal status")

    def _cancel_cooldown(self):
        self._cooldown_active = False
        if self._cooldown_timer is not None:
            self._cooldown_timer.cancel()
            self.destroy_timer(self._cooldown_timer)
            self._cooldown_timer = None


# =====================================================================
# Entry point
# =====================================================================


def main(args=None):
    rclpy.init(args=args)
    node = WaypointManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Waypoint Manager")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
