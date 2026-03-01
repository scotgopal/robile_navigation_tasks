#!/usr/bin/env python3
"""
Waypoint Follower Node

Subscribes to /waypoints (PoseStamped) to collect waypoints from the
A* planner, then publishes them one-by-one to /goal_pose.  After each
waypoint is published, the node waits for a goal-reached confirmation
on /goal_status (GoalStatusArray) before advancing to the next waypoint.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatusArray, GoalStatus


class WaypointFollower(Node):

    def __init__(self):
        super().__init__('waypoint_follower')

        # ---- State -----------------------------------------------------
        self.waypoints: list[PoseStamped] = []
        self.current_index: int = 0
        self.navigating: bool = False

        # After publishing a new goal we ignore status messages for a
        # short cooldown so stale SUCCEEDED statuses from the *previous*
        # goal don't cause an immediate skip.
        self._cooldown_active: bool = False
        self._cooldown_timer = None

        # Timer used to detect the end of a waypoint batch (all waypoints
        # are published in rapid succession on /waypoints).
        self._batch_timer = None

        # ---- Publisher --------------------------------------------------
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # ---- Subscribers ------------------------------------------------
        self.waypoint_sub = self.create_subscription(
            PoseStamped, '/waypoints', self._waypoint_cb, 10
        )
        self.status_sub = self.create_subscription(
            GoalStatusArray, '/goal_status', self._status_cb, 10
        )

        self.get_logger().info('Waypoint Follower initialised')
        self.get_logger().info('  Listening for waypoints on /waypoints')
        self.get_logger().info('  Publishing goals to    /goal_pose')
        self.get_logger().info('  Monitoring status on   /goal_status')

    # ------------------------------------------------------------------
    # /waypoints callback  (PoseStamped, many arrive in quick succession)
    # ------------------------------------------------------------------
    def _waypoint_cb(self, msg: PoseStamped):
        """Collect incoming waypoints into a batch."""

        # If a new set of waypoints arrives while navigating, reset
        if self.navigating:
            self.get_logger().info(
                'New waypoints received — resetting navigation'
            )
            self.navigating = False
            self.current_index = 0
            self.waypoints = []
            self._cancel_cooldown()

        self.waypoints.append(msg)

        # (Re)start the batch-collection timer.  Once 0.5 s passes with
        # no new waypoint, we consider the batch complete and start
        # navigating.
        if self._batch_timer is not None:
            self._batch_timer.cancel()
            self.destroy_timer(self._batch_timer)
        self._batch_timer = self.create_timer(0.5, self._on_batch_complete)

    # ------------------------------------------------------------------
    def _on_batch_complete(self):
        """Called ~0.5 s after the last waypoint in a batch."""
        # One-shot: cancel immediately
        if self._batch_timer is not None:
            self._batch_timer.cancel()
            self.destroy_timer(self._batch_timer)
            self._batch_timer = None

        if len(self.waypoints) == 0:
            return

        self.get_logger().info(
            f'Batch complete — {len(self.waypoints)} waypoints collected. '
            f'Starting navigation.'
        )
        self.navigating = True
        self.current_index = 0
        self._publish_current_waypoint()

    # ------------------------------------------------------------------
    # Goal publishing
    # ------------------------------------------------------------------
    def _publish_current_waypoint(self):
        """Publish the waypoint at ``current_index`` to /goal_pose."""
        if self.current_index >= len(self.waypoints):
            self.get_logger().info('All waypoints reached!')
            self.navigating = False
            return

        wp = self.waypoints[self.current_index]
        # Refresh the timestamp so Nav2 accepts it
        wp.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(wp)

        self.get_logger().info(
            f'Published waypoint {self.current_index + 1}/{len(self.waypoints)}: '
            f'({wp.pose.position.x:.2f}, {wp.pose.position.y:.2f})'
        )

        # Start cooldown — ignore status callbacks for 2 s so that a
        # stale SUCCEEDED from the previous goal doesn't trigger a skip.
        self._start_cooldown()

    # ------------------------------------------------------------------
    # Cooldown helpers (prevent stale status from skipping a waypoint)
    # ------------------------------------------------------------------
    def _start_cooldown(self):
        self._cancel_cooldown()
        self._cooldown_active = True
        self._cooldown_timer = self.create_timer(2.0, self._end_cooldown)

    def _end_cooldown(self):
        """One-shot timer callback."""
        if self._cooldown_timer is not None:
            self._cooldown_timer.cancel()
            self.destroy_timer(self._cooldown_timer)
            self._cooldown_timer = None
        self._cooldown_active = False

    def _cancel_cooldown(self):
        self._cooldown_active = False
        if self._cooldown_timer is not None:
            self._cooldown_timer.cancel()
            self.destroy_timer(self._cooldown_timer)
            self._cooldown_timer = None

    # ------------------------------------------------------------------
    # /goal_status callback
    # ------------------------------------------------------------------
    def _status_cb(self, msg: GoalStatusArray):
        """Advance to the next waypoint when the current goal succeeds."""
        if not self.navigating or self._cooldown_active:
            return

        if len(msg.status_list) == 0:
            return

        latest = msg.status_list[-1]

        if latest.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f'Waypoint {self.current_index + 1}/{len(self.waypoints)} '
                f'reached!'
            )
            self.current_index += 1
            self._publish_current_waypoint()

        elif latest.status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn(
                f'Waypoint {self.current_index + 1}/{len(self.waypoints)} '
                f'ABORTED — skipping to next waypoint'
            )
            self.current_index += 1
            self._publish_current_waypoint()

        elif latest.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn(
                'Navigation CANCELED — stopping waypoint follower'
            )
            self.navigating = False
            self._cancel_cooldown()


# ============================================================================
# Entry point
# ============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Waypoint Follower')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
