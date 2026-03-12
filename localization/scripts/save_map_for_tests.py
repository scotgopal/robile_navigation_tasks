#!/usr/bin/env python3
"""
Subscribe once to /map and save grid + metadata as .npz for unit tests.
Run while your simulation (or map server) is running:

  cd robile_navigation_tasks/localization && python3 scripts/save_map_for_tests.py

Output: tests/fixtures/map_sample.npz (grid, origin_x, origin_y, resolution, width, height).
"""

import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)


def main():
    rclpy.init()
    node = Node("save_map_once")
    received = []

    def cb(msg: OccupancyGrid):
        received.append(msg)

    # Match /map publisher: latched (TRANSIENT_LOCAL), RELIABLE, so we get the map even if we join after it was published.
    qos_map = QoSProfile(
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    node.create_subscription(
        OccupancyGrid,
        "/map",
        cb,
        qos_map,
    )
    node.get_logger().info("Waiting for one /map message (run your sim or map server)...")
    timeout_sec = 15
    t0 = time.monotonic()
    while not received and (time.monotonic() - t0) < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.5)
    if not received:
        node.get_logger().error("No /map message received in 15 s. Is the sim running and /map published?")
        rclpy.shutdown()
        return 1
    msg = received[0]
    grid = np.array(msg.data, dtype=np.int32).reshape(msg.info.height, msg.info.width)
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "map_sample.npz"
    np.savez(
        out_file,
        grid=grid,
        origin_x=float(msg.info.origin.position.x),
        origin_y=float(msg.info.origin.position.y),
        resolution=float(msg.info.resolution),
        width=int(msg.info.width),
        height=int(msg.info.height),
    )
    node.get_logger().info(f"Saved map to {out_file} (shape {grid.shape}, res={msg.info.resolution})")
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    exit(main())
