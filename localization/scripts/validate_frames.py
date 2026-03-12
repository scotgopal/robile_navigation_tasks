#!/usr/bin/env python3
"""
Validate frames and topics used by MCL: /scan frame_id and laser origin, /map frame and metadata,
and TF base_footprint -> scan_frame. Run with your simulation (and optionally MCL) running:

  cd robile_navigation_tasks/localization && python3 scripts/validate_frames.py

Prints a report and a short checklist you can use to confirm frames are correct.
"""

import sys
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from tf2_ros import Buffer, TransformListener


# Default frame names used by MCL (override via params or env if needed)
BASE_FRAME = "base_footprint"
MAP_FRAME = "map"
ODOM_FRAME = "odom"


def main():
    rclpy.init()
    node = Node("validate_frames")
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    scan_msg = [None]
    map_msg = [None]

    def cb_scan(msg: LaserScan):
        scan_msg[0] = msg

    def cb_map(msg: OccupancyGrid):
        map_msg[0] = msg

    qos_map = QoSProfile(
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    qos_scan = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    node.create_subscription(LaserScan, "/scan", cb_scan, qos_scan)
    node.create_subscription(OccupancyGrid, "/map", cb_map, qos_map)

    node.get_logger().info("Waiting for /scan and /map (run your sim)...")
    timeout = 15.0
    t0 = time.monotonic()
    while (time.monotonic() - t0) < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
        if scan_msg[0] is not None and map_msg[0] is not None:
            break
    if scan_msg[0] is None or map_msg[0] is None:
        node.get_logger().error("Did not get both /scan and /map in time.")
        rclpy.shutdown()
        return 1

    scan = scan_msg[0]
    map_ = map_msg[0]

    # --- Report ---
    lines = []
    lines.append("")
    lines.append("============ MCL frame validation report ============")
    lines.append("")

    # Scan
    scan_frame = getattr(scan.header, "frame_id", "") or ""
    n_beams = len(scan.ranges)
    angle_min = getattr(scan, "angle_min", 0.0)
    angle_increment = getattr(scan, "angle_increment", 0.0)
    angle_max = getattr(scan, "angle_max", 0.0)
    range_min = getattr(scan, "range_min", 0.0)
    range_max = getattr(scan, "range_max", float("inf"))

    lines.append("[1] /scan")
    lines.append(f"    header.frame_id:  {repr(scan_frame)}")
    lines.append(f"    num beams:        {n_beams}")
    lines.append(f"    angle_min:       {angle_min:.4f} rad")
    lines.append(f"    angle_increment:  {angle_increment:.6f} rad")
    lines.append(f"    angle_max:       {angle_max:.4f} rad")
    lines.append(f"    range_min/max:   {range_min:.3f} / {range_max:.3f} m")
    lines.append("")

    # Map
    map_frame = getattr(map_.header, "frame_id", "") or ""
    ox = map_.info.origin.position.x
    oy = map_.info.origin.position.y
    res = map_.info.resolution
    w = map_.info.width
    h = map_.info.height
    lines.append("[2] /map")
    lines.append(f"    header.frame_id: {repr(map_frame)}")
    lines.append(f"    origin (x,y):    ({ox:.4f}, {oy:.4f}) m")
    lines.append(f"    resolution:      {res:.6f} m/cell")
    lines.append(f"    width x height:  {w} x {h} cells")
    lines.append("")

    # TF base_footprint -> scan frame (laser origin in base)
    # First lookups often fail while the buffer fills; retry several times before concluding failure.
    lines.append("[3] TF laser origin (base_footprint -> scan frame)")
    if not scan_frame:
        lines.append("    Scan has no frame_id; MCL will use laser_origin_x/y from params (default 0,0).")
    elif scan_frame == BASE_FRAME:
        lines.append(f"    Scan frame equals base_frame ({BASE_FRAME}); laser origin = (0, 0). OK.")
    else:
        t = None
        num_retries = 5
        for attempt in range(num_retries):
            try:
                t = tf_buffer.lookup_transform(
                    BASE_FRAME,
                    scan_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
                break
            except Exception:
                if attempt < num_retries - 1:
                    time.sleep(0.5)
                    for _ in range(5):
                        rclpy.spin_once(node, timeout_sec=0.1)
        if t is not None:
            lox = t.transform.translation.x
            loy = t.transform.translation.y
            lines.append(f"    Lookup {BASE_FRAME} -> {scan_frame}:  translation (x,y) = ({lox:.4f}, {loy:.4f}) m")
            lines.append("    -> Set in mcl_params.yaml:  laser_origin_x: {:.4f},  laser_origin_y: {:.4f}".format(lox, loy))
        else:
            lines.append(f"    Lookup FAILED after {num_retries} attempts (early TF lookups often fail; run script again or set laser_origin_x/y from URDF).")
            lines.append("    MCL will use laser_origin_x/y from params (default 0,0). If laser is not at base, set them manually.")
    lines.append("")

    # TF map -> odom -> base (optional); retry a few times (buffer may not have it yet)
    lines.append("[4] TF map -> odom -> base_footprint (optional)")
    map_ok = False
    for attempt in range(3):
        try:
            tf_buffer.lookup_transform(MAP_FRAME, BASE_FRAME, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            map_ok = True
            break
        except Exception:
            if attempt < 2:
                time.sleep(0.3)
                for _ in range(3):
                    rclpy.spin_once(node, timeout_sec=0.1)
    if map_ok:
        lines.append("    map -> base_footprint: available (e.g. from MCL or static map).")
    else:
        lines.append("    map -> base_footprint: not available. Normal if MCL not running or no initial pose yet.")
    lines.append("")

    lines.append("============ Checklist (answer for yourself) ============")
    lines.append("")
    lines.append("(a) Is the map you use in MCL the same as the simulation environment?")
    lines.append("    (Same file / same map_server? Wrong map => particles never converge.)")
    lines.append("")
    lines.append("(b) Does the scan frame_id match base_footprint, or do you have TF base_footprint -> scan_frame?")
    lines.append("    If not: set laser_origin_x and laser_origin_y in mcl_params.yaml to the laser position in base frame.")
    lines.append("")
    lines.append("(c) In RViz, when you set '2D Pose Estimate', are you clicking roughly where the robot actually is in the sim?")
    lines.append("    (Large initial error can prevent convergence.)")
    lines.append("")
    lines.append("(d) Run:  ros2 topic echo /scan --once   and  ros2 topic echo /map --once")
    lines.append("    Compare header.frame_id and map origin/resolution with the report above.")
    lines.append("")
    lines.append("==========================================================")
    lines.append("")

    for line in lines:
        print(line)
        node.get_logger().info(line.strip() if line.strip() else "")

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
