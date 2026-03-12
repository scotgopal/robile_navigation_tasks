#!/usr/bin/env python3
"""Thin launcher for the MCL localization node."""

import argparse
import os
import sys

# Allow importing mcl_node from the same directory when run from anywhere
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import rclpy  # noqa: E402
from rclpy.logging import LoggingSeverity, set_logger_level  # noqa: E402
from mcl_node import MCLNode  # noqa: E402


def parse_config_args(args=None):
    """
    Parse command line for --config <path> and -h/--help; convert --config to ROS2 --params-file.

    Allows
      python3 localization.py --config config/mcl_params.yaml
      python3 localization.py -h
    or standard:
      python3 localization.py --ros-args --params-file config/mcl_params.yaml
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Launch MCL (Monte Carlo Localization) node. Use --config to pass a params YAML."
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="Path to MCL params YAML (emits --ros-args --params-file FILE)",
    )
    parsed, remaining = parser.parse_known_args(args)
    out = []
    if parsed.config is not None:
        out.append("--ros-args")
        out.append("--params-file")
        out.append(parsed.config)
    out.extend(remaining)
    return out


def main(args=None):
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
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(args=sys.argv[1:])
