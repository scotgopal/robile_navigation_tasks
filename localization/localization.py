#!/usr/bin/env python3
"""Thin launcher for MCL / AMCL localization nodes."""

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import rclpy  # noqa: E402
from rclpy.logging import LoggingSeverity, set_logger_level  # noqa: E402

_DEFAULT_CONFIGS = {
    "mcl":  os.path.join(_script_dir, "config", "mcl_params.yaml"),
    "amcl": os.path.join(_script_dir, "config", "amcl_params.yaml"),
}

_NODE_CLASSES = {
    "mcl":  ("mcl_node",  "MCLNode"),
    "amcl": ("amcl_node", "AMCLNode"),
}


def parse_args(args=None):
    """
    Parse ``--method``, ``--config``, and ``--debug`` flags.

    Parameters
    ----------
    args : list of str or None
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    method : str
        ``"mcl"`` or ``"amcl"``.
    debug : bool
        Whether debug logging was requested.
    ros_args : list of str
        Remaining argument list suitable for ``rclpy.init(args=...)``.
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Launch an MCL or AMCL localization node. "
                    "Use --method to select the algorithm and "
                    "--config to pass a params YAML.")
    parser.add_argument(
        "--method", choices=["mcl", "amcl"], default="amcl",
        help="Localization algorithm to use (default: amcl)")
    parser.add_argument(
        "--config", metavar="FILE", default=None,
        help="Path to params YAML. Defaults to the built-in config "
             "for the chosen method.")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging")
    parsed, remaining = parser.parse_known_args(args)

    config_path = parsed.config or _DEFAULT_CONFIGS[parsed.method]

    ros_args = ["--ros-args", "--params-file", config_path] + remaining
    return parsed.method, parsed.debug, ros_args


def parse_config_args(args=None, *, method="amcl"):
    """
    Backward-compatible helper for standalone node entry points.

    Parses ``--config`` and ``--debug`` and returns a flat list of ROS args
    (with ``--debug`` left in-band so callers can strip it themselves).

    Parameters
    ----------
    args : list of str or None
        Command-line arguments.  Defaults to ``sys.argv[1:]``.
    method : str
        ``"mcl"`` or ``"amcl"`` — used to pick the default config file
        when ``--config`` is not provided.

    Returns
    -------
    list of str
        Argument list suitable for ``rclpy.init(args=...)``, with an
        optional ``--debug`` sentinel that callers should strip.
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", metavar="FILE", default=None)
    parsed, remaining = parser.parse_known_args(args)

    config_path = parsed.config or _DEFAULT_CONFIGS[method]

    out = ["--ros-args", "--params-file", config_path]
    out.extend(remaining)
    return out


def _import_node_class(method):
    """Lazily import and return the node class for *method*."""
    module_name, class_name = _NODE_CLASSES[method]
    module = __import__(module_name)
    return getattr(module, class_name)


def main(args=None):
    method, debug, ros_args = parse_args(args)
    rclpy.init(args=ros_args)

    NodeClass = _import_node_class(method)
    node = NodeClass()

    if debug:
        set_logger_level(node.get_name(), LoggingSeverity.DEBUG)
        node.log.debug("Debug logging enabled.")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.log.info("Shutting down (keyboard interrupt).")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(args=sys.argv[1:])
